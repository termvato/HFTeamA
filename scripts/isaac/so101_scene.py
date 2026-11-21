"""Three SO-101 arms on a table in Isaac Sim, driven by joint positions, rendered with RTX.

Feasibility spike for issue #33 (RTX rehearsal, sim first then real). Sim only: this script
never talks to a real arm or a serial port.

What it does:
  1. Converts the MuJoCo Menagerie SO-101 MJCF (robotstudio_so101/so101.xml, Apache-2.0) to USD
     with the Isaac Sim MJCF importer, once, into a cache folder.
  2. Builds a stage: floor, table, 3 SO-101 arms at different base poses, small cubes, 2 bins.
  3. Drives every arm by setting joint positions each frame (scripted sweep, or a live joint
     stream over UDP with --listen).
  4. Renders with RTX (RaytracedLighting) at 1280x720, saves a few PNG frames, and prints
     startup time, frames per second and GPU memory (nvidia-smi).

Run (from any folder; ROS on PYTHONPATH breaks Isaac's Python, hence env -u):
  env -u PYTHONPATH <isaacsim>/python.sh scripts/isaac/so101_scene.py \
      --mjcf <menagerie>/robotstudio_so101/so101.xml --out /tmp/isaac_frames
Options:
  --gui            open the Isaac Sim window instead of headless (live view for a demo screen)
  --frames N       number of simulated frames (default 300)
  --save-every K   save a PNG every K frames (default 60; 0 = never)
  --listen PORT    also read joint positions from UDP 127.0.0.1:PORT (see joint_stream_sender.py);
                   arms with no recent packet keep running the scripted sweep
  --cache DIR      where the converted USD goes (default ~/.cache/hfteama_isaac)

Gotchas found in the spike (Isaac Sim 5.1, MJCF importer 2.5.13):
  - the importer's fixed root joint is anchored at the world origin; moved arms explode unless it is
    re-anchored at the base pose (done below);
  - colliders in the scene push the teleported arms and also make them explode, so the table, cubes
    and bins are visual only and gravity is off (kinematic playback, no contacts, no grasping);
  - the articulation root ends up on <arm>/base/base, not on <arm>/worldBody.
"""

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time

T0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mjcf", default=os.environ.get("SO101_MJCF", ""), help="path to robotstudio_so101/so101.xml"
)
parser.add_argument("--out", default="/tmp/isaac_frames")
parser.add_argument("--cache", default=os.path.expanduser("~/.cache/hfteama_isaac"))
parser.add_argument("--gui", action="store_true")
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--save-every", type=int, default=60)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--listen", type=int, default=0)
args, _unknown = parser.parse_known_args()
if not args.mjcf or not os.path.isfile(args.mjcf):
    sys.exit("give --mjcf <menagerie>/robotstudio_so101/so101.xml (or set SO101_MJCF)")


def gpu_mem_mib():
    """Total GPU memory in use (all processes), in MiB, from nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


GPU_BEFORE = gpu_mem_mib()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "renderer": "RaytracedLighting",
        "width": args.width,
        "height": args.height,
    }
)
T_APP = time.time()

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

# ---------------------------------------------------------------- 1. MJCF -> USD (cached)
os.makedirs(args.cache, exist_ok=True)
usd_path = os.path.join(args.cache, "so101", "so101.usd")
if not os.path.isfile(usd_path):
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    _, cfg = omni.kit.commands.execute("MJCFCreateImportConfig")
    cfg.set_fix_base(True)
    cfg.set_import_inertia_tensor(True)
    cfg.set_make_default_prim(True)
    cfg.set_create_physics_scene(False)
    cfg.set_self_collision(False)
    omni.kit.commands.execute(
        "MJCFCreateAsset",
        mjcf_path=args.mjcf,
        import_config=cfg,
        prim_path="/so101",
        dest_path=usd_path,
    )
    print(f"[so101_scene] converted MJCF -> {usd_path}", flush=True)
T_IMPORT = time.time()

# ---------------------------------------------------------------- 2. stage
create_new_stage()
simulation_app.update()
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
world.get_physics_context().set_gravity(0.0)  # kinematic playback: joints are set, not simulated
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

world.scene.add_default_ground_plane()
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(400)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1200)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40, 20, 0))

TABLE_Z = 0.75  # table top height (m)
# Visual only (no collider): during kinematic playback the arms must not be pushed by contacts.
VisualCuboid(
    "/World/table",
    name="table",
    position=np.array([0.0, 0.0, TABLE_Z - 0.02]),
    scale=np.array([1.4, 0.8, 0.04]),
    color=np.array([0.55, 0.42, 0.30]),
)

# Arm bases: (x, y, yaw in degrees). Arms sit along the back edge and face the front of the table.
ARM_POSES = [(-0.45, 0.25, -90.0), (0.0, 0.28, -90.0), (0.45, 0.25, -90.0)]
for i, (x, y, yaw) in enumerate(ARM_POSES):
    prim = add_reference_to_stage(usd_path=usd_path, prim_path=f"/World/arm_{i}")
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, TABLE_Z))
    xf.AddRotateZOp().Set(yaw)
    # The importer's fixed root joint anchors the base to the WORLD ORIGIN (localPos0 = 0). With the
    # arm moved, that joint would fight the placement and the arm explodes, so re-anchor it here.
    root_joint = stage.GetPrimAtPath(f"/World/arm_{i}/joints/rootJoint_base")
    root_joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(x, y, TABLE_Z))
    half = math.radians(yaw) / 2
    root_joint.GetAttribute("physics:localRot0").Set(
        Gf.Quatf(math.cos(half), 0.0, 0.0, math.sin(half))
    )

# Cubes in front of the arms, bins between them.
CUBE_COLORS = [
    (0.85, 0.15, 0.15),
    (0.15, 0.6, 0.2),
    (0.15, 0.3, 0.85),
    (0.9, 0.75, 0.1),
    (0.6, 0.2, 0.7),
]
for k, c in enumerate(CUBE_COLORS):
    VisualCuboid(
        f"/World/cube_{k}",
        position=np.array([-0.5 + 0.25 * k, 0.0, TABLE_Z + 0.0125]),
        scale=np.array([0.025, 0.025, 0.025]),
        color=np.array(c),
    )


def make_bin(name, cx, cy, w=0.14, d=0.10, h=0.05, t=0.004, color=(0.25, 0.25, 0.28)):
    col = np.array(color)
    parts = {
        "floor": ((cx, cy, TABLE_Z + t / 2), (w, d, t)),
        "wall_n": ((cx, cy + d / 2, TABLE_Z + h / 2), (w, t, h)),
        "wall_s": ((cx, cy - d / 2, TABLE_Z + h / 2), (w, t, h)),
        "wall_e": ((cx + w / 2, cy, TABLE_Z + h / 2), (t, d, h)),
        "wall_w": ((cx - w / 2, cy, TABLE_Z + h / 2), (t, d, h)),
    }
    for part, (pos, size) in parts.items():
        VisualCuboid(
            f"/World/{name}/{part}", position=np.array(pos), scale=np.array(size), color=col
        )


make_bin("bin_a", -0.22, 0.12)
make_bin("bin_b", 0.22, 0.12)
make_bin("bin_out", 0.0, -0.2, w=0.2, color=(0.1, 0.35, 0.45))

# The importer's PhysX variant puts the articulation root on the base link.
arms = Articulation(prim_paths_expr="/World/arm_.*/base/base", name="so101_arms")
world.scene.add(arms)

# Camera + render product at the requested resolution (this is what we capture and time).
CAM_EYE, CAM_TARGET = (0.0, -1.25, TABLE_Z + 0.75), (0.0, 0.08, TABLE_Z + 0.05)
cam = rep.create.camera(position=CAM_EYE, look_at=CAM_TARGET)
rp = rep.create.render_product(cam, (args.width, args.height))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

if args.gui:  # point the GUI viewport at the same view as the capture camera
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=np.array(CAM_EYE), target=np.array(CAM_TARGET))

world.reset()
dof_names = list(arms.dof_names)
n_arms, n_dof = arms.count, arms.num_dof
print(f"[so101_scene] arms={n_arms} dof={n_dof} names={dof_names}", flush=True)

# Joint limits (radians) from the articulation, used to keep the sweep inside 60% of range.
limits = arms.get_dof_limits()
limits = np.asarray(limits.cpu() if hasattr(limits, "cpu") else limits)[0]
lo, hi = limits[:, 0], limits[:, 1]
mid, amp = (lo + hi) / 2, 0.6 * (hi - lo) / 2


def sweep(t, arm_index):
    """Scripted joint targets: each joint is a sine with its own period, each arm phase-shifted."""
    periods = np.array([6.0, 4.0, 3.5, 3.0, 5.0, 2.0])[:n_dof]
    phase = arm_index * 1.3
    return mid + amp * np.sin(2 * math.pi * t / periods + phase)


# ---------------------------------------------------------------- optional live joint stream
sock = None
latest = {}  # arm index -> (time received, np.array of joint positions)
if args.listen:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.listen))
    sock.setblocking(False)
    print(f"[so101_scene] listening for joint packets on udp://127.0.0.1:{args.listen}", flush=True)


def drain_socket():
    """Read every pending packet: {"arm": i, "q": [6 radians in dof_names order]}."""
    while sock is not None:
        try:
            data, _ = sock.recvfrom(4096)
        except BlockingIOError:
            return
        try:
            msg = json.loads(data)
            latest[int(msg["arm"])] = (time.time(), np.asarray(msg["q"], dtype=np.float32))
        except Exception as exc:  # bad packet: ignore, keep rendering
            print(f"[so101_scene] bad packet: {exc}", flush=True)


# ---------------------------------------------------------------- 3+4. drive, render, measure
os.makedirs(args.out, exist_ok=True)
for _ in range(10):  # warm-up so shaders and the first frame are ready
    world.step(render=True)
T_FIRST = time.time()
GPU_SCENE = gpu_mem_mib()

saved, gpu_peak, streamed_frames = [], GPU_SCENE, 0
t_loop = time.time()
for f in range(args.frames):
    t = f / 60.0
    drain_socket()
    q = np.stack([sweep(t, i) for i in range(n_arms)]).astype(np.float32)
    now = time.time()
    for i, (t_rx, q_rx) in latest.items():
        if i < n_arms and now - t_rx < 0.5:
            q[i, : len(q_rx)] = np.clip(q_rx[:n_dof], lo[: len(q_rx)], hi[: len(q_rx)])
            streamed_frames += 1
    arms.set_joint_positions(q)  # kinematic: teleport the joints...
    arms.set_joint_position_targets(q)  # ...and hold them there with the drives
    world.step(render=True)
    if args.save_every and f % args.save_every == 0:
        img = rgb.get_data()
        if img is not None and img.size:
            path = os.path.join(args.out, f"so101_scene_{f:04d}.png")
            Image.fromarray(np.asarray(img)[:, :, :3]).save(path)
            saved.append(path)
    if f % 50 == 0:
        gpu_peak = max(gpu_peak, gpu_mem_mib())
    if args.gui and not simulation_app.is_running():
        break
loop_s = time.time() - t_loop
frames_done = f + 1

if args.gui:  # evidence that the GUI viewport shows the scene: grab what the window displays
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    vp_png = os.path.join(args.out, "gui_viewport.png")
    capture_viewport_to_file(get_active_viewport(), vp_png)
    for _ in range(20):
        simulation_app.update()
    saved.append(vp_png)

report = {
    "mode": "gui" if args.gui else "headless",
    "resolution": f"{args.width}x{args.height}",
    "arms": n_arms,
    "app_start_s": round(T_APP - T0, 1),
    "import_s": round(T_IMPORT - T_APP, 1),
    "to_first_frame_s": round(T_FIRST - T0, 1),
    "frames": frames_done,
    "fps_incl_png_saves": round(frames_done / loop_s, 1),
    "gpu_mib_before": GPU_BEFORE,
    "gpu_mib_scene": GPU_SCENE,
    "gpu_mib_peak": gpu_peak,
    "streamed_arm_frames": streamed_frames,
    "pngs": saved,
}
print("[so101_scene] REPORT " + json.dumps(report), flush=True)
with open(os.path.join(args.out, "report.json"), "w") as fh:
    json.dump(report, fh, indent=2)

if sock is not None:
    sock.close()
simulation_app.close()
