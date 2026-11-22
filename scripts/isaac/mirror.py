"""Isaac Sim view of a running MuJoCo simulation (the sender is hfteama.view.isaac_mirror).

MuJoCo runs the task; this only draws it, with RTX: the same arm meshes, objects and location
discs, moved to MuJoCo's body poses every frame. Nothing is simulated here (no PhysX), so the
two can never disagree. Lighting is Isaac Sim's default rig (dome + distant light); surfaces are
glossy PBR materials, so the table and parts show reflections.

Live, in two terminals (ROS on PYTHONPATH breaks Isaac's Python, hence env -u):
    env -u PYTHONPATH ~/isaac-sim/python.sh scripts/isaac/mirror.py
    .venv/bin/python scripts/sim_demo.py --isaac 9871          # Vivek's 3-arm sort, sort12.json
Either can start first. A new sim run (another task or layout) rebuilds the scene.

Video, from a recording (scripts/isaac/pyramid_demo.py --out DIR writes DIR/states.bin):
    env -u PYTHONPATH ~/isaac-sim/python.sh scripts/isaac/mirror.py --replay DIR/states.bin \\
        --record DIR/frames
Every recorded pose (30 per sim second) becomes one frame, however slow the rendering, and
DIR/frames/frames.csv lists each frame's sim time (scripts/isaac/make_video.py uses it).

Options:
  --port 9871        UDP port to listen on (the sim's --isaac)
  --headless         no window (implied by --replay)
  --replay FILE      render a recording instead of listening
  --record DIR       save the camera view as PNG frames in DIR
  --record-every K   live mode: save every K-th rendered frame (default 2)
  --subframes N      render passes per saved frame (default 4: cleaner reflections)
  --exit-idle S      live mode: quit S seconds after the last packet (default 0 = never)
  --view NAME        camera: front (default), top, side, close; framed on the scene
  --orbit DEG        circle the camera slowly around the scene, DEG degrees per sim second
                     (default 0 = still; 1.5 turns it half round in two minutes)
  --same-colour      keep the model's yellow on every arm (default: arm 1 black, 2 green, 3 white)
  --wood PNG/JPG     table-top texture (default: the oak planks that ship with Isaac Sim)
  --tags DIR         AprilTag images, tag36h11_<n>.png, put on top of box_<n> (default:
                     ~/.cache/hfteama_isaac/apriltags, written by make_apriltags.py)
  --width/--height   render size (default 1280x720)
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import time

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--port", type=int, default=9871)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--replay", default="")
parser.add_argument("--record", default="")
parser.add_argument("--record-every", type=int, default=2)
parser.add_argument("--subframes", type=int, default=4)
parser.add_argument("--exit-idle", type=float, default=0.0)
parser.add_argument("--view", choices=["front", "top", "side", "close"], default="front")
parser.add_argument("--orbit", type=float, default=0.0)
parser.add_argument("--same-colour", action="store_true")
parser.add_argument(
    "--wood",
    default=os.path.expanduser(
        "~/isaac-sim/extscache/omni.mdl.usd_converter-1.0.40+00c488ae/data/usd/simple_bitmap/"
        "Materials/Material__42/wooden_floor.jpg"
    ),
)
parser.add_argument("--tags", default=os.path.expanduser("~/.cache/hfteama_isaac/apriltags"))
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
args, _unknown = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402  (must come before any omni / pxr import)

app = SimulationApp(
    {
        "headless": args.headless or bool(args.replay),
        "renderer": "RaytracedLighting",
        "width": args.width,
        "height": args.height,
    }
)

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

STATE_MAGIC, SCENE_MAGIC = b"MJM1", b"MJS1"  # see hfteama/view/isaac_mirror.py
SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 2, 3, 4, 5, 6, 7  # mjtGeom
TABLE_H = 0.75  # the MuJoCo floor (z = 0) is the table top; the room floor is this far below
VIEW_DIRS = {  # camera direction from the scene centre (x, y, z), scaled to the scene size
    "front": (0.6, -1.2, 1.0),
    "side": (1.15, -0.25, 0.65),
    "top": (0.0, -0.05, 1.45),
    "close": (0.2, -0.42, 0.34),  # near the pyramid: the boxes' AprilTags are readable
}

omni.usd.get_context().new_stage()
app.update()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
settings = carb.settings.get_settings()
for key_, value in {
    "/rtx/reflections/enabled": True,
    "/rtx/indirectDiffuse/enabled": True,
    "/rtx/ambientOcclusion/enabled": True,
    "/rtx/shadows/enabled": True,
    "/rtx/directLighting/sampledLighting/enabled": True,
    "/rtx/post/aa/op": 3,
}.items():
    settings.set(key_, value)


def xform(path, pos=(0, 0, 0), quat=(1, 0, 0, 0), scale=None):
    """Translate + orient (+ scale) ops on a prim; returns (translate op, orient op)."""
    x = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    x.ClearXformOpOrder()
    t = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    o = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    t.Set(Gf.Vec3d(*map(float, pos)))
    o.Set(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:]))))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
    return t, o


_materials = {}


def material(rgb, rough, metallic=0.0, clearcoat=0.0):
    """A UsdPreviewSurface (RTX renders it directly), shared by everything with the same look."""
    key = (*(round(float(c), 3) for c in rgb[:3]), rough, metallic, clearcoat)
    if key not in _materials:
        path = f"/World/Looks/m{len(_materials):03d}"
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, f"{path}/pbr")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*key[:3]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(clearcoat))
        sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.05)
        sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.6, 0.6))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        _materials[key] = mat
    return _materials[key]


def paint(prim, rgb, rough=0.35, metallic=0.0, clearcoat=0.0):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, rgb[:3]))]))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material(rgb, rough, metallic, clearcoat))


_textures = {}


def textured(image, rough=0.3, clearcoat=0.0):
    """A UsdPreviewSurface whose colour is an image, read with the mesh's `st` coordinates."""
    key = (image, rough, clearcoat)
    if key not in _textures:
        path = f"/World/Looks/t{len(_textures):03d}"
        mat = UsdShade.Material.Define(stage, path)
        st = UsdShade.Shader.Define(stage, f"{path}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/image")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(image)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st.ConnectableAPI(), "result"
        )
        sh = UsdShade.Shader.Define(stage, f"{path}/pbr")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(clearcoat))
        sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.05)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        _textures[key] = mat
    return _textures[key]


def quad(path, center, sx, sy, mat, uv=(1.0, 1.0)):
    """A flat rectangle facing up (+z), sx by sy metres, the image repeated uv times."""
    m = UsdGeom.Mesh.Define(stage, path)
    hx, hy = float(sx) / 2, float(sy) / 2
    m.CreatePointsAttr(
        [Gf.Vec3f(-hx, -hy, 0), Gf.Vec3f(hx, -hy, 0), Gf.Vec3f(hx, hy, 0), Gf.Vec3f(-hx, hy, 0)]
    )
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    u, v = uv
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(u, 0), Gf.Vec2f(u, v), Gf.Vec2f(0, v)])
    xform(path, center)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


def box(path, center, size, rgb, **look):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform(path, center, scale=size)
    paint(cube.GetPrim(), rgb, **look)


# ---------------------------------------------------------------- lights: Isaac's default rig
# omni.kit.viewport.menubar.lighting data/usd/Default.usda: a dome light and a distant light
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome.CreateIntensityAttr(1.0)
dome.CreateExposureAttr(9.0)
dome.CreateEnableColorTemperatureAttr(True)
dome.CreateColorTemperatureAttr(6150.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
sun.CreateIntensityAttr(1.0)
sun.CreateExposureAttr(10.0)
sun.CreateAngleAttr(0.53)
sun.CreateEnableColorTemperatureAttr(True)
sun.CreateColorTemperatureAttr(7250.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(40.0, 0.0, 135.0))


def build_room(cx, cy, half):
    """Lacquered wooden table under the scene (top at z = 0), legs, a satin floor."""
    root = "/World/Room"
    if stage.GetPrimAtPath(root):
        stage.RemovePrim(root)
    w = 2 * half + 0.3
    box(
        f"{root}/table_top",
        (cx, cy, -0.015),
        (w, w, 0.03),
        (0.55, 0.36, 0.2),
        rough=0.18,
        clearcoat=0.6,
    )
    if os.path.isfile(args.wood):  # oak planks; the image is about 1.2 m of floor
        quad(
            f"{root}/table_wood",
            (cx, cy, 0.0002),
            w,
            w,
            textured(args.wood, 0.22, 0.5),
            uv=(w / 1.2, w / 1.2),
        )
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        box(
            f"{root}/leg_{i}",
            (cx + sx * (w / 2 - 0.05), cy + sy * (w / 2 - 0.05), -0.03 - (TABLE_H - 0.03) / 2),
            (0.04, 0.04, TABLE_H - 0.03),
            (0.12, 0.12, 0.13),
            rough=0.25,
            metallic=0.8,
        )
    box(f"{root}/floor", (cx, cy, -TABLE_H - 0.01), (10, 10, 0.02), (0.38, 0.38, 0.4), rough=0.45)


cam_path = "/World/Camera"
cam = UsdGeom.Camera.Define(stage, cam_path)
cam.CreateFocalLengthAttr(21.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))


def look_at(eye, target):
    """Orient the camera (looks down its -z, y up) from eye to target, z up in the world."""
    eye, target = np.array(eye, float), np.array(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    if np.linalg.norm(right) < 1e-6:  # looking straight down
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rot = Gf.Matrix3d(*right, *up, *(-fwd))  # rows: camera x, y, z axes in the world
    q = rot.ExtractRotation().GetQuat()
    xform(cam_path, eye, (q.GetReal(), *q.GetImaginary()))


if not (args.headless or args.replay):
    from omni.kit.viewport.utility import get_active_viewport

    get_active_viewport().camera_path = cam_path

writer = None
if args.record:
    import omni.replicator.core as rep
    from PIL import Image

    os.makedirs(args.record, exist_ok=True)
    rp = rep.create.render_product(cam_path, (args.width, args.height))
    writer = rep.AnnotatorRegistry.get_annotator("rgb")
    writer.attach([rp])
    rep.orchestrator.set_capture_on_play(False)  # capture only when stepped below


# ---------------------------------------------------------------- the mirrored scene
ARM_TINTS = [
    (0.045, 0.045, 0.05),
    (0.12, 0.6, 0.28),
    (0.9, 0.9, 0.88),
    (0.85, 0.3, 0.2),
]  # 1 black, 2 green, 3 white
PRINTED = np.array([1.0, 0.82, 0.12])  # the SO-101 model's printed-part colour


class Scene:
    """The shapes of one MuJoCo scene file, one Xform per body, moved by pose packets."""

    def __init__(self, scene_id, path):
        self.id = scene_id
        self.framed = False
        self.t0 = None
        z = np.load(path)
        root = "/World/Sim"
        if stage.GetPrimAtPath(root):
            stage.RemovePrim(root)
        UsdGeom.Xform.Define(stage, root)
        self.ops = []
        self.names = names = [str(n) for n in z["body_names"]]
        arms = sorted({n.split("/")[0] for n in names if "/" in n})
        for b, name in enumerate(names):
            path_ = f"{root}/b{b:02d}_{_safe(name)}"
            UsdGeom.Xform.Define(stage, path_)
            self.ops.append(xform(path_))
        mv, mf = z["mesh_vert"], z["mesh_face"]
        for g in range(len(z["g_type"])):
            body = z["g_body"][g]
            gp = f"{root}/b{body:02d}_{_safe(names[body])}/g{g:03d}"
            kind, size, rgba = int(z["g_type"][g]), z["g_size"][g], z["g_rgba"][g]
            pos, quat = z["g_pos"][g], z["g_quat"][g]
            if kind == MESH:
                k = z["g_mesh"][g]
                va, vn = z["mesh_vadr"][k], z["mesh_vnum"][k]
                fa, fn = z["mesh_fadr"][k], z["mesh_fnum"][k]
                mesh = UsdGeom.Mesh.Define(stage, gp)
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mv[va : va + vn]))
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(fn, 3, np.int32)))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mf[fa : fa + fn].ravel()))
                mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                prim, scale = mesh.GetPrim(), None
            elif kind == BOX:
                prim, scale = UsdGeom.Cube.Define(stage, gp).GetPrim(), 2 * size
                UsdGeom.Cube(prim).CreateSizeAttr(1.0)
            elif kind == CYLINDER:
                c = UsdGeom.Cylinder.Define(stage, gp)
                c.CreateRadiusAttr(float(size[0]))
                c.CreateHeightAttr(float(2 * size[1]))
                c.CreateAxisAttr("Z")
                prim, scale = c.GetPrim(), None
            elif kind == CAPSULE:
                c = UsdGeom.Capsule.Define(stage, gp)
                c.CreateRadiusAttr(float(size[0]))
                c.CreateHeightAttr(float(2 * size[1]))
                c.CreateAxisAttr("Z")
                prim, scale = c.GetPrim(), None
            elif kind in (SPHERE, ELLIPSOID):
                s = UsdGeom.Sphere.Define(stage, gp)
                s.CreateRadiusAttr(1.0)
                prim, scale = s.GetPrim(), size if kind == ELLIPSOID else (size[0],) * 3
            else:
                continue
            xform(gp, pos, quat, scale)
            self._look(prim, names[body], rgba, kind, arms)
            if kind == BOX and names[body].startswith(("box_", "block_")):
                self._tag(gp.rsplit("/", 1)[0], names[body], pos, size, g)
        print(
            f"[mirror] scene {scene_id:08x}: {len(names)} bodies, {len(z['g_type'])} shapes",
            flush=True,
        )

    @staticmethod
    def _look(prim, body_name, rgba, kind, arms):
        rgb = np.asarray(rgba[:3], float)
        if "/" in body_name:  # an arm part
            if np.abs(rgb - PRINTED).max() <= 0.02:  # printed plastic: glossy, one colour per arm
                if not args.same_colour:
                    rgb = ARM_TINTS[arms.index(body_name.split("/")[0]) % len(ARM_TINTS)]
                paint(prim, rgb, rough=0.28, clearcoat=0.4)
            else:  # servos and screws: dark, shiny
                paint(prim, rgb, rough=0.18, metallic=0.3)
        elif kind == CYLINDER and body_name == "world":  # location discs on the table
            paint(prim, rgb, rough=0.5)
        elif body_name.startswith(("box_", "block_")):  # boxes: white, satin
            paint(prim, (0.93, 0.93, 0.91), rough=0.35, clearcoat=0.2)
        else:
            paint(prim, rgb, rough=0.3)

    @staticmethod
    def _tag(body_path, name, pos, size, g):
        """AprilTag tag36h11 <n> on the top face of box_<n> (the face the overhead camera sees)."""
        n = int("".join(c for c in name if c.isdigit()) or 0)
        image = os.path.join(args.tags, f"tag36h11_{n:02d}.png")
        if not os.path.isfile(image):
            return
        top = (float(pos[0]), float(pos[1]), float(pos[2] + size[2] + 0.0003))
        quad(f"{body_path}/tag{g:03d}", top, 2 * size[0], 2 * size[1], textured(image, 0.5))

    def frame(self, poses):
        """Fit the room and the camera to where the bodies are (first pose only)."""
        p = poses[[i for i, n in enumerate(self.names) if n != "world"], :3]
        lo, hi = p.min(axis=0), p.max(axis=0)
        cx, cy = (lo[:2] + hi[:2]) / 2
        half = float(max(hi[:2] - lo[:2]) / 2 + 0.12)
        build_room(cx, cy, half)
        d = np.array(VIEW_DIRS[args.view]) * (half / 0.55)
        self.cam = (np.array([cx + d[0], cy + d[1], d[2]]), np.array([cx, cy, 0.02]))
        look_at(*self.cam)
        self.framed = True

    def apply(self, poses, sim_t=None):
        if not self.framed:
            self.frame(poses)
            self.t0 = sim_t
        if args.orbit and sim_t is not None and self.t0 is not None:
            # turn the camera about the vertical through the scene centre, same height and distance
            eye, target = self.cam
            a = np.radians(args.orbit * (sim_t - self.t0))
            rx, ry = eye[0] - target[0], eye[1] - target[1]
            ca, sa = np.cos(a), np.sin(a)
            look_at((target[0] + ca * rx - sa * ry, target[1] + sa * rx + ca * ry, eye[2]), target)
        with Sdf.ChangeBlock():
            for (t, o), p in zip(self.ops, poses, strict=True):
                t.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                o.Set(Gf.Quatd(float(p[3]), Gf.Vec3d(float(p[4]), float(p[5]), float(p[6]))))


def _safe(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def save_frame(index, sim_t, csv):
    rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0, pause_timeline=False)
    img = writer.get_data()
    if img is None or not getattr(img, "size", 0):
        return False
    name = f"frame_{index:05d}.png"
    Image.fromarray(np.asarray(img)[:, :, :3]).save(os.path.join(args.record, name))
    csv.write(f"{name},{sim_t:.4f}\n")
    return True


def parse_state(data):
    sid, t_, n = struct.unpack_from("<IdI", data, 4)
    return sid, t_, np.frombuffer(data, np.float32, offset=20).reshape(n, 7)


csv = open(os.path.join(args.record, "frames.csv"), "w") if args.record else None  # noqa: SIM115
if csv:
    csv.write("file,sim_t\n")

if args.replay:  # ------------------------------------------------ a recording, frame by frame
    with open(args.replay, "rb") as fh:
        packets = []
        while head := fh.read(4):
            packets.append(fh.read(struct.unpack("<I", head)[0]))
    scene_packet = packets[0]
    assert scene_packet.startswith(SCENE_MAGIC), "a recording starts with its scene packet"
    scene = Scene(struct.unpack_from("<I", scene_packet, 4)[0], scene_packet[8:].decode())
    states = [parse_state(p) for p in packets[1:] if p.startswith(STATE_MAGIC)]
    for _ in range(20):  # warm-up: shaders and materials
        app.update()
    t_start, saved = time.time(), 0
    for k, (_sid, sim_t, poses) in enumerate(states):
        scene.apply(poses, sim_t)
        app.update()
        if writer is not None:
            saved += save_frame(saved, sim_t, csv)
        if k % 150 == 0:
            rate = (k + 1) / (time.time() - t_start)
            print(f"[mirror] frame {k + 1}/{len(states)} ({rate:.1f} fps)", flush=True)
    print(f"[mirror] replay done: {saved} frames in {args.record}", flush=True)
else:  # ------------------------------------------------------------------------ live
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.setblocking(False)
    print(
        f"[mirror] listening on udp://127.0.0.1:{args.port} (sim: --isaac {args.port})",
        flush=True,
    )
    scene, latest, sim_t, last_rx, frame, saved = None, None, 0.0, 0.0, 0, 0
    t_report = time.time()
    while app.is_running():
        while True:  # drain: only the newest pose matters
            try:
                data, _ = sock.recvfrom(65535)
            except BlockingIOError:
                break
            last_rx = time.time()
            if data.startswith(SCENE_MAGIC):
                sid = struct.unpack_from("<I", data, 4)[0]
                if scene is None or scene.id != sid:
                    path = data[8:].decode()
                    if os.path.isfile(path):
                        scene, latest = Scene(sid, path), None
            elif data.startswith(STATE_MAGIC):
                sid, t_, poses = parse_state(data)
                if scene is not None and sid == scene.id:
                    latest, sim_t = poses, t_
        if scene is not None and latest is not None:
            scene.apply(latest, sim_t)
            latest = None
        app.update()
        frame += 1
        if writer is not None and scene is not None and frame % args.record_every == 0:
            saved += save_frame(saved, sim_t, csv)
        if time.time() - t_report > 5:
            state = f"sim t={sim_t:.1f}s" if scene else "waiting for the sim"
            print(f"[mirror] {state}, {frame} frames rendered, {saved} saved", flush=True)
            t_report = time.time()
        if args.exit_idle and last_rx and time.time() - last_rx > args.exit_idle:
            break
    sock.close()

if csv:
    csv.close()
app.close()
