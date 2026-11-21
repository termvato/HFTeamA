"""Real table, dry run: top camera -> marker calibration -> Qwen object positions -> reachability
-> executor split across the arms. The executor runs fake workers (no motion) standing in for the
real arms, so this checks perception and planning end to end before any arm moves.

    python scripts/real_table_plan.py --device "HD Pro Webcam C920"      # grab a frame (ffmpeg)
    python scripts/real_table_plan.py --image frame.jpg                    # or a file / stream URL
        [--table configs/hacklab_table.json] [--travel 4.0] [--out plan.png] [--no-run]

Prints the calibration (markers, focal length, fit error), every object in table coordinates with
its distance to each arm, which arms may do each step, and the split with 1..N arms. Saves the
frame with reach arcs, bases, bowl and every object coloured by the arm that did it.
Needs: pip install -e ".[vlm,camera]".
"""

from __future__ import annotations

import argparse
import io
import math
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hfteama.arms.fake import FakeArm
from hfteama.clock import Clock
from hfteama.events import EventLog
from hfteama.executor import Executor
from hfteama.perception import (
    Decomposer,
    DecompositionError,
    DetectionError,
    QwenVLM,
    VLMError,
    VLMPointDetector,
    build_scene,
    load_bins,
)
from hfteama.perception.markers import calibrate_from_markers
from hfteama.registry import load_catalog, resolve

ROOT = Path(__file__).resolve().parent.parent
PALETTE = [(30, 30, 30), (40, 200, 90), (40, 90, 220), (220, 120, 30)]


def grab(device: str, size: tuple[int, int]) -> Image.Image:
    """One frame from a camera through ffmpeg (DirectShow on Windows, V4L2 on Linux)."""
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = ["-f", "dshow", "-i", f"video={device}"] if sys.platform == "win32" else [
        "-f", "v4l2", "-i", device]  # fmt: skip
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-video_size", f"{size[0]}x{size[1]}", *src,
           "-ss", "2", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, timeout=30, check=True).stdout
    return Image.open(io.BytesIO(out)).convert("RGB")


def load(src: str) -> Image.Image:
    if not src.startswith(("http://", "https://")):
        return Image.open(src).convert("RGB")
    with urllib.request.urlopen(src, timeout=10) as r:
        buf = r.read(5_000_000)
    s = buf.find(b"\xff\xd8")
    e = buf.find(b"\xff\xd9", s + 2)
    return Image.open(io.BytesIO(buf[s : e + 2])).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--device", help='camera name (Windows) or /dev/video path, e.g. "HD Pro Webcam C920"'
    )
    ap.add_argument("--image", help="frame file or URL instead of a camera")
    ap.add_argument("--table", default=str(ROOT / "configs/hacklab_table.json"))
    ap.add_argument("--travel", type=float, default=4.0, help="s per metre from arm to step")
    ap.add_argument(
        "--clearance",
        type=float,
        default=0.15,
        help="min distance (m) between two arms working at once; 0 = off",
    )
    ap.add_argument("--image-side", type=int, default=1920, help="pixels sent to Qwen")
    ap.add_argument("--out", default="real_table_plan.png")
    ap.add_argument("--no-run", action="store_true", help="skip the executor runs")
    a = ap.parse_args()

    import json

    cfg = json.loads(Path(a.table).read_text())
    if a.image:
        img = load(a.image)
    elif a.device:
        img = grab(a.device, tuple(cfg.get("camera", {}).get("size", (1920, 1080))))
    else:
        ap.error("give --device or --image")

    mk = cfg["markers"]
    m = calibrate_from_markers(
        np.asarray(img), mk["size_m"], mk.get("reference_id"), dictionary=mk["dictionary"]
    )
    cal = m.calib
    print("calibration:", m.summary(), sep="\n")
    arms = list(load_catalog(a.table).values())
    bins = load_bins(a.table)
    for arm in arms:
        print(f"arm {arm.arm_id}: base {arm.base_xy}, yaw {arm.base_yaw_deg}")

    vlm = QwenVLM(max_image_side=a.image_side)
    try:
        det = VLMPointDetector(vlm, cal, object_height_m=0.02)
        t0 = time.monotonic()
        scene = build_scene(det.detect(img), cal)
        print(f"\nQwen: {len(scene.objects)} objects in {time.monotonic() - t0:.1f} s")
        for o in scene.objects.values():
            dist = "".join(
                f"  {r.arm_id}: {math.dist(o.xyz[:2], r.pan_axis_xy):.2f} m"
                f"{'' if r.reaches(o.xyz) else ' (out of reach)'}"
                for r in arms
            )
            print(f"  {o.id} {o.label!r:16} ({o.xyz[0]:+.3f}, {o.xyz[1]:+.3f}) m{dist}")
        if det.dropped:
            print("  dropped:", det.dropped)
        dec = Decomposer(vlm, bins).decompose(img, scene)
        print("grouping:", json.dumps(dec.grouping()))
    except (VLMError, DetectionError, DecompositionError) as e:
        print("VLM refused:", e)
        return 2
    finally:
        vlm.unload()

    rr = resolve(dec.graph, arms, travel_s_per_m=a.travel, clearance_m=a.clearance)
    print("\nsteps (allowed arms: planned seconds):")
    for tid, t in rr.graph.tasks.items():
        times = ", ".join(f"{arm} {t.duration_on(arm):.1f}" for arm in t.allowed_arms)
        print(f"  {tid:18} {times or 'BLOCKED'}")

    done_by: dict[str, str] = {}
    if not a.no_run:
        for n in range(1, len(arms) + 1):
            clock = Clock(scale=50.0)
            log = EventLog(Path(tempfile.mkdtemp()) / "events.jsonl")
            ex = Executor(rr.graph, log, clock=clock, catalog=[r.arm_id for r in arms])
            for r in arms[:n]:
                ex.add_arm(
                    FakeArm(
                        r.arm_id, clock=clock, duration_fn=lambda t, i=r.arm_id: t.duration_on(i)
                    )
                )
            ex.start()
            finished = ex.wait(60)
            ends = [e for e in log.read() if e.kind == "task_end" and e.outcome == "ok"]
            ex.stop()
            split: dict[str, list[str]] = {}
            for e in ends:
                obj = rr.graph.tasks[e.task].args["object"]
                if e.task.startswith("pick"):
                    split.setdefault(e.arm, []).append(obj)
                done_by[obj] = e.arm
            t_end = max((e.data.get("end", 0.0) for e in ends), default=0.0)
            print(f"\n{n} arm(s): {'done' if finished else 'NOT finished'} in {t_end:.1f} s")
            for arm, objs in split.items():
                print(f"  {arm}: {', '.join(objs)}")

    colour = {r.arm_id: PALETTE[k % len(PALETTE)] for k, r in enumerate(arms)}

    def px(x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        q = cal.K @ (np.linalg.inv(cal.T_table_cam) @ np.array([x, y, z, 1.0]))[:3]
        return (q[0] / q[2], q[1] / q[2])

    draw = ImageDraw.Draw(img)
    for r in arms:
        cx, cy = r.pan_axis_xy
        lim = r.reach.pan_limit_deg if r.reach.r_max < math.inf else 0
        arc = [
            px(cx + r.reach.r_max * math.cos(math.radians(r.base_yaw_deg + k)),
               cy + r.reach.r_max * math.sin(math.radians(r.base_yaw_deg + k)))
            for k in range(-int(lim), int(lim) + 1, 5)
        ]  # fmt: skip
        if len(arc) > 1:
            draw.line(arc, fill=colour[r.arm_id], width=4)
        u, v = px(*r.base_xy)
        draw.ellipse((u - 10, v - 10, u + 10, v + 10), outline=(255, 0, 0), width=4)
        draw.text((u + 12, v - 20), r.arm_id, fill=(255, 0, 0))
    for b in bins:
        u, v = px(b.xyz[0], b.xyz[1])
        draw.ellipse((u - 8, v - 8, u + 8, v + 8), outline=(0, 0, 255), width=4)
    for o in scene.objects.values():
        u, v = o.pixel
        draw.ellipse((u - 9, v - 9, u + 9, v + 9), fill=colour.get(done_by.get(o.id), (255, 0, 0)),
                     outline=(255, 255, 0), width=2)  # fmt: skip
        draw.text((u + 12, v - 6), f"{o.id} -> {done_by.get(o.id, '?')}", fill=(255, 255, 0))
    img.save(a.out)
    print("\npicture:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
