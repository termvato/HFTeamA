"""Run the real Qwen VLM (with its GPU safeguards) through detect -> sort -> step check.

    python scripts/vlm_smoke.py [--model Qwen/Qwen3-VL-4B-Instruct] [--quantize auto|4bit|none]
                                [--image photo.png --camera configs/camera.json]

Without --image, renders a synthetic top-down table through configs/camera.example.json (blocks
at known table positions, three bins), so it can report the pointing error in cm and whether
the grouping follows colour. With --image, runs the same pipeline on a real photo.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hfteama.perception import (
    Calibration,
    Decomposer,
    DecompositionError,
    DetectionError,
    QwenVLM,
    StepChecker,
    VLMError,
    VLMPointDetector,
    build_scene,
    load_bins,
)
from hfteama.registry import load_catalog

ROOT = Path(__file__).resolve().parent.parent
BLOCK = 0.04  # m, cube side
BLOCKS = [  # (colour, rgb, table x, y)
    ("red", (210, 40, 40), 0.05, 0.05),
    ("blue", (40, 70, 210), 0.30, 0.02),
    ("red", (200, 50, 45), 0.62, 0.06),
    ("green", (40, 170, 70), 0.20, 0.45),
    ("blue", (50, 80, 200), 0.55, 0.43),
    ("green", (45, 160, 60), 0.85, 0.47),
]


def project(cal: Calibration, x: float, y: float, z: float) -> tuple[float, float]:
    p = np.linalg.inv(cal.T_table_cam) @ np.array([x, y, z, 1.0])
    uvw = np.asarray(cal.K) @ p[:3]
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def render(cal: Calibration, bins) -> Image.Image:
    img = Image.new("RGB", tuple(cal.image_size), (155, 150, 140))
    d = ImageDraw.Draw(img)

    def square(x, y, z, half, **kw):
        corners = [project(cal, x + sx * half, y + sy * half, z) for sx, sy in ((-1, -1), (1, 1))]
        (u0, v0), (u1, v1) = corners
        d.rectangle((min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)), **kw)

    for b in bins:  # open grey boxes
        square(b.xyz[0], b.xyz[1], 0.0, 0.08, fill=(90, 90, 95), outline=(40, 40, 40), width=4)
        square(b.xyz[0], b.xyz[1], 0.0, 0.065, fill=(60, 60, 65))
    for _, rgb, x, y in BLOCKS:
        square(x, y, BLOCK, BLOCK / 2, fill=rgb, outline=(25, 25, 25), width=2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--quantize", default="auto", choices=["auto", "4bit", "none"])
    ap.add_argument("--image", default=None)
    ap.add_argument("--camera", default=str(ROOT / "configs/camera.example.json"))
    ap.add_argument("--arms", default=str(ROOT / "configs/arms.example.json"))
    ap.add_argument("--bins", default=str(ROOT / "configs/bins.example.json"))
    a = ap.parse_args()

    cal = Calibration.load(a.camera, load_catalog(a.arms))
    bins = load_bins(a.bins)
    image = Image.open(a.image).convert("RGB") if a.image else render(cal, bins)

    vlm = QwenVLM(a.model, quantize=a.quantize)
    ok = True
    try:
        t0 = time.monotonic()
        detector = VLMPointDetector(vlm, cal, object_height_m=BLOCK)
        scene = build_scene(detector.detect(image), cal)
        print(f"loaded as {vlm.loaded_as}; detect {time.monotonic() - t0:.1f} s incl. load")
        print("  stats:", {k: round(v, 2) for k, v in vlm.last_stats.items()})
        for o in scene.objects.values():
            print(f"  {o.id}: {o.label!r} at ({o.xyz[0]:.3f}, {o.xyz[1]:.3f}) px {o.pixel}")
        if detector.dropped:
            print("  dropped:", detector.dropped)

        truth: dict[str, str] = {}
        if not a.image:
            errs = []
            for oid, o in scene.objects.items():
                c, _, x, y = min(BLOCKS, key=lambda b: math.dist((b[2], b[3]), o.xyz[:2]))
                errs.append(math.dist((x, y), o.xyz[:2]))
                truth[oid] = c
            found = len(scene.objects) == len(BLOCKS) and len(set(truth.values())) == 3
            print(
                f"  found {len(scene.objects)}/{len(BLOCKS)} blocks; pointing error "
                f"mean {100 * np.mean(errs):.1f} cm, max {100 * max(errs):.1f} cm"
                if errs
                else "  found nothing"
            )
            ok &= found and bool(errs) and max(errs) < 0.03

        t0 = time.monotonic()
        d = Decomposer(vlm, bins).decompose(image, scene)
        print(f"decompose {time.monotonic() - t0:.1f} s, attempts {d.attempts}")
        print("  stats:", {k: round(v, 2) for k, v in vlm.last_stats.items()})
        print("  grouping:", json.dumps(d.grouping()))
        if truth:
            groups: dict[str, set[str]] = {}
            for oid, cat in d.sorting.assignment.items():
                groups.setdefault(cat, set()).add(truth.get(oid, "?"))
            by_colour = all(len(c) == 1 for c in groups.values()) and len(groups) == 3
            print("  grouped by colour:", by_colour)
            ok &= by_colour

        first_place = next(t for t in d.graph.tasks.values() if t.skill == "place")
        t0 = time.monotonic()
        chk = StepChecker(vlm, bins).check(first_place, image)
        print(f"step check {time.monotonic() - t0:.1f} s (object still on the table): {chk}")
        print("  stats:", {k: round(v, 2) for k, v in vlm.last_stats.items()})
        ok &= not chk.done
        print("SMOKE", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    except (VLMError, DecompositionError, DetectionError) as e:
        print("VLM refused:", e)
        return 2
    finally:
        vlm.unload()


if __name__ == "__main__":
    sys.exit(main())
