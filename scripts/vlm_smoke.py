"""Run the real Qwen VLM (with its GPU safeguards) on a synthetic top-down table image.

    python scripts/vlm_smoke.py [--model Qwen/Qwen3-VL-4B-Instruct] [--quantize auto|4bit|none]
                                [--image photo.png]

Without --image, draws coloured blocks on a grey table, with object positions known, and checks
that the decomposer groups them by colour and that the step check answers. With --image, only
the pixel labels are synthetic (a 3x2 grid), so use it to eyeball answers on a real photo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from hfteama.perception import (
    Bin,
    Calibration,
    Decomposer,
    DecompositionError,
    Detection,
    QwenVLM,
    StepChecker,
    VLMError,
    build_scene,
)

BLOCKS = [  # (colour name, rgb, pixel centre)
    ("red", (210, 40, 40), (160, 140)),
    ("blue", (40, 70, 210), (320, 120)),
    ("red", (200, 50, 45), (470, 160)),
    ("green", (40, 170, 70), (200, 330)),
    ("blue", (50, 80, 200), (380, 300)),
    ("green", (45, 160, 60), (520, 340)),
]
BINS = [
    Bin("bin_1", "the left bin", (0.10, 0.25, 0.05)),
    Bin("bin_2", "the middle bin", (0.45, 0.25, 0.05)),
    Bin("bin_3", "the right bin", (0.80, 0.25, 0.05)),
]


def table_image() -> Image.Image:
    img = Image.new("RGB", (640, 480), (150, 150, 145))
    d = ImageDraw.Draw(img)
    for _, rgb, (u, v) in BLOCKS:
        d.rectangle((u - 22, v - 22, u + 22, v + 22), fill=rgb, outline=(30, 30, 30))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--quantize", default="auto", choices=["auto", "4bit", "none"])
    ap.add_argument("--image", default=None)
    a = ap.parse_args()

    image = Image.open(a.image).convert("RGB") if a.image else table_image()
    # fake SAM-6D/OnePose output: one detection per block, table frame = camera frame here
    dets = []
    for i, (_, _, (u, v)) in enumerate(BLOCKS):
        T = np.eye(4)
        T[:3, 3] = (0.1 + 0.15 * (i % 3), -0.15 - 0.1 * (i // 3), 0.02)
        dets.append(Detection(T_cam_obj=T, model="block", pixel=(float(u), float(v))))
    scene = build_scene(dets, Calibration(np.eye(4)))

    vlm = QwenVLM(a.model, quantize=a.quantize)
    try:
        t0 = time.monotonic()
        d = Decomposer(vlm, BINS).decompose(image, scene)
        print(f"loaded as {vlm.loaded_as}; decompose {time.monotonic() - t0:.1f} s incl. load")
        print("stats:", {k: round(v, 2) for k, v in vlm.last_stats.items()})
        print("attempts:", d.attempts)
        print("grouping:", json.dumps(d.grouping(), indent=1))

        ok = True
        if not a.image:
            truth = {f"obj_{i + 1:02d}": c for i, (c, _, _) in enumerate(BLOCKS)}
            groups = {}
            for oid, cat in d.sorting.assignment.items():
                groups.setdefault(cat, set()).add(truth[oid])
            ok = all(len(colours) == 1 for colours in groups.values()) and len(groups) == 3
            print("grouped by colour:", ok)

        chk = StepChecker(vlm, BINS).check(d.graph.tasks["place_obj_01"], image)
        print("step check (object is still on the table, expect done=False):", chk)
        print("stats:", {k: round(v, 2) for k, v in vlm.last_stats.items()})
        return 0 if ok and not chk.done else 1
    except (VLMError, DecompositionError) as e:
        print("VLM refused:", e)
        return 2
    finally:
        vlm.unload()


if __name__ == "__main__":
    sys.exit(main())
