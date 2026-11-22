"""AprilTag images for the boxes in the Isaac Sim view: tag36h11, one id per box.

    python scripts/isaac/make_apriltags.py [--ids 0-29] [--out ~/.cache/hfteama_isaac/apriltags]

Needs OpenCV with the aruco module (the lerobot env has it). Each PNG is the tag with a one-cell
white border, as printed tags have, so it can be detected. mirror.py puts tag <n> on the top face
of box_<n>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ids", default="0-29")
    ap.add_argument("--out", default=str(Path.home() / ".cache" / "hfteama_isaac" / "apriltags"))
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.ids.split("-"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    cell = 32  # px per tag cell: 8 cells + 2 border cells = 320 px
    for i in range(lo, hi + 1):
        tag = cv2.aruco.generateImageMarker(d, i, 8 * cell)
        img = np.full((10 * cell, 10 * cell), 255, np.uint8)
        img[cell:-cell, cell:-cell] = tag
        cv2.imwrite(str(out / f"tag36h11_{i:02d}.png"), img)
    # check: every tag is found again with its own id
    det = cv2.aruco.ArucoDetector(d)
    ok = sum(
        1
        for i in range(lo, hi + 1)
        if (ids := det.detectMarkers(cv2.imread(str(out / f"tag36h11_{i:02d}.png"), 0))[1])
        is not None
        and int(ids[0][0]) == i
    )
    print(f"{hi - lo + 1} tags in {out}, {ok} detected back with the right id")


if __name__ == "__main__":
    main()
