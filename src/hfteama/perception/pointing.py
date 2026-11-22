"""Object detection by the VLM itself: zero-shot, one RGB camera, no per-object models.

1. The VLM points at every pickable object (pass 1 of the architecture doc, section 3.3) and
   gives each a short description ("red cube").
2. Each point is projected onto the table: the camera ray through the pixel meets the plane at
   the objects' top height (camera pose from calibration, intrinsics K). Everything relevant lies
   on the table, so no depth camera is needed.
3. Points off the table (table_bounds_m) are dropped; points within merge_radius_m are merged.

Yaw (Blockers #4): for elongated objects the VLM also gives two points at the ends of the long
axis (`axis_2d`); both are projected the same way and their direction is the object's yaw, which
the decomposer passes on as `pick_yaw_deg`. Without them (round objects, or a malformed axis)
the object gets the identity orientation and no yaw: the pick skill keeps its default heading.
Qwen3-VL gives points on a relative 0..1000 scale, which
also makes them independent of the safeguard's image downscaling.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

from hfteama.perception.annotate import to_pil
from hfteama.perception.frames import Calibration
from hfteama.perception.prompts import POINT_SYSTEM, point_prompt
from hfteama.perception.scene import Detection
from hfteama.perception.vlm import VLM, ParseError


class DetectionError(RuntimeError):
    """The VLM gave no usable list of points after all retries."""


Point = tuple[float, float]


def _is_point(pt: Any) -> bool:
    return (
        isinstance(pt, list)
        and len(pt) == 2
        and all(isinstance(c, int | float) and not isinstance(c, bool) for c in pt)
    )


def parse_points(text: str) -> list[tuple[str, Point]]:
    """(label, (x, y)) from {"objects": [...]} or a bare list of {"label", "point_2d"} items."""
    return [(label, pt) for label, pt, _ in parse_objects(text)]


def parse_objects(text: str) -> list[tuple[str, Point, tuple[Point, Point] | None]]:
    """(label, centre, long-axis end points or None) per object. A malformed optional
    `axis_2d` is ignored (the object keeps its centre, without yaw), never a parse error."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ParseError("no JSON in the answer")
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e.msg}") from e
    items = value.get("objects") if isinstance(value, dict) else value
    if not isinstance(items, list):
        raise ParseError('expected {"objects": [...]}')
    out = []
    for it in items:
        if not isinstance(it, dict):
            raise ParseError(f"object {it!r} is not a JSON object")
        pt = it.get("point_2d", it.get("point"))
        if not _is_point(pt):
            raise ParseError(f'object {it!r} needs "point_2d": [x, y]')
        ax = it.get("axis_2d")
        axis = None
        if isinstance(ax, list) and len(ax) == 2 and all(_is_point(p) for p in ax):
            axis = ((float(ax[0][0]), float(ax[0][1])), (float(ax[1][0]), float(ax[1][1])))
        out.append((str(it.get("label", "")).strip(), (float(pt[0]), float(pt[1])), axis))
    return out


class VLMPointDetector:
    """perception.scene.ObjectDetector backed by the VLM's pointing."""

    def __init__(
        self,
        vlm: VLM,
        calib: Calibration,
        object_height_m: float = 0.04,
        coord_scale: float = 1000.0,
        merge_radius_m: float = 0.02,
        max_retries: int = 2,
        model: str = "object",
        min_axis_m: float = 0.01,
    ) -> None:
        if calib.K is None:
            raise ValueError("VLMPointDetector needs camera intrinsics K in the calibration")
        self.vlm = vlm
        self.calib = calib
        self.object_height_m = object_height_m
        self.coord_scale = coord_scale
        self.merge_radius_m = merge_radius_m
        self.max_retries = max_retries
        self.model = model
        self.min_axis_m = min_axis_m
        self.dropped: list[tuple[str, str]] = []  # (label, why) from the last call

    def detect(self, rgb: Any, depth: Any = None) -> list[Detection]:
        img = to_pil(rgb)
        w, h = img.size
        errors: list[str] = []
        for _ in range(self.max_retries + 1):
            text = self.vlm.ask(img, point_prompt(errors or None), POINT_SYSTEM)  # VLMError: up
            try:
                points = parse_objects(text)
                break
            except ParseError as e:
                errors = [str(e)]
        else:
            raise DetectionError("no usable points from the VLM: " + "; ".join(errors))

        self.dropped = []
        kept: list[tuple[str, Point, tuple[float, float, float], float | None]] = []
        s = self.coord_scale
        for label, (x, y), axis in points:
            if not (0 <= x <= s and 0 <= y <= s):
                self.dropped.append((label, f"point {x:.0f},{y:.0f} outside the image"))
                continue
            u, v = x / s * w, y / s * h
            top = self.calib.pixel_to_table(u, v, self.object_height_m, (w, h))
            if top is None or not self.calib.on_table(top):
                self.dropped.append((label, "not on the table"))
                continue
            xyz = (top[0], top[1], self.object_height_m / 2)
            if any(math.dist(xyz[:2], k[2][:2]) < self.merge_radius_m for k in kept):
                self.dropped.append((label, "duplicate point"))
                continue
            kept.append((label, (u, v), xyz, self._axis_yaw(axis, (w, h))))

        out = []
        for label, pixel, xyz, yaw in kept:
            T = np.eye(4)
            if yaw is not None:
                c, sn = math.cos(yaw), math.sin(yaw)
                T[:3, :3] = [[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]
            T[:3, 3] = xyz
            out.append(
                Detection(
                    T_cam_obj=self.calib.to_camera(T),
                    model=self.model,
                    pixel=pixel,
                    label=label,
                    has_yaw=yaw is not None,
                )
            )
        return out

    def _axis_yaw(self, axis: tuple[Point, Point] | None, size: tuple[int, int]) -> float | None:
        """Yaw (rad) of the line between the two axis points, projected onto the object tops.

        None when there is no axis, an end misses the table plane, or the ends are closer than
        `min_axis_m` (too short to give a direction)."""
        if axis is None:
            return None
        w, h = size
        s = self.coord_scale
        ends = [
            self.calib.pixel_to_table(x / s * w, y / s * h, self.object_height_m, size)
            for x, y in axis
        ]
        if any(e is None for e in ends):
            return None
        dx, dy = ends[1][0] - ends[0][0], ends[1][1] - ends[0][1]  # type: ignore[index]
        if math.hypot(dx, dy) < self.min_axis_m:
            return None
        return math.atan2(dy, dx)
