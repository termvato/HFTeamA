"""Camera frame -> table frame. The camera pose in an arm's base frame comes from calibration
(Hardware). The table frame is the frame of configs/arms.json (base_pose_m), so every arm and
object share one frame. Camera axes follow OpenCV: x right, y down, z forward (optical axis).

configs/camera.json, either form (see configs/camera.example.json):
    {"T_table_cam": 4x4}                                  camera pose in the table frame
    {"base_arm": "arm_a", "T_base_cam": 4x4}              camera pose in one arm's base frame
                                                          (the base pose comes from the catalog)
plus, for turning pixels into table positions:
    "K": 3x3 intrinsics, "image_size": [width, height]   (the resolution K was calibrated at)
    "table_bounds_m": [[xmin, ymin], [xmax, ymax]]      optional workspace check
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hfteama.registry.arms import ArmNode


def base_transform(arm: ArmNode) -> np.ndarray:
    """4x4 pose of an arm's base in the table frame (translation + yaw about z)."""
    c, s = math.cos(math.radians(arm.base_yaw_deg)), math.sin(math.radians(arm.base_yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = arm.base_pose_m
    return T


def quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    m = np.asarray(R, float)

    def part(a: float, b: float, c: float) -> float:
        return math.sqrt(max(0.0, 1.0 + a + b + c)) / 2.0

    w = part(m[0, 0], m[1, 1], m[2, 2])
    x = math.copysign(part(m[0, 0], -m[1, 1], -m[2, 2]), m[2, 1] - m[1, 2])
    y = math.copysign(part(-m[0, 0], m[1, 1], -m[2, 2]), m[0, 2] - m[2, 0])
    z = math.copysign(part(-m[0, 0], -m[1, 1], m[2, 2]), m[1, 0] - m[0, 1])
    return (w, x, y, z)


@dataclass
class Calibration:
    T_table_cam: np.ndarray  # 4x4, camera pose in the table frame
    K: np.ndarray | None = None  # 3x3 intrinsics at image_size
    image_size: tuple[int, int] | None = None  # (width, height) K belongs to
    table_bounds_m: tuple[tuple[float, float], tuple[float, float]] | None = None

    def to_table(self, T_cam_obj: np.ndarray) -> np.ndarray:
        return self.T_table_cam @ np.asarray(T_cam_obj, float)

    def to_camera(self, T_table_obj: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.T_table_cam) @ np.asarray(T_table_obj, float)

    def on_table(self, xy: tuple[float, ...]) -> bool:
        if self.table_bounds_m is None:
            return True
        (x0, y0), (x1, y1) = self.table_bounds_m
        return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1

    def pixel_to_table(
        self, u: float, v: float, z: float, image_size: tuple[int, int] | None = None
    ) -> tuple[float, float, float] | None:
        """Where the camera ray through pixel (u, v) meets the plane at height z (table frame).

        `image_size` is the size of the image the pixel is in; K is rescaled if it differs from
        the calibrated size. None if the ray does not hit the plane in front of the camera.
        """
        if self.K is None:
            raise ValueError("camera intrinsics K missing from the calibration")
        K = np.asarray(self.K, float).copy()
        if image_size is not None and self.image_size is not None:
            K[0] *= image_size[0] / self.image_size[0]
            K[1] *= image_size[1] / self.image_size[1]
        ray_cam = np.linalg.solve(K, np.array([u, v, 1.0]))
        R, o = self.T_table_cam[:3, :3], self.T_table_cam[:3, 3]
        d = R @ ray_cam
        if abs(d[2]) < 1e-9:
            return None
        t = (z - o[2]) / d[2]
        if t <= 0:
            return None
        p = o + t * d
        return (float(p[0]), float(p[1]), float(p[2]))

    @classmethod
    def from_base(cls, arm: ArmNode, T_base_cam: np.ndarray, **kw: object) -> Calibration:
        return cls(base_transform(arm) @ np.asarray(T_base_cam, float), **kw)

    @classmethod
    def load(cls, path: str | Path, catalog: dict[str, ArmNode] | None = None) -> Calibration:
        d = json.loads(Path(path).read_text())
        if "T_table_cam" in d:
            T = np.asarray(d["T_table_cam"], float)
        else:
            if catalog is None or d["base_arm"] not in catalog:
                raise ValueError(f"{path}: base_arm {d.get('base_arm')!r} needs the arm catalog")
            T = cls.from_base(catalog[d["base_arm"]], d["T_base_cam"]).T_table_cam
        if T.shape != (4, 4):
            raise ValueError(f"{path}: camera transform must be 4x4")
        K = np.asarray(d["K"], float) if "K" in d else None
        if K is not None and K.shape != (3, 3):
            raise ValueError(f"{path}: K must be 3x3")
        size = tuple(int(v) for v in d["image_size"]) if "image_size" in d else None
        bounds = d.get("table_bounds_m")
        if bounds is not None:
            bounds = (tuple(map(float, bounds[0])), tuple(map(float, bounds[1])))
        return cls(T, K=K, image_size=size, table_bounds_m=bounds)
