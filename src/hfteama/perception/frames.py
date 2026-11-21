"""Camera frame -> table frame. Object poses come from SAM-6D in the camera frame; the
camera-to-robot-base extrinsic comes from calibration (Hardware). The table frame is the frame
of configs/arms.json (base_pose_m), so every arm and object share one frame.

configs/camera.json, either form (see configs/camera.example.json):
    {"T_table_cam": 4x4}                                  camera pose in the table frame
    {"base_arm": "arm_a", "T_base_cam": 4x4}              camera pose in one arm's base frame
                                                          (the base pose comes from the catalog)
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

    def to_table(self, T_cam_obj: np.ndarray) -> np.ndarray:
        return self.T_table_cam @ np.asarray(T_cam_obj, float)

    @classmethod
    def from_base(cls, arm: ArmNode, T_base_cam: np.ndarray) -> Calibration:
        return cls(base_transform(arm) @ np.asarray(T_base_cam, float))

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
        return cls(T)
