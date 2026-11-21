"""Sorter config, where each arm stands on the table, and fitting that from touched points.

An arm's placement is its base_link origin (x, y) in the table frame and its yaw: the heading,
in the table frame, of the arm's +x axis (the way it points at shoulder_pan = 0). The arms face
the camera, so yaw is about -90 deg. Table and arm z are both 0 on the table top.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sorter.json"
# Written by `hfteama sort touch`; overrides each arm's base/yaw_deg from the config.
PLACEMENTS_FILE = PROJECT_ROOT / "configs" / "sorter_placements.json"


def load_config(path: str | Path | None = None, placements: str | Path | None = None) -> dict:
    """The sorter config with fitted arm placements (if any) applied on top."""
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    pfile = Path(placements or PLACEMENTS_FILE)
    if pfile.exists():
        for side, fitted in json.loads(pfile.read_text()).items():
            if side in cfg["arms"]:
                cfg["arms"][side].update({k: fitted[k] for k in ("base", "yaw_deg") if k in fitted})
    return cfg


def resolve(path: str | Path) -> Path:
    """Config paths are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class Placement:
    x: float
    y: float
    yaw: float  # rad

    @classmethod
    def from_config(cls, arm: dict) -> Placement:
        return cls(x=arm["base"][0], y=arm["base"][1], yaw=np.radians(arm["yaw_deg"]))

    def _R(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s], [s, c]])

    def to_arm(self, p) -> np.ndarray:
        """Table point (x, y[, z]) -> arm frame (same length)."""
        p = np.asarray(p, float)
        xy = self._R().T @ (p[:2] - [self.x, self.y])
        return np.concatenate([xy, p[2:]])

    def to_table(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        xy = self._R() @ p[:2] + [self.x, self.y]
        return np.concatenate([xy, p[2:]])

    def yaw_to_arm(self, yaw_table: float) -> float:
        return yaw_table - self.yaw


def fit_placement(arm_xy, table_xy, yaw_guess: float | None = None) -> tuple[Placement, float]:
    """Rigid 2D fit table = R(yaw) arm + (x, y) from matched points (tool tip in the arm frame,
    from FK, and the same point in the table frame, from the camera).

    With one point only the translation is fitted and yaw_guess is kept. Returns the placement
    and the rms residual (m).
    """
    A = np.atleast_2d(np.asarray(arm_xy, float))[:, :2]
    B = np.atleast_2d(np.asarray(table_xy, float))[:, :2]
    if len(A) == 1:
        if yaw_guess is None:
            raise ValueError("one point needs a yaw guess")
        p = Placement(0.0, 0.0, yaw_guess)
        t = B[0] - p.to_table(A[0])
        return Placement(float(t[0]), float(t[1]), yaw_guess), 0.0
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    yaw = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
    p = Placement(0.0, 0.0, yaw)
    t = cb - p.to_table(ca)
    p = Placement(float(t[0]), float(t[1]), yaw)
    res = np.array([p.to_table(a) - b for a, b in zip(A, B, strict=True)])
    return p, float(np.sqrt(np.mean(np.sum(res**2, axis=1))))


def save_placement(side: str, p: Placement, rms: float, n: int, path: str | Path | None = None):
    path = Path(path or PLACEMENTS_FILE)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[side] = {
        "base": [round(p.x, 4), round(p.y, 4)],
        "yaw_deg": round(float(np.degrees(p.yaw)), 2),
        "fit_rms_mm": round(rms * 1000, 1),
        "points": n,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path
