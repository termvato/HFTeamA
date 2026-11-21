"""Named places and objects on the table, in the shared table frame (metres).

A task folder keeps them in `tasks/<name>/poses.json`, so the core stays task-agnostic:

    {"table_z": 0.0,
     "locations": {"input_bin": [0.30, 0.20, 0.02]},
     "per_arm": {"arm_a": {"own_output_bin": [0.10, -0.15, 0.02]}},
     "objects": {"block_01": [0.31, 0.21, 0.015]},
     "arms": {"arm_a": {"base": [0.0, 0.26, 0.0], "yaw": 1.57}}}

`arms` (base poses) is only read by the simulator. Real arms get theirs from Hardware's config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hfteama.arms.primitives import SkillFailed

Xyz = list[float]


@dataclass
class Poses:
    locations: dict[str, Xyz] = field(default_factory=dict)
    per_arm: dict[str, dict[str, Xyz]] = field(default_factory=dict)
    objects: dict[str, Xyz] = field(default_factory=dict)
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    table_z: float = 0.0

    @classmethod
    def load(cls, path: str | Path) -> Poses:
        raw = json.loads(Path(path).read_text())
        return cls(
            locations=raw.get("locations", {}),
            per_arm=raw.get("per_arm", {}),
            objects=raw.get("objects", {}),
            arms=raw.get("arms", {}),
            table_z=float(raw.get("table_z", 0.0)),
        )

    @classmethod
    def empty(cls) -> Poses:
        return cls()

    def _lookup(self, name: str, arm_id: str) -> Xyz | None:
        for source in (self.per_arm.get(arm_id, {}), self.locations, self.objects):
            if name in source:
                return source[name]
        return None

    def has(self, name: str, arm_id: str) -> bool:
        return self._lookup(name, arm_id) is not None

    def point(self, name: str, arm_id: str) -> np.ndarray:
        """Table-frame point for `name`: this arm's own entry, then locations, then objects."""
        xyz = self._lookup(name, arm_id)
        if xyz is None:
            raise SkillFailed(f"unknown location: {name}")
        return np.asarray(xyz, float)
