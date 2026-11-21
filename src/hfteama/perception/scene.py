"""What is on the table: detections turned into objects with stable ids in the table frame.

A detector (perception.pointing: the VLM points, the pixel is projected onto the table) gives a
pose per detected instance in the camera frame, but no identity across frames (two red cubes
look the same). Ids are kept stable by matching each new detection to the nearest
known object of the same model within `match_radius`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from hfteama.perception.frames import Calibration, quat_wxyz


@dataclass
class Detection:
    """One detected object instance."""

    T_cam_obj: np.ndarray  # 4x4 pose in the camera frame, metres
    model: str = ""  # kind of object, used to match ids across frames
    score: float = 1.0
    pixel: tuple[float, float] | None = None  # centre in the image, for labelling it to the VLM
    label: str = ""  # free description from the detector, e.g. "red cube" (shown to the VLM)


class ObjectDetector(Protocol):
    """Any object detector / pose estimator behind one call."""

    def detect(self, rgb: Any, depth: Any = None) -> list[Detection]: ...


@dataclass(frozen=True)
class SceneObject:
    id: str
    model: str
    xyz: tuple[float, float, float]  # table frame, metres
    quat_wxyz: tuple[float, float, float, float]
    pixel: tuple[float, float] | None = None
    score: float = 1.0
    label: str = ""


@dataclass
class Scene:
    objects: dict[str, SceneObject] = field(default_factory=dict)
    t: float = field(default_factory=time.time)
    next_index: int = 1  # for new ids, kept across re-scans

    def ids(self) -> list[str]:
        return sorted(self.objects)


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.dist(a[:3], b[:3])


def build_scene(
    detections: list[Detection],
    calib: Calibration,
    previous: Scene | None = None,
    match_radius: float = 0.05,
    min_score: float = 0.5,
) -> Scene:
    """Detections -> Scene in the table frame, reusing ids from `previous` where they match."""
    found = []
    for d in detections:
        if d.score < min_score:
            continue
        T = calib.to_table(d.T_cam_obj)
        xyz = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        found.append((d, xyz, quat_wxyz(T[:3, :3])))

    prev = dict(previous.objects) if previous else {}
    index = previous.next_index if previous else 1
    # greedy nearest match, closest pairs first
    pairs = sorted(
        (_dist(xyz, o.xyz), i, oid)
        for i, (d, xyz, _) in enumerate(found)
        for oid, o in prev.items()
        if o.model == d.model and _dist(xyz, o.xyz) <= match_radius
    )
    ids: dict[int, str] = {}
    used: set[str] = set()
    for _, i, oid in pairs:
        if i not in ids and oid not in used:
            ids[i] = oid
            used.add(oid)

    objects: dict[str, SceneObject] = {}
    for i, (d, xyz, q) in enumerate(found):
        oid = ids.get(i)
        if oid is None:
            oid = f"obj_{index:02d}"
            index += 1
        objects[oid] = SceneObject(oid, d.model, xyz, q, d.pixel, d.score, d.label)
    return Scene(objects=objects, next_index=index)


@dataclass(frozen=True)
class SceneDiff:
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    moved: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.moved)


def diff_scenes(expected: Scene, observed: Scene, move_tol: float = 0.015) -> SceneDiff:
    """Compare the scene we expect (after the steps done so far) with what the camera sees."""
    added = tuple(sorted(set(observed.objects) - set(expected.objects)))
    removed = tuple(sorted(set(expected.objects) - set(observed.objects)))
    moved = tuple(
        sorted(
            oid
            for oid in set(expected.objects) & set(observed.objects)
            if _dist(expected.objects[oid].xyz, observed.objects[oid].xyz) > move_tol
        )
    )
    return SceneDiff(added, removed, moved)
