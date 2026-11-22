"""Object detection from fiducial tags (AprilTag or ArUco) printed on top of each box.

The alternative to perception.pointing (the VLM points): one RGB camera, no model, a few ms per
frame, and every box gets its identity and its yaw from its tag.

1. OpenCV finds the tags (perception.markers.detect_markers; AprilTag families such as
   DICT_APRILTAG_36h11 are in cv2.aruco too).
2. Each tag's 4 corners are projected onto the plane of the box top (camera ray meets the plane
   z = box height, as in pointing.py). Centre = mean of the corners; yaw = direction of the tag's
   top edge (corner 0 -> corner 1) in the table frame. The camera pose from the table markers is
   all that is needed: the tag's printed size plays no part, so a slightly wrong size costs
   nothing.
3. The box pose is its centre at half its height (pointing.py's convention), turned by the yaw.
   The detection carries `key = "tag_<id>"`, so the box keeps that id in the Scene across
   re-scans, however far it moves. The yaw reaches the pick step as `pick_yaw_deg` (the jaws
   close across it), so stick each tag with its top edge along the box's long side.

Table markers (the calibration ones) are skipped by id. Give the boxes a different tag family,
or ids outside the table set, and list the table ids in the config ("markers": {"table_ids"}).

Config (in the table file, e.g. configs/hacklab_table.json):

    "tags": {
      "dictionary": "DICT_APRILTAG_36h11",
      "height_m": 0.04,                 # default box height = height of the tag plane
      "only_listed": false,             # true: ignore tags not in "boxes"
      "boxes": {
        "5": {"label": "small red box", "model": "box", "height_m": 0.03, "bin": "bowl"}
      }
    }

A per-box "bin" lets the plan be built without the VLM (tag_sorting).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hfteama.perception.frames import Calibration
from hfteama.perception.scene import Detection, Scene


@dataclass(frozen=True)
class TagBox:
    """What a tag id stands for. Every field is optional."""

    label: str = ""
    model: str = "box"
    height_m: float | None = None  # None: the detector's default
    bin: str | None = None  # where this box goes, for tag_sorting


@dataclass
class TagConfig:
    dictionary: str = "DICT_APRILTAG_36h11"
    height_m: float = 0.04
    only_listed: bool = False
    boxes: dict[int, TagBox] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TagConfig:
        boxes = {
            int(k): TagBox(
                label=str(v.get("label", "")),
                model=str(v.get("model", "box")),
                height_m=None if v.get("height_m") is None else float(v["height_m"]),
                bin=v.get("bin"),
            )
            for k, v in d.get("boxes", {}).items()
        }
        return cls(
            dictionary=d.get("dictionary", "DICT_APRILTAG_36h11"),
            height_m=float(d.get("height_m", 0.04)),
            only_listed=bool(d.get("only_listed", False)),
            boxes=boxes,
        )


def tag_key(tag_id: int) -> str:
    return f"tag_{tag_id:02d}"


class TagDetector:
    """perception.scene.ObjectDetector backed by fiducial tags on the boxes."""

    def __init__(
        self,
        calib: Calibration,
        config: TagConfig | None = None,
        ignore_ids: Iterable[int] = (),
    ) -> None:
        if calib.K is None:
            raise ValueError("TagDetector needs camera intrinsics K in the calibration")
        self.calib = calib
        self.config = config or TagConfig()
        self.ignore_ids = frozenset(ignore_ids)
        self.dropped: list[tuple[int, str]] = []  # (tag id, why) from the last call

    def detect(self, rgb: Any, depth: Any = None) -> list[Detection]:
        from hfteama.perception.markers import detect_markers

        img = np.asarray(rgb)
        h_px, w_px = img.shape[:2]
        cfg = self.config
        self.dropped = []
        out = []
        for tid, corners in sorted(detect_markers(img, cfg.dictionary).items()):
            if tid in self.ignore_ids:
                continue
            box = cfg.boxes.get(tid)
            if box is None and cfg.only_listed:
                self.dropped.append((tid, "not listed in the config"))
                continue
            box = box or TagBox()
            height = cfg.height_m if box.height_m is None else box.height_m
            pts = [self.calib.pixel_to_table(u, v, height, (w_px, h_px)) for u, v in corners]
            if any(p is None for p in pts):
                self.dropped.append((tid, "corner ray misses the box-top plane"))
                continue
            p = np.array(pts)
            cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
            if not self.calib.on_table((cx, cy)):
                self.dropped.append((tid, "not on the table"))
                continue
            edge = p[1] - p[0]  # tag's top edge, along its +x
            yaw = math.atan2(edge[1], edge[0])
            c, s = math.cos(yaw), math.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
            T[:3, 3] = (cx, cy, height / 2)
            u, v = corners.mean(axis=0)
            out.append(
                Detection(
                    T_cam_obj=self.calib.to_camera(T),
                    model=box.model,
                    pixel=(float(u), float(v)),
                    label=box.label or f"box with tag {tid}",
                    key=tag_key(tid),
                    has_yaw=True,
                )
            )
        return out


def tag_yaw_deg(quat_wxyz: tuple[float, float, float, float]) -> float:
    """Yaw about table z of a SceneObject's orientation, degrees."""
    w, x, y, z = quat_wxyz
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tag_sorting(scene: Scene, bins: list[Any], config: TagConfig, name: str = "sort") -> Any:
    """Plan straight from the config, no VLM: each listed box with a "bin" goes to that bin.

    Returns a perception.decomposer.Decomposition (one category per bin). Boxes without a bin in
    the config, and boxes already inside a bin, get no steps.
    """
    from hfteama.perception.decomposer import Category, Decomposition, Sorting, sort_steps
    from hfteama.taskgraph import TaskGraph

    by_id = {b.id: b for b in bins}
    by_key = {tag_key(t): b for t, b in config.boxes.items() if b.bin}
    for b in by_key.values():
        if b.bin not in by_id:
            raise ValueError(f"tag config names unknown bin {b.bin!r} (bins: {sorted(by_id)})")
    categories: dict[str, Category] = {}
    assignment: dict[str, str] = {}
    tasks = {}
    done = []
    for oid in scene.ids():
        box = by_key.get(oid)
        if box is None:
            continue
        obj, bin_ = scene.objects[oid], by_id[box.bin]
        if bin_.contains(obj.xyz):
            done.append(oid)
            continue
        cat = f"to_{bin_.id}"
        categories.setdefault(cat, Category(cat, f"tags listed for {bin_.id}", bin_.id))
        assignment[oid] = cat
        for t in sort_steps(obj, cat, bin_):
            tasks[t.id] = t
    graph = TaskGraph(name=name, tasks=tasks)
    graph.validate()
    return Decomposition(Sorting(categories, assignment), graph, done, attempts=0)


def tag_decomposer(bins: list[Any], config: TagConfig) -> Any:
    """A perception.decomposer.Decomposer that sorts from the tag config instead of asking a VLM,
    so `perception.rescan` (new, moved, dropped boxes) works the same way with tags. Only boxes
    listed with a "bin" are work; every other tag is left alone, as in tag_sorting."""
    from hfteama.perception.decomposer import Category, Decomposer, Sorting

    by_key = {tag_key(t): b.bin for t, b in config.boxes.items() if b.bin}

    class TagDecomposer(Decomposer):
        def unsorted(self, scene: Scene) -> tuple[list[str], list[str]]:
            todo, done = super().unsorted(scene)
            return [o for o in todo if o in by_key], [o for o in done if o in by_key]

        def categorize(
            self, image: Any, scene: Scene, object_ids: list[str], fixed: Any = None
        ) -> tuple[Any, int]:
            categories = dict(fixed or {})
            assignment = {}
            for oid in object_ids:
                bin_id = by_key[oid]
                cat = f"to_{bin_id}"
                categories.setdefault(cat, Category(cat, f"tags listed for {bin_id}", bin_id))
                assignment[oid] = cat
            return Sorting(categories, assignment), 0

    dec = TagDecomposer(None, bins)  # type: ignore[arg-type]
    for bin_id in by_key.values():
        if bin_id not in dec.bins_by_id:
            raise ValueError(
                f"tag config names unknown bin {bin_id!r} (bins: {sorted(dec.bins_by_id)})"
            )
    return dec
