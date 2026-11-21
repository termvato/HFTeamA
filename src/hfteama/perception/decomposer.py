"""Decomposer: camera + Go -> categories -> steps (a task graph in the format on main).

1. SAM-6D gives the objects and their table-frame poses (perception.scene).
2. The VLM sees the image with object ids drawn on it and decides the categories and which
   object goes to which bin. It never chooses arms, positions or timings.
3. The answer is validated (format, known ids, one bin per category, every object assigned);
   on error the VLM is asked again with the errors, up to 3 retries, then DecompositionError
   (the operator decides).
4. Code builds the steps: per object a pick, then a place on the same arm (same_arm_as), with
   the table coordinates in args (pick_xyz, place_xyz), so reachability can fill allowed_arms.

Objects already inside a bin count as sorted and get no steps.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hfteama.perception.annotate import label_objects
from hfteama.perception.prompts import SYSTEM, sort_prompt
from hfteama.perception.scene import Scene, SceneObject
from hfteama.perception.vlm import VLM, ParseError, VLMError, parse_json
from hfteama.taskgraph import Task, TaskGraph

# Placeholder step durations (s), as in tasks/sort/sort12.json, until measured (backlog #13).
DEFAULT_DURATIONS = {"pick": 4.0, "place": 6.0}


class DecompositionError(RuntimeError):
    """The VLM gave no valid sorting after all retries; ask the operator."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("no valid sorting from the VLM: " + "; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Bin:
    id: str
    label: str
    xyz: tuple[float, float, float]  # where to drop, table frame, metres
    radius_m: float = 0.08  # an object within this distance (in x, y) is inside the bin
    shared: bool = True  # a shared area: one arm at a time (a resource in the task graph)

    def contains(self, xyz: tuple[float, ...]) -> bool:
        return math.hypot(xyz[0] - self.xyz[0], xyz[1] - self.xyz[1]) <= self.radius_m


def load_bins(path: str | Path) -> list[Bin]:
    raw = json.loads(Path(path).read_text())
    bins = [
        Bin(
            id=d["id"],
            label=d.get("label", d["id"]),
            xyz=tuple(float(v) for v in d["xyz"]),
            radius_m=float(d.get("radius_m", 0.08)),
            shared=bool(d.get("shared", True)),
        )
        for d in raw["bins"]
    ]
    if len({b.id for b in bins}) != len(bins):
        raise ValueError(f"{path}: duplicate bin id")
    return bins


@dataclass(frozen=True)
class Category:
    name: str
    rule: str
    bin: str


@dataclass
class Sorting:
    categories: dict[str, Category]  # name -> category
    assignment: dict[str, str]  # object id -> category name

    def bin_of(self, obj_id: str) -> str:
        return self.categories[self.assignment[obj_id]].bin


def validate_sorting(
    raw: dict[str, Any],
    object_ids: list[str],
    bins: list[Bin],
    fixed: dict[str, Category] | None = None,
) -> tuple[Sorting | None, list[str]]:
    """Check a VLM answer. Returns (sorting, []) or (None, errors to send back)."""
    errors: list[str] = []
    bin_ids = {b.id for b in bins}
    fixed = fixed or {}
    cats_raw, asg_raw = raw.get("categories"), raw.get("assignments")
    if not isinstance(cats_raw, list) or not isinstance(asg_raw, list):
        return None, ['"categories" and "assignments" must both be lists']

    categories: dict[str, Category] = dict(fixed)
    for c in cats_raw:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not c["name"].strip():
            errors.append(f"category {c!r} needs a non-empty name")
            continue
        name, b = c["name"].strip(), c.get("bin")
        if name in fixed:
            if b != fixed[name].bin:
                errors.append(f"category {name!r} must keep bin {fixed[name].bin!r}")
            continue
        if name in categories:
            errors.append(f"category {name!r} is listed twice")
            continue
        if b not in bin_ids:
            errors.append(f"category {name!r} uses unknown bin {b!r}")
            continue
        categories[name] = Category(name, str(c.get("rule", "")), b)

    used: dict[str, str] = {}
    for cat in categories.values():
        if cat.bin in used:
            errors.append(f"bin {cat.bin!r} is used by {used[cat.bin]!r} and {cat.name!r}")
        used[cat.bin] = cat.name

    assignment: dict[str, str] = {}
    wanted = set(object_ids)
    for a in asg_raw:
        if not isinstance(a, dict):
            errors.append(f"assignment {a!r} is not an object")
            continue
        oid, cat = a.get("object"), a.get("category")
        cat = cat.strip() if isinstance(cat, str) else cat
        if oid not in wanted:
            errors.append(f"unknown object {oid!r}")
        elif oid in assignment:
            errors.append(f"object {oid!r} is assigned twice")
        elif cat not in categories:
            errors.append(f"object {oid!r} has unknown category {cat!r}")
        else:
            assignment[oid] = cat
    missing = sorted(wanted - set(assignment))
    if missing:
        errors.append(f"objects not assigned: {', '.join(missing)}")
    if errors:
        return None, errors
    return Sorting(categories, assignment), []


@dataclass
class Decomposition:
    sorting: Sorting
    graph: TaskGraph
    already_sorted: list[str] = field(default_factory=list)  # objects already inside a bin
    attempts: int = 1  # VLM calls needed (1 = valid on the first try)

    def grouping(self) -> dict[str, Any]:
        """For the screen: the VLM's categories and which objects are in each."""
        return {
            name: {
                "rule": c.rule,
                "bin": c.bin,
                "objects": sorted(o for o, n in self.sorting.assignment.items() if n == name),
            }
            for name, c in self.sorting.categories.items()
        }


def sort_steps(
    obj: SceneObject,
    category: str,
    bin_: Bin,
    durations: dict[str, float] | None = None,
    suffix: str = "",
) -> list[Task]:
    """Pick then place for one object (the arm holding it also places it)."""
    d = {**DEFAULT_DURATIONS, **(durations or {})}
    pick_id, place_id = f"pick_{obj.id}{suffix}", f"place_{obj.id}{suffix}"
    return [
        Task(
            id=pick_id,
            skill="pick",
            args={
                "object": obj.id,
                "model": obj.model,
                "category": category,
                "pick_xyz": list(obj.xyz),
                "pick_quat_wxyz": list(obj.quat_wxyz),
            },
            duration=d["pick"],
        ),
        Task(
            id=place_id,
            skill="place",
            args={
                "object": obj.id,
                "category": category,
                "target": bin_.id,
                "place_xyz": list(bin_.xyz),
            },
            deps=(pick_id,),
            resources=(bin_.id,) if bin_.shared else (),
            duration=d["place"],
            same_arm_as=pick_id,
        ),
    ]


class Decomposer:
    def __init__(
        self,
        vlm: VLM,
        bins: list[Bin],
        max_retries: int = 3,
        durations: dict[str, float] | None = None,
        annotate: bool = True,
    ) -> None:
        if not bins:
            raise ValueError("no bins configured")
        self.vlm = vlm
        self.bins = bins
        self.bins_by_id = {b.id: b for b in bins}
        self.max_retries = max_retries
        self.durations = durations
        self.annotate = annotate

    def unsorted(self, scene: Scene) -> tuple[list[str], list[str]]:
        """(objects to sort, objects already inside a bin)."""
        todo, done = [], []
        for oid in scene.ids():
            inside = any(b.contains(scene.objects[oid].xyz) for b in self.bins)
            (done if inside else todo).append(oid)
        return todo, done

    def categorize(
        self,
        image: Any,
        scene: Scene,
        object_ids: list[str],
        fixed: dict[str, Category] | None = None,
    ) -> tuple[Sorting, int]:
        """Ask the VLM for categories of `object_ids`. Returns (sorting, attempts)."""
        if not object_ids:
            return Sorting(dict(fixed or {}), {}), 0
        img = label_objects(image, scene, object_ids) if self.annotate else image
        objs = [(o, scene.objects[o].model, scene.objects[o].pixel) for o in object_ids]
        bins = [(b.id, b.label) for b in self.bins]
        fixed_txt = {n: (c.rule, c.bin) for n, c in (fixed or {}).items()}
        errors: list[str] = []
        for attempt in range(1, self.max_retries + 2):
            try:
                text = self.vlm.ask(img, sort_prompt(objs, bins, fixed_txt, errors or None), SYSTEM)
            except VLMError as e:
                raise DecompositionError([f"VLM unavailable: {e}"]) from e
            try:
                sorting, errors = validate_sorting(parse_json(text), object_ids, self.bins, fixed)
            except ParseError as e:
                sorting, errors = None, [str(e)]
            if sorting is not None:
                return sorting, attempt
        raise DecompositionError(errors)

    def decompose(self, image: Any, scene: Scene, name: str = "sort") -> Decomposition:
        """Go: the whole table -> categories -> task graph."""
        todo, done = self.unsorted(scene)
        sorting, attempts = self.categorize(image, scene, todo)
        tasks: dict[str, Task] = {}
        for oid in todo:
            cat = sorting.assignment[oid]
            for t in sort_steps(
                scene.objects[oid], cat, self.bins_by_id[sorting.bin_of(oid)], self.durations
            ):
                tasks[t.id] = t
        graph = TaskGraph(name=name, tasks=tasks)
        graph.validate()
        return Decomposition(sorting, graph, done, attempts)


def graph_to_dict(graph: TaskGraph) -> dict[str, Any]:
    """Task graph -> the JSON format of hfteama.taskgraph (what load() reads)."""
    tasks = []
    for t in graph.tasks.values():
        d = asdict(t)
        d["deps"] = list(t.deps)
        d["resources"] = list(t.resources)
        d["allowed_arms"] = t.allowed_arms if t.allowed_arms == "*" else list(t.allowed_arms)
        tasks.append({k: v for k, v in d.items() if v is not None})
    return {"name": graph.name, "tasks": tasks}
