"""Re-scan: the camera notices new, moved or vanished objects mid-run and the steps follow.

Call it with a fresh SAM-6D scene (built with previous=the last scene, so ids carry over).
Only objects outside the bins are work. Rules:
- An object with a running or finished step is left alone (running work is never preempted).
- Object gone (or put in a bin by hand): its pending steps are dropped.
- Object moved: its pending pick gets the new pose.
- New object: the VLM sorts it into the existing categories (a new one only if a bin is free),
  and it gets a pick and a place.
- A sorted object seen outside the bins again (dropped): it is sorted again (steps "_r<n>").
- Objects higher than `max_z` above the table (table frame z = 0 at the surface) are being held
  by an arm and are ignored, so a carried object is not mistaken for a new one.

Returns the updated source graph; run registry.resolve() on it, then re-plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from hfteama.perception.decomposer import Decomposer, Sorting, pick_pose_args, sort_steps
from hfteama.perception.scene import Scene, SceneDiff, diff_scenes
from hfteama.taskgraph import TaskGraph


@dataclass
class Rescan:
    scene: Scene  # what the camera sees now; the `expected` scene for the next re-scan
    diff: SceneDiff  # objects outside the bins: added / removed / moved
    graph: TaskGraph
    sorting: Sorting
    added: list[str] = field(default_factory=list)  # new task ids
    dropped: list[str] = field(default_factory=list)  # pending task ids removed
    updated: list[str] = field(default_factory=list)  # pending task ids with a new pose

    @property
    def changed(self) -> bool:
        return bool(self.added or self.dropped or self.updated)


def _outside(dec: Decomposer, scene: Scene, max_z: float) -> Scene:
    """Objects that are work: on the table, not in a bin, not held up by an arm."""
    todo, _ = dec.unsorted(scene)
    keep = {o: scene.objects[o] for o in todo if scene.objects[o].xyz[2] <= max_z}
    return Scene(keep, scene.t, scene.next_index)


def rescan(
    dec: Decomposer,
    image: Any,
    observed: Scene,
    expected: Scene,
    graph: TaskGraph,
    sorting: Sorting,
    done: set[str],
    running: set[str],
    move_tol: float = 0.015,
    max_z: float = 0.06,
) -> Rescan:
    obs, exp = _outside(dec, observed, max_z), _outside(dec, expected, max_z)
    diff = diff_scenes(exp, obs, move_tol)

    by_obj: dict[str, list[str]] = {}
    for tid, t in graph.tasks.items():
        if "object" in t.args:
            by_obj.setdefault(t.args["object"], []).append(tid)
    started = done | running

    tasks = dict(graph.tasks)
    dropped, updated = [], []
    for oid, tids in by_obj.items():
        pending = [t for t in tids if t not in started]
        if len(pending) != len(tids):
            continue  # in progress or finished: leave it
        if oid not in obs.objects:
            for tid in pending:
                del tasks[tid]
            dropped += pending
        elif oid in diff.moved:
            o = obs.objects[oid]
            for tid in pending:
                t = tasks[tid]
                if "pick_xyz" in t.args:
                    args = {k: v for k, v in t.args.items() if k != "pick_yaw_deg"}
                    tasks[tid] = replace(t, args={**args, **pick_pose_args(o)})
                    updated.append(tid)

    # objects that need (new) steps: never seen, or finished but outside the bins again
    need: list[str] = []
    for oid in obs.ids():
        tids = by_obj.get(oid, [])
        if not tids or all(t in done for t in tids):
            need.append(oid)

    new_sorting = sorting
    added: list[str] = []
    if need:
        part, _ = dec.categorize(image, observed, need, fixed=sorting.categories)
        new_sorting = Sorting(
            categories=part.categories, assignment={**sorting.assignment, **part.assignment}
        )
        for oid in need:
            n = 0
            suffix = ""
            while f"pick_{oid}{suffix}" in tasks:
                n += 1
                suffix = f"_r{n}"
            cat = part.assignment[oid]
            bin_ = dec.bins_by_id[part.categories[cat].bin]
            for t in sort_steps(obs.objects[oid], cat, bin_, dec.durations, suffix):
                tasks[t.id] = t
                added.append(t.id)

    new_graph = TaskGraph(name=graph.name, tasks=tasks)
    new_graph.validate()
    return Rescan(observed, diff, new_graph, new_sorting, added, dropped, updated)
