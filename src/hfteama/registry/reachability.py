"""Reachability: which connected arms can reach each step; steps no arm reaches are BLOCKED.

Owner: Operations (#21, backlog #3).

Task-agnostic convention: every task arg whose name ends in "_xyz" is a table-frame point
(metres) the arm has to reach, e.g. pick_xyz, place_xyz. A task is allowed on an arm when the
arm reaches all of its points (and the arm is in the task's own allowed_arms, if it lists any).
Tasks linked by same_arm_as (the arm holding a part also places it) share one arm, so their
allowed sets are intersected. A task with no allowed arm is BLOCKED, and so is everything that
depends on it. Blocked steps are not dropped: they unblock when an arm that reaches them joins.

No handover chains in v1 (two-arm steps are a later extension, docs/plan.md).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

from hfteama.registry.arms import ArmNode
from hfteama.taskgraph import Task, TaskGraph


def task_points(task: Task) -> list[tuple[float, ...]]:
    """Table-frame points a task needs to reach (args ending in _xyz)."""
    return [tuple(v) for k, v in sorted(task.args.items()) if k.endswith("_xyz")]


def arms_reaching(task: Task, arms: Iterable[ArmNode]) -> set[str]:
    pts = task_points(task)
    return {
        a.arm_id
        for a in arms
        if task.can_run_on(a.arm_id, a.capabilities or None) and all(a.reaches(p) for p in pts)
    }


def _same_arm_groups(graph: TaskGraph) -> list[set[str]]:
    parent = {tid: tid for tid in graph.tasks}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tid, t in graph.tasks.items():
        if t.same_arm_as is not None:
            parent[find(tid)] = find(t.same_arm_as)
    groups: dict[str, set[str]] = {}
    for tid in graph.tasks:
        groups.setdefault(find(tid), set()).add(tid)
    return list(groups.values())


@dataclass
class ReachResult:
    graph: TaskGraph  # every task, allowed_arms set to the connected arms that reach it
    blocked: frozenset[str]  # no connected arm reaches it, or it depends on a blocked task
    unreachable: frozenset[str]  # the blocked tasks that are blocked by reach itself (the cause)

    def runnable(self) -> TaskGraph:
        """The graph without blocked tasks: what the scheduler can plan now."""
        tasks = {tid: t for tid, t in self.graph.tasks.items() if tid not in self.blocked}
        return TaskGraph(name=self.graph.name, tasks=tasks)


def clearance_zones(points: list[tuple[float, ...]], clearance_m: float) -> list[str]:
    """Table cells (size clearance/4) touched by a disc of radius clearance/2 around each point.

    Two points closer than clearance_m have overlapping discs, so they share a cell: their steps
    lock a common resource and never run at the same time. Points farther apart than
    clearance_m + one cell diagonal (about 1.35 x clearance_m) never share one.
    """
    cell = clearance_m / 4.0
    r = clearance_m / 2.0
    out: list[str] = []
    for p in points:
        x, y = p[0], p[1]
        for i in range(math.floor((x - r) / cell), math.floor((x + r) / cell) + 1):
            for j in range(math.floor((y - r) / cell), math.floor((y + r) / cell) + 1):
                # nearest point of cell (i, j) to the disc centre
                nx = min(max(x, i * cell), (i + 1) * cell)
                ny = min(max(y, j * cell), (j + 1) * cell)
                z = f"zone:{i},{j}"
                if math.hypot(nx - x, ny - y) <= r and z not in out:
                    out.append(z)
    return out


def resolve(
    graph: TaskGraph,
    arms: Iterable[ArmNode],
    travel_s_per_m: float | None = None,
    clearance_m: float | None = None,
) -> ReachResult:
    """Fill allowed_arms from reach, and find the steps none of `arms` can do.

    With the planner on orchestrator, pass the whole catalog (every arm we own, plugged in or
    not): allowed_arms then lists every arm that could reach a step, and the planner's own
    blocked_tasks / unblocked_by work out what the connected arms can do and what a newly
    plugged-in arm unblocks. Passing only the connected arms gives the same `blocked` here, but
    then a new arm would be missing from allowed_arms until resolve is run again.

    Always pass the source graph (as decomposed), not a previous result's graph.

    travel_s_per_m: if set, each allowed arm gets its own duration for the step (task.durations):
    the step's duration plus this many seconds per metre from the arm's pan axis to the step's
    points (architecture doc 4.3), so the planner prefers the nearer arm. A prior only: once an
    arm has measured runs of a skill, the executor's DurationModel uses those instead.

    clearance_m: if set, a step that two or more arms can reach locks the table cells around its
    points (resources "zone:i,j", see clearance_zones), so two arms never work on points closer
    than clearance_m at the same time. Steps only one arm can reach take no zone locks.
    """
    arms = list(arms)
    allowed = {tid: arms_reaching(t, arms) for tid, t in graph.tasks.items()}
    for group in _same_arm_groups(graph):
        common = set.intersection(*(allowed[tid] for tid in group))
        for tid in group:
            allowed[tid] = common

    unreachable = {tid for tid, a in allowed.items() if not a}
    blocked = set(unreachable)
    for tid in graph.topo_order():
        if any(d in blocked for d in graph.tasks[tid].deps):
            blocked.add(tid)

    by_id = {a.arm_id: a for a in arms}
    tasks = {}
    for tid, t in graph.tasks.items():
        durations = dict(t.durations)
        resources = list(t.resources)
        pts = task_points(t)
        if travel_s_per_m and pts:
            for arm_id in allowed[tid]:
                a = by_id[arm_id]
                reach = sum(math.dist(a.pan_axis_xy, p[:2]) for p in pts) / len(pts)
                durations[arm_id] = t.duration_on(arm_id) + travel_s_per_m * reach
        if clearance_m and pts and len(allowed[tid]) >= 2:
            resources += [z for z in clearance_zones(pts, clearance_m) if z not in resources]
        tasks[tid] = replace(
            t,
            allowed_arms=tuple(sorted(allowed[tid])),
            durations=durations,
            resources=tuple(resources),
        )
    return ReachResult(
        graph=TaskGraph(name=graph.name, tasks=tasks),
        blocked=frozenset(blocked),
        unreachable=frozenset(unreachable),
    )


def newly_unblocked(before: ReachResult, after: ReachResult) -> set[str]:
    """Steps that only became possible because of an arm change (shown on screen)."""
    return {tid for tid in before.blocked - after.blocked if tid in after.graph.tasks}
