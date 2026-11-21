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


def resolve(graph: TaskGraph, arms: Iterable[ArmNode]) -> ReachResult:
    """Fill allowed_arms for the connected arms and find the blocked steps.

    Call again whenever an arm joins or leaves (fast: tasks x arms radius checks). Always pass
    the source graph (as decomposed), not a previous result's graph: a result's allowed_arms
    lists only the arms connected at that time, so a new arm would be filtered out.
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

    tasks = {
        tid: replace(t, allowed_arms=tuple(sorted(allowed[tid]))) for tid, t in graph.tasks.items()
    }
    return ReachResult(
        graph=TaskGraph(name=graph.name, tasks=tasks),
        blocked=frozenset(blocked),
        unreachable=frozenset(unreachable),
    )


def newly_unblocked(before: ReachResult, after: ReachResult) -> set[str]:
    """Steps that only became possible because of an arm change (shown on screen)."""
    return {tid for tid in before.blocked - after.blocked if tid in after.graph.tasks}
