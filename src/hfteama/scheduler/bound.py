"""Lower bound on finish time, and which limit binds. Shown live on the demo screen.

finish >= max(total work / N, longest chain of ordered steps, busiest shared resource).
Binding term: "arms" (more arms help), "order" (dependencies cap it), "resource:<name>".
Each task counts at its fastest allowed arm, so the bound stays valid with unequal arms.
"""

from __future__ import annotations

from dataclasses import dataclass

from hfteama.taskgraph import Task, TaskGraph


@dataclass(frozen=True)
class LowerBound:
    value: float
    binding: str  # "arms" | "order" | "resource:<name>"
    terms: dict[str, float]


def _best(task: Task, arms: list[str]) -> float:
    options = [task.duration_on(a) for a in arms if task.can_run_on(a)]
    if not options:
        raise ValueError(f"no available arm can run {task.id}")
    return min(options) * task.needs_arms


def lower_bound(
    graph: TaskGraph, arms: list[str], done: frozenset[str] = frozenset()
) -> LowerBound:
    """Bound on the remaining work (tasks not in `done`), measured from now."""
    if not arms:
        raise ValueError("no arms")
    todo = {tid: t for tid, t in graph.tasks.items() if tid not in done}
    cost = {tid: _best(t, arms) for tid, t in todo.items()}

    chain: dict[str, float] = {}
    for tid in graph.topo_order():
        if tid in todo:
            prev = [chain[d] for d in todo[tid].deps if d in chain]
            chain[tid] = cost[tid] / todo[tid].needs_arms + max(prev, default=0.0)

    terms = {"arms": sum(cost.values()) / len(arms), "order": max(chain.values(), default=0.0)}
    for tid, t in todo.items():
        for r in t.resources:
            key = f"resource:{r}"
            terms[key] = terms.get(key, 0.0) + cost[tid] / t.needs_arms
    binding = max(terms, key=lambda k: terms[k])
    return LowerBound(value=terms[binding], binding=binding, terms=terms)
