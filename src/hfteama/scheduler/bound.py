"""Lower bound on finish time, and which limit binds. Shown live on the demo screen.

finish >= max(total work / N, longest chain of ordered steps, busiest shared resource), where a
shared resource counts the earliest its steps can start, their total time, and the shortest work
that must still follow one of them.
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

    own = {tid: cost[tid] / todo[tid].needs_arms for tid in todo}
    chain: dict[str, float] = {}
    for tid in graph.topo_order():
        if tid in todo:
            prev = [chain[d] for d in todo[tid].deps if d in chain]
            chain[tid] = cost[tid] / todo[tid].needs_arms + max(prev, default=0.0)

    # tail[t]: the longest chain of work that must still follow t (dependants and the step tied
    # to it by same_arm_as), not counting t itself
    after: dict[str, list[str]] = {tid: [] for tid in todo}
    for tid, t in todo.items():
        for d in {*t.deps, *([t.same_arm_as] if t.same_arm_as else [])}:
            if d in after:
                after[d].append(tid)
    tail: dict[str, float] = {}
    for tid in reversed([t for t in graph.topo_order() if t in todo]):
        tail[tid] = max((own[c] + tail[c] for c in after[tid]), default=0.0)

    terms = {"arms": sum(cost.values()) / len(arms), "order": max(chain.values(), default=0.0)}
    # A shared resource is used by one step at a time: its steps take their total time, and that
    # block can start no sooner than the earliest of them may start and is followed by at least
    # the shortest work that must come after one of them (the classic one-machine bound).
    by_res: dict[str, list[str]] = {}
    for tid, t in todo.items():
        for r in t.resources:
            by_res.setdefault(r, []).append(tid)
    for r, tids in by_res.items():
        head = min(chain[t] - own[t] for t in tids)
        terms[f"resource:{r}"] = head + sum(own[t] for t in tids) + min(tail[t] for t in tids)
    binding = max(terms, key=lambda k: terms[k])
    return LowerBound(value=terms[binding], binding=binding, terms=terms)
