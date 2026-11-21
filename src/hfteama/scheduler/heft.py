"""HEFT (backlog #6): full schedule, the default planner and CP-SAT's warm start.

Steps are ranked by their longest remaining path to the end of the batch (critical work first),
then each goes to the arm that finishes it earliest, filling idle gaps where it fits.
"""

from __future__ import annotations

from hfteama.scheduler.builder import Builder
from hfteama.scheduler.plan import Plan
from hfteama.scheduler.problem import PlanError, Problem


def upward_rank(p: Problem) -> dict[str, float]:
    children: dict[str, list[str]] = {t: [] for t in p.graph.tasks}
    for t in p.graph.tasks.values():
        for d in t.deps:
            children[d].append(t.id)
    rank: dict[str, float] = {}
    for tid in reversed(p.graph.topo_order()):
        arms = p.capable_arms(tid) or p.arms
        avg = sum(p.duration(tid, a) for a in arms) / len(arms) if arms else 0.0
        rank[tid] = avg + max([rank[c] for c in children[tid]], default=0.0)
    return rank


def heft(p: Problem) -> Plan:
    b = Builder(p)
    b.place_held()
    rank = upward_rank(p)
    topo = {t: i for i, t in enumerate(p.graph.topo_order())}
    order = sorted(b.todo(), key=lambda t: (-rank[t], topo[t]))
    for tid in order:
        if tid in b.placed or b.is_chained_follower(tid):
            continue
        arms = b.arms_for(tid)
        if not arms:
            raise PlanError(f"heft: no arm for {tid}")
        best = min((b.trial(tid, a) for a in arms), key=lambda pl: (pl[-1].end, pl[0].arm_id))
        b.book(best)
    return b.plan("heft")
