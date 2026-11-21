"""Greedy allocator (backlog #6, architecture doc L1): the never-fails fallback.

Simulates the online rule "an idle arm takes its cheapest ready step" to the end of the batch, so
it still yields a full schedule and a predicted finish time. Milliseconds at hackathon sizes.
"""

from __future__ import annotations

from hfteama.scheduler.builder import EPS, Builder
from hfteama.scheduler.plan import Plan
from hfteama.scheduler.problem import PlanError, Problem


def greedy(p: Problem) -> Plan:
    b = Builder(p)
    b.place_held()
    free_at = {a: b.arm_free(a) for a in p.arms}
    active = set(p.arms)
    while True:
        todo = [t for t in b.todo() if not b.is_chained_follower(t)]
        if not todo:
            return b.plan("greedy")
        if not active:
            raise PlanError(f"greedy: no arm left for {todo}")
        arm = min(active, key=lambda a: (free_at[a], a))
        now = free_at[arm]
        mine = [t for t in todo if arm in b.arms_for(t)]
        if not mine:
            active.discard(arm)
            continue
        ready = [t for t in mine if b.deps_placed(t) and b.ready(t) <= now + EPS]
        if not ready:
            later = [b.ready(t) for t in mine if b.deps_placed(t)]
            ends = [a.end for a in b.placed.values() if a.end > now + EPS]
            nxt = min(later) if later else (min(ends) if ends else None)
            if nxt is None or nxt <= now + EPS:
                active.discard(arm)
            else:
                free_at[arm] = nxt
            continue
        tid = min(ready, key=lambda t: (p.duration(t, arm), t))
        placement = b.trial(tid, arm, not_before=max(now, b.arm_free(arm)))
        b.book(placement)
        free_at[arm] = b.arm_free(arm)
