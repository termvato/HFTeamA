"""Placement engine shared by the greedy fallback, HEFT and re-timing an old plan.

It books time on arms and shared resources and enforces the physical rules every plan needs:
- an arm does one step at a time;
- a shared resource (bin, fixture, tray) is used by one step at a time;
- a step starts after its dependencies end;
- an arm that picks a part (`same_arm_as` partner) holds it until it places it: nothing else is
  booked on that arm in between.
Running steps are pre-booked and never moved.
"""

from __future__ import annotations

from collections import defaultdict

from hfteama.scheduler.plan import Assignment, Plan
from hfteama.scheduler.problem import PlanError, Problem, blocked_tasks, check_chains

EPS = 1e-9


class Builder:
    def __init__(self, p: Problem) -> None:
        check_chains(p)
        self.p = p
        self.busy: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.placed: dict[str, Assignment] = {}
        self.blocked = blocked_tasks(p)
        for a in p.running.values():
            self.busy[_arm(a.arm_id)].append((a.start, max(a.end, p.now)))
            for r in p.task(a.task_id).resources:
                self.busy[_res(r)].append((a.start, max(a.end, p.now)))

    # ---- queries

    def todo(self) -> list[str]:
        return [t for t in self.p.pending() if t not in self.placed and t not in self.blocked]

    def end_of(self, tid: str) -> float | None:
        if tid in self.p.done:
            return self.p.now
        if tid in self.p.running:
            return max(self.p.running[tid].end, self.p.now)
        if tid in self.placed:
            return self.placed[tid].end
        return None

    def deps_placed(self, tid: str) -> bool:
        return all(self.end_of(d) is not None for d in self.p.task(tid).deps)

    def ready(self, tid: str) -> float:
        ends = [self.end_of(d) for d in self.p.task(tid).deps]
        return max([self.p.now, *[e for e in ends if e is not None]])

    def arms_for(self, tid: str) -> list[str]:
        arms = self.p.capable_arms(tid)
        pin = self.p.pinned_arm(tid)
        partner = self.p.task(tid).same_arm_as
        if pin is None and partner in self.placed:
            pin = self.placed[partner].arm_id
        return [a for a in arms if a == pin] if pin is not None else arms

    def arm_free(self, arm: str) -> float:
        return max([self.p.now, *[e for _, e in self.busy[_arm(arm)]]])

    def is_chained_follower(self, tid: str) -> bool:
        """Placed together with its partner, not on its own."""
        partner = self.p.task(tid).same_arm_as
        return bool(partner) and partner not in self.p.done and partner not in self.p.running

    # ---- placement

    def trial(self, tid: str, arm: str, not_before: float = 0.0) -> list[Assignment]:
        """Earliest placement of `tid` on `arm` (with its held follower, if any). Not booked."""
        ready = max(self.ready(tid), not_before)
        res = [_res(r) for r in self.p.task(tid).resources]
        d = self.p.duration(tid, arm)
        f = self.p.follower_of(tid)
        if f is None or f in self.placed or f in self.p.done or f in self.blocked:
            s = self._free_from([_arm(arm), *res], ready, d)
            return [Assignment(tid, arm, s, s + d)]
        f_res = [_res(r) for r in self.p.task(f).resources]
        df = self.p.duration(f, arm)
        f_other = [self.end_of(x) for x in self.p.task(f).deps if x != tid]
        if any(e is None for e in f_other):
            raise PlanError(f"{f}: dependencies not placed before its partner {tid}")
        f_ready = max([0.0, *f_other])
        for s in self._candidates([_arm(arm), *res], ready):
            if not self._free([_arm(arm), *res], s, s + d):
                continue
            sf = self._free_from(f_res, max(s + d, f_ready), df)
            if self._free([_arm(arm)], s, sf + df):
                return [Assignment(tid, arm, s, s + d), Assignment(f, arm, sf, sf + df)]
        raise PlanError(f"no placement for {tid} on {arm}")  # unreachable: last candidate fits

    def book(self, assignments: list[Assignment]) -> None:
        first, last = assignments[0], assignments[-1]
        self.busy[_arm(first.arm_id)].append((first.start, last.end))  # held in between
        for a in assignments:
            self.placed[a.task_id] = a
            for r in self.p.task(a.task_id).resources:
                self.busy[_res(r)].append((a.start, a.end))

    def place_held(self) -> None:
        """Arms holding a part (or still running its pick) place it before anything else."""
        firsts = dict(self.p.held)
        for tid, a in self.p.running.items():
            f = self.p.follower_of(tid)
            if f and f not in self.p.done and f not in self.p.running and f not in self.blocked:
                firsts[a.arm_id] = f
        for arm, f in firsts.items():
            if f in self.placed or f in self.blocked or arm not in self.p.arms:
                continue
            ready = max(self.ready(f), self.arm_free(arm))
            res = [_res(r) for r in self.p.task(f).resources]
            d = self.p.duration(f, arm)
            s = self._free_from(res, ready, d)
            self.busy[_arm(arm)].append((self.arm_free(arm), s + d))
            self.placed[f] = Assignment(f, arm, s, s + d)
            for r in self.p.task(f).resources:
                self.busy[_res(r)].append((s, s + d))

    def plan(self, solver: str) -> Plan:
        fixed = list(self.p.running.values())
        out = sorted([*fixed, *self.placed.values()], key=lambda a: (a.start, a.arm_id))
        makespan = max([self.p.now, *[a.end for a in out]])
        return Plan(out, solver, makespan, blocked=sorted(self.blocked))

    # ---- interval helpers

    def _candidates(self, keys: list[str], ready: float) -> list[float]:
        ends = {e for k in keys for _, e in self.busy[k] if e > ready}
        return sorted({ready, *ends})

    def _free(self, keys: list[str], s: float, e: float) -> bool:
        for k in keys:
            for a, b in self.busy[k]:
                if s < b - EPS and a < e - EPS and e - s > EPS:
                    return False
                if e - s <= EPS and a < s - EPS < b - EPS:  # zero-length inside a busy span
                    return False
        return True

    def _free_from(self, keys: list[str], ready: float, d: float) -> float:
        for s in self._candidates(keys, ready):
            if self._free(keys, s, s + d):
                return s
        raise PlanError("no free window")  # unreachable: after the last end everything is free


def _arm(a: str) -> str:
    return f"arm:{a}"


def _res(r: str) -> str:
    return f"res:{r}"
