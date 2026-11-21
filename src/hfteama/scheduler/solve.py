"""One entry point for the executor and the screen (backlog #6, #7, #12, #16).

- solve(): run the chosen planner, fall back to greedy if it fails or the validator rejects the
  plan, and attach the lower bound, the binding limit and the blocked steps.
- retime(): keep an old plan's decisions (arm and per-arm order) and recompute its times from now;
  used for the 10% re-plan rule.
- whatif(): predicted time to finish with 1, 2, 3, 4 ... arms, for the estimate before Go.
"""

from __future__ import annotations

import time
from dataclasses import replace

from hfteama.scheduler.bound import lower_bound
from hfteama.scheduler.builder import Builder
from hfteama.scheduler.check import check
from hfteama.scheduler.greedy import greedy
from hfteama.scheduler.heft import heft
from hfteama.scheduler.plan import Plan
from hfteama.scheduler.problem import PlanError, Problem

METHODS = ("greedy", "heft", "cpsat")  # the executor also takes "auto" (see Executor)


def solve(p: Problem, method: str = "heft", time_limit: float = 5.0) -> Plan:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    t0 = time.perf_counter()
    plan: Plan | None = None
    notes: list[str] = []
    if method == "cpsat":
        try:
            from hfteama.scheduler.cpsat import cpsat

            plan = cpsat(p, time_limit=time_limit, hint=_safe(heft, p))
        except ImportError:
            notes.append("ortools not installed")
        except PlanError as e:
            notes.append(str(e))
    if plan is None and method in ("heft", "cpsat"):
        plan = _safe(heft, p)
    if plan is not None and check(plan, p):
        notes.append("rejected by validator: " + "; ".join(check(plan, p))[:200])
        plan = None
    if plan is None:
        plan = greedy(p)
        errors = check(plan, p)
        if errors:
            raise PlanError("no valid plan: " + "; ".join(errors))
    _attach_bound(plan, p)
    plan.info["solve_time"] = time.perf_counter() - t0
    if notes:
        plan.info["fallbacks"] = float(len(notes))
    return plan


def retime(old: Plan, p: Problem) -> Plan | None:
    """The old plan's assignments, re-timed from now. None if it no longer fits (arms changed,
    new or retried steps)."""
    b = Builder(p)
    b.place_held()
    keep = [a for a in sorted(old.assignments, key=lambda a: a.start) if a.task_id in b.todo()]
    if {a.task_id for a in keep} != set(b.todo()):
        return None
    for a in keep:
        if a.task_id in b.placed:
            continue
        if b.is_chained_follower(a.task_id):
            continue
        if a.arm_id not in b.arms_for(a.task_id) or not b.deps_placed(a.task_id):
            return None
        try:
            b.book(b.trial(a.task_id, a.arm_id, not_before=b.arm_free(a.arm_id)))
        except PlanError:
            return None
    plan = b.plan(f"retimed:{old.solver.removeprefix('retimed:')}")
    if check(plan, p):
        return None
    _attach_bound(plan, p)
    return plan


def whatif(
    p: Problem, counts: tuple[int, ...] = (1, 2, 3, 4), spare: list[str] | None = None
) -> dict[int, float]:
    """Seconds from now to finish with n arms. Beyond the connected arms it adds `spare` arms
    first (known arms not plugged in, e.g. from the reachability catalog, so their reach counts),
    then virtual copies that can do any step open to every arm."""
    out: dict[int, float] = {}
    busy = [a.arm_id for a in p.running.values()]
    order = list(dict.fromkeys([*busy, *p.arms, *(spare or [])]))
    for n in counts:
        arms = order[:n] + [f"virtual-{k}" for k in range(1, n - len(order) + 1)]
        keep_running = {t: a for t, a in p.running.items() if a.arm_id in arms}
        held = {a: f for a, f in p.held.items() if a in arms}
        sub = replace(p, arms=arms, running=keep_running, held=held)
        try:
            plan = heft(sub)
        except PlanError:
            continue
        if plan.blocked:
            continue
        out[n] = plan.makespan - p.now
    return out


def _attach_bound(plan: Plan, p: Problem) -> None:
    """Lower bound on the finish time (absolute) and the binding limit, from the remaining work."""
    excluded = frozenset(p.done | set(p.running) | set(plan.blocked))
    if not p.arms or len(excluded) == len(p.graph.tasks):
        plan.lower_bound = max([p.now, *[a.end for a in p.running.values()]])
        plan.binding = "" if not p.arms else "arms"
        return
    lb = lower_bound(p.graph, p.arms, done=excluded)
    running_end = max([p.now, *[a.end for a in p.running.values()]])
    plan.lower_bound = max(p.now + lb.value, running_end)
    plan.binding = lb.binding
    plan.info.update({f"bound_{k}": v for k, v in lb.terms.items()})


def _safe(fn, p: Problem) -> Plan | None:
    try:
        return fn(p)
    except PlanError:
        return None
