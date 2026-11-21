"""CP-SAT planner (backlog #7, architecture doc 5.2): best plan within a time limit.

Flexible job shop in OR-Tools CP-SAT:
- one optional interval per (step, capable arm), exactly one chosen per step;
- NoOverlap per arm and per shared resource (resources are capacity 1);
- precedence: a step starts after its dependencies end;
- an arm that picks a part holds it until it places it: same arm, and the arm is reserved from
  the pick's start to the place's end (one "hold" interval on that arm);
- running steps are fixed; parts already held are placed first by their arm;
- minimise the finish time; warm-started from the HEFT plan; keeps the best plan found when the
  time limit hits. Times are in tenths of a second internally.
"""

from __future__ import annotations

import math
import os

from ortools.sat.python import cp_model

from hfteama.scheduler.plan import Assignment, Plan
from hfteama.scheduler.problem import PlanError, Problem, blocked_tasks, check_chains

SCALE = 10  # ticks per second


def cpsat(p: Problem, time_limit: float = 5.0, hint: Plan | None = None) -> Plan:
    check_chains(p)
    blocked = blocked_tasks(p)
    todo = [t for t in p.pending() if t not in blocked]
    m = cp_model.CpModel()

    def up(x: float) -> int:
        return math.ceil(x * SCALE - 1e-9)

    now = up(p.now)
    running_end = {t: max(up(a.end), now) for t, a in p.running.items()}
    arm_free = {a: now for a in p.arms}
    for tid, a in p.running.items():
        if a.arm_id in arm_free:
            arm_free[a.arm_id] = max(arm_free[a.arm_id], running_end[tid])
    dur = {(t, a): up(p.duration(t, a)) for t in todo for a in p.arms}
    horizon = (
        max([now, *running_end.values()])
        + sum(max((dur[(t, a)] for a in p.arms), default=0) for t in todo)
        + 1
    )

    # arms each step may use: capable, and pinned where its partner is running or the part held
    first_on: dict[str, str] = {}  # step -> arm that must do it first (part held or being picked)
    for arm, f in p.held.items():
        if f in todo and arm in p.arms:
            first_on[f] = arm
    for tid, a in p.running.items():
        f = p.follower_of(tid)
        if f in todo and a.arm_id in p.arms:
            first_on[f] = a.arm_id
    arms_of: dict[str, list[str]] = {}
    for t in todo:
        arms = p.capable_arms(t)
        if t in first_on:
            arms = [a for a in arms if a == first_on[t]]
        if not arms:
            raise PlanError(f"cpsat: no arm for {t}")
        arms_of[t] = arms

    start = {t: m.new_int_var(now, horizon, f"s_{t}") for t in todo}
    end = {t: m.new_int_var(now, horizon, f"e_{t}") for t in todo}
    pick: dict[tuple[str, str], cp_model.IntVar] = {}
    by_arm: dict[str, list[cp_model.IntervalVar]] = {a: [] for a in p.arms}
    by_res: dict[str, list[cp_model.IntervalVar]] = {}

    chained = {t for t in todo if p.task(t).same_arm_as in todo}  # follower placed with partner
    partner_of = {t: p.task(t).same_arm_as for t in chained}

    for t in todo:
        lits = []
        for a in arms_of[t]:
            x = m.new_bool_var(f"x_{t}_{a}")
            pick[(t, a)] = x
            lits.append(x)
            m.add(end[t] == start[t] + dur[(t, a)]).only_enforce_if(x)
        m.add_exactly_one(lits)
        size = m.new_int_var(0, horizon, f"z_{t}")
        whole = m.new_interval_var(start[t], size, end[t], f"i_{t}")
        for r in p.task(t).resources:
            by_res.setdefault(r, []).append(whole)

    for t in todo:
        f = p.follower_of(t)
        if t in chained:
            continue  # its arm time is covered by the partner's hold interval
        if f in chained:  # t picks, f places on the same arm; arm reserved from s_t to e_f
            common = [a for a in arms_of[t] if a in arms_of[f]]
            if not common:
                raise PlanError(f"cpsat: {t} and {f} share no arm")
            for a in arms_of[t]:
                if a not in arms_of[f]:
                    m.add(pick[(t, a)] == 0)
            for a in arms_of[f]:
                if a not in arms_of[t]:
                    m.add(pick[(f, a)] == 0)
            for a in common:
                m.add(pick[(t, a)] == pick[(f, a)])
                h = m.new_int_var(0, horizon, f"h_{t}_{a}")
                by_arm[a].append(
                    m.new_optional_interval_var(start[t], h, end[f], pick[(t, a)], f"hold_{t}_{a}")
                )
        elif t in first_on:  # arm holds the part (or is picking it now): reserved until e_t
            a = first_on[t]
            h = m.new_int_var(0, horizon, f"h_{t}")
            by_arm[a].append(m.new_interval_var(arm_free[a], h, end[t], f"held_{t}"))
            m.add(start[t] >= arm_free[a])
        else:
            for a in arms_of[t]:
                by_arm[a].append(
                    m.new_optional_fixed_size_interval_var(
                        start[t], dur[(t, a)], pick[(t, a)], f"a_{t}_{a}"
                    )
                )

    for tid, a in p.running.items():
        s0 = math.floor(a.start * SCALE + 1e-9)
        iv = m.new_fixed_size_interval_var(s0, running_end[tid] - s0, f"run_{tid}")
        if a.arm_id in by_arm:
            by_arm[a.arm_id].append(iv)
        for r in p.task(tid).resources:
            by_res.setdefault(r, []).append(iv)

    for t in todo:
        for d in p.task(t).deps:
            if d in end:
                m.add(start[t] >= end[d])
            elif d in running_end:
                m.add(start[t] >= running_end[d])
    for t, partner in partner_of.items():
        m.add(start[t] >= end[partner])

    for ivs in [*by_arm.values(), *by_res.values()]:
        if len(ivs) > 1:
            m.add_no_overlap(ivs)

    makespan = m.new_int_var(now, horizon, "makespan")
    m.add_max_equality(makespan, [now, *running_end.values(), *end.values()])
    m.minimize(makespan)

    if hint is not None:
        for a in hint.assignments:
            if a.task_id in start:
                m.add_hint(start[a.task_id], up(a.start))
                m.add_hint(end[a.task_id], up(a.start) + dur[(a.task_id, a.arm_id)])
                for b in arms_of[a.task_id]:
                    m.add_hint(pick[(a.task_id, b)], int(b == a.arm_id))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = min(8, os.cpu_count() or 1)
    status = solver.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise PlanError(f"cpsat: {solver.status_name(status)}")

    out = list(p.running.values())
    for t in todo:
        arm = next(a for a in arms_of[t] if solver.value(pick[(t, a)]))
        s = solver.value(start[t]) / SCALE
        out.append(Assignment(t, arm, s, s + p.duration(t, arm)))
    out.sort(key=lambda a: (a.start, a.arm_id))
    obj, bound = solver.objective_value, solver.best_objective_bound
    plan = Plan(out, "cpsat", max([p.now, *[a.end for a in out]]), blocked=sorted(blocked))
    plan.info.update(
        {
            "optimal": float(status == cp_model.OPTIMAL),
            "gap": (obj - bound) / obj if obj else 0.0,
            "cpsat_bound": bound / SCALE,
        }
    )
    return plan
