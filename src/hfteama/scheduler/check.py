"""Plan validator (backlog #16): an independent check run in tests and after every re-plan.

Returns a list of violations (empty = valid). It does not share code with the planners on purpose.
"""

from __future__ import annotations

from collections import defaultdict

from hfteama.scheduler.plan import Assignment, Plan
from hfteama.scheduler.problem import Problem, blocked_tasks

TOL = 1e-6


def check(plan: Plan, p: Problem) -> list[str]:
    errors: list[str] = []
    blocked = blocked_tasks(p)
    by_task: dict[str, list[Assignment]] = defaultdict(list)
    for a in plan.assignments:
        by_task[a.task_id].append(a)

    for tid in p.graph.tasks:
        n = len(by_task.get(tid, []))
        if tid in p.done or tid in blocked:
            if n:
                errors.append(f"{tid}: done or blocked but planned")
        elif n != 1:
            errors.append(f"{tid}: planned {n} times")
    if set(blocked) != set(plan.blocked):
        errors.append(f"blocked list {plan.blocked} != {sorted(blocked)}")

    end = {a.task_id: a.end for a in plan.assignments}
    for a in plan.assignments:
        t = p.task(a.task_id)
        fixed = p.running.get(a.task_id)
        if fixed is not None:
            if fixed != a:
                errors.append(f"{a.task_id}: running step was moved")
            continue
        if a.arm_id not in p.capable_arms(a.task_id):
            errors.append(f"{a.task_id}: arm {a.arm_id} cannot do it")
        if a.start < p.now - TOL:
            errors.append(f"{a.task_id}: starts in the past")
        if a.end - a.start < p.duration(a.task_id, a.arm_id) - TOL:
            errors.append(f"{a.task_id}: shorter than its duration")
        for d in t.deps:
            if d in end and a.start < end[d] - TOL:
                errors.append(f"{a.task_id}: starts before dependency {d} ends")
        partner = t.same_arm_as
        if partner and partner in end:
            pa = next(x for x in plan.assignments if x.task_id == partner)
            if pa.arm_id != a.arm_id:
                errors.append(f"{a.task_id}: not on the arm holding the part ({pa.arm_id})")
    for arm, f in p.held.items():
        if f in by_task and by_task[f][0].arm_id != arm:
            errors.append(f"{f}: must be placed by {arm}, which holds the part")

    lanes: dict[str, list[Assignment]] = defaultdict(list)
    for a in plan.assignments:
        lanes[f"arm {a.arm_id}"].append(a)
        for r in p.task(a.task_id).resources:
            lanes[f"resource {r}"].append(a)
    for lane, items in lanes.items():
        items.sort(key=lambda a: (a.start, a.end))
        for x, y in zip(items, items[1:], strict=False):
            if y.start < x.end - TOL and y.end - y.start > TOL and x.end - x.start > TOL:
                errors.append(f"{lane}: {x.task_id} and {y.task_id} overlap")

    # an arm holding a part does nothing else until it places it
    for a in plan.assignments:
        f = p.follower_of(a.task_id)
        if f and f in end:
            fa = by_task[f][0]
            for other in plan.assignments:
                if other.arm_id == a.arm_id and other.task_id not in (a.task_id, f):
                    if other.start < fa.start - TOL and other.end > a.end + TOL:
                        errors.append(f"arm {a.arm_id}: {other.task_id} while holding {f}")
    if plan.makespan + TOL < max([p.now, *end.values()]):
        errors.append("makespan before the last step ends")
    return errors
