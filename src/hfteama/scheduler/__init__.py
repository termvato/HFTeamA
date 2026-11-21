"""Scheduler: assign every remaining task to an arm and a start time, minimising finish time.

Owner: Operations (#21, backlog #6 to #9, #12, #16).

Contract used by the executor:
    solve(graph, arms, now, fixed) -> Plan
- arms: ids of READY arms; fixed: Assignments already running (kept as-is on a re-plan).

Modules:
- plan.py   Plan / Assignment dataclasses (done).
- bound.py  lower bound and which term binds: arms, order or a resource (done).
- check.py  independent validator: deps respected, no overlap per arm and per resource.
- heft.py   greedy HEFT: baseline and fallback.
            Done when: every graph in tasks/ plans on 1-8 arms and check.py passes.
- cpsat.py  OR-Tools CP-SAT flexible job shop: optional interval per (task, arm), NoOverlap per arm
            and per resource, precedence, minimise makespan, HEFT as hint,
            5 s limit (1-2 s on re-plan).
            Done when: under 5 s at demo size (else live demo uses HEFT, CP-SAT shown offline).
"""

from hfteama.scheduler.bound import LowerBound, lower_bound
from hfteama.scheduler.plan import Assignment, Plan

__all__ = ["Assignment", "LowerBound", "Plan", "lower_bound"]
