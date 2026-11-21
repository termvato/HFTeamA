"""Arm registry and reachability graph. Owner: Operations (#21, backlog #3).

- arms.py          ArmNode, Reach, ArmRegistry (join / leave / fail / heartbeat), load_catalog,
                   load_real_arms (same catalog from Action's configs/real_arms.json).
- reachability.py  resolve(graph, arms) -> allowed_arms per task, BLOCKED steps, newly_unblocked.
"""

from hfteama.registry.arms import ArmNode, ArmRegistry, Reach, load_catalog, load_real_arms
from hfteama.registry.reachability import ReachResult, newly_unblocked, resolve, task_points

__all__ = [
    "ArmNode",
    "ArmRegistry",
    "Reach",
    "ReachResult",
    "load_catalog",
    "load_real_arms",
    "newly_unblocked",
    "resolve",
    "task_points",
]
