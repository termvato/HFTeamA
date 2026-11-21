"""Executor: runs the plan and re-splits the remaining work when something changes.

Owner: Operations (#21, backlog #10, #11).

- Keeps the solver's decisions (which arm, per-arm order, per-resource order), not its timestamps.
- Starts a task the moment its deps are done, its arm is free and its resources are free.
- Resource locks are taken again at runtime, right before motion: two arms never share one.
- Re-plan triggers: arm_added, arm_removed, arm_failed, task failed (retry on same or other arm),
  large overrun. Running tasks stay fixed; solver gets 1-2 s, HEFT on timeout.
- Writes every plan, start, end and arm event to events.EventLog.
- Removal is a slow stop: "remove arm N" -> arm finishes or aborts its skill -> parks -> unplug.

Done when: a batch on 3 fake arms survives "add arm" and "remove arm" mid-run, every plan passes
scheduler/check.py, and the event log replays into the view.
"""

from hfteama.executor.executor import Executor, TaskState

__all__ = ["Executor", "TaskState"]
