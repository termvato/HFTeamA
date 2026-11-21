"""Task graph: the work to do, as JSON. Interface 1 of 3 (frozen; change only with team agreement).

Owner: Operations (#21, backlog #2).

Task-agnostic: any job that splits into skills, order and shared zones fits.
Examples: tasks/sort/sort12.json (independent units), tasks/assemble/assemble3.json (ordered).

File format:
    {
      "name": "...",
      "tasks": [
        {"id": "pick_01", "skill": "pick_place", "args": {"object": "block_01"},
         "deps": [], "zones": ["input_bin"], "allowed_arms": "*",
         "duration": 4.0, "durations": {"arm_b": 4.5}, "needs_arms": 1},
        {"id": "place_01", "skill": "place", "deps": ["pick_01"], "same_arm_as": "pick_01", ...}
      ]
    }

- deps: ids that must finish before this task starts.
- zones: shared table regions this task occupies; capacity 1 (a lock).
- allowed_arms: "*" or a list of arm ids that can reach the task.
- duration: default expected seconds; durations: per-arm override (measured p80 or twin estimate).
- same_arm_as: task id that must run on the same arm (a place after its pick: the arm holds it).
- needs_arms: 1, or 2 for a two-arm step (handoff, hold-and-insert); stretch goal.
"""

from hfteama.taskgraph.schema import Task, TaskGraph, TaskGraphError, load

__all__ = ["Task", "TaskGraph", "TaskGraphError", "load"]
