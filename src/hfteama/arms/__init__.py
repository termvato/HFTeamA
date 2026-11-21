"""Arm workers: one process per arm, same contract for every backend. Interface 2 of 3 (frozen).

Task-agnostic: a worker runs whatever skill a task names, looked up in arms.skills.REGISTRY.
Tasks can bring their own skills (see tasks/README.md); nothing here knows about a task.

Owners:
- Hardware (#19): real arms, safe operation, detecting arms plugged in or out.
- Action (#20): the skills, on real and simulated arms. Scripted, learned or both is their call.
- Operations (#21): the contract in base.py and a fake arm for testing (backlog #4, #14).

Every backend implements arms.base.ArmWorker: state(), execute(task) -> Result, heartbeat(),
cancel(task_id), park().
"""

from hfteama.arms.base import ArmState, ArmWorker, Outcome, Result
from hfteama.arms.skills import REGISTRY, skill

__all__ = ["REGISTRY", "ArmState", "ArmWorker", "Outcome", "Result", "skill"]
