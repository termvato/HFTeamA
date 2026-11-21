"""Arm workers: one process per arm, same contract for every backend. Interface 2 of 3 (frozen).

Task-agnostic: a worker runs whatever skill a task names, looked up in arms.skills.REGISTRY.
Tasks bring their own skills and poses (see tasks/README.md); nothing here knows about a task.

Owners: hardware (H1, H2) for real.py, primitives, discovery.py; sim (S) for mujoco_arm.py.
fake.py is shared and used by everyone for tests.

Backends, all implementing arms.base.ArmWorker:
- fake.py        sleeps for the predicted duration; optional random failures. Lets the planner,
                 executor and view run with no hardware.
                 Done when: the executor runs any task graph on 3 fakes.
- mujoco_arm.py  MuJoCo SO-101 copy at 1x wall clock. Done when: pick_place runs in sim.
- real.py        LeRobot SOFollower on its own USB port.
                 Done when: pick_place on 1 real arm, durations logged.
- skills.py      skill registry (done) + generic primitives every task can reuse: move_to(pose),
                 grip, release, grasp check (gripper position + Present_Current, thresholds
                 from 10 empty / 10 loaded closes per arm), park.
- discovery.py   polls /dev/serial/by-id; known serial -> start worker -> safety check ->
                 slow home -> READY -> event arm_added. Port gone or heartbeat lost ->
                 arm_removed. Done when: plugging arm 2 mid-run triggers a re-plan.
"""

from hfteama.arms.base import ArmState, ArmWorker, Outcome, Result
from hfteama.arms.skills import REGISTRY, skill

__all__ = ["REGISTRY", "ArmState", "ArmWorker", "Outcome", "Result", "skill"]
