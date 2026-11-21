"""Generic skills: pick, place, pick_place, inspect, park. Written once for every backend.

A skill is `(arm: ArmDriver, task, poses) -> None`; it returns when the step worked and raises
`SkillFailed(reason)` when it did not. It calls only `ArmDriver` primitives. Where to go comes from
the task's args: table coordinates (`pick_xyz`, `place_xyz`) first, else a name looked up in `poses`
(`object` / `from` to pick, `target` / `to` to place).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from hfteama.arms.poses import Poses
from hfteama.arms.primitives import ArmDriver, SkillFailed
from hfteama.arms.skills import REGISTRY, SkillFn
from hfteama.taskgraph import Task

APPROACH_DZ = 0.08  # hover height above the point before descending, m
LIFT_DZ = 0.10  # height to lift a grasped part to, m
GRASP_DZ = 0.0  # tool-centre offset at the point itself, m
INSPECT_HOLD_S = 0.5


def _register(name: str) -> Callable[[SkillFn], SkillFn]:
    """Register once; importing this module again must not raise."""

    def deco(fn: SkillFn) -> SkillFn:
        REGISTRY.setdefault(name, fn)
        return fn

    return deco


def pick_point(task: Task, poses: Poses, arm_id: str) -> np.ndarray:
    args = task.args
    if "pick_xyz" in args:
        return np.asarray(args["pick_xyz"], float)
    for key in ("object", "from"):
        if key in args and poses.has(args[key], arm_id):
            return poses.point(args[key], arm_id)
    name = args.get("object") or args.get("from")
    if name is None:
        raise SkillFailed("no pick location in args (pick_xyz, object or from)")
    raise SkillFailed(f"unknown location: {name}")


def place_point(task: Task, poses: Poses, arm_id: str) -> np.ndarray:
    args = task.args
    if "place_xyz" in args:
        return np.asarray(args["place_xyz"], float)
    name = args.get("target") or args.get("to")
    if name is None:
        raise SkillFailed("no place location in args (place_xyz, target or to)")
    return poses.point(name, arm_id)


def _above(p: np.ndarray, dz: float) -> np.ndarray:
    return p + np.array([0.0, 0.0, dz])


@_register("pick")
def pick(arm: ArmDriver, task: Task, poses: Poses) -> None:
    p = pick_point(task, poses, arm.arm_id)
    arm.open_gripper()
    arm.move_to(_above(p, APPROACH_DZ))
    arm.move_to(_above(p, GRASP_DZ))
    if not arm.close_gripper().holding:
        arm.open_gripper()  # missed: leave the shared area before reporting, so others can enter
        arm.move_to(_above(p, APPROACH_DZ))
        raise SkillFailed("grasp")
    arm.move_to(_above(p, LIFT_DZ))
    if not arm.holding():
        raise SkillFailed("dropped")


@_register("place")
def place(arm: ArmDriver, task: Task, poses: Poses) -> None:
    p = place_point(task, poses, arm.arm_id)
    arm.move_to(_above(p, APPROACH_DZ))
    arm.move_to(_above(p, GRASP_DZ))
    arm.open_gripper()
    arm.move_to(_above(p, APPROACH_DZ))


@_register("pick_place")
def pick_place(arm: ArmDriver, task: Task, poses: Poses) -> None:
    pick(arm, task, poses)
    place(arm, task, poses)


@_register("inspect")
def inspect(arm: ArmDriver, task: Task, poses: Poses) -> None:
    if poses.has("inspect", arm.arm_id):
        target = poses.point("inspect", arm.arm_id)
    else:
        target = _above(pick_point(task, poses, arm.arm_id), APPROACH_DZ)
    arm.move_to(target)
    time.sleep(INSPECT_HOLD_S)


@_register("park")
def park(arm: ArmDriver, task: Task, poses: Poses) -> None:
    arm.park()
