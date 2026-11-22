"""Generic skills: pick, place, pick_place, inspect, park. Written once for every backend.

A skill is `(arm: ArmDriver, task, poses) -> None`; it returns when the step worked and raises
`SkillFailed(reason)` when it did not. It calls only `ArmDriver` primitives. Where to go comes from
the task's args: table coordinates (`pick_xyz`, `place_xyz`) first, else a name looked up in `poses`
(`object` / `from` to pick, `target` / `to` to place).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

import numpy as np

from hfteama.arms.poses import Poses
from hfteama.arms.primitives import ArmDown, ArmDriver, MotionError, SkillFailed, check_point
from hfteama.arms.skills import REGISTRY, SkillFn
from hfteama.taskgraph import Task

APPROACH_DZ = 0.08  # hover height above the point before descending, m
LIFT_DZ = 0.10  # height to lift a grasped part to, m
GRASP_DZ = 0.0  # tool-centre offset at the point itself, m
INSPECT_HOLD_S = 0.5
WARN_DISAGREE_M = (
    0.02  # pick_xyz this far from the named object's pose is probably a bad coordinate
)

log = logging.getLogger(__name__)


def _register(name: str) -> Callable[[SkillFn], SkillFn]:
    """Register once; importing this module again must not raise."""

    def deco(fn: SkillFn) -> SkillFn:
        REGISTRY.setdefault(name, fn)
        return fn

    return deco


def _xyz(value: object, what: str) -> np.ndarray:
    """Three finite numbers, or SkillFailed before anything moves."""
    try:
        arr = np.asarray(value, float)
    except (TypeError, ValueError):
        raise SkillFailed(f"bad coordinates: {what}={value!r}") from None
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise SkillFailed(f"bad coordinates: {what}={value!r}")
    return arr


def pick_point(task: Task, poses: Poses, arm_id: str) -> np.ndarray:
    args = task.args
    if "pick_xyz" in args:
        xyz = _xyz(args["pick_xyz"], "pick_xyz")
        name = args.get("object")
        if name is not None and poses.has(name, arm_id):
            gap = float(np.linalg.norm(xyz - poses.point(name, arm_id)))
            if gap > WARN_DISAGREE_M:
                log.warning(
                    "pick_xyz is %.3f m from %s's known pose: check the coordinate", gap, name
                )
        return xyz
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
        return _xyz(args["place_xyz"], "place_xyz")
    for key in ("target", "to"):
        if key in args and poses.has(args[key], arm_id):
            return poses.point(args[key], arm_id)
    name = args.get("target") or args.get("to")
    if name is None:
        raise SkillFailed("no place location in args (place_xyz, target or to)")
    raise SkillFailed(f"unknown location: {name}")


def _above(p: np.ndarray, dz: float) -> np.ndarray:
    return p + np.array([0.0, 0.0, dz])


Waypoint = tuple[np.ndarray, float | None]  # table-frame point, jaw-closing heading (rad) or None


def _validate(arm: ArmDriver, waypoints: list[Waypoint]) -> None:
    """Refuse a step whose every waypoint cannot be reached, before anything moves."""
    for w, yaw in waypoints:
        check_point(w, arm.table_z)
        if not arm.reachable(w, yaw):
            raise MotionError("unreachable")


def _carry_point(poses: Poses, arm_id: str) -> np.ndarray | None:
    """Where pick leaves the arm: a per-arm `carry` pose outside the shared area, if defined."""
    return poses.point("carry", arm_id) if poses.has("carry", arm_id) else None


def pick_yaw(task: Task) -> float | None:
    """Jaw-closing heading (rad) for `pick_yaw_deg`, the object's long axis: close across it."""
    if "pick_yaw_deg" not in task.args:
        return None
    try:
        deg = float(task.args["pick_yaw_deg"])
    except (TypeError, ValueError):
        raise SkillFailed(f"bad coordinates: pick_yaw_deg={task.args['pick_yaw_deg']!r}") from None
    if not math.isfinite(deg):
        raise SkillFailed(f"bad coordinates: pick_yaw_deg={deg!r}")
    return math.radians(deg) + math.pi / 2


def _move(arm: ArmDriver, xyz: np.ndarray, yaw: float | None = None) -> None:
    arm.move_to(xyz) if yaw is None else arm.move_to(xyz, yaw=yaw)


def _pick_waypoints(p: np.ndarray, yaw: float | None, carry: np.ndarray | None) -> list[Waypoint]:
    points: list[Waypoint] = [
        (_above(p, APPROACH_DZ), yaw),
        (_above(p, GRASP_DZ), yaw),
        (_above(p, LIFT_DZ), yaw),
    ]
    return points + ([(carry, None)] if carry is not None else [])


def _place_waypoints(p: np.ndarray, carry: np.ndarray | None = None) -> list[Waypoint]:
    points: list[Waypoint] = [(_above(p, APPROACH_DZ), None), (_above(p, GRASP_DZ), None)]
    return points + ([(carry, None)] if carry is not None else [])


def _retreat_on_failure(
    arm: ArmDriver,
    p: np.ndarray,
    run: Callable[[], None],
    yaw: float | None = None,
    safe: np.ndarray | None = None,
) -> None:
    """Run a motion sequence; if it fails part-way, get clear of the shared area first.

    Clear means the arm's own `carry` pose when it has one, else straight up. A stopped or lost
    arm is left alone: the stop is what the caller asked for.
    """
    try:
        run()
    except MotionError as exc:
        if exc.reason != "stopped":
            try:
                _move(arm, _above(p, APPROACH_DZ), yaw)  # straight up first: a sweep from a low
                if safe is not None:  # pose clips the table
                    _move(arm, safe)
            except (MotionError, ArmDown):
                pass
        raise


def _do_pick(
    arm: ArmDriver, p: np.ndarray, carry: np.ndarray | None, yaw: float | None = None
) -> None:
    def run() -> None:
        arm.open_gripper()
        if carry is not None:
            # Stage over the arm's own side first: a joint-space sweep from wherever the arm
            # rests straight to the bin would pass through other arms' areas.
            _move(arm, carry)
        _move(arm, _above(p, APPROACH_DZ), yaw)
        _move(arm, _above(p, GRASP_DZ), yaw)
        if not arm.close_gripper().holding:
            arm.open_gripper()  # missed: leave the shared area before reporting
            _move(arm, _above(p, APPROACH_DZ), yaw)  # straight up first,
            if carry is not None:  # then out of the shared area
                _move(arm, carry)
            raise SkillFailed("grasp")
        _move(arm, _above(p, LIFT_DZ), yaw)
        if not arm.holding():
            raise SkillFailed("dropped")
        if carry is not None:
            # The executor frees the shared area lock when this step returns: be out by then.
            _move(arm, carry)

    _retreat_on_failure(arm, p, run, yaw, safe=carry)


def _do_place(arm: ArmDriver, p: np.ndarray, carry: np.ndarray | None = None) -> None:
    def run() -> None:
        _move(arm, _above(p, APPROACH_DZ))
        _move(arm, _above(p, GRASP_DZ))
        arm.open_gripper()
        _move(arm, _above(p, APPROACH_DZ))  # straight up first,
        if carry is not None:  # then out of the shared area: the executor frees its lock now
            _move(arm, carry)

    _retreat_on_failure(arm, p, run, safe=carry)


@_register("pick")
def pick(arm: ArmDriver, task: Task, poses: Poses) -> None:
    p, carry = pick_point(task, poses, arm.arm_id), _carry_point(poses, arm.arm_id)
    yaw = pick_yaw(task)
    _validate(arm, _pick_waypoints(p, yaw, carry))
    _do_pick(arm, p, carry, yaw)


@_register("place")
def place(arm: ArmDriver, task: Task, poses: Poses) -> None:
    p, carry = place_point(task, poses, arm.arm_id), _carry_point(poses, arm.arm_id)
    _validate(arm, _place_waypoints(p, carry))
    _do_place(arm, p, carry)


@_register("pick_place")
def pick_place(arm: ArmDriver, task: Task, poses: Poses) -> None:
    pp, qq = pick_point(task, poses, arm.arm_id), place_point(task, poses, arm.arm_id)
    carry, yaw = _carry_point(poses, arm.arm_id), pick_yaw(task)
    _validate(arm, _pick_waypoints(pp, yaw, carry) + _place_waypoints(qq, carry))  # before a grasp
    _do_pick(arm, pp, carry, yaw)
    _do_place(arm, qq, carry)


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
