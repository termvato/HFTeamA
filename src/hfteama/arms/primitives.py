"""Arm primitives: what a skill may ask an arm to do. The Action <-> Hardware interface.

Every backend (MuJoCo, LeRobot, test double) implements `ArmDriver`. Skills call only these methods
and never branch on the backend. Safety checks live here, in the drivers, so no skill can bypass
them: joint limits, nothing below the table, a slow stop.

Positions are metres in the shared table frame; joint angles are radians in the model's convention
(`kinematics.JOINTS` order: five body joints, then the gripper). Real backends convert with
`kinematics.ArmFrame`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from hfteama.arms.kinematics import JOINTS, LIMITS_RAD


class _Reasoned(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MotionError(_Reasoned):
    """A motion was refused or failed: unreachable, joint limit, below table, stalled, stopped."""


class ArmDown(_Reasoned):
    """The arm is gone: bus error, lost heartbeat, removed. Every later call fails too."""


class SkillFailed(_Reasoned):
    """A skill could not do its step (no grasp, dropped part, unknown location)."""


@dataclass(frozen=True)
class Grasp:
    holding: bool  # True when the gripper stalled on something instead of closing fully
    position: float  # gripper joint angle after closing, rad
    effort: float = 0.0  # servo load or current when available, else 0


@runtime_checkable
class ArmDriver(Protocol):
    arm_id: str
    table_z: float  # table height in the table frame, m

    def read_joints(self) -> np.ndarray:
        """Current joints, 6 entries (five body joints and the gripper), rad."""
        ...

    def set_joint_target(self, q: np.ndarray) -> None:
        """Non-blocking, limit-checked target for streaming control (e.g. a learned policy)."""
        ...

    def move_joints(self, q: np.ndarray, speed: float = 1.0) -> None:
        """Blocking, interpolated move. Raises MotionError on a limit, stall or stop."""
        ...

    def move_to(self, xyz: np.ndarray, speed: float = 1.0) -> None:
        """Move the tool centre to a table-frame point, gripper pointing down.

        Raises MotionError("below table") or MotionError("unreachable") before moving.
        """
        ...

    def open_gripper(self) -> None: ...

    def close_gripper(self) -> Grasp: ...

    def holding(self) -> bool:
        """Whether the gripper currently holds something (re-check after lifting)."""
        ...

    def stop(self) -> None:
        """Freeze in place; a blocked call in another thread raises MotionError("stopped")."""
        ...

    def park(self) -> None:
        """Slow stop: freeze, move to the parked pose, then torque off. Never cut torque mid-air."""
        ...

    def alive(self) -> float:
        """Time (time.time()) of the arm's last sign of life; feeds the worker's heartbeat."""
        ...


def check_joints(q: np.ndarray) -> None:
    """Raise MotionError("joint limit: <name>") for the first joint outside its limits."""
    for name, value in zip(JOINTS, np.asarray(q, float), strict=False):
        lo, hi = LIMITS_RAD[name]
        if not lo <= value <= hi:
            raise MotionError(f"joint limit: {name}")


def check_point(xyz: np.ndarray, table_z: float, clearance: float = 0.01) -> None:
    """Raise MotionError("below table") if the point is less than `clearance` above the table."""
    if float(np.asarray(xyz, float)[2]) < table_z + clearance:
        raise MotionError("below table")
