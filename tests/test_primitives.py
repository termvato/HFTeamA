"""Primitives: errors, limit and table checks, and that the test driver satisfies the protocol."""

import numpy as np
import pytest

from hfteama.arms.kinematics import JOINTS, LIMITS_RAD
from hfteama.arms.primitives import (
    ArmDown,
    ArmDriver,
    Grasp,
    MotionError,
    SkillFailed,
    check_joints,
    check_point,
)
from tests.fake_driver import FakeDriver


@pytest.mark.parametrize("k", range(len(JOINTS)))
def test_check_joints_rejects_each_limit(k: int) -> None:
    for side, delta in ((1, 0.01), (0, -0.01)):
        q = np.zeros(6)
        q[k] = LIMITS_RAD[JOINTS[k]][side] + delta
        with pytest.raises(MotionError, match=f"joint limit: {JOINTS[k]}"):
            check_joints(q)


def test_check_joints_accepts_zero_and_five_joint_vectors() -> None:
    check_joints(np.zeros(6))
    check_joints(np.zeros(5))


def test_check_point_below_table() -> None:
    with pytest.raises(MotionError, match="below table"):
        check_point(np.array([0.2, 0.0, 0.005]), table_z=0.0)
    check_point(np.array([0.2, 0.0, 0.02]), table_z=0.0)


def test_errors_keep_their_reason() -> None:
    for exc in (MotionError, ArmDown, SkillFailed):
        assert exc("because").reason == "because"


def test_grasp_defaults() -> None:
    assert Grasp(holding=True, position=0.4).effort == 0.0


def test_fake_driver_satisfies_protocol() -> None:
    assert isinstance(FakeDriver(), ArmDriver)
