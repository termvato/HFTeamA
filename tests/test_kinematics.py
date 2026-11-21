"""LeRobot <-> kinematic joint-angle conversion."""

import pytest

np = pytest.importorskip("numpy")

from hfteama.arms.kinematics import ArmFrame  # noqa: E402


def test_round_trip_and_elbow_zero() -> None:
    frame = ArmFrame()
    obs = {
        "shoulder_pan.pos": 10.0,
        "shoulder_lift.pos": -20.0,
        "elbow_flex.pos": 6.9,
        "wrist_flex.pos": 30.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 50.0,
    }
    q = frame.to_kinematic(obs)
    assert abs(q[2]) < 1e-12, "LeRobot elbow 6.9 deg is the kinematic zero"
    back = frame.to_lerobot(q)
    assert all(abs(back[k] - obs[k]) < 1e-9 for k in obs)


def test_targets_clipped_to_joint_limits() -> None:
    out = ArmFrame().to_lerobot(np.array([10.0, 0, 0, 0, 0]))
    assert out["shoulder_pan.pos"] == pytest.approx(np.degrees(1.91986))
