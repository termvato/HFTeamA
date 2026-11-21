"""hfteama.arm_check: config validation without an arm, and staged supervised commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hfteama import arm_check  # noqa: E402
from hfteama.arms.ik import ArmBase, Kinematics  # noqa: E402
from tests.fake_robot import FakeRobot, make_driver  # noqa: E402

pytestmark = pytest.mark.sim

ROOT = Path(__file__).parent.parent
POSES_REAL = ROOT / "tasks" / "sort" / "poses_real.example.json"
ARM = "rakesh_follower"
BASE = {"base": [0.0, 0.26, 0.0], "yaw": 4.7124}


def _write(tmp_path: Path, arm: dict | None = None, arm_id: str = ARM) -> Path:
    entry = {
        "pair": "rakesh",
        "table_z": 0.0,
        "park_q": [1.85, -1.6, 1.6, 1.0, 0.0],
        "grip_holding_min_pct": 15.0,
        **BASE,
        **(arm or {}),
    }
    path = tmp_path / "real_arms.json"
    path.write_text(json.dumps({"arms": {arm_id: entry}}))
    return path


def _pairs(tmp_path: Path, follower_id: str = ARM) -> Path:
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps({"pairs": {"rakesh": {"follower": {"id": follower_id, "serial": "S"}}}})
    )
    return path


def test_a_consistent_config_has_no_problems(tmp_path: Path) -> None:
    assert arm_check.check_config(_write(tmp_path), POSES_REAL, _pairs(tmp_path)) == []


@pytest.mark.parametrize(
    "arm, expect",
    [
        ({"grip_holding_min_pct": None}, "grip_holding_min_pct"),
        ({"park_q": [9.0, 0, 0, 0, 0]}, "park_q"),
        ({"park_q": [0, 0, 0]}, "park_q"),
        ({"base": [3.0, 3.0, 0.0]}, "cannot reach"),
        ({"pair": ""}, "pair"),
    ],
)
def test_problems_are_named(tmp_path: Path, arm: dict, expect: str) -> None:
    problems = arm_check.check_config(_write(tmp_path, arm), POSES_REAL, _pairs(tmp_path))
    assert problems and any(expect in p for p in problems), problems


def test_an_arm_id_missing_from_the_pairs_or_the_poses_is_reported(tmp_path: Path) -> None:
    problems = arm_check.check_config(
        _write(tmp_path), POSES_REAL, _pairs(tmp_path, "someone_else")
    )
    assert any("teleop_pairs" in p for p in problems)
    problems = arm_check.check_config(
        _write(tmp_path, arm_id="arm_a"), POSES_REAL, _pairs(tmp_path)
    )
    assert any("no entry for arm 'arm_a'" in p for p in problems)


def _connect(robot: FakeRobot):  # noqa: ANN202
    base = ArmBase(tuple(BASE["base"]), BASE["yaw"])
    return lambda arm_id, path: make_driver(robot, arm_id=arm_id, kin=Kinematics(base))


def _run(argv: list[str], robot: FakeRobot, ask=lambda _p: "yes") -> int:  # noqa: ANN001
    return arm_check.main(argv, ask=ask, connect=_connect(robot))


def test_read_is_read_only_and_prints_joints_and_tcp(
    capsys: pytest.CaptureFixture[str],
) -> None:
    robot = FakeRobot()
    assert _run(["read", "--arm", ARM], robot) == 0
    out = capsys.readouterr().out
    for word in ("shoulder_pan", "gripper", "TCP", "alive"):
        assert word in out
    assert robot.actions == []  # reading never commands the arm


@pytest.mark.parametrize(
    "cmd", [["joint", "--joint", "shoulder_pan"], ["point", "--xyz", "0.0", "0.0", "0.12"]]
)
def test_motion_commands_need_the_typed_yes(cmd: list[str]) -> None:
    robot = FakeRobot()
    assert _run([cmd[0], "--arm", ARM, *cmd[1:]], robot, ask=lambda _p: "no") == 2
    assert robot.actions == []


def test_joint_moves_only_that_joint_by_the_delta() -> None:
    from hfteama.arms.kinematics import BODY, ArmFrame

    robot = FakeRobot()
    start = ArmFrame().to_kinematic({**{f"{j}.pos": 0.0 for j in BODY}, "gripper.pos": 50.0})
    assert _run(["joint", "--arm", ARM, "--joint", "wrist_flex", "--delta", "0.1"], robot) == 0
    path = np.array([ArmFrame().to_kinematic(a) for a in robot.actions]) - start
    assert path[:, 3].max() == pytest.approx(0.1, abs=0.01)  # out by the delta
    assert abs(path[-1, 3]) < 0.01  # and back
    assert np.abs(np.delete(path, 3, axis=1)).max() < 1e-6  # nothing else ever moved


def test_a_point_below_the_table_is_refused_and_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    robot = FakeRobot()
    assert _run(["point", "--arm", ARM, "--xyz", "0.0", "0.0", "0.0"], robot) == 1
    assert "refused" in capsys.readouterr().out
    assert robot.actions == []


def test_descend_goes_one_stage_at_a_time_and_stops_when_told() -> None:
    robot = FakeRobot()
    answers = iter(["yes", "", "stop"])  # confirm, continue after stage 1, stop after stage 2
    rc = arm_check.main(
        ["descend", "--arm", ARM, "--xy", "0.0", "0.0"],
        ask=lambda _p: next(answers),
        connect=_connect(robot),
    )
    assert rc == 0
    zs = [
        Kinematics(ArmBase(tuple(BASE["base"]), BASE["yaw"])).fk(
            __import__("hfteama.arms.kinematics", fromlist=["ArmFrame"])
            .ArmFrame()
            .to_kinematic(a)[:5]
        )[0][2]
        for a in robot.actions
    ]
    assert min(zs) > 0.04  # stopped after the 0.10 and 0.05 stages: never went down to 0.03
