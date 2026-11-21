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


class FakeServos:
    """A serial port with Feetech STS3215 servos 1..6: answers pings and reads, ignores writes."""

    def __init__(self, ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6)) -> None:
        self.regs = {i: {40: 1, 56: 2048 + i * 10, 62: 121, 63: 31} for i in ids}
        self.sent: list[bytes] = []
        self.instructions: list[int] = []
        self.writes: list[tuple[int, int, int]] = []
        self._out = b""
        self.timeout = 0.05

    def write(self, data: bytes) -> int:
        self.sent.append(bytes(data))
        assert data[:2] == b"\xff\xff"
        sid, inst = data[2], data[4]
        assert (~(sum(data[2:-1])) & 0xFF) == data[-1], "bad checksum from the probe"
        self.instructions.append(inst)
        if inst == 0x03:  # write: keep it, answer with a status packet
            addr, value = data[5], data[6] | data[7] << 8
            self.writes.append((sid, addr, value))
            self.regs[sid][addr] = value
            body = bytes([sid, 2, 0])
            self._out += b"\xff\xff" + body + bytes([~sum(body) & 0xFF])
            return len(data)
        if sid not in self.regs:
            return len(data)
        if inst == 0x01:
            payload = b""
        else:
            addr, n = data[5], data[6]
            raw = bytes(self.regs[sid].get(addr + k, 0) if n == 1 else 0 for k in range(n))
            if n == 2:
                v = self.regs[sid][addr]
                raw = bytes([v & 0xFF, v >> 8])
            payload = raw
        body = bytes([sid, len(payload) + 2, 0]) + payload
        self._out += b"\xff\xff" + body + bytes([~sum(body) & 0xFF])
        return len(data)

    def read(self, n: int) -> bytes:
        out, self._out = self._out[:n], self._out[n:]
        return out

    def reset_input_buffer(self) -> None:
        self._out = b""

    def __enter__(self) -> FakeServos:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def close(self) -> None:
        pass


def test_probe_reads_every_servo_without_ever_writing() -> None:
    port = FakeServos()
    found = arm_check.probe(port, ids=range(1, 7))
    assert sorted(found) == [1, 2, 3, 4, 5, 6]
    assert found[3]["position"] == 2048 + 30 and found[3]["torque"] is True
    assert found[1]["volts"] == pytest.approx(12.1) and found[1]["temp_c"] == 31
    assert all(p[4] in (0x01, 0x02) for p in port.sent)  # pings and reads only
    assert port.writes == []


def test_probe_reports_a_missing_servo_as_absent() -> None:
    found = arm_check.probe(FakeServos(ids=(1, 2, 3)), ids=range(1, 7))
    assert sorted(found) == [1, 2, 3]


def test_hold_present_pose_writes_only_goal_equals_present_for_every_servo() -> None:
    from hfteama.arms import feetech

    port = FakeServos()
    for servo in port.regs.values():
        servo[40] = 0  # torque off: the dangerous moment
        servo[42] = 4000  # a stale goal from an earlier session
    feetech.hold_present_pose(port, ids=range(1, 7))
    assert sorted(port.writes) == [(i, 42, 2048 + i * 10) for i in range(1, 7)]
    assert all(servo[42] == servo[56] for servo in port.regs.values())


def test_hold_present_pose_refuses_when_a_servo_does_not_answer() -> None:
    from hfteama.arms import feetech

    port = FakeServos(ids=(1, 2, 3))
    with pytest.raises(feetech.ArmNotReady, match="4"):
        feetech.hold_present_pose(port, ids=range(1, 7))
    assert port.writes == []  # a half-answering arm is never partially written


def test_connecting_an_arm_first_holds_its_present_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    serial = pytest.importorskip(
        "serial"
    )  # pyserial: comes with LeRobot, not with the base install
    from hfteama.arms import real

    port = FakeServos()
    for servo in port.regs.values():
        servo[42] = 4000  # stale goal: torque-on would lunge toward it
    opened: list[tuple] = []
    monkeypatch.setattr(serial, "Serial", lambda *a, **k: (opened.append(a), port)[1])
    real.prepare_port("COM7")
    assert opened and opened[0][0] == "COM7"
    assert all(servo[42] == servo[56] for servo in port.regs.values())
