"""Timing of the real-arm motion loops (phase improvement plan E): measurement only, no hardware."""

from __future__ import annotations

import json
import threading
import time

import pytest

np = pytest.importorskip("numpy")

from hfteama.arms.kinematics import BODY  # noqa: E402
from hfteama.runner import _ProcessProbe, _save_rescan_output  # noqa: E402
from hfteama.sorter.robot import ArmDriver  # noqa: E402


class FakeRobot:
    """A follower that is wherever it was last told to be; `send_s` makes the bus slow."""

    def __init__(self, send_s: float = 0.0):
        self.pos = {f"{j}.pos": 0.0 for j in BODY} | {"gripper.pos": 50.0}
        self.sent: list[dict] = []
        self.send_s = send_s

    def get_observation(self):
        return dict(self.pos)

    def send_action(self, action):
        time.sleep(self.send_s)
        self.sent.append(dict(action))
        self.pos.update(action)


def driver(robot, rate=200) -> tuple[ArmDriver, list[dict]]:
    d = object.__new__(ArmDriver)  # no pairs config, calibration file or bus
    d.side, d.robot = "test", robot
    d.motion = {"rate_hz": rate, "max_joint_speed_deg_s": 400.0}
    d.stop_event = threading.Event()
    d.last_io, d.gripper_target = 0.0, 50.0
    notes: list[dict] = []
    d.on_move = notes.append
    return d, notes


def test_move_deg_reports_its_ticks_and_sends_the_same_ramp() -> None:
    robot = FakeRobot()
    d, notes = driver(robot)
    d.move_deg({"shoulder_pan.pos": 20.0})
    assert robot.sent[-1]["shoulder_pan.pos"] == pytest.approx(20.0)  # the ramp ends on the goal
    (n,) = notes
    assert n["kind"] == "move_deg" and n["ticks"] == 10 == n["sends"]  # 20 deg at 2 deg a tick
    assert n["planned_s"] == pytest.approx(0.05)
    assert n["err_deg"] == pytest.approx(0.0) and n["settle_s"] < 0.05
    assert n["send_ms"] is not None and n["read_ms"] is not None


def test_a_slow_bus_shows_as_late_ticks_and_send_time() -> None:
    d, notes = driver(FakeRobot(send_s=0.012))  # 12 ms a command against a 5 ms slot
    d.move_deg({"shoulder_pan.pos": 10.0})
    (n,) = notes
    assert n["late_ticks"] == n["ticks"] == 5
    assert n["send_ms"] >= 12.0 and n["ramp_s"] > 2 * n["planned_s"]


def test_a_failing_logger_never_stops_a_move() -> None:
    robot = FakeRobot()
    d, _ = driver(robot)
    d.on_move = lambda rec: 1 / 0
    d.move_deg({"shoulder_pan.pos": 4.0})
    assert robot.pos["shoulder_pan.pos"] == pytest.approx(4.0)


def test_gripper_reports_where_it_got() -> None:
    d, notes = driver(FakeRobot())
    d.motion["max_relative_target"] = 10.0
    assert d.gripper(80.0) == pytest.approx(80.0, abs=1.0)
    assert notes[-1]["kind"] == "gripper" and notes[-1]["asked"] == 80.0


def test_process_probe_logs_cpu_threads_and_lateness(tmp_path) -> None:
    probe = _ProcessProbe(tmp_path / "process.jsonl", period_s=0.1, tick_s=0.01)
    probe.start()
    time.sleep(0.4)
    probe.stop()
    rec = json.loads((tmp_path / "process.jsonl").read_text().splitlines()[0])
    assert {"t", "cpu", "threads", "late_mean_ms", "late_max_ms", "scan_max_ms"} <= set(rec)
    assert rec["threads"] >= 2


def test_rescan_output_is_kept_in_the_run_folder(tmp_path) -> None:
    note = _save_rescan_output(tmp_path, "timed out after 120 s", "table", b"Thread 0x1 ...")
    (saved,) = tmp_path.glob("rescan-*.log")
    text = saved.read_text()
    assert "timed out" in text and "table" in text and "Thread 0x1" in text
    assert str(saved) in note
    assert _save_rescan_output(None, "x", "", "") == ""
