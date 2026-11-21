"""Fake arm (#14): contract behaviour and failure injection."""

import threading

from hfteama.arms.base import ArmState, Outcome
from hfteama.arms.fake import FakeArm
from hfteama.clock import Clock
from hfteama.taskgraph import Task

# 10x: a 4 s step sleeps 0.4 s; timer ticks (~15.6 ms on Windows) stay small next to it
FAST = Clock(scale=10.0)


def task(tid: str = "t1", skill: str = "pick") -> Task:
    return Task(id=tid, skill=skill)


def test_ok_takes_the_skill_duration() -> None:
    arm = FakeArm("a", clock=FAST, durations={"pick": 4.0})
    r = arm.execute(task())
    assert r.outcome is Outcome.OK and r.arm_id == "a" and r.task_id == "t1"
    assert 3.9 <= r.duration <= 4.5
    assert arm.state() is ArmState.READY


def test_fail_next_then_ok() -> None:
    arm = FakeArm("a", clock=FAST, default_duration=1.0)
    arm.fail_next("grasp")
    r = arm.execute(task())
    assert r.outcome is Outcome.FAILED and r.reason == "grasp"
    assert arm.execute(task()).outcome is Outcome.OK


def test_cancel_interrupts() -> None:
    arm = FakeArm("a", clock=Clock(scale=1.0), default_duration=30.0)
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("r", arm.execute(task("x"))))
    th.start()
    while arm.state() is not ArmState.BUSY:
        pass
    arm.cancel("x")
    th.join(timeout=2)
    assert out["r"].outcome is Outcome.FAILED and out["r"].reason == "cancelled"


def test_go_down_stops_heartbeat_and_reports_arm_down() -> None:
    clock = Clock(scale=10.0)
    arm = FakeArm("a", clock=clock, default_duration=1.0)
    arm.go_down()
    beat = arm.heartbeat()
    clock.sleep(5.0)
    assert arm.heartbeat() == beat
    assert arm.execute(task()).outcome is Outcome.ARM_DOWN


def test_duration_fn_and_slow() -> None:
    arm = FakeArm("a", clock=FAST, duration_fn=lambda t: 2.0)
    arm.set_slow(3.0)
    assert 5.9 <= arm.execute(task()).duration <= 6.5
