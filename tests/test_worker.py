"""SkillWorker: outcome mapping, deadline, cancel, lost arm, and that it never raises."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from hfteama.arms.base import ArmState, Outcome
from hfteama.arms.poses import Poses
from hfteama.arms.primitives import MotionError
from hfteama.arms.skills import REGISTRY
from hfteama.arms.worker import SkillWorker
from hfteama.taskgraph import Task
from tests.fake_driver import FakeDriver

POSES = Poses(locations={"input_bin": [0.2, 0.0, 0.02]}, objects={"block_01": [0.21, 0.0, 0.015]})
PICK = Task(id="t1", skill="pick", args={"object": "block_01"})


def _worker(driver: FakeDriver | None = None, **kw: object) -> SkillWorker:
    return SkillWorker("arm_a", driver or FakeDriver(), POSES, backend="fake", **kw)


def _wait(cond: Callable[[], bool], timeout: float = 3.0) -> None:
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, "condition not reached"
        time.sleep(0.005)


def _run_in_thread(w: SkillWorker, task: Task) -> tuple[threading.Thread, dict]:
    box: dict = {}
    th = threading.Thread(target=lambda: box.setdefault("r", w.execute(task)))
    th.start()
    return th, box


def test_ok_result_has_ids_and_times() -> None:
    r = _worker().execute(PICK)
    assert (r.outcome, r.task_id, r.arm_id, r.reason) == (Outcome.OK, "t1", "arm_a", "")
    assert r.started_at <= r.finished_at


def test_skill_failed_maps_reason() -> None:
    r = _worker(FakeDriver(fail_grasp=True)).execute(PICK)
    assert (r.outcome, r.reason) == (Outcome.FAILED, "grasp")


def test_motion_error_maps_to_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreachable(arm, task, poses):  # noqa: ANN001, ANN202
        raise MotionError("unreachable")

    monkeypatch.setitem(REGISTRY, "_t_unreachable", unreachable)
    r = _worker().execute(Task(id="t2", skill="_t_unreachable"))
    assert (r.outcome, r.reason) == (Outcome.FAILED, "unreachable")


def test_unknown_skill() -> None:
    r = _worker().execute(Task(id="t3", skill="juggle"))
    assert (r.outcome, r.reason) == (Outcome.FAILED, "unknown skill: juggle")


def test_unknown_location_is_a_failed_step_not_a_crash() -> None:
    r = _worker().execute(Task(id="t4", skill="pick", args={"object": "mystery"}))
    assert (r.outcome, r.reason) == (Outcome.FAILED, "unknown location: mystery")


def test_timeout_stops_the_arm() -> None:
    d = FakeDriver(hang_on="move_to")
    start = time.time()
    r = _worker(d, deadline_s=0.2).execute(PICK)
    assert r.outcome == Outcome.TIMEOUT
    assert time.time() - start < 3.0
    assert ("stop", ()) in d.calls


def test_per_skill_deadline_overrides_default() -> None:
    d = FakeDriver(hang_on="move_to")
    r = _worker(d, deadline_s=30.0, skill_deadlines={"pick": 0.2}).execute(PICK)
    assert r.outcome == Outcome.TIMEOUT


def test_arm_down_is_sticky() -> None:
    d = FakeDriver(down_on="close_gripper")
    w = _worker(d)
    assert w.execute(PICK).outcome == Outcome.ARM_DOWN
    assert w.state() == ArmState.DOWN
    n = len(d.calls)
    again = w.execute(PICK)
    assert again.outcome == Outcome.ARM_DOWN
    assert len(d.calls) == n  # nothing was sent to a lost arm


def test_cancel_of_another_task_or_when_idle_is_a_noop() -> None:
    d = FakeDriver(hang_on="move_to")
    w = _worker(d)
    w.cancel("t1")  # idle
    assert ("stop", ()) not in d.calls
    th, box = _run_in_thread(w, PICK)
    _wait(lambda: any(n == "move_to" for n, _ in d.calls))
    w.cancel("some-other-task")
    assert ("stop", ()) not in d.calls
    w.cancel("t1")
    th.join(3.0)
    assert box["r"].outcome == Outcome.FAILED


def test_cancel_running_task() -> None:
    d = FakeDriver(hang_on="move_to")
    w = _worker(d)
    th, box = _run_in_thread(w, PICK)
    _wait(lambda: any(n == "move_to" for n, _ in d.calls))
    assert w.state() == ArmState.BUSY
    w.cancel("t1")
    th.join(3.0)
    assert not th.is_alive()
    assert (box["r"].outcome, box["r"].reason) == (Outcome.FAILED, "cancelled")
    assert w.state() == ArmState.READY


def test_after_cancel_the_skill_cannot_move_the_arm_again(monkeypatch: pytest.MonkeyPatch) -> None:
    def stubborn(arm, task, poses):  # noqa: ANN001, ANN202
        try:
            arm.move_to([0.2, 0.0, 0.1])
        except MotionError:
            pass
        arm.close_gripper()  # a skill that ignores the stop must still be refused

    monkeypatch.setitem(REGISTRY, "_t_stubborn", stubborn)
    d = FakeDriver(hang_on="move_to")
    w = _worker(d)
    th, box = _run_in_thread(w, Task(id="t5", skill="_t_stubborn"))
    _wait(lambda: any(n == "move_to" for n, _ in d.calls))
    w.cancel("t5")
    th.join(3.0)
    assert box["r"].outcome == Outcome.FAILED
    assert "close_gripper" not in [n for n, _ in d.calls]


def test_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(arm, task, poses):  # noqa: ANN001, ANN202
        return 1 / 0

    monkeypatch.setitem(REGISTRY, "_t_broken", broken)
    r = _worker().execute(Task(id="t6", skill="_t_broken"))
    assert r.outcome == Outcome.FAILED and "ZeroDivisionError" in r.reason


def test_park_goes_through_the_driver_and_marks_the_arm_down() -> None:
    d = FakeDriver()
    w = _worker(d)
    w.park()
    assert ("park", ()) in d.calls
    assert w.state() == ArmState.DOWN


def test_heartbeat_is_the_drivers_alive() -> None:
    assert abs(_worker().heartbeat() - time.time()) < 1.0


def test_has_every_contract_member() -> None:
    w = _worker()
    assert w.arm_id == "arm_a" and w.backend == "fake"
    for member in ("state", "execute", "heartbeat", "cancel", "park"):
        assert callable(getattr(w, member))
