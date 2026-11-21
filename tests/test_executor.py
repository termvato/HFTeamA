"""Executor (#10) and live re-split (#11) on fake arms, with the clock sped up."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hfteama.arms.fake import FakeArm
from hfteama.clock import Clock
from hfteama.events import Event, EventLog
from hfteama.executor import Executor
from hfteama.taskgraph import Task, TaskGraph, load

SORT12 = load(Path(__file__).parent.parent / "tasks" / "sort" / "sort12.json")
SCALE = 100.0  # functional tests; timing is checked at a slower speed-up


def arm(arm_id: str, clock: Clock) -> FakeArm:
    return FakeArm(arm_id, clock=clock, duration_fn=lambda t: t.duration_on(arm_id))


def make(
    tmp_path: Path, graph: TaskGraph = SORT12, scale: float = SCALE, **kw
) -> tuple[Executor, EventLog, Clock]:
    clock = Clock(scale=scale)
    log = EventLog(tmp_path / "events.jsonl")
    return Executor(graph, log, clock=clock, **kw), log, clock


def until(pred, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


def kinds(log: EventLog, kind: str) -> list[Event]:
    return [e for e in log.read() if e.kind == kind]


def ok_ends(log: EventLog) -> list[Event]:
    return [e for e in kinds(log, "task_end") if e.outcome == "ok"]


def test_runs_a_batch_to_completion(tmp_path: Path) -> None:
    # At 25x, a few ms of re-planning and thread wake-ups per step cost little batch time.
    ex, log, clock = make(tmp_path, scale=25.0)
    ex.add_arm(arm("a", clock))
    ex.add_arm(arm("b", clock))
    ex.start()
    assert ex.wait(10)
    assert set(ex.snapshot()["status"].values()) == {"done"}
    assert len(ok_ends(log)) == 24
    finish = kinds(log, "replan")[-1].data["finish"]
    assert finish == pytest.approx(64.0, abs=8.0)  # plan says 64 s for two arms


def test_shared_bin_never_used_twice_at_once(tmp_path: Path) -> None:
    ex, log, clock = make(tmp_path)
    for a in "abc":
        ex.add_arm(arm(a, clock))
    ex.start()
    assert ex.wait(10)
    picks = sorted(
        (e.data["start"], e.data["end"]) for e in ok_ends(log) if e.task and "pick" in e.task
    )
    for (_, e1), (s2, _) in zip(picks, picks[1:], strict=False):
        assert s2 >= e1 - 0.5


def test_adding_an_arm_mid_run_resplits(tmp_path: Path) -> None:
    ex, log, clock = make(tmp_path)
    ex.add_arm(arm("a", clock))
    ex.start()
    until(lambda: ex.snapshot()["now"] > 20)
    ex.add_arm(arm("b", clock))
    assert ex.wait(10)
    added = [e for e in kinds(log, "replan") if e.data.get("reason") == "arm_added"]
    assert added[-1].data["committed"] and set(added[-1].data["arms"]) == {"a", "b"}
    assert {e.arm for e in ok_ends(log)} == {"a", "b"}
    assert kinds(log, "replan")[-1].data["finish"] < 110


def test_removing_an_arm_is_a_slow_stop(tmp_path: Path) -> None:
    ex, log, clock = make(tmp_path)
    ex.add_arm(arm("a", clock))
    ex.add_arm(arm("b", clock))
    ex.start()
    until(lambda: any(e.arm == "b" and "pick" in (e.task or "") for e in kinds(log, "task_start")))
    ex.remove_arm("b")
    assert ex.wait(10)
    removed_at = kinds(log, "arm_removed")[0].t
    b_ends = [e for e in ok_ends(log) if e.arm == "b"]
    assert b_ends and b_ends[-1].task.startswith("place")  # placed the part it held, then parked
    assert all(e.t <= removed_at for e in b_ends)
    assert set(ex.snapshot()["status"].values()) == {"done"}


def test_failed_step_is_retried(tmp_path: Path) -> None:
    ex, log, clock = make(tmp_path)
    a = arm("a", clock)
    a.fail_next("grasp")
    ex.add_arm(a)
    ex.add_arm(arm("b", clock))
    ex.start()
    assert ex.wait(10)
    failed = [e for e in kinds(log, "task_end") if e.outcome == "failed"]
    assert failed and failed[0].data["reason"] == "grasp"
    assert set(ex.snapshot()["status"].values()) == {"done"}


def test_dead_arm_work_flows_to_the_others(tmp_path: Path) -> None:
    ex, log, clock = make(tmp_path)
    a, b = arm("a", clock), arm("b", clock)
    ex.add_arm(a)
    ex.add_arm(b)
    ex.start()
    until(lambda: ex.snapshot()["now"] > 15)
    b.go_down()
    assert ex.wait(10)
    assert [e.arm for e in kinds(log, "arm_failed")] == ["b"]
    assert set(ex.snapshot()["status"].values()) == {"done"}
    down_at = kinds(log, "arm_failed")[0].t
    assert all(e.arm == "a" for e in kinds(log, "task_start") if e.t > down_at)


def test_stuck_step_times_out_and_moves(tmp_path: Path) -> None:
    ex, log, clock = make(tmp_path, timeout_factor=2.0)
    slow = arm("b", clock)
    slow.set_slow(10.0)
    ex.add_arm(arm("a", clock))
    ex.add_arm(slow)
    ex.start()
    until(lambda: any(e.outcome == "timeout" for e in kinds(log, "task_end")))
    slow.set_slow(1.0)
    assert ex.wait(10)
    assert set(ex.snapshot()["status"].values()) == {"done"}


def test_verifier_rejection_retries(tmp_path: Path) -> None:
    seen: set[str] = set()

    def verifier(task, result) -> tuple[bool, str]:
        if task.id == "place_01" and task.id not in seen:
            seen.add(task.id)
            return False, "part not in bin"
        return True, ""

    ex, log, clock = make(tmp_path, verifier=verifier)
    ex.add_arm(arm("a", clock))
    ex.add_arm(arm("b", clock))
    ex.start()
    assert ex.wait(10)
    rejected = [e for e in kinds(log, "task_end") if "verification" in e.data.get("reason", "")]
    assert rejected and rejected[0].task == "place_01"
    assert set(ex.snapshot()["status"].values()) == {"done"}


def test_new_arm_unblocks_steps(tmp_path: Path) -> None:
    g = TaskGraph(
        "reach",
        {
            "near": Task("near", "pick_place", duration=2.0),
            "far": Task("far", "pick_place", allowed_arms=("z",), duration=3.0),
        },
    )
    ex, log, clock = make(tmp_path, graph=g)
    ex.add_arm(arm("a", clock))
    ex.start()
    until(lambda: ex.snapshot()["status"]["near"] == "done")
    assert not ex.finished and ex.snapshot()["blocked"] == ["far"]
    ex.add_arm(arm("z", clock))
    assert ex.wait(10)
    added = [e for e in kinds(log, "replan") if e.data.get("arm") == "z"]
    assert added[0].data["unblocked"] == ["far"]


def test_estimate_before_go(tmp_path: Path) -> None:
    ex, log, clock = make(tmp_path)
    ex.add_arm(arm("a", clock))
    plan = ex.estimate()
    assert plan.makespan == pytest.approx(120.0)
    last = kinds(log, "replan")[-1].data
    assert last["reason"] == "estimate" and last["binding"] == "arms"
    assert float(last["whatif"]["1"]) > float(last["whatif"]["2"]) > float(last["whatif"]["3"])
