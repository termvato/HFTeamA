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
PLACE = frozenset({"place"})
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
    ex, log, clock = make(tmp_path, timeout_guess=2.0)
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


def test_timeout_is_3x_guess_then_1_5x_measured(tmp_path: Path) -> None:
    ex, _, clock = make(tmp_path / "guess")
    ex.add_arm(arm("a", clock))
    with ex._lock:
        ex._launch("pick_01", "a")
    assert ex._runs["pick_01"].limit == pytest.approx(3.0 * 4.0)  # task file says 4 s

    ex2, _, clock2 = make(tmp_path / "measured")
    ex2.add_arm(arm("a", clock2))
    for _ in range(3):
        ex2.durations.record("a", "pick", 5.0)
    with ex2._lock:
        ex2._launch("pick_01", "a")
    assert ex2._runs["pick_01"].limit == pytest.approx(1.5 * 5.0)


def _more(graph: TaskGraph, n_new: int, start: int = 13) -> TaskGraph:
    tasks = dict(graph.tasks)
    for i in range(start, start + n_new):
        k = f"{i:02d}"
        tasks[f"pick_{k}"] = Task(f"pick_{k}", "pick", resources=("input_bin",), duration=4.0)
        tasks[f"place_{k}"] = Task(
            f"place_{k}", "place", deps=(f"pick_{k}",), same_arm_as=f"pick_{k}", duration=6.0
        )
    return TaskGraph(graph.name, tasks)


def test_update_graph_mid_run_adds_and_drops(tmp_path: Path) -> None:
    ex, log, clock = make(tmp_path)
    ex.add_arm(arm("a", clock))
    ex.add_arm(arm("b", clock))
    ex.start()
    until(lambda: ex.snapshot()["now"] > 10)
    status = ex.snapshot()["status"]
    pending = [t for t in ("pick_12", "place_12") if status[t] == "pending"]
    g = _more(SORT12, 1)
    for t in pending:
        del g.tasks[t]
    ex.update_graph(g)
    assert ex.wait(10)
    final = ex.snapshot()["status"]
    assert final["pick_13"] == final["place_13"] == "done"
    for t in pending:
        assert t not in final
    upd = [e for e in kinds(log, "replan") if e.data.get("reason") == "graph_updated"]
    assert upd and "pick_13" in upd[0].data["added"]


def test_update_graph_after_finish_runs_again(tmp_path: Path) -> None:
    ex, log, clock = make(tmp_path)
    ex.add_arm(arm("a", clock))
    ex.add_arm(arm("b", clock))
    ex.start()
    assert ex.wait(10)
    ex.update_graph(_more(SORT12, 1))
    until(lambda: ex.snapshot()["status"].get("place_13") == "done")
    assert ex.wait(5)


def test_whatif_uses_spare_catalog_arms(tmp_path: Path) -> None:
    g = TaskGraph(
        "reach",
        {
            "near": Task("near", "pick_place", allowed_arms=("a", "z"), duration=2.0),
            "far": Task("far", "pick_place", allowed_arms=("z",), duration=3.0),
        },
    )
    ex, log, clock = make(tmp_path, graph=g, catalog=["a", "z"])
    ex.add_arm(arm("a", clock))
    ex.estimate()
    w = kinds(log, "replan")[-1].data["whatif"]
    assert "2" in w  # with the spare arm z the far step becomes possible
    assert "1" not in w  # arm a alone cannot finish (far is blocked)


def test_camera_checks_only_place_steps_in_demo_setting(tmp_path: Path) -> None:
    from hfteama.executor import PLACE_SKILLS

    checked: list[str] = []

    def verifier(task, result) -> bool:
        checked.append(task.skill)
        return True

    ex, _, clock = make(tmp_path, verifier=verifier, verify_skills=PLACE_SKILLS)
    ex.add_arm(arm("a", clock))
    ex.add_arm(arm("b", clock))
    ex.start()
    assert ex.wait(10)
    assert checked and set(checked) == {"place"} and len(checked) == 12


def test_a_slow_place_check_does_not_rerun_the_place_and_stop_still_parks(tmp_path: Path) -> None:
    """Live demo 22 Nov: the arm kept "holding" a place while the camera checked it, so the
    dispatcher re-ran the place every few seconds, the check's answer was thrown away, and a
    remove never parked the arm. A finished place is not held any more."""
    import time as _time

    g = TaskGraph(
        "one",
        {
            "pick_a": Task("pick_a", "pick", args={"pick_xyz": [0.1, 0.1, 0.02]}, duration=1.0),
            "place_a": Task(
                "place_a", "place", deps=("pick_a",), same_arm_as="pick_a", duration=1.0
            ),
        },
    )

    def slow_check(task, result) -> tuple[bool, str]:
        _time.sleep(0.5)  # Qwen takes seconds; the fake arm is much faster than that
        return True, ""

    ex, log, clock = make(
        tmp_path, graph=g, verifier=slow_check, verify_skills=frozenset({"place"})
    )
    ex.add_arm(arm("a", clock))
    ex.start()
    assert ex.wait(10)
    starts = [e.task for e in kinds(log, "task_start")]
    assert starts == ["pick_a", "place_a"], starts
    assert set(ex.snapshot()["status"].values()) == {"done"}

    # a stop during the check parks the arm instead of re-running the place
    ex2, log2, clock2 = make(
        tmp_path / "b", graph=g, verifier=slow_check, verify_skills=frozenset({"place"})
    )
    ex2.add_arm(arm("a", clock2))
    ex2.start()
    end = _time.monotonic() + 10
    while ex2.snapshot()["status"]["place_a"] != "verifying" and _time.monotonic() < end:
        _time.sleep(0.01)
    ex2.remove_arm("a")
    end = _time.monotonic() + 10
    while not kinds(log2, "arm_removed") and _time.monotonic() < end:
        _time.sleep(0.01)
    assert kinds(log2, "arm_removed"), "the arm was never parked"
    assert [e.task for e in kinds(log2, "task_start")].count("place_a") == 1
    ex2.stop()


def test_lost_part_waits_for_a_rescan_before_the_pick_is_retried(tmp_path: Path) -> None:
    from dataclasses import replace as dc_replace

    g = TaskGraph(
        "one",
        {
            "pick_a": Task("pick_a", "pick", args={"pick_xyz": [0.1, 0.1, 0.02]}, duration=4.0),
            "place_a": Task(
                "place_a", "place", deps=("pick_a",), same_arm_as="pick_a", duration=4.0
            ),
        },
    )
    seen: list[str] = []
    calls: list[float] = []

    def verifier(task, result) -> tuple[bool, str]:
        seen.append(task.id)
        return (seen.count("place_a") > 1, "not in the bin")  # first place is rejected

    def rescan() -> TaskGraph:
        calls.append(time.monotonic())
        time.sleep(0.05)  # the camera and the VLM take a moment
        moved = dc_replace(g.tasks["pick_a"], args={"pick_xyz": [0.3, 0.2, 0.02]})
        return TaskGraph("one", {**g.tasks, "pick_a": moved})

    ex, log, clock = make(tmp_path, graph=g, verifier=verifier, verify_skills=PLACE, rescan=rescan)
    ex.add_arm(arm("a", clock))
    ex.start()
    assert ex.wait(10)
    assert len(calls) == 1
    assert ex.graph.tasks["pick_a"].args["pick_xyz"] == [0.3, 0.2, 0.02]
    order = [
        (e.kind, e.task or e.data.get("reason"))
        for e in log.read()
        if e.kind == "task_start" or e.data.get("reason") in ("rescan_requested", "graph_updated")
    ]
    assert order == [
        ("task_start", "pick_a"),
        ("task_start", "place_a"),
        ("replan", "rescan_requested"),
        ("replan", "graph_updated"),
        ("task_start", "pick_a"),
        ("task_start", "place_a"),
    ]
