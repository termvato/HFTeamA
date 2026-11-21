"""Screen state from the event log (#12, #17)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hfteama.arms.fake import FakeArm
from hfteama.clock import Clock
from hfteama.events import EventLog
from hfteama.executor import Executor
from hfteama.taskgraph import load
from hfteama.view.state import build, clock_text, limit_text

SORT12 = load(Path(__file__).parent.parent / "tasks" / "sort" / "sort12.json")


def fake(arm_id: str, clock: Clock) -> FakeArm:
    return FakeArm(arm_id, clock=clock, duration_fn=lambda t: t.duration_on(arm_id))


def test_limit_sentences() -> None:
    assert "More arms" in limit_text("arms")
    assert "order of the steps" in limit_text("order")
    assert limit_text("resource:input_bin").startswith("The input bin is the bottleneck")
    assert limit_text("") == ""


def test_clock_text() -> None:
    assert clock_text(None) == "--:--"
    assert clock_text(64.4) == "1:04"
    assert clock_text(-3) == "0:00"


def test_estimate_before_go(tmp_path: Path) -> None:
    clock = Clock(100.0)
    log = EventLog(tmp_path / "e.jsonl")
    ex = Executor(SORT12, log, clock=clock)
    ex.add_arm(fake("a", clock))
    sc = build(log.read())
    assert sc.phase == "estimate"
    assert sc.estimate == pytest.approx(120.0)
    assert sc.whatif[1] == pytest.approx(120.0) and sc.whatif[3] < sc.whatif[2] < sc.whatif[1]
    assert sc.arms == {"a": "fake"} and len(sc.plan) == 24
    assert sc.limit == "More arms would make this faster."


def test_running_then_finished(tmp_path: Path) -> None:
    clock = Clock(100.0)
    log = EventLog(tmp_path / "e.jsonl")
    ex = Executor(SORT12, log, clock=clock)
    for a in "abc":
        ex.add_arm(fake(a, clock))
    ex.start()
    deadline = time.monotonic() + 5
    while ex.snapshot()["now"] < 20 and time.monotonic() < deadline:
        time.sleep(0.005)
    mid = build(log.read(), clock_now=clock.time())
    assert mid.phase == "running" and mid.finish is not None
    assert mid.remaining is not None and 0 < mid.remaining < 60
    assert mid.binding == "resource:input_bin" and "input bin" in mid.limit
    assert any(b.kind in ("actual", "running") for b in mid.actual)
    assert ex.wait(10)
    end = build(log.read())
    assert end.phase == "finished" and end.done == end.total == 24
    assert end.feed[0].startswith("Finished in")
    assert sum(1 for b in end.actual if b.outcome == "ok") == 24
