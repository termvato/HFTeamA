"""Demo runner wiring: executor + watcher + screen controls, on fake arms."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hfteama.clock import Clock
from hfteama.demo import build, fake_factory, hook
from hfteama.events import EventLog
from hfteama.taskgraph import load

SORT12 = load(Path(__file__).parent.parent / "tasks" / "sort" / "sort12.json")
KNOWN = {"S1": "rakesh_follower", "S2": "vato_follower"}


def until(pred, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.005)


def test_manual_plug_go_plug_unplug(tmp_path: Path) -> None:
    clock = Clock(100.0)
    log = EventLog(tmp_path / "e.jsonl")
    ex, watcher, rig = build(SORT12, log, KNOWN, fake_factory(clock, noise=0.0), clock=clock)
    assert rig.add() == "rakesh_follower"
    rig.go()
    until(lambda: ex.snapshot()["now"] > 15)
    assert rig.add() == "vato_follower" and rig.add() == "no arm left"
    until(lambda: set(ex.snapshot()["arms"]) == set(KNOWN.values()))
    rig.down("vato_follower")
    assert ex.wait(10)
    kinds = [(e.kind, e.arm) for e in log.read() if e.kind.startswith("arm_")]
    assert ("arm_added", "vato_follower") in kinds and ("arm_failed", "vato_follower") in kinds
    assert set(ex.snapshot()["status"].values()) == {"done"}


def test_removed_arm_is_not_reported_lost_and_can_rejoin(tmp_path: Path) -> None:
    clock = Clock(100.0)
    present = {"S1", "S2"}
    ex, watcher, rig = build(
        SORT12,
        EventLog(tmp_path / "e.jsonl"),
        KNOWN,
        fake_factory(clock, noise=0.0),
        clock=clock,
        scan=lambda: set(present),
    )
    watcher.debounce = 1
    watcher.poll_once()
    assert watcher.joined() == set(KNOWN.values())
    rig.remove("vato_follower")  # slow stop while still plugged in
    watcher.poll_once()
    assert watcher.joined() == {"rakesh_follower"}  # not re-added while the cable stays in
    present.discard("S2")
    watcher.poll_once()
    present.add("S2")
    watcher.poll_once()
    assert "vato_follower" in watcher.joined()  # replugged: joins again


def test_hook_spec() -> None:
    assert hook("hfteama.taskgraph:load") is load
    with pytest.raises(ValueError):
        hook("no_colon")
