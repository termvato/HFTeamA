"""Plug-in detection: a known USB serial appearing adds the arm, disappearing drops it."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from hfteama.arms.fake import FakeArm
from hfteama.clock import Clock
from hfteama.discovery import ArmWatcher, followers_from_pairs
from hfteama.events import EventLog
from hfteama.executor import Executor
from hfteama.registry import load_real_arms
from hfteama.taskgraph import load

ROOT = Path(__file__).parent.parent
SORT12 = load(ROOT / "tasks" / "sort" / "sort12.json")


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def add_arm(self, worker) -> None:
        self.calls.append(("add", worker.arm_id))

    def arm_lost(self, arm_id: str, reason: str = "unplugged") -> None:
        self.calls.append(("lost", arm_id))


def watcher(ex, present: set[str], **kw) -> ArmWatcher:
    return ArmWatcher(
        ex,
        {"S1": "arm_one", "S2": "arm_two"},
        factory=lambda a: FakeArm(a, clock=Clock(100.0)),
        scan=lambda: set(present),
        **kw,
    )


def test_pairs_config_maps_follower_serials() -> None:
    known = followers_from_pairs(ROOT / "configs" / "teleop_pairs.json")
    assert known["5B79014975"] == "rakesh_follower"
    assert "vato_follower" in known.values() and "rakesh_leader" not in known.values()


def test_plug_and_unplug_are_debounced() -> None:
    ex, present = Recorder(), {"S1", "UNKNOWN"}
    w = watcher(ex, present)
    w.poll_once()
    assert ex.calls == []  # seen once: wait for a second scan
    w.poll_once()
    assert ex.calls == [("add", "arm_one")]
    w.poll_once()
    assert ex.calls == [("add", "arm_one")]  # no repeat while it stays plugged in
    present.add("S2")
    present.discard("S1")
    w.poll_once()
    present.add("S1")  # a one-scan glitch must not drop the arm
    w.poll_once()
    assert ("lost", "arm_one") not in ex.calls and ("add", "arm_two") in ex.calls


def test_failed_scan_and_failed_start_are_survived() -> None:
    ex = Recorder()
    errors: list[str] = []

    def boom() -> set[str]:
        raise OSError("usb hiccup")

    w = ArmWatcher(ex, {"S1": "a"}, factory=lambda a: FakeArm(a), scan=boom, debounce=1)
    w.poll_once()
    assert ex.calls == []

    def bad_factory(arm_id: str):
        raise RuntimeError("not calibrated")

    w2 = ArmWatcher(
        ex,
        {"S1": "a"},
        factory=bad_factory,
        scan=lambda: {"S1"},
        debounce=1,
        on_error=lambda a, e: errors.append(f"{a}: {e}"),
    )
    w2.poll_once()
    assert ex.calls == [] and errors == ["a: not calibrated"] and w2.joined() == set()


def test_plugging_an_arm_mid_run_resplits_and_unplugging_recovers(tmp_path: Path) -> None:
    clock = Clock(100.0)
    log = EventLog(tmp_path / "events.jsonl")
    ex = Executor(SORT12, log, clock=clock)
    present = {"S1"}
    w = ArmWatcher(
        ex,
        {"S1": "arm_one", "S2": "arm_two"},
        factory=lambda a: FakeArm(a, clock=clock, duration_fn=lambda t: t.duration),
        scan=lambda: set(present),
        debounce=1,
    )
    w.poll_once()
    ex.start()
    end = time.monotonic() + 5
    while ex.snapshot()["now"] < 15 and time.monotonic() < end:
        time.sleep(0.005)
    present.add("S2")
    w.poll_once()
    while ex.snapshot()["now"] < 35 and time.monotonic() < end:
        time.sleep(0.005)
    present.discard("S2")
    w.poll_once()
    assert ex.wait(10)
    kinds = [(e.kind, e.arm) for e in log.read() if e.kind in ("arm_added", "arm_failed")]
    assert kinds == [("arm_added", "arm_one"), ("arm_added", "arm_two"), ("arm_failed", "arm_two")]
    assert set(ex.snapshot()["status"].values()) == {"done"}


def test_catalog_from_actions_real_arms_file(tmp_path: Path) -> None:
    p = tmp_path / "real_arms.json"
    arms = {
        "rakesh_follower": {"pair": "rakesh", "base": [0.089, 0.42, 0.0], "yaw": -math.pi / 2},
        "vato_follower": {"pair": "hacklab", "base": [0.488, 0.44, 0.0], "yaw": -math.pi / 2},
    }
    p.write_text(json.dumps({"arms": arms}))
    cat = load_real_arms(p)
    left = cat["rakesh_follower"]
    assert left.base_yaw_deg == pytest.approx(-90.0) and left.base_pose_m[0] == 0.089
    assert left.reaches((0.1, 0.15)) and not left.reaches((0.1, 0.9))  # faces -y, 0.44 m reach
