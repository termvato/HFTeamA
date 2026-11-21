"""hfteama.runner: the whole wiring (watcher, factory, worker, executor) with no hardware."""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from hfteama import runner  # noqa: E402
from hfteama.arms.ik import ArmBase, Kinematics  # noqa: E402
from hfteama.events import EventLog  # noqa: E402
from tests.fake_robot import FakeRobot, make_driver  # noqa: E402

pytestmark = pytest.mark.sim

ROOT = Path(__file__).parent.parent
SORT = ROOT / "tasks" / "sort"


def _small_graph(tmp_path: Path, pairs: int = 2) -> Path:
    full = json.loads((SORT / "sort12.json").read_text())
    keep = {f"{k}_{i:02d}" for i in range(1, pairs + 1) for k in ("pick", "place")}
    graph = {**full, "tasks": [t for t in full["tasks"] if t["id"] in keep]}
    path = tmp_path / "small.json"
    path.write_text(json.dumps(graph))
    return path


def _ends(log: Path) -> list:
    return [e for e in EventLog(log).read() if e.kind == "task_end"]


def test_sim_backend_runs_a_batch_through_the_executor(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    argv = ["--backend", "sim", "--task", str(_small_graph(tmp_path)), "--yes"]
    argv += ["--poses", str(SORT / "poses.json"), "--arms", "arm_a,arm_b", "--log", str(log)]
    assert runner.main(argv) == 0
    assert sorted(e.task for e in _ends(log) if e.outcome == "ok") == [
        "pick_01",
        "pick_02",
        "place_01",
        "place_02",
    ]
    assert any(e.kind == "arm_removed" for e in EventLog(log).read())  # arms were parked at the end


def test_real_arms_are_not_connected_without_the_typed_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("an arm was connected without confirmation")

    monkeypatch.setattr("hfteama.arms.real.from_config", boom)
    argv = ["--backend", "real", "--task", str(_small_graph(tmp_path)), "--arms", "rakesh_follower"]
    argv += ["--poses", str(SORT / "poses_real.example.json"), "--log", str(tmp_path / "e.jsonl")]
    assert runner.main(argv, ask=lambda _prompt: "y") == 2  # "y" is not "yes"


def test_real_backend_refuses_the_simulator_only_yes_flag(tmp_path: Path) -> None:
    argv = ["--backend", "real", "--yes", "--arms", "rakesh_follower"]
    with pytest.raises(SystemExit):
        runner.main(argv)


def test_declining_go_parks_the_connected_arm_and_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = FakeRobot()
    robot.grip_floor = 30.0
    monkeypatch.setattr("hfteama.arms.real.from_config", lambda arm_id, path: make_driver(robot))
    answers = iter(["yes", "no"])
    argv = ["--backend", "real", "--task", str(_small_graph(tmp_path)), "--arms", "rakesh_follower"]
    argv += ["--poses", str(SORT / "poses_real.example.json"), "--log", str(tmp_path / "e.jsonl")]
    assert runner.main(argv, ask=lambda _prompt: next(answers)) == 2
    assert robot.calls[-1] == "disable_torque"  # connected, then parked: never left holding
    assert not _ends(tmp_path / "e.jsonl")


def test_real_path_end_to_end_on_a_fake_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    robot = FakeRobot()
    robot.grip_floor = 30.0  # the fingers stall on the part
    base = ArmBase(
        (0.0, 0.26, 0.0), 4.7124
    )  # arm_a of the sim layout, standing in for the real arm
    monkeypatch.setattr(
        "hfteama.arms.real.from_config",
        lambda arm_id, path: make_driver(robot, arm_id=arm_id, kin=Kinematics(base)),
    )
    answers = iter(["yes", "go"])
    log = tmp_path / "events.jsonl"
    argv = ["--backend", "real", "--task", str(_small_graph(tmp_path)), "--arms", "rakesh_follower"]
    argv += ["--poses", str(SORT / "poses_real.example.json"), "--log", str(log)]
    assert runner.main(argv, ask=lambda _prompt: next(answers)) == 0
    ends = _ends(log)
    assert sorted(e.task for e in ends if e.outcome == "ok") == [
        "pick_01",
        "pick_02",
        "place_01",
        "place_02",
    ]
    assert {e.arm for e in ends} == {"rakesh_follower"}
    assert robot.calls[-1] == "disable_torque"  # parked lowered, then torque off


def test_ctrl_c_is_a_slow_stop_not_a_kill(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    argv = ["--backend", "sim", "--task", str(SORT / "sort12.json"), "--yes", "--log", str(log)]
    argv += ["--poses", str(SORT / "poses.json"), "--arms", "arm_a,arm_b"]
    timer = threading.Timer(4.0, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    try:
        assert runner.main(argv) == 130
    finally:
        timer.cancel()
    kinds = [e.kind for e in EventLog(log).read()]
    assert kinds.count("arm_removed") == 2  # both arms were parked, nothing was left running


def test_sim_can_inject_a_failure_and_the_executor_retries(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    argv = ["--backend", "sim", "--task", str(_small_graph(tmp_path)), "--yes"]
    argv += ["--poses", str(SORT / "poses.json"), "--arms", "arm_a,arm_b", "--log", str(log)]
    argv += ["--inject", "arm_a:grasp", "--inject", "arm_b:grasp"]
    assert runner.main(argv) == 0  # every step still ends ok
    ends = _ends(log)
    assert any(e.outcome == "failed" and e.data.get("reason") == "grasp" for e in ends)


def test_inject_is_for_the_simulator_only(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.main(["--backend", "real", "--arms", "rakesh_follower", "--inject", "arm_a:grasp"])


def test_the_timeout_factor_reaches_the_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}
    real_executor = runner.Executor

    def spy(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(runner, "Executor", spy)
    argv = ["--backend", "sim", "--task", str(_small_graph(tmp_path, pairs=1)), "--yes"]
    argv += ["--poses", str(SORT / "poses.json"), "--arms", "arm_a", "--log", str(tmp_path / "e")]
    argv += ["--timeout-factor", "4"]
    assert runner.main(argv) == 0
    assert seen["timeout_measured"] == 4.0


def test_hacklab_two_adjacent_arms_share_the_bowl_without_colliding(tmp_path: Path) -> None:
    hack = ROOT / "tasks" / "hacklab"
    full = json.loads((hack / "hacklab12.json").read_text())
    keep = {f"{k}_{i:02d}" for i in (1, 2, 7, 8) for k in ("pick", "place")}  # two blocks per arm
    graph = tmp_path / "hacklab4.json"
    graph.write_text(json.dumps({**full, "tasks": [t for t in full["tasks"] if t["id"] in keep]}))
    log = tmp_path / "events.jsonl"
    argv = ["--backend", "sim", "--task", str(graph), "--poses", str(hack / "poses.json")]
    argv += ["--arms", "rakesh_follower,vato_follower", "--yes", "--log", str(log)]
    assert runner.main(argv) == 0
    ends = _ends(log)
    assert {e.arm for e in ends} == {"rakesh_follower", "vato_follower"}
    assert all(e.outcome == "ok" for e in ends)  # no stall, no timeout: nobody hit the other arm


def test_reusing_a_log_path_does_not_mix_runs(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    argv = ["--backend", "sim", "--task", str(_small_graph(tmp_path, pairs=1)), "--yes"]
    argv += ["--poses", str(SORT / "poses.json"), "--arms", "arm_a", "--log", str(log)]
    assert runner.main(argv) == 0
    assert runner.main(argv) == 0  # same path again
    assert len([e for e in _ends(log) if e.outcome == "ok"]) == 2  # this run's, not 4


def test_real_runs_default_to_poses_real_and_fall_back_to_poses(tmp_path: Path) -> None:
    (tmp_path / "poses.json").write_text("{}")
    assert runner.default_poses(tmp_path, real=True) == str(tmp_path / "poses.json")
    (tmp_path / "poses_real.json").write_text("{}")
    assert runner.default_poses(tmp_path, real=True) == str(tmp_path / "poses_real.json")
    assert runner.default_poses(tmp_path, real=False) == str(tmp_path / "poses.json")
