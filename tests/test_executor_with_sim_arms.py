"""Operations' Executor drives our SkillWorker on MuJoCo arms: the contract holds end to end."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from hfteama.arms import generic_skills  # noqa: E402, F401  (registers the skills)
from hfteama.arms.poses import Poses  # noqa: E402
from hfteama.arms.sim import SimTable  # noqa: E402
from hfteama.arms.worker import SkillWorker  # noqa: E402
from hfteama.clock import Clock  # noqa: E402
from hfteama.events import EventLog  # noqa: E402
from hfteama.executor import Executor  # noqa: E402
from hfteama.taskgraph import load  # noqa: E402

pytestmark = pytest.mark.sim

SORT = Path(__file__).parent.parent / "tasks" / "sort"
SPEED = 1.0  # real-arm clock scale: the executor compares heartbeats to time.time()


def test_executor_runs_sort12_on_three_sim_arms(tmp_path: Path) -> None:
    poses = Poses.load(SORT / "poses.json")
    graph = load(SORT / "sort12.json")
    log = EventLog(tmp_path / "events.jsonl")
    with SimTable(poses, speed=SPEED) as table:
        ex = Executor(graph, log, clock=Clock(scale=SPEED))
        for arm_id in table.arm_ids:
            ex.add_arm(SkillWorker(arm_id, table.driver(arm_id), poses, "sim"))
        ex.start()
        try:
            assert ex.wait(timeout=180), "batch did not finish"
        finally:
            ex.stop()
    ends = [e for e in log.read() if e.kind == "task_end"]
    ok = {e.task for e in ends if e.outcome == "ok"}
    assert ok == set(graph.tasks), f"not ok: {sorted(set(graph.tasks) - ok)}"
    assert len({e.arm for e in ends}) >= 2  # really split (the shared bin may cap the useful arms)
