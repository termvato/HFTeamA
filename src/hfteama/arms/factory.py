"""Build workers for the plug-in watcher: `factory(arm_id) -> ArmWorker`.

Operations' `ArmWatcher` (and the runner) call a factory when an arm appears. Building a real worker
connects to the arm, which enables torque (the arm holds its current pose), so it only happens after
the human has confirmed. The arm id is the catalog id (`rakesh_follower`, ...): the planner, the
reach graph, the executor and `configs/real_arms.json` all use it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from hfteama.arms.poses import Poses
from hfteama.arms.sim import SimTable
from hfteama.arms.worker import SkillWorker

WorkerFactory = Callable[[str], SkillWorker]


def real_worker_factory(config_path: str | Path, poses: Poses) -> WorkerFactory:
    """Real SO-101 workers from `configs/real_arms.json`. LeRobot is imported only when called."""

    def make(arm_id: str) -> SkillWorker:
        from hfteama.arms import real  # looked up at call time so tests can substitute from_config

        return SkillWorker(arm_id, real.from_config(arm_id, config_path), poses, "real")

    return make


def sim_worker_factory(table: SimTable, poses: Poses) -> WorkerFactory:
    """Simulated workers on a running `SimTable`: the same wiring, no hardware."""

    def make(arm_id: str) -> SkillWorker:
        return SkillWorker(arm_id, table.driver(arm_id), poses, "sim")

    return make
