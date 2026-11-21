"""The arm contract: execute(task) -> Result. Real, MuJoCo and fake arms all implement it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from hfteama.taskgraph import Task


class Outcome(str, Enum):
    OK = "ok"
    FAILED = "failed"  # the step did not succeed (e.g. part not grasped); see Result.reason
    TIMEOUT = "timeout"  # skill ran past ~1.5x predicted duration
    ARM_DOWN = "arm_down"  # bus error, heartbeat lost, or arm removed


class ArmState(str, Enum):
    STARTING = "starting"  # process up, safety checks and homing not done
    READY = "ready"
    BUSY = "busy"
    PARKING = "parking"  # removal requested: finish or abort skill, then park slowly
    DOWN = "down"


@dataclass(frozen=True)
class Result:
    task_id: str
    arm_id: str
    outcome: Outcome
    started_at: float  # wall clock, time.time()
    finished_at: float
    reason: str = ""  # free text for FAILED / ARM_DOWN, e.g. "grasp", "bus error"

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at


class ArmWorker(Protocol):
    arm_id: str
    backend: str  # "real" | "sim" | "fake"; shown as REAL/SIM on every Gantt bar

    def state(self) -> ArmState: ...

    def execute(self, task: Task) -> Result:
        """Run one task's skill. Blocks until done. Never raises for a task failure; returns it.

        Reads only task.id, task.skill and task.args (object id, target bin, table coordinates);
        the rest of the task is the planner's business.
        """
        ...

    def heartbeat(self) -> float:
        """Time of the arm's last sign of life; the executor marks it down if this goes stale."""
        ...

    def cancel(self, task_id: str) -> None:
        """Stop a task no longer assigned to this arm (slow stop, then park or hold safely)."""
        ...

    def park(self) -> None:
        """Slow stop: freeze, move to the parked pose, then torque off. No torque cut mid-air."""
        ...
