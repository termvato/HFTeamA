"""The arm contract: execute(task) -> Result. Real, MuJoCo and fake arms all implement it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from hfteama.taskgraph import Task


class Outcome(str, Enum):
    OK = "ok"
    GRASP_FAIL = "grasp_fail"  # gripper closed to near-empty, or object lost in transport
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
    started: float  # wall clock, time.time()
    ended: float

    @property
    def duration(self) -> float:
        return self.ended - self.started


class ArmWorker(Protocol):
    arm_id: str
    backend: str  # "real" | "sim" | "fake"; shown as REAL/SIM on every Gantt bar

    def state(self) -> ArmState: ...

    def execute(self, task: Task) -> Result:
        """Run one task's skill. Blocks until done. Never raises for a task failure; returns it."""
        ...

    def park(self) -> None:
        """Slow stop: freeze, move to the parked pose, then torque off. No torque cut mid-air."""
        ...
