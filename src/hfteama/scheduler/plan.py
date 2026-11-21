"""Plan format shared by the scheduler, executor and view."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Assignment:
    task_id: str
    arm_id: str
    start: float  # seconds, same clock as the executor's `now`
    end: float


@dataclass
class Plan:
    assignments: list[Assignment]
    solver: str  # "greedy" | "heft" | "cpsat" (or "retimed:<solver>")
    makespan: float  # predicted finish time
    lower_bound: float = 0.0
    binding: str = ""  # "arms" | "order" | "resource:<name>"
    info: dict[str, float] = field(default_factory=dict)  # solve time, gap, ...
    blocked: list[str] = field(default_factory=list)  # steps no available arm can do

    def by_arm(self) -> dict[str, list[Assignment]]:
        out: dict[str, list[Assignment]] = {}
        for a in sorted(self.assignments, key=lambda a: a.start):
            out.setdefault(a.arm_id, []).append(a)
        return out
