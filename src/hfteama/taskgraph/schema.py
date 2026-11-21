"""Task graph dataclasses, JSON loader and validator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class TaskGraphError(ValueError):
    """The task graph is malformed (unknown dep, cycle, bad field)."""


@dataclass(frozen=True)
class Task:
    id: str
    skill: str
    args: dict[str, Any] = field(default_factory=dict)
    deps: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()  # shared zones / tools, capacity 1
    allowed_arms: str | tuple[str, ...] = "*"
    capability: str | None = None  # kind of arm needed; None = any
    duration: float = 0.0
    durations: dict[str, float] = field(default_factory=dict)
    needs_arms: int = 1
    same_arm_as: str | None = None

    def can_run_on(self, arm_id: str, capabilities: frozenset[str] | None = None) -> bool:
        """True if the arm may run this task (listed, and has the capability when one is needed)."""
        if self.allowed_arms != "*" and arm_id not in self.allowed_arms:
            return False
        if self.capability is None or capabilities is None:
            return True
        return self.capability in capabilities

    def duration_on(self, arm_id: str) -> float:
        return self.durations.get(arm_id, self.duration)


@dataclass
class TaskGraph:
    name: str
    tasks: dict[str, Task]

    def topo_order(self) -> list[str]:
        """Task ids in dependency order. Raises TaskGraphError on a cycle."""
        indeg = {tid: len(t.deps) for tid, t in self.tasks.items()}
        children: dict[str, list[str]] = {tid: [] for tid in self.tasks}
        for tid, t in self.tasks.items():
            for d in t.deps:
                children[d].append(tid)
        ready = [tid for tid, n in indeg.items() if n == 0]
        order: list[str] = []
        while ready:
            tid = ready.pop()
            order.append(tid)
            for c in children[tid]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        if len(order) != len(self.tasks):
            stuck = sorted(set(self.tasks) - set(order))
            raise TaskGraphError(f"dependency cycle among: {stuck}")
        return order

    def validate(self) -> None:
        for tid, t in self.tasks.items():
            for d in t.deps:
                if d not in self.tasks:
                    raise TaskGraphError(f"{tid}: unknown dep {d!r}")
            if t.duration < 0 or any(v < 0 for v in t.durations.values()):
                raise TaskGraphError(f"{tid}: negative duration")
            if t.same_arm_as is not None and t.same_arm_as not in self.tasks:
                raise TaskGraphError(f"{tid}: unknown same_arm_as {t.same_arm_as!r}")
            if t.needs_arms not in (1, 2):
                raise TaskGraphError(f"{tid}: needs_arms must be 1 or 2")
        self.topo_order()


def _task_from_dict(d: dict[str, Any]) -> Task:
    arms = d.get("allowed_arms", "*")
    return Task(
        id=d["id"],
        skill=d["skill"],
        args=dict(d.get("args", {})),
        deps=tuple(d.get("deps", ())),
        resources=tuple(d.get("resources", ())),
        capability=d.get("capability"),
        allowed_arms=arms if arms == "*" else tuple(arms),
        duration=float(d.get("duration", 0.0)),
        durations={k: float(v) for k, v in d.get("durations", {}).items()},
        needs_arms=int(d.get("needs_arms", 1)),
        same_arm_as=d.get("same_arm_as"),
    )


def load(path: str | Path) -> TaskGraph:
    """Load and validate a task graph JSON file."""
    raw = json.loads(Path(path).read_text())
    tasks: dict[str, Task] = {}
    for d in raw["tasks"]:
        t = _task_from_dict(d)
        if t.id in tasks:
            raise TaskGraphError(f"duplicate task id {t.id!r}")
        tasks[t.id] = t
    graph = TaskGraph(name=raw.get("name", Path(path).stem), tasks=tasks)
    graph.validate()
    return graph
