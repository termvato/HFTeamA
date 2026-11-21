"""Event log, one JSON object per line. Interface 3 of 3 (frozen).

The executor writes it; the view (live Gantt) and the scaling analysis read it.
Kinds:
- plan:         one per task after each (re)solve: arm, planned_start, planned_end, solver name
- task_start / task_end:  actual times; task_end carries outcome
- arm_added / arm_removed / arm_failed
- replan:       why (reason), predicted finish (makespan) and lower bound with its binding term
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Event:
    kind: str
    t: float = field(default_factory=time.time)
    arm: str | None = None
    task: str | None = None
    planned_start: float | None = None
    planned_end: float | None = None
    outcome: str | None = None
    plan_version: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


class EventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: Event) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def read(self) -> list[Event]:
        if not self.path.exists():
            return []
        return [Event(**json.loads(line)) for line in self.path.read_text().splitlines() if line]
