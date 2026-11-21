"""Duration estimates (backlog #13): plans are only as good as their durations.

Starts from the task file's defaults; once an arm has run a skill a few times, plans use the 80th
percentile of what that arm actually took. An external estimator (e.g. a digital twin) can be
plugged in through the same `estimate(task, arm)` call.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable

from hfteama.events import EventLog
from hfteama.taskgraph import Task

MIN_SAMPLES = 3
PERCENTILE = 0.8


class DurationModel:
    def __init__(
        self,
        prior: Callable[[Task, str], float] | None = None,
        min_samples: int = MIN_SAMPLES,
        percentile: float = PERCENTILE,
    ) -> None:
        self.prior = prior
        self.min_samples = min_samples
        self.percentile = percentile
        self._samples: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, arm: str, skill: str, seconds: float) -> None:
        if seconds >= 0:
            with self._lock:
                self._samples[(arm, skill)].append(seconds)

    def measured(self, arm: str, skill: str) -> float | None:
        with self._lock:
            xs = sorted(self._samples.get((arm, skill), []))
        if len(xs) < self.min_samples:
            return None
        k = min(len(xs) - 1, max(0, round(self.percentile * (len(xs) - 1))))
        return xs[k]

    def estimate(self, task: Task, arm: str) -> float:
        m = self.measured(arm, task.skill)
        if m is not None:
            return m
        return self.prior(task, arm) if self.prior else task.duration_on(arm)

    def load(self, log: EventLog) -> int:
        """Learn from a previous run's event log (successful task_end events). Returns count."""
        n = 0
        for e in log.read():
            if e.kind == "task_end" and e.outcome == "ok" and e.arm and "skill" in e.data:
                self.record(e.arm, e.data["skill"], float(e.data.get("duration", -1)))
                n += 1
        return n

    def summary(self) -> dict[str, dict[str, float]]:
        """Per arm, per skill: measured estimate (only where enough samples exist)."""
        out: dict[str, dict[str, float]] = defaultdict(dict)
        with self._lock:
            keys = list(self._samples)
        for arm, skill in keys:
            m = self.measured(arm, skill)
            if m is not None:
                out[arm][skill] = m
        return dict(out)
