"""Fake arm (backlog #14): implements the arm contract without hardware.

It "runs" a step by sleeping for the step's duration (plus noise) on a Clock, so the planner,
executor and screen can be built and tested before any real or simulated arm works.
Failures can be injected: a failed step, a dead arm (heartbeat stops), or a slow arm.

Like every worker it reads only task.id, task.skill and task.args. How long a step takes is a
property of the fake arm (`durations` per skill, or a `duration_fn` given by the test harness).
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable

from hfteama.arms.base import ArmState, Outcome, Result
from hfteama.clock import Clock
from hfteama.taskgraph import Task


class FakeArm:
    backend = "fake"

    def __init__(
        self,
        arm_id: str,
        clock: Clock | None = None,
        durations: dict[str, float] | None = None,
        default_duration: float = 5.0,
        duration_fn: Callable[[Task], float] | None = None,
        noise: float = 0.0,
        fail_prob: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.arm_id = arm_id
        self.clock = clock or Clock()
        self.durations = dict(durations or {})
        self.default_duration = default_duration
        self.duration_fn = duration_fn
        self.noise = noise
        self.fail_prob = fail_prob
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._state = ArmState.READY
        self._cancel = threading.Event()
        self._current: str | None = None
        self._down = False
        self._down_at = 0.0
        self._fail_next: str | None = None
        self._slow = 1.0

    # ---- contract

    def state(self) -> ArmState:
        with self._lock:
            return self._state

    def execute(self, task: Task) -> Result:
        with self._lock:
            if self._down:
                now = self.clock.time()
                return Result(task.id, self.arm_id, Outcome.ARM_DOWN, now, now, "arm down")
            self._state = ArmState.BUSY
            self._current = task.id
            self._cancel.clear()
            base = self.duration_fn(task) if self.duration_fn else self._skill_duration(task.skill)
            jitter = self._rng.uniform(-self.noise, self.noise) if self.noise else 0.0
            seconds = max(0.0, base * (1.0 + jitter) * self._slow)
        start = self.clock.time()
        interrupted = self.clock.sleep(seconds, self._cancel)
        end = self.clock.time()
        with self._lock:
            self._current = None
            if self._down:
                return Result(task.id, self.arm_id, Outcome.ARM_DOWN, start, end, "arm down")
            self._state = ArmState.READY
            if interrupted:
                return Result(task.id, self.arm_id, Outcome.FAILED, start, end, "cancelled")
            if self._fail_next is not None:
                reason, self._fail_next = self._fail_next, None
                return Result(task.id, self.arm_id, Outcome.FAILED, start, end, reason)
            if self.fail_prob and self._rng.random() < self.fail_prob:
                return Result(task.id, self.arm_id, Outcome.FAILED, start, end, "grasp")
            return Result(task.id, self.arm_id, Outcome.OK, start, end)

    def heartbeat(self) -> float:
        with self._lock:
            return self._down_at if self._down else self.clock.time()

    def cancel(self, task_id: str) -> None:
        with self._lock:
            if self._current == task_id:
                self._cancel.set()

    def park(self) -> None:
        with self._lock:
            self._cancel.set()
            self._state = ArmState.DOWN

    # ---- failure injection (tests and demo controls)

    def fail_next(self, reason: str = "grasp") -> None:
        """The next step that finishes reports FAILED with this reason."""
        with self._lock:
            self._fail_next = reason

    def go_down(self) -> None:
        """Simulate an unplugged or dead arm: heartbeat stops, the running step ends as ARM_DOWN."""
        with self._lock:
            self._down = True
            self._down_at = self.clock.time()
            self._state = ArmState.DOWN
            self._cancel.set()

    def set_slow(self, factor: float) -> None:
        """Multiply the duration of later steps (e.g. 5.0 to trigger a timeout)."""
        with self._lock:
            self._slow = factor

    def _skill_duration(self, skill: str) -> float:
        return self.durations.get(skill, self.default_duration)
