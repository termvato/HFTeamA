"""SkillWorker: implements the frozen `ArmWorker` contract over any `ArmDriver`.

`execute(task)` runs the task's skill (looked up in `REGISTRY`) in a helper thread and always
returns a `Result`; it never raises for a task failure. The worker reads only `task.id`,
`task.skill` and `task.args` (Operations' locked contract). Its `deadline_s` is a hard safety cap
on one step; the planned-duration timeout (factor x estimate) belongs to the executor's watchdog.

A stop (deadline, `cancel`, `park`) sets a flag that makes every later primitive call from the
skill raise `MotionError("stopped")`, so a skill that ignores the first stop cannot move the arm
again.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from hfteama.arms import generic_skills  # noqa: F401  (registers pick, place, ...)
from hfteama.arms.base import ArmState, Outcome, Result
from hfteama.arms.poses import Poses
from hfteama.arms.primitives import ArmDown, ArmDriver, MotionError, SkillFailed
from hfteama.arms.skills import REGISTRY
from hfteama.taskgraph import Task

_JOIN_AFTER_STOP_S = 2.0
_PASSTHROUGH = frozenset({"stop", "alive", "read_joints"})


class _Guard:
    """The driver as a skill sees it: refuses every motion once a stop was requested."""

    def __init__(self, driver: ArmDriver, stopped: threading.Event) -> None:
        self._driver = driver
        self._stopped = stopped

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._driver, name)
        if name in _PASSTHROUGH or not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            if self._stopped.is_set():
                raise MotionError("stopped")
            return attr(*args, **kwargs)

        return call


class SkillWorker:
    def __init__(
        self,
        arm_id: str,
        driver: ArmDriver,
        poses: Poses,
        backend: str = "sim",
        deadline_s: float = 30.0,
        skill_deadlines: dict[str, float] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.arm_id = arm_id
        self.backend = backend  # "real" | "sim" | "fake"
        self._driver = driver
        self._poses = poses
        self._deadline_s = deadline_s
        self._skill_deadlines = dict(skill_deadlines or {})
        self._clock = clock
        self._lock = threading.Lock()
        self._state = ArmState.READY
        self._current: str | None = None
        self._abort: str | None = None  # "timeout" | "cancelled" while a stop is in effect
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- contract

    def state(self) -> ArmState:
        with self._lock:
            return self._state

    def heartbeat(self) -> float:
        return self._driver.alive()

    def cancel(self, task_id: str) -> None:
        with self._lock:
            running = self._state == ArmState.BUSY and self._current == task_id
        if running:
            self._request_stop("cancelled")

    def park(self) -> None:
        with self._lock:
            busy, th = self._state == ArmState.BUSY, self._thread
        if busy:
            self._request_stop("cancelled")
            if th is not None:
                th.join(_JOIN_AFTER_STOP_S)
        with self._lock:
            self._state = ArmState.PARKING
        try:
            self._driver.park()
        finally:
            with self._lock:
                self._state = ArmState.DOWN

    def execute(self, task: Task) -> Result:
        with self._lock:
            now = self._clock()
            if self._state == ArmState.DOWN:
                return Result(task.id, self.arm_id, Outcome.ARM_DOWN, now, now, "arm down")
            fn = REGISTRY.get(task.skill)
            if fn is None:
                reason = f"unknown skill: {task.skill}"
                return Result(task.id, self.arm_id, Outcome.FAILED, now, now, reason)
            self._state, self._current, self._abort = ArmState.BUSY, task.id, None
            self._stopped.clear()
        box: dict[str, BaseException | bool] = {}

        def run() -> None:
            try:
                fn(_Guard(self._driver, self._stopped), task, self._poses)
                box["ok"] = True
            except BaseException as exc:  # noqa: BLE001  (a worker never raises for a step)
                box["exc"] = exc

        started = self._clock()
        th = threading.Thread(target=run, name=f"{self.arm_id}:{task.id}", daemon=True)
        self._thread = th
        th.start()
        th.join(self._skill_deadlines.get(task.skill, self._deadline_s))
        if th.is_alive():
            self._request_stop("timeout")
            th.join(_JOIN_AFTER_STOP_S)
        finished = self._clock()
        outcome, reason = self._classify(box, stuck=th.is_alive())
        with self._lock:
            self._current = None
            if outcome == Outcome.ARM_DOWN or th.is_alive():
                self._state = ArmState.DOWN
            else:
                self._state = ArmState.READY
        return Result(task.id, self.arm_id, outcome, started, finished, reason)

    # ---- internals

    def _request_stop(self, why: str) -> None:
        with self._lock:
            if self._abort is None:
                self._abort = why
            self._stopped.set()
        self._driver.stop()

    def _classify(self, box: dict[str, BaseException | bool], stuck: bool) -> tuple[Outcome, str]:
        exc = box.get("exc")
        if "ok" in box:
            return Outcome.OK, ""
        if isinstance(exc, ArmDown):
            return Outcome.ARM_DOWN, exc.reason
        if self._abort == "timeout":
            return Outcome.TIMEOUT, "arm unresponsive" if stuck else "deadline"
        if self._abort == "cancelled":
            return Outcome.FAILED, "cancelled"
        if isinstance(exc, SkillFailed | MotionError):
            return Outcome.FAILED, exc.reason
        return Outcome.FAILED, repr(exc)
