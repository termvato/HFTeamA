"""Executor (backlog #10) and live re-split (backlog #11).

Runs a plan as a partial order and re-plans the remaining work whenever something happens
(architecture doc, sections 5.3, 6 and 7):

- Dispatch: a step starts when its dependencies are done, its arm is free (in the plan's order for
  that arm) and its shared resources are free. Resource locks are taken right before the step and
  released when it ends, so two arms never use one resource at once even if durations drift.
- Every event re-plans. Running steps are never moved. The new plan replaces the old one only if
  it finishes more than 10% sooner, unless the arm set changed or something failed (hysteresis).
- An arm holding a part (its `same_arm_as` partner just finished) places it before anything else.
  If that arm is lost, the part's pick is redone.
- Failures: a failed step is retried (same or another arm) up to `max_retries`; an arm whose
  heartbeat goes stale is dropped and its running step goes back to the pool.
- Timeout (decided 21 Nov, adaptive): a step is cancelled and counted as a timeout after
  `timeout_guess` x its estimate while durations are still guesses (default 3x), or
  `timeout_measured` x the arm's measured 80th-percentile time once it has one (default 1.5x).
- Removing an arm is a slow stop: it finishes its step (and places a held part), then parks.
- Step check: an optional `verifier(task, result)` (e.g. the VLM looking at the camera) confirms
  a successful step before it counts as done; `verify_skills=PLACE_SKILLS` limits it to placing
  steps (the demo setting).

Every plan, start, end, arm change and re-plan goes to the event log; the screen reads only that.
"""

from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from hfteama.arms.base import ArmState, ArmWorker, Outcome, Result
from hfteama.clock import Clock
from hfteama.events import Event, EventLog
from hfteama.scheduler.durations import DurationModel
from hfteama.scheduler.plan import Assignment, Plan
from hfteama.scheduler.problem import Problem, unblocked_by
from hfteama.scheduler.solve import retime, solve, whatif
from hfteama.taskgraph import Task, TaskGraph

Verifier = Callable[[Task, Result], "bool | tuple[bool, str]"]


PLACE_SKILLS = frozenset({"place", "pick_place"})  # steps the demo's camera check confirms


class TaskState(str, Enum):
    PENDING = "pending"  # waiting to be dispatched (planned, or blocked until an arm can reach)
    RUNNING = "running"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"  # gave up (retries used up, or a dependency failed)


@dataclass
class _Run:
    task_id: str
    arm_id: str
    token: int
    start: float  # run seconds
    estimate: float
    limit: float = 0.0  # seconds after start before the step counts as a timeout
    timed_out: bool = False
    overrun: bool = False
    aborted: bool = False


class Executor:
    def __init__(
        self,
        graph: TaskGraph,
        log: EventLog,
        clock: Clock | None = None,
        method: str = "heft",
        time_limit: float = 1.0,
        timeout_guess: float = 3.0,  # x estimate while durations are guesses
        timeout_measured: float = 1.5,  # x measured 80th percentile once measured
        hysteresis: float = 0.10,
        max_retries: int = 2,
        heartbeat_timeout: float = 1.5,
        verifier: Verifier | None = None,
        verify_skills: frozenset[str] | None = None,
        durations: DurationModel | None = None,
        whatif_counts: tuple[int, ...] = (1, 2, 3, 4),
        catalog: list[str] | None = None,  # every known arm id (for the what-if row)
        poll: float = 0.02,
    ) -> None:
        graph.validate()
        self.graph = graph
        self.log = log
        self.clock = clock or Clock()
        self.method = method
        self.time_limit = time_limit
        self.timeout_guess = timeout_guess
        self.timeout_measured = timeout_measured
        self.hysteresis = hysteresis
        self.max_retries = max_retries
        self.heartbeat_timeout = heartbeat_timeout
        self.verifier = verifier
        # Which skills the camera confirms; None = every step. The demo passes PLACE_SKILLS
        # (decided 21 Nov: only placing steps; a failed pick shows up as a grasp failure).
        self.verify_skills = verify_skills
        self.durations = durations or DurationModel()
        self.whatif_counts = whatif_counts
        self.catalog = list(catalog or [])
        self.poll = poll

        self._lock = threading.RLock()
        self._inbox: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._tokens = itertools.count(1)
        self._workers: dict[str, ArmWorker] = {}
        self._active: set[str] = set()
        self._parking: set[str] = set()
        self._status = {t: TaskState.PENDING for t in graph.tasks}
        self._retries = {t: 0 for t in graph.tasks}
        self._runs: dict[str, _Run] = {}
        self._busy: dict[str, str] = {}  # arm -> running task
        self._held: dict[str, str] = {}  # arm -> task it must do next (holds the part)
        self._locks: dict[str, str] = {}  # resource -> task using it
        self._plan: Plan | None = None
        self._version = 0
        self._queues: dict[str, list[str]] = {}
        self._t0: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._finished = threading.Event()

    # ------------------------------------------------------------------ public (thread-safe)

    def add_arm(self, worker: ArmWorker) -> None:
        """An arm became available (plugged in and ready). Re-splits the remaining work."""
        self._submit(("add", worker))

    def update_graph(self, graph: TaskGraph) -> None:
        """Replace the batch mid-run (e.g. after a camera re-scan): new steps join, dropped
        pending steps go, changed steps take their new args. Done and running steps are kept
        even if the new graph leaves them out. Re-plans at once."""
        self._submit(("graph", graph))

    def arm_lost(self, arm_id: str, reason: str = "unplugged") -> None:
        """The arm vanished (USB unplugged, process died): drop it now, without waiting for its
        heartbeat to go stale. Its running step and any part it held go back to the pool."""
        self._submit(("lost", arm_id, reason))

    def remove_arm(self, arm_id: str, abort: bool = False) -> None:
        """Slow stop: no new steps; finish (abort=True: cancel) the current one, then park."""
        self._submit(("remove", arm_id, abort))

    def start(self) -> None:
        """Go: start the batch."""
        with self._lock:
            if self._thread is not None:
                return
            self._t0 = self.clock.time()
            self._emit(Event("replan", data={"reason": "go", "t0": self._t0}))
            self._replan("start", force=True)
            self._thread = threading.Thread(target=self._loop, name="executor", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every step is done or given up. `timeout` in real seconds."""
        return self._finished.wait(timeout)

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def estimate(self) -> Plan:
        """Plan and what-if for the current arms without starting (the estimate before Go)."""
        with self._lock:
            return self._replan("estimate", force=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "now": self.now(),
                "arms": sorted(self._active),
                "parking": sorted(self._parking),
                "status": {t: s.value for t, s in self._status.items()},
                "busy": dict(self._busy),
                "held": dict(self._held),
                "plan_version": self._version,
                "predicted_finish": self._plan.makespan if self._plan else None,
                "binding": self._plan.binding if self._plan else "",
                "blocked": list(self._plan.blocked) if self._plan else [],
            }

    def now(self) -> float:
        return 0.0 if self._t0 is None else self.clock.time() - self._t0

    # ------------------------------------------------------------------ loop

    def _submit(self, msg: tuple[Any, ...]) -> None:
        if self._thread is None or self._finished.is_set():  # before Go or after the end
            with self._lock:
                self._handle(msg)
        else:
            self._inbox.put(msg)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._inbox.get(timeout=self.poll)
            except queue.Empty:
                msg = None
            with self._lock:
                if msg is not None:
                    self._handle(msg)
                    while True:
                        try:
                            self._handle(self._inbox.get_nowait())
                        except queue.Empty:
                            break
                self._watch()
                self._dispatch()
                if all(s in (TaskState.DONE, TaskState.FAILED) for s in self._status.values()):
                    self._emit(
                        Event(
                            "replan",
                            plan_version=self._version,
                            data={"reason": "finished", "finish": self.now()},
                        )
                    )
                    self._finished.set()
                    return

    def _handle(self, msg: tuple[Any, ...]) -> None:
        kind = msg[0]
        if kind == "add":
            self._on_add(msg[1])
        elif kind == "remove":
            self._on_remove(msg[1], msg[2])
        elif kind == "result":
            self._on_result(msg[1], msg[2])
        elif kind == "lost":
            if msg[1] in self._active:
                self._lose_arm(msg[1], msg[2])
        elif kind == "graph":
            self._on_graph(msg[1])
        elif kind == "verified":
            self._on_verified(*msg[1:])

    # ------------------------------------------------------------------ arms

    def _on_graph(self, new: TaskGraph) -> None:
        started = {
            t
            for t, s in self._status.items()
            if s in (TaskState.RUNNING, TaskState.VERIFYING, TaskState.DONE)
        }
        tasks = dict(new.tasks)
        for t in started - set(tasks):
            tasks[t] = self.graph.tasks[t]  # never drop work that ran or is running
        merged = TaskGraph(new.name, tasks)
        merged.validate()
        added = sorted(set(tasks) - set(self.graph.tasks))
        dropped = sorted(set(self.graph.tasks) - set(tasks))
        changed = sorted(
            t
            for t in set(tasks) & set(self.graph.tasks) - started
            if tasks[t] != self.graph.tasks[t]
        )
        self.graph = merged
        self._status = {t: self._status.get(t, TaskState.PENDING) for t in tasks}
        self._retries = {t: self._retries.get(t, 0) for t in tasks}
        for arm, f in list(self._held.items()):
            if f not in tasks:
                del self._held[arm]
        self._finished.clear()
        self._replan(
            "graph_updated",
            force=True,
            extra={"added": added, "dropped": dropped, "changed": changed},
        )
        if self._thread is not None and not self._thread.is_alive():  # batch had finished
            self._thread = threading.Thread(target=self._loop, name="executor", daemon=True)
            self._thread.start()

    def _on_add(self, worker: ArmWorker) -> None:
        arm = worker.arm_id
        if arm in self._active:
            return
        # only meaningful when the arm joins others (the first arm "unblocks" everything)
        unblocked = unblocked_by(self._problem(), arm) if self._active else set()
        self._workers[arm] = worker
        self._active.add(arm)
        self._emit(Event("arm_added", arm=arm, data={"backend": worker.backend}))
        self._replan("arm_added", force=True, extra={"arm": arm, "unblocked": sorted(unblocked)})

    def _on_remove(self, arm: str, abort: bool) -> None:
        if arm not in self._active or arm in self._parking:
            return
        self._parking.add(arm)
        tid = self._busy.get(arm)
        if tid is not None and abort:
            self._runs[tid].aborted = True
            self._workers[arm].cancel(tid)
        if tid is None and arm not in self._held:
            self._park(arm)
        else:
            self._replan("arm_removing", force=True, extra={"arm": arm})

    def _park(self, arm: str) -> None:
        worker = self._workers.pop(arm)
        self._active.discard(arm)
        self._parking.discard(arm)
        threading.Thread(target=worker.park, name=f"park-{arm}", daemon=True).start()
        self._emit(Event("arm_removed", arm=arm))
        self._replan("arm_removed", force=True, extra={"arm": arm})

    def _lose_arm(self, arm: str, why: str) -> None:
        """Arm died or vanished: its running step and any part it held go back to the pool."""
        self._workers.pop(arm, None)
        self._active.discard(arm)
        self._parking.discard(arm)
        tid = self._busy.pop(arm, None)
        if tid is not None:
            self._runs.pop(tid, None)
            self._release(tid)
            self._status[tid] = TaskState.PENDING
            self._redo_partner(tid)
        held = self._held.pop(arm, None)
        if held is not None:
            self._redo_partner(held)
        self._emit(Event("arm_failed", arm=arm, data={"reason": why}))
        self._replan("arm_failed", force=True, extra={"arm": arm})

    # ------------------------------------------------------------------ steps

    def _dispatch(self) -> None:
        for arm in sorted(self._active):
            if arm not in self._active or arm in self._busy:
                continue
            worker = self._workers[arm]
            if worker.state() is not ArmState.READY:
                continue
            if arm in self._held:
                tid: str | None = self._held[arm]
            elif arm in self._parking:
                self._park(arm)
                continue
            else:
                tid = next(
                    (t for t in self._queues.get(arm, []) if self._status[t] is TaskState.PENDING),
                    None,
                )
            if tid is None or not self._can_start(tid):
                continue
            self._launch(tid, arm)

    def _can_start(self, tid: str) -> bool:
        t = self.graph.tasks[tid]
        if any(self._status[d] is not TaskState.DONE for d in t.deps):
            return False
        return all(r not in self._locks for r in t.resources)

    def _launch(self, tid: str, arm: str) -> None:
        task = self.graph.tasks[tid]
        for r in task.resources:
            self._locks[r] = tid
        token = next(self._tokens)
        est = self.durations.estimate(task, arm)
        measured = self.durations.measured(arm, task.skill) is not None
        factor = self.timeout_measured if measured else self.timeout_guess
        self._runs[tid] = _Run(tid, arm, token, self.now(), est, limit=factor * max(est, 1e-3))
        self._busy[arm] = tid
        self._status[tid] = TaskState.RUNNING
        planned = self._planned(tid)
        self._emit(
            Event(
                "task_start",
                arm=arm,
                task=tid,
                planned_start=planned.start if planned else None,
                planned_end=planned.end if planned else None,
                plan_version=self._version,
                data={"skill": task.skill, "start": self.now(), "estimate": est},
            )
        )
        worker = self._workers[arm]

        def run() -> None:
            try:
                res = worker.execute(task)
            except Exception as e:  # contract says execute never raises; be safe anyway
                now = self.clock.time()
                res = Result(tid, arm, Outcome.FAILED, now, now, f"exception: {e}")
            self._inbox.put(("result", token, res))

        threading.Thread(target=run, name=f"{arm}-{tid}", daemon=True).start()

    def _on_result(self, token: int, res: Result) -> None:
        run = self._runs.get(res.task_id)
        if run is None or run.token != token:
            return  # stale: the arm was already dropped
        tid, arm = res.task_id, run.arm_id
        del self._runs[tid]
        self._busy.pop(arm, None)
        self._release(tid)
        outcome, reason = res.outcome, res.reason
        if outcome is not Outcome.OK and run.timed_out:
            outcome, reason = Outcome.TIMEOUT, "timeout"
        if outcome is not Outcome.OK and run.aborted:
            reason = "aborted: arm removed"
        t0 = self._t0 or 0.0
        task = self.graph.tasks[tid]
        self._emit(
            Event(
                "task_end",
                arm=arm,
                task=tid,
                outcome=outcome.value,
                plan_version=self._version,
                data={
                    "skill": task.skill,
                    "start": res.started_at - t0,
                    "end": res.finished_at - t0,
                    "duration": res.duration,
                    "reason": reason,
                },
            )
        )
        if outcome is Outcome.OK:
            self.durations.record(arm, task.skill, res.duration)
            if self.verifier is None or (
                self.verify_skills is not None and task.skill not in self.verify_skills
            ):
                self._done(tid, arm)
                self._replan("task_done", force=False)
            else:
                self._status[tid] = TaskState.VERIFYING
                self._verify(tid, arm, token, res)
        elif outcome is Outcome.ARM_DOWN:
            self._status[tid] = TaskState.PENDING
            self._redo_partner(tid)
            self._lose_arm(arm, reason or "arm down")
        elif run.aborted:
            self._status[tid] = TaskState.PENDING
            self._redo_partner(tid)
            self._replan("task_aborted", force=True)
        else:
            self._retry(tid, reason or outcome.value)

    def _verify(self, tid: str, arm: str, token: int, res: Result) -> None:
        task = self.graph.tasks[tid]

        def run() -> None:
            try:
                out = self.verifier(task, res)  # type: ignore[misc]
                ok, why = out if isinstance(out, tuple) else (bool(out), "")
            except Exception as e:
                ok, why = False, f"verifier error: {e}"
            self._inbox.put(("verified", tid, arm, token, ok, why))

        threading.Thread(target=run, name=f"verify-{tid}", daemon=True).start()

    def _on_verified(self, tid: str, arm: str, token: int, ok: bool, why: str) -> None:
        if self._status[tid] is not TaskState.VERIFYING:
            return
        if ok:
            self._done(tid, arm)
            self._replan("task_done", force=False)
        else:
            self._emit(
                Event(
                    "task_end",
                    arm=arm,
                    task=tid,
                    outcome=Outcome.FAILED.value,
                    plan_version=self._version,
                    data={"reason": f"verification: {why}" if why else "verification"},
                )
            )
            self._retry(tid, "verification")

    def _done(self, tid: str, arm: str) -> None:
        self._status[tid] = TaskState.DONE
        if self._held.get(arm) == tid:
            del self._held[arm]
        follower = next((t.id for t in self.graph.tasks.values() if t.same_arm_as == tid), None)
        if follower and self._status[follower] is TaskState.PENDING and arm in self._active:
            self._held[arm] = follower

    def _retry(self, tid: str, reason: str) -> None:
        self._retries[tid] += 1
        if self._retries[tid] <= self.max_retries:
            self._redo_partner(tid)
            self._status[tid] = TaskState.PENDING
        else:
            self._give_up(tid, reason)
        self._replan("task_failed", force=True, extra={"task": tid, "why": reason})

    def _give_up(self, tid: str, reason: str) -> None:
        self._status[tid] = TaskState.FAILED
        for t in self.graph.topo_order():
            task = self.graph.tasks[t]
            deps = {*task.deps, *([task.same_arm_as] if task.same_arm_as else [])}
            if self._status[t] is TaskState.PENDING and any(
                self._status[d] is TaskState.FAILED for d in deps
            ):
                self._status[t] = TaskState.FAILED
                why = {"reason": "dependency"}
                self._emit(Event("task_end", task=t, outcome="failed", data=why))

    def _redo_partner(self, tid: str) -> None:
        """A placing step failed or its arm was lost: the part is no longer held, redo the pick."""
        partner = self.graph.tasks[tid].same_arm_as
        for arm, f in list(self._held.items()):
            if f == tid:
                del self._held[arm]
        if partner and self._status[partner] is TaskState.DONE:
            self._status[partner] = TaskState.PENDING

    def _release(self, tid: str) -> None:
        for r, holder in list(self._locks.items()):
            if holder == tid:
                del self._locks[r]

    # ------------------------------------------------------------------ watchdog

    def _watch(self) -> None:
        wall = self.clock.time()
        for arm in sorted(self._active):
            if arm not in self._active:
                continue
            if wall - self._workers[arm].heartbeat() > self.heartbeat_timeout:
                self._lose_arm(arm, "heartbeat lost")
        now = self.now()
        for run in list(self._runs.values()):
            if not run.timed_out and now > run.start + run.limit:
                run.timed_out = True
                self._workers[run.arm_id].cancel(run.task_id)
            elif not run.overrun and now > run.start + run.estimate * 1.5 + 1.0:
                run.overrun = True
                self._replan("overrun", force=False, extra={"task": run.task_id})

    # ------------------------------------------------------------------ planning

    def _problem(self) -> Problem:
        now = self.now()
        done = frozenset(
            t for t, s in self._status.items() if s in (TaskState.DONE, TaskState.VERIFYING)
        )
        running = {
            tid: Assignment(tid, r.arm_id, r.start, max(r.start + r.estimate, now))
            for tid, r in self._runs.items()
        }
        held = {
            a: t
            for a, t in self._held.items()
            if a in self._active and self._status[t] is TaskState.PENDING
        }
        restrict = {a: frozenset([held[a]] if a in held else []) for a in self._parking}
        dropped = frozenset(t for t, s in self._status.items() if s is TaskState.FAILED)
        return Problem(
            self.graph,
            sorted(self._active),
            now,
            done,
            running,
            held,
            restrict,
            estimate=self.durations.estimate,
            dropped=dropped,
        )

    def _replan(self, reason: str, force: bool, extra: dict[str, Any] | None = None) -> Plan:
        p = self._problem()
        new = solve(p, self.method, self.time_limit)
        chosen, committed = new, True
        if not force and self._plan is not None:
            old = retime(self._plan, p)
            if old is not None:
                old_left, new_left = old.makespan - p.now, new.makespan - p.now
                if not new_left < (1.0 - self.hysteresis) * old_left:
                    chosen, committed = old, False
        self._plan = chosen
        self._version += 1
        by_arm: dict[str, list[str]] = {}
        for a in sorted(chosen.assignments, key=lambda a: a.start):
            if a.task_id not in p.running:
                by_arm.setdefault(a.arm_id, []).append(a.task_id)
        self._queues = by_arm
        for a in chosen.assignments:
            self._emit(
                Event(
                    "plan",
                    arm=a.arm_id,
                    task=a.task_id,
                    planned_start=a.start,
                    planned_end=a.end,
                    plan_version=self._version,
                    data={"running": a.task_id in p.running},
                )
            )
        data: dict[str, Any] = {
            "reason": reason,
            "committed": committed,
            "solver": chosen.solver,
            "now": p.now,
            "makespan": chosen.makespan,
            "lower_bound": chosen.lower_bound,
            "binding": chosen.binding,
            "blocked": list(chosen.blocked),
            "arms": {a: self._workers[a].backend for a in sorted(self._active)},
            "whatif": {
                str(k): v
                for k, v in whatif(
                    p, self.whatif_counts, [a for a in self.catalog if a not in self._active]
                ).items()
            },
            "solve_time": chosen.info.get("solve_time", 0.0),
            "done": sum(s is TaskState.DONE for s in self._status.values()),
            "total": len(self._status),
        }
        data.update(extra or {})
        self._emit(Event("replan", plan_version=self._version, data=data))
        return chosen

    def _planned(self, tid: str) -> Assignment | None:
        if self._plan is None:
            return None
        return next((a for a in self._plan.assignments if a.task_id == tid), None)

    def _emit(self, event: Event) -> None:
        event.t = self.clock.time()
        self.log.write(event)
