"""Run a task graph on simulated SO-101 arms through SkillWorker (proof on MuJoCo, issue #20).

This is a test harness, not the executor (Operations owns that): one thread per arm takes the next
ready lead step (a step nobody must follow on the same arm), holds the locks of its `resources`
while it runs, then runs the steps tied to it by `same_arm_as`. A failed lead step goes back on the
list once. Exit code 0 only if every step ends `ok`.

    python scripts/sim_demo.py                          # tasks/sort/sort12.json on 3 arms
    python scripts/sim_demo.py --viewer --speed 1       # watch it
    python scripts/sim_demo.py --fail arm_b:grasp       # arm_b misses its first grasp
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from hfteama.arms import generic_skills  # noqa: F401  (registers the skills)
from hfteama.arms.base import ArmState, Outcome
from hfteama.arms.poses import Poses
from hfteama.arms.sim import SimTable
from hfteama.arms.worker import SkillWorker
from hfteama.events import Event, EventLog
from hfteama.taskgraph import Task, TaskGraph, load

ROOT = Path(__file__).resolve().parent.parent


class Board:
    """Shared list of lead steps, with each step's dependencies and retry count."""

    def __init__(self, graph: TaskGraph) -> None:
        self.graph = graph
        self.lock = threading.Lock()
        follows = {t.id: [] for t in graph.tasks.values() if t.same_arm_as is None}
        for tid in graph.topo_order():
            t = graph.tasks[tid]
            if t.same_arm_as is not None:
                follows[t.same_arm_as].append(tid)
        self.followers = follows
        self.pending = [tid for tid in graph.topo_order() if tid in follows]
        self.running = 0
        self.done: set[str] = set()
        self.last: dict[str, Outcome] = {}
        self.retries: dict[str, int] = {}

    def take(self) -> str | None | bool:
        """Next ready lead step; False if work remains but none is ready yet; None when finished."""
        with self.lock:
            for tid in self.pending:
                if all(d in self.done for d in self.graph.tasks[tid].deps):
                    self.pending.remove(tid)
                    self.running += 1
                    return tid
            return False if (self.pending or self.running) else None

    def finish(self, tid: str, ok: bool) -> None:
        with self.lock:
            self.running -= 1
            if ok:
                self.done.add(tid)
            elif self.retries.get(tid, 0) < 1:
                self.retries[tid] = self.retries.get(tid, 0) + 1
                self.pending.insert(0, tid)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--task", default=str(ROOT / "tasks" / "sort" / "sort12.json"))
    ap.add_argument("--poses", help="default: poses.json next to the task file")
    ap.add_argument("--speed", type=float, default=1.0, help="sim speed (1 = real time)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument(
        "--fail", action="append", default=[], metavar="ARM:KIND", help="grasp|down|slow"
    )
    ap.add_argument("--events", default=str(ROOT / "runs" / "sim_demo" / "events.jsonl"))
    args = ap.parse_args(argv)

    task_path = Path(args.task)
    graph = load(task_path)
    poses = Poses.load(args.poses or task_path.with_name("poses.json"))
    events = Path(args.events)
    events.parent.mkdir(parents=True, exist_ok=True)
    events.unlink(missing_ok=True)
    log = EventLog(events)
    board = Board(graph)
    locks: dict[str, threading.Lock] = {}
    errors: list[str] = []
    per_arm: dict[str, list[float]] = {}

    with SimTable(poses, speed=args.speed, viewer=args.viewer) as table:
        workers = {
            a: SkillWorker(a, table.driver(a), poses, backend="sim", clock=table.sim_time)
            for a in table.arm_ids
        }
        for spec in args.fail:
            arm, kind = spec.split(":")
            table.inject(arm, kind)  # type: ignore[arg-type]

        def run_step(w: SkillWorker, task: Task) -> Outcome:
            held = [locks.setdefault(r, threading.Lock()) for r in sorted(task.resources)]
            for lk in held:
                lk.acquire()
            try:
                log.write(Event(kind="task_start", t=table.sim_time(), arm=w.arm_id, task=task.id))
                res = w.execute(task)
                if res.outcome != Outcome.OK and w.state() != ArmState.DOWN:
                    # Clear the shared area before releasing its lock, as an executor would.
                    w.execute(Task(id=f"{task.id}:recover", skill="park"))
                log.write(
                    Event(
                        kind="task_end",
                        t=table.sim_time(),
                        arm=w.arm_id,
                        task=task.id,
                        outcome=res.outcome.value,
                        data={"reason": res.reason, "duration": res.duration},
                    )
                )
            finally:
                for lk in held:
                    lk.release()
            board.last[task.id] = res.outcome
            per_arm.setdefault(w.arm_id, []).append(res.duration)
            return res.outcome

        def arm_loop(w: SkillWorker) -> None:
            try:
                while True:
                    tid = board.take()
                    if tid is None:
                        return
                    if tid is False:
                        time.sleep(0.01)
                        continue
                    ok = run_step(w, graph.tasks[tid]) == Outcome.OK
                    for follower in board.followers[tid] if ok else []:
                        ok = run_step(w, graph.tasks[follower]) == Outcome.OK and ok
                    board.finish(tid, ok)
            except Exception as exc:  # noqa: BLE001  (report, don't hang the other arms)
                errors.append(f"{w.arm_id}: {exc!r}")

        t0 = table.sim_time()
        threads = [threading.Thread(target=arm_loop, args=(w,)) for w in workers.values()]
        [t.start() for t in threads]
        [t.join() for t in threads]
        makespan = table.sim_time() - t0

    bad = [tid for tid in graph.tasks if board.last.get(tid) != Outcome.OK] + errors
    print(
        f"{graph.name}: {len(graph.tasks) - len(bad)}/{len(graph.tasks)} steps ok on "
        f"{len(workers)} arms in {makespan:.1f} s of sim time"
    )
    for arm, ds in sorted(per_arm.items()):
        print(f"  {arm}: {len(ds)} steps, {sum(ds):.1f} s busy")
    if bad:
        print("  not ok:", ", ".join(bad))
    print(f"  events: {events}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
