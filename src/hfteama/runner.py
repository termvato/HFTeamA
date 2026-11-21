"""Run a task graph on real (or simulated) SO-101 arms through Operations' executor.

    python -m hfteama.runner --backend real --arms rakesh_follower,vato_follower \
        --task tasks/sort/sort12.json --poses tasks/sort/poses_real.json
    python -m hfteama.runner --backend sim --arms arm_a,arm_b --yes     # same wiring, no hardware

The flow, and where a human stands between the code and the arms:
1. Load the task, the poses and the arm config; build the executor and the planner catalog.
2. REAL only: print what is about to happen and wait for the typed word `yes`. Connecting an arm
   enables its torque, so it holds whatever pose it is in: support it first.
3. Connect each arm (`--arms`), print the estimate (predicted finish, what limits it).
4. REAL only: wait for the typed word `go` (or the Go button with `--view`).
5. Run. Ctrl-C does NOT kill anything: it asks every arm for a slow stop (finish the step, lower,
   torque off). A second Ctrl-C is ignored with a reminder. When the batch ends the arms are parked.

Exit code 0 only if every step ended `ok`.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path

from hfteama.arms.base import ArmState
from hfteama.arms.factory import real_worker_factory, sim_worker_factory
from hfteama.arms.poses import Poses
from hfteama.arms.sim import SimTable
from hfteama.arms.worker import SkillWorker
from hfteama.clock import Clock
from hfteama.discovery import ArmWatcher, followers_from_pairs
from hfteama.events import EventLog
from hfteama.executor import Executor
from hfteama.taskgraph import load

ROOT = Path(__file__).resolve().parents[2]
WARNING = """\
This will CONNECT the arms and then MOVE THEM.
  - Connecting enables torque: each arm holds the pose it is in. Support any raised arm first.
  - Clear the table of hands and objects that are not part of the batch.
  - Keep a hand on the power. Ctrl-C asks for a slow stop (lower, torque off): do not kill it.
"""


class _RealRig:
    """Controls for the live screen: its Go button starts the executor."""

    def __init__(self, executor: Executor) -> None:
        self.ex = executor
        self._started = False

    def go(self) -> None:
        if not self._started:
            self._started = True
            self.ex.start()

    def started(self) -> bool:
        return self._started


def default_poses(task_dir: Path, real: bool) -> str:
    """poses_real.json for a real run when the task has one, else poses.json."""
    real_file = task_dir / "poses_real.json"
    return str(real_file if real and real_file.exists() else task_dir / "poses.json")


def _summary(log: EventLog, graph_size: int) -> tuple[int, list[str]]:
    ends = [e for e in log.read() if e.kind == "task_end"]
    ok = {e.task for e in ends if e.outcome == "ok"}
    bad = sorted({e.task for e in ends if e.task and e.task not in ok})
    return len(ok), bad


def main(argv: list[str] | None = None, ask: Callable[[str], str] = input) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--backend",
        choices=("real", "sim", "sorter"),
        default="real",
        help="sorter: real arms through hfteama.sorter (calibrated IK, per-arm aim corrections)",
    )
    ap.add_argument("--task", default=str(ROOT / "tasks" / "sort" / "sort12.json"))
    ap.add_argument(
        "--poses", default=None, help="default: poses_real.json (real) / poses.json (sim)"
    )
    ap.add_argument("--config", default=str(ROOT / "configs" / "real_arms.json"))
    ap.add_argument("--pairs", default=str(ROOT / "configs" / "teleop_pairs.json"))
    ap.add_argument("--arms", default="", help="comma-separated arm ids to connect (catalog ids)")
    ap.add_argument("--watch", action="store_true", help="plug arms in as their USB serial appears")
    ap.add_argument("--method", default="heft", choices=("greedy", "heft", "cpsat"))
    ap.add_argument("--log", default=None, help="event log (default runs/<time>/events.jsonl)")
    ap.add_argument(
        "--view", action="store_true", help="serve the live screen; its Go button starts"
    )
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument(
        "--timeout-factor",
        type=float,
        default=2.5,
        help="a step is cancelled after this x its arm's measured 80th-percentile time "
        "(Operations' open decision: 1.5 cancels the occasional slow step)",
    )
    ap.add_argument("--yes", action="store_true", help="sim only: skip the typed confirmations")
    ap.add_argument(
        "--inject",
        action="append",
        default=[],
        metavar="ARM:KIND",
        help="sim only: inject a failure (grasp | down | slow) to watch the executor recover",
    )
    args = ap.parse_args(argv)

    real = args.backend in ("real", "sorter")
    if real and args.yes:
        ap.error("--yes is for the simulator only: real arms always need the typed confirmations")
    if real and args.inject:
        ap.error("--inject is for the simulator only")
    task_dir = Path(args.task).parent
    poses_path = args.poses or default_poses(task_dir, real)
    graph, poses = load(args.task), Poses.load(poses_path)
    arm_ids = [a for a in args.arms.split(",") if a]
    if not arm_ids and not args.watch:
        ap.error("give --arms (ids) or --watch")

    log_path = Path(args.log or ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S") / "events.jsonl")
    log_path.unlink(missing_ok=True)  # a log is one run: never mix this run's steps with an old one
    log = EventLog(log_path)
    clock = Clock()  # real arms always run on the scale-1 clock (heartbeats are wall times)
    ex = Executor(
        graph,
        log,
        clock=clock,
        method=args.method,
        catalog=arm_ids,
        timeout_measured=args.timeout_factor,
    )
    stop_requested = threading.Event()
    sigints = 0

    def on_sigint(signum: int, frame: object) -> None:
        nonlocal sigints
        sigints += 1
        if sigints == 1:
            print("\nslow stop requested: finishing the current steps, then parking every arm...")
            stop_requested.set()
        else:
            print("\nalready stopping: do not kill the process while arms move; wait for the park.")

    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, on_sigint)

    table: SimTable | None = None
    watcher: ArmWatcher | None = None
    made: dict[str, SkillWorker] = {}  # every worker we built, to wait for its park
    try:
        if real:
            print(WARNING)
            if ask("Type 'yes' to connect the arms: ").strip() != "yes":
                print("not confirmed: nothing was connected.")
                return 2
            if args.backend == "sorter":
                from hfteama.sorter.bridge import sorter_worker_factory

                factory = sorter_worker_factory(poses, pairs_config=args.pairs)
            else:
                factory = real_worker_factory(args.config, poses)
        else:
            table = SimTable(poses, arm_ids=arm_ids or None, speed=1.0).__enter__()
            factory = sim_worker_factory(table, poses)
        known = followers_from_pairs(args.pairs) if args.watch else {}

        def tracked(arm_id: str) -> SkillWorker:
            made[arm_id] = factory(arm_id)
            return made[arm_id]

        watcher = ArmWatcher(ex, known=known, factory=tracked)
        for arm_id in arm_ids:
            if not watcher.plug(arm_id):
                print(f"could not start {arm_id}: aborting (nothing has moved).")
                _park_all(ex, watcher, made)
                return 3
        for spec in args.inject:
            arm_id, kind = spec.split(":")
            table.inject(arm_id, kind)  # type: ignore[arg-type,union-attr]
        if args.watch:
            watcher.start()

        plan = ex.estimate()
        print(f"estimate: finish in {plan.makespan:.0f} s; limit: {plan.binding or 'n/a'}")
        rig = _RealRig(ex)
        if args.view:
            _serve(log, rig, clock, args.port)
            print(f"live screen: http://127.0.0.1:{args.port}  (press Go there to start)")
        elif real and ask("Type 'go' to start the batch: ").strip() != "go":
            print("not started: parking the arms.")
            _park_all(ex, watcher, made)
            return 2
        else:
            rig.go()

        while not ex.wait(timeout=0.5):
            if stop_requested.is_set():
                break
        if stop_requested.is_set():
            _park_all(ex, watcher, made)
            return 130
        _park_all(ex, watcher, made)
    finally:
        if watcher is not None:
            watcher.stop()
        ex.stop()
        if table is not None:
            table.__exit__(None, None, None)
        if previous is not None:
            signal.signal(signal.SIGINT, previous)

    n_ok, bad = _summary(log, len(graph.tasks))
    print(f"{n_ok}/{len(graph.tasks)} steps ok. events: {log_path}")
    if bad or n_ok < len(graph.tasks):
        print("not ok:", ", ".join(bad) or "unfinished steps")
        return 1
    return 0


def _park_all(
    ex: Executor, watcher: ArmWatcher, made: dict[str, SkillWorker], wait_s: float = 180.0
) -> None:
    """Slow stop for every connected arm: finish the step, lower, torque off. Waits until done.

    The executor hands each arm to a background park thread and forgets it at once, so waiting on
    the executor is not enough: wait for every worker to report DOWN (parked, or lost).
    """
    for arm_id in sorted(watcher.joined()):
        ex.remove_arm(arm_id)
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        if all(w.state() == ArmState.DOWN for w in made.values()):
            return
        time.sleep(0.2)
    print(
        "warning: some arms did not finish parking:",
        [a for a, w in made.items() if w.state() != ArmState.DOWN],
    )


def _serve(log: EventLog, rig: _RealRig, clock: Clock, port: int) -> None:
    from hfteama.view.app import create_app

    app = create_app(log, controls=rig, clock=clock)
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False), daemon=True
    ).start()


if __name__ == "__main__":
    raise SystemExit(main())
