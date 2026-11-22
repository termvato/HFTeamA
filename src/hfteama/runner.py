"""Run a task graph on real (or simulated) SO-101 arms through Operations' executor.

    python -m hfteama.runner --backend real --arms rakesh_follower,vato_follower \
        --task tasks/sort/sort12.json --poses tasks/sort/poses_real.json
    python -m hfteama.runner --backend sim --arms arm_a,arm_b --yes     # same wiring, no hardware

The flow, and where a human stands between the code and the arms:
1. Load the task, the poses and the arm config; build the executor and the planner catalog.
2. REAL only: print what is about to happen and wait for the typed word `yes`. Connecting an arm
   enables its torque, so it holds whatever pose it is in: support it first.
3. Connect each arm (`--arms`), print the estimate (predicted finish, what limits it).
4. REAL only: wait for the typed word `go` (or the dashboard's Go button with `--view`).
   Real arms: Qwen checks each place from the overhead camera (`--no-vlm` to skip it).
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
from hfteama.discovery import ArmWatcher, connected_serials, followers_from_pairs
from hfteama.events import EventLog
from hfteama.executor import PLACE_SKILLS, Executor
from hfteama.taskgraph import load

ROOT = Path(__file__).resolve().parents[2]
WARNING = """\
This will CONNECT the arms and then MOVE THEM.
  - Connecting enables torque: each arm holds the pose it is in. Support any raised arm first.
  - Clear the table of hands and objects that are not part of the batch.
  - Keep a hand on the power. Ctrl-C asks for a slow stop (lower, torque off): do not kill it.
"""


class _RealRig:
    """Controls for the dashboard (`--view`, served by hfteama.orchestra): Go starts the
    executor; Remove / Stop all is a slow stop per arm, and when no arm is left in the run the
    runner parks everything and exits, exactly as one Ctrl-C does."""

    def __init__(
        self,
        executor: Executor,
        watcher: ArmWatcher | None = None,
        workers: dict | None = None,
        known: list[str] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self.ex = executor
        self.watcher = watcher
        self.workers = workers if workers is not None else {}
        self.known = list(known or [])
        self.on_stop = on_stop
        self._started = False

    def go(self) -> None:
        if not self._started:
            self._started = True
            self.ex.start()

    def started(self) -> bool:
        return self._started

    def plug(self, arm_id: str) -> bool:
        return self.watcher.plug(arm_id) if self.watcher is not None else False

    def remove(self, arm_id: str, abort: bool = False) -> None:
        self.ex.remove_arm(arm_id, abort=abort)
        if self.watcher is not None:
            self.watcher.forget(arm_id)
            left = self.watcher.joined()
        else:
            left = set(self.ex.snapshot()["arms"]) - {arm_id}
        if not left and self.on_stop is not None:
            self.on_stop()

    def down(self, arm_id: str) -> None:  # test-only control: never offered on real arms
        raise RuntimeError("not offered on real arms")

    def fail(self, arm_id: str) -> None:  # fake arms only
        raise RuntimeError("only fake arms can be told to fail")


class _Overhead:
    """The overhead camera, shared by the Qwen place check and the re-scan: opened on first use
    (one process may hold it; on Windows a second open fails), one frame at a time."""

    def __init__(self, root: Path) -> None:
        import json

        self.cfg = json.loads((root / "configs" / "sorter.json").read_text())
        self._cam = None
        self._lock = threading.Lock()

    def grab(self):
        from hfteama.vision.camera import OverheadCamera

        with self._lock:
            if self._cam is None:
                self._cam = OverheadCamera(self.cfg["camera"]).open()
            return self._cam.frame()

    def close(self) -> None:
        with self._lock:
            if self._cam is not None:
                self._cam.close()
                self._cam = None


class _ProcessProbe:
    """Is this Python process keeping up? (measurement only, phase improvement plan E)

    A thread asks to sleep 20 ms, like an arm's 50 Hz motion loop, and records how late it
    wakes: lateness here with no bus involved means Python itself was busy (another thread held
    the interpreter). Every `period_s` it appends one JSON line to `path`: process CPU (1.0 =
    one core), thread count, mean and worst lateness, and the slowest USB scan of the plug-in
    watcher. Read next to `motion.jsonl` (each move's bus times) to tell the two causes apart.
    """

    def __init__(self, path: Path, period_s: float = 5.0, tick_s: float = 0.02) -> None:
        self.path, self.period_s, self.tick_s = path, period_s, tick_s
        self._scan_max = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def timed_scan(self) -> set[str]:
        t = time.perf_counter()
        try:
            return connected_serials()
        finally:
            self._scan_max = max(self._scan_max, time.perf_counter() - t)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="process-probe", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import json

        wall0, cpu0 = time.perf_counter(), time.process_time()
        late_sum, late_max, n = 0.0, 0.0, 0
        while not self._stop.is_set():
            t = time.perf_counter()
            time.sleep(self.tick_s)
            late = time.perf_counter() - t - self.tick_s
            late_sum, late_max, n = late_sum + late, max(late_max, late), n + 1
            wall = time.perf_counter() - wall0
            if wall < self.period_s:
                continue
            rec = {
                "t": round(time.time(), 3),
                "cpu": round((time.process_time() - cpu0) / wall, 2),
                "threads": threading.active_count(),
                "late_mean_ms": round(1000 * late_sum / n, 2),
                "late_max_ms": round(1000 * late_max, 1),
                "scan_max_ms": round(1000 * self._scan_max, 1),
            }
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass
            wall0, cpu0 = time.perf_counter(), time.process_time()
            late_sum, late_max, n, self._scan_max = 0.0, 0.0, 0, 0.0


WARM_SKILLS = ("pick", "place", "pick_place")


def _warm_ik(graph, workers: dict[str, SkillWorker], poses: Poses, stop: threading.Event) -> None:
    """Before Go: solve each arm's IK for every step it may take, so its first moves after Go
    start at once (22 Nov 12:17: 20 s and 36 s of IK before the arms' first moves, then 35 s
    before a later one; the planner itself takes ~1 ms).

    It makes the reach checks a skill makes before it moves (generic_skills: every waypoint
    through `ArmDriver.reachable`), plus the sorter bridge's drop over the bowl. Those fill the
    arm driver's IK cache, and the moves reuse the same entries. Nothing moves and the bus is
    not used. A step whose check fails here fails the same way at run time, so failures are
    only counted. Not covered: the short vertical lines (seeded from the live pose). An arm that
    joins later (USB, --watch) is not warmed."""
    from hfteama.arms import generic_skills as gs

    for arm_id, worker in workers.items():
        drv = getattr(worker, "_driver", None)
        if drv is None or not hasattr(drv, "reachable"):
            continue
        carry = gs._carry_point(poses, arm_id)
        steps = [t for t in graph.tasks.values() if t.skill in WARM_SKILLS and t.can_run_on(arm_id)]
        if not steps:
            continue
        print(f"IK for {arm_id}: solving {len(steps)} steps before Go...", flush=True)
        t0, bad = time.perf_counter(), 0
        for task in steps:
            if stop.is_set():
                print("IK warm-up interrupted.")
                return
            try:
                points, drops = [], []
                if task.skill in ("pick", "pick_place"):
                    p = gs.pick_point(task, poses, arm_id)
                    points += gs._pick_waypoints(p, gs.pick_yaw(task), carry)
                if task.skill in ("place", "pick_place"):
                    drops = gs._place_waypoints(gs.place_point(task, poses, arm_id), carry)
                    points += drops
                if not all(drv.reachable(w, yaw) for w, yaw in points):
                    bad += 1
                # carrying a box, the sorter bridge solves the bowl as a drop, a separate
                # cache entry the reach check does not fill (offline: the first pick's IK
                # 2.8 s -> 0 s with it, hacklab's whole batch 7.0 s -> 2.3 s)
                if hasattr(drv, "_command"):
                    for w, _ in drops:
                        drv._command(w, None, "drop")
            except Exception:  # noqa: BLE001  (the step reports it at run time, as before)
                bad += 1
        took = time.perf_counter() - t0
        if hasattr(drv, "_ik_s"):  # the motion log's per-move IK time: count run time only
            drv._ik_s = 0.0
        note = f" ({bad} out of this arm's reach)" if bad else ""
        print(f"IK for {arm_id}: ready in {took:.0f} s{note}", flush=True)


def _vlm_place_check(cam: _Overhead):
    """Qwen confirms each place from a fresh overhead frame (the architecture's VLM layer).
    Returns (verifier, unload)."""
    from hfteama.perception.boxes import place_checker
    from hfteama.perception.decomposer import Bin
    from hfteama.perception.vlm import QwenVLM

    print("VLM: loading Qwen for the place check (first time ~30 s)...")
    vlm = QwenVLM()
    t0 = time.monotonic()
    vlm._load()  # now, not at the first place: that step would time out waiting for it
    print(f"VLM: Qwen ready ({vlm.loaded_as}, {time.monotonic() - t0:.0f} s)")
    label = cam.cfg.get("box", {}).get("label", "the round bowl in the middle")
    return place_checker(vlm, [Bin("bowl", label, (0, 0, 0))], cam.grab), vlm.unload


# The export through the normal CLI, with every thread's stack written to stderr if it is still
# running after 100 s (the runner gives it 120 s): a stuck re-scan then says where it is stuck.
_EXPORT_CHILD = (
    "import faulthandler, sys; faulthandler.dump_traceback_later(100); "
    "from hfteama.__main__ import main; sys.exit(main(sys.argv[1:]))"
)


def _sorter_rescan(cam: _Overhead, args, log_dir: Path | None = None) -> Callable[[], object]:
    """After a missed grasp: a fresh frame through Hardware's own export (tags, IK reach,
    angles), positions only (no Qwen), as the task graph the executor merges in."""
    import subprocess
    import sys
    import tempfile

    import cv2

    one_at_a_time = threading.Lock()

    def rescan():
        # In a child process, one at a time: the export is ~15 s of pure-Python IK, and in this
        # process it held the GIL the arms' 50 Hz loops need (22 Nov 09:29: a re-scan took 130 s,
        # steps took 43-66 s instead of 10-20, and re-scans piled up after every miss).
        with one_at_a_time:
            out = Path(tempfile.mkdtemp(prefix="rescan_"))
            frame = out / "frame.png"
            cv2.imwrite(str(frame), cam.grab())
            cmd = [
                sys.executable, "-c", _EXPORT_CHILD, "--config", str(args.pairs),
                "sort", "export", "--image", str(frame), "--out", str(out),
                "--place-above", "0.03", "--duration", "30", "--no-vlm",
            ]  # fmt: skip
            print()
            print("re-scan after a missed grasp (child process):")
            t0 = time.monotonic()
            try:
                done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired as e:
                saved = _save_rescan_output(log_dir, "timed out after 120 s", e.stdout, e.stderr)
                raise RuntimeError(f"re-scan export timed out (120 s){saved}") from None
            took = f"exit {done.returncode} after {time.monotonic() - t0:.1f} s"
            _save_rescan_output(log_dir, took, done.stdout, done.stderr)
            if done.returncode != 0 or not (out / "boxes.json").exists():
                tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
                raise RuntimeError(f"re-scan export failed: {tail[0]}")
            return load(str(out / "boxes.json"))

    return rescan


def _save_rescan_output(log_dir: Path | None, how: str, stdout, stderr) -> str:
    """Keep a re-scan's console output in the run folder (it was lost before). Returns a note
    for the error message; never raises."""
    if log_dir is None:
        return ""

    def text(x) -> str:
        return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")

    path = log_dir / f"rescan-{time.strftime('%H%M%S')}.log"
    try:
        path.write_text(
            f"{how}\n--- stdout\n{text(stdout)}\n--- stderr\n{text(stderr)}\n", encoding="utf-8"
        )
    except OSError:
        return ""
    return f"; output in {path}"


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
        "--view",
        "--orchestra",  # the old name: there is one dashboard now
        dest="view",
        action="store_true",
        help="serve the dashboard: its Go starts the run, Stop all is a slow stop",
    )
    ap.add_argument("--port", "--orchestra-port", dest="port", type=int, default=8050)
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="dashboard address: 0.0.0.0 shows it to other laptops on the network (view only; "
        "the buttons need --token, made up and printed if not given)",
    )
    ap.add_argument("--token", default=None, help="operator token for the dashboard's buttons")
    ap.add_argument(
        "--no-rescan",
        action="store_true",
        help="sorter: retry a missed grasp at the old spot instead of re-scanning first",
    )
    ap.add_argument(
        "--no-vlm",
        action="store_true",
        help="real arms: skip Qwen's camera check after each place (on by default)",
    )
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
    close_vlm: Callable[[], None] = lambda: None  # noqa: E731
    close_cam: Callable[[], None] = lambda: None  # noqa: E731
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
    probe = _ProcessProbe(log_path.parent / "process.jsonl")
    made: dict[str, SkillWorker] = {}  # every worker we built, to wait for its park
    try:
        if real:
            print(WARNING)
            if ask("Type 'yes' to connect the arms: ").strip() != "yes":
                print("not confirmed: nothing was connected.")
                return 2
            cam = _Overhead(ROOT)
            close_cam = cam.close
            # the overhead camera belongs to the sorter rig: Qwen and the re-scan need it
            if args.backend == "sorter" and not args.no_vlm:  # once confirmed: Qwen takes ~30 s
                try:
                    ex.verifier, close_vlm = _vlm_place_check(cam)
                    ex.verify_skills = PLACE_SKILLS
                except (Exception, SystemExit) as e:  # noqa: BLE001  (no GPU / model / camera)
                    print(f"VLM: place check off ({e}); the run goes on without it")
            if args.backend == "sorter" and not args.no_rescan:
                ex.rescan = _sorter_rescan(cam, args, log_path.parent)
                print("re-scan: a missed grasp is retried where the camera sees the box now")
            if args.backend == "sorter":
                from hfteama.sorter.bridge import sorter_worker_factory

                factory = sorter_worker_factory(
                    poses, pairs_config=args.pairs, motion_log=log_path.parent / "motion.jsonl"
                )
            else:
                factory = real_worker_factory(args.config, poses)
        else:
            table = SimTable(poses, arm_ids=arm_ids or None, speed=1.0).__enter__()
            factory = sim_worker_factory(table, poses)
        known = followers_from_pairs(args.pairs) if args.watch else {}

        def tracked(arm_id: str) -> SkillWorker:
            made[arm_id] = factory(arm_id)
            return made[arm_id]

        def refused(arm_id: str, e: BaseException) -> None:
            print(f"{arm_id} could not join: {e or type(e).__name__}", flush=True)

        watcher = ArmWatcher(
            ex, known=known, factory=tracked, scan=probe.timed_scan, on_error=refused
        )
        for arm_id in arm_ids:
            if not watcher.plug(arm_id):
                print(f"could not start {arm_id}: aborting (nothing has moved).")
                _park_all(ex, watcher, made)
                return 3
        for spec in args.inject:
            arm_id, kind = spec.split(":")
            table.inject(arm_id, kind)  # type: ignore[arg-type,union-attr]
        probe.start()
        # before the dashboard (and its Go) is up, so no arm moves while its IK is solved
        _warm_ik(graph, dict(made), poses, stop_requested)
        if stop_requested.is_set():
            _park_all(ex, watcher, made)
            return 130
        if args.watch:
            watcher.start()

        plan = ex.estimate()
        print(f"estimate: finish in {plan.makespan:.0f} s; limit: {plan.binding or 'n/a'}")
        rig = _RealRig(ex, watcher, made, arm_ids or list(known.values()), stop_requested.set)
        if args.view:
            _serve(log, ex, watcher, rig, clock, args)
            local = f"http://127.0.0.1:{args.port}/"
            if args.token:
                local += f"?token={args.token}"
            print(f"dashboard: {local}  (Go there starts the run; Stop all = slow stop, then park)")
            if args.host not in ("127.0.0.1", "localhost", "::1"):
                print(
                    f"other laptops (view only, no buttons): http://{_lan_ip()}:{args.port}/  "
                    "(same network; allow Python through the Windows firewall if asked)"
                )
        screen = args.view  # the dashboard's Go button starts the run
        if not screen and real and ask("Type 'go' to start the batch: ").strip() != "go":
            print("not started: parking the arms.")
            _park_all(ex, watcher, made)
            return 2
        if not screen:
            rig.go()

        while not ex.wait(timeout=0.5):
            if stop_requested.is_set():
                break
        if stop_requested.is_set():
            _park_all(ex, watcher, made)
            return 130
        _park_all(ex, watcher, made)
    finally:
        probe.stop()
        if watcher is not None:
            watcher.stop()
        ex.stop()
        if table is not None:
            table.__exit__(None, None, None)
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
        close_vlm()
        close_cam()

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


def _serve(log, ex, watcher, rig, clock, args) -> None:
    from hfteama.orchestra import StateHub
    from hfteama.orchestra.server import start

    hub = StateHub(
        log,
        ex,
        watcher,
        rig,
        clock,
        args.backend,
        args.pairs,
        watching=args.watch,
        task_name=Path(args.task).stem,
    )
    host = getattr(args, "host", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1") and not getattr(args, "token", None):
        import secrets

        # on the network the buttons (Go, Stop all, Plug) need a token; viewing does not
        args.token = secrets.token_urlsafe(8)
    start(hub, rig, host, args.port, getattr(args, "token", None))


def _lan_ip() -> str:
    """This laptop's address on the local network (no packet is sent: UDP connect only)."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


if __name__ == "__main__":
    raise SystemExit(main())
