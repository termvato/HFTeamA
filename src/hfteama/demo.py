"""The demo runner: executor + plug-in detection + live screen, on fake, simulated or real arms.

    python -m hfteama.demo --backend fake --plugged rakesh_follower     # no hardware
    python -m hfteama.demo --backend real                               # arms join when plugged in
    python -m hfteama.demo --backend fake --planner pkg.module:function # batch from a planner hook

Then open http://127.0.0.1:8050 and press Go.

- Which arms exist: the followers in Hardware's `configs/teleop_pairs.json` (USB serial -> arm id).
- Arms join and leave through `hfteama.discovery.ArmWatcher`: by USB serial (`--watch`, default for
  sim and real), or by hand from the screen's control panel (the stage fallback, always there).
- What an arm is (`--backend`): `fake` sleeps for each step's duration; `sim` and `real` are
  Action's `SkillWorker` on its MuJoCo or LeRobot driver (imported only when asked for, so this
  file runs without them). `real` moves real arms: a person must be at the rig.
- `--table configs/hacklab_table.json`: the real table layout. Each step's allowed arms then come
  from reach, and the screen shows a top view of who takes what.
- What the batch is: a task file (`--task`), or a planner hook `module:function` returning a
  `TaskGraph` (the camera + VLM pipeline plugs in here), optionally with `--verifier
  module:function` returning the executor's step-check function.
"""

from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hfteama.arms.fake import FakeArm
from hfteama.clock import Clock
from hfteama.discovery import ArmWatcher, connected_serials, followers_from_pairs
from hfteama.events import Event, EventLog
from hfteama.executor import PLACE_SKILLS, Executor
from hfteama.taskgraph import Task, TaskGraph, load

BACKENDS = ("fake", "sim", "real")


def hook(spec: str) -> Callable[..., Any]:
    """'package.module:function' -> the function."""
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"expected module:function, got {spec!r}")
    return getattr(importlib.import_module(module), name)


def fake_factory(clock: Clock, noise: float = 0.1) -> Callable[[str], Any]:
    def make(arm_id: str) -> FakeArm:
        def seconds(task: Task) -> float:
            return task.duration_on(arm_id)

        return FakeArm(arm_id, clock=clock, duration_fn=seconds, noise=noise)

    return make


def action_factory(backend: str, poses_path: str, arms_config: str) -> Callable[[str], Any]:
    """Action's worker on its sim or real driver. Imported here, so `fake` needs none of it."""
    try:
        from hfteama.arms import generic_skills  # noqa: F401  (registers the skills)
        from hfteama.arms.poses import Poses
        from hfteama.arms.worker import SkillWorker
    except ImportError as e:
        raise SystemExit(
            f"--backend {backend} needs Action's arm code (arms/worker.py, branch "
            f"action-arm-skills), which is not in this checkout: {e}"
        ) from e
    poses = Poses.load(poses_path)
    if backend == "real":
        from hfteama.arms.real import from_config

        def make(arm_id: str) -> Any:
            return SkillWorker(arm_id, from_config(arm_id, arms_config), poses, backend="real")

        return make
    from hfteama.arms.sim import SimTable

    table = SimTable(poses, speed=1.0, viewer=False)
    table.__enter__()  # lives for the whole run

    def make_sim(arm_id: str) -> Any:
        return SkillWorker(arm_id, table.driver(arm_id), poses, backend="sim", clock=table.sim_time)

    return make_sim


def with_table(graph: TaskGraph, table: str, arm_ids: list[str]) -> tuple[TaskGraph, dict]:
    """Fill each step's allowed arms from reach on the real table layout, and describe the table
    for the screen's top view (arms with their reach, bins, objects with their steps)."""
    import json

    from hfteama.registry import load_catalog, resolve

    catalog = {a: n for a, n in load_catalog(table).items() if a in arm_ids}
    resolved = resolve(graph, catalog.values(), travel_s_per_m=10.0).graph
    objects: dict[str, dict] = {}
    for t in resolved.tasks.values():
        oid = t.args.get("object")
        if oid is None:
            continue
        o = objects.setdefault(oid, {"id": oid, "label": t.args.get("label", ""), "tasks": []})
        o["tasks"].append(t.id)
        if "pick_xyz" in t.args:
            o["xy"] = [float(v) for v in t.args["pick_xyz"][:2]]
    scene = {
        "arms": [
            {
                "id": a,
                "xy": list(n.pan_axis_xy),
                "r_min": n.reach.r_min,
                "r_max": n.reach.r_max,
            }
            for a, n in catalog.items()
        ],
        "bins": [
            {"id": b["id"], "xy": b["xyz"][:2], "radius": b.get("radius_m", 0.08)}
            for b in json.loads(Path(table).read_text()).get("bins", [])
        ],
        "objects": [o for o in objects.values() if "xy" in o],
    }
    return resolved, scene


class Rig:
    """Go plus the manual plug / unplug controls the screen shows."""

    def __init__(self, executor: Executor, watcher: ArmWatcher, arm_ids: list[str]) -> None:
        self.ex = executor
        self.watcher = watcher
        self.known = list(arm_ids)
        self.workers: dict[str, Any] = {}
        self._started = False
        inner = watcher.factory

        def remember(arm_id: str) -> Any:
            self.workers[arm_id] = inner(arm_id)
            return self.workers[arm_id]

        watcher.factory = remember

    def go(self) -> None:
        if not self._started:
            self._started = True
            self.ex.start()

    def started(self) -> bool:
        return self._started

    def add(self) -> str:
        """Plug in the next known arm that is not running yet."""
        for arm_id in self.known:
            if arm_id not in self.watcher.joined():
                return arm_id if self.watcher.plug(arm_id) else f"{arm_id} (could not start)"
        return "no arm left"

    def remove(self, arm_id: str) -> None:
        """Slow stop: finish the step, park, then it may be unplugged."""
        self.ex.remove_arm(arm_id)
        self.watcher.forget(arm_id)

    def down(self, arm_id: str) -> None:
        """As if its cable was pulled."""
        worker = self.workers.get(arm_id)
        if hasattr(worker, "go_down"):
            worker.go_down()
        self.watcher.unplug(arm_id)

    def fail(self, arm_id: str) -> None:
        worker = self.workers.get(arm_id)
        if hasattr(worker, "fail_next"):  # only fake arms can be told to fail
            worker.fail_next("grasp")

    def arm_ids(self) -> list[str]:
        return sorted(self.ex.snapshot()["arms"])


def build(
    graph: TaskGraph,
    log: EventLog,
    known: dict[str, str],
    factory: Callable[[str], Any],
    clock: Clock | None = None,
    scan: Callable[[], Any] | None = None,
    verifier: Callable[..., Any] | None = None,
    method: str = "heft",
) -> tuple[Executor, ArmWatcher, Rig]:
    """Wire executor, watcher and controls. `scan=None` means manual plug-in only."""
    arm_ids = list(known.values())
    ex = Executor(
        graph,
        log,
        clock=clock,
        method=method,
        catalog=arm_ids,
        verifier=verifier,
        verify_skills=PLACE_SKILLS if verifier else None,
    )
    watcher = ArmWatcher(ex, known, factory, scan=scan or (lambda: ()))
    return ex, watcher, Rig(ex, watcher, arm_ids)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m hfteama.demo", description=__doc__)
    ap.add_argument("--backend", choices=BACKENDS, default="fake")
    ap.add_argument("--task", default="tasks/sort/sort12.json")
    ap.add_argument("--planner", help="module:function returning a TaskGraph (overrides --task)")
    ap.add_argument("--verifier", help="module:function returning verifier(task, result)")
    ap.add_argument("--pairs", default="configs/teleop_pairs.json")
    ap.add_argument("--arms-config", default="configs/real_arms.json")
    ap.add_argument("--poses", default="tasks/sort/poses.json")
    ap.add_argument("--plugged", default="", help="arm ids to plug in at start (comma separated)")
    ap.add_argument("--watch", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--method", default="auto", choices=["auto", "greedy", "heft", "cpsat"])
    ap.add_argument("--scale", type=float, default=1.0, help="clock speed-up (fake arms only)")
    ap.add_argument(
        "--table", help="table layout JSON (arm poses, bins): fills reach and draws the map"
    )
    ap.add_argument("--log", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args(argv)

    from hfteama.view.app import create_app

    clock = Clock(args.scale if args.backend == "fake" else 1.0)
    known = followers_from_pairs(args.pairs)
    graph = hook(args.planner)() if args.planner else load(args.task)
    verifier = hook(args.verifier)() if args.verifier else None
    scene = None
    if args.table:
        graph, scene = with_table(graph, args.table, list(known.values()))
    if args.backend == "fake":
        factory = fake_factory(clock)
    else:
        factory = action_factory(args.backend, args.poses, args.arms_config)
    watch = args.watch if args.watch is not None else args.backend == "real"
    path = Path(args.log or f"runs/{time.strftime('%Y%m%d-%H%M%S')}/events.jsonl")
    log = EventLog(path)
    ex, watcher, rig = build(
        graph,
        log,
        known,
        factory,
        clock=clock,
        scan=connected_serials if watch else None,
        verifier=verifier,
        method=args.method,
    )
    if scene:
        log.write(Event("scene", data=scene))
    for arm_id in filter(None, args.plugged.split(",")):
        watcher.plug(arm_id.strip())
    if not watcher.joined():
        ex.estimate()
    if watch:
        watcher.start()
    print(f"arms: {', '.join(known.values())} | backend: {args.backend} | USB watch: {watch}")
    print(f"event log: {path}")
    create_app(log, controls=rig, clock=clock).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
