"""Run the live screen on fake arms (development) or, later, with real arms added by discovery.

    python -m hfteama.view --task tasks/sort/sort12.json --fake-arms 1
    python -m hfteama.view --task tasks/sort/sort12.json --fake-arms 0   # wait for arms

Then open http://127.0.0.1:8050. The screen reads only the event log (runs/<time>/events.jsonl).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hfteama.arms.fake import FakeArm
from hfteama.clock import Clock
from hfteama.events import EventLog
from hfteama.executor import Executor
from hfteama.taskgraph import load


class FakeRig:
    """Go plus test controls for fake arms (not part of the live product)."""

    def __init__(self, executor: Executor, clock: Clock, noise: float = 0.1) -> None:
        self.ex = executor
        self.clock = clock
        self.noise = noise
        self.arms: dict[str, FakeArm] = {}
        self._n = 0
        self._started = False

    def go(self) -> None:
        if not self._started:
            self._started = True
            self.ex.start()

    def started(self) -> bool:
        return self._started

    def add(self) -> str:
        self._n += 1
        arm_id = f"fake-{self._n}"
        arm = FakeArm(
            arm_id,
            clock=self.clock,
            duration_fn=lambda t, a=arm_id: t.duration_on(a),
            noise=self.noise,
            seed=self._n,
        )
        self.arms[arm_id] = arm
        self.ex.add_arm(arm)
        return arm_id

    def remove(self, arm_id: str) -> None:
        self.ex.remove_arm(arm_id)

    def fail(self, arm_id: str) -> None:
        if arm_id in self.arms:
            self.arms[arm_id].fail_next("grasp")

    def down(self, arm_id: str) -> None:
        if arm_id in self.arms:
            self.arms[arm_id].go_down()

    def arm_ids(self) -> list[str]:
        return sorted(self.ex.snapshot()["arms"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m hfteama.view", description=__doc__)
    ap.add_argument("--task", default="tasks/sort/sort12.json")
    ap.add_argument("--fake-arms", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="clock speed-up for fake arms")
    ap.add_argument("--method", default="heft", choices=["greedy", "heft", "cpsat"])
    ap.add_argument("--log", default=None, help="event log path (default runs/<time>/events.jsonl)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args(argv)

    from hfteama.view.app import create_app

    path = Path(args.log or f"runs/{time.strftime('%Y%m%d-%H%M%S')}/events.jsonl")
    clock = Clock(args.scale)
    log = EventLog(path)
    ex = Executor(load(args.task), log, clock=clock, method=args.method)
    rig = FakeRig(ex, clock)
    for _ in range(args.fake_arms):
        rig.add()
    if args.fake_arms == 0:
        ex.estimate()
    print(f"event log: {path}")
    create_app(log, controls=rig, clock=clock).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
