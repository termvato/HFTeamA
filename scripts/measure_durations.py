"""Measure how long pick and place take on each arm: p50 / p80 per (skill, arm).

Feeds Operations' duration estimates (#13): they plan with the 80th percentile. Times are in
simulated seconds (the sim's clock), so they are comparable to a real arm's wall-clock durations.

    python scripts/measure_durations.py --cycles 5 --out runs/durations/sort_sim.json
    python scripts/measure_durations.py --backend real --arms arm_a   # MOVES A REAL ARM, see below

`--backend real` drives real arms from `configs/real_arms.json` and is only for the staged
Hardware checklist (`docs/handoff/action-to-hardware.md`), after steps 1-4 passed. It asks for a
typed confirmation, then asks you to place each block at its marked spot before every cycle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hfteama.arms import generic_skills  # noqa: F401  (registers the skills)
from hfteama.arms.base import Outcome
from hfteama.arms.poses import Poses
from hfteama.arms.sim import SimTable
from hfteama.arms.worker import SkillWorker
from hfteama.taskgraph import Task

ROOT = Path(__file__).resolve().parent.parent


def _steps(obj: str) -> tuple[tuple[str, dict], tuple[str, dict]]:
    return (
        ("pick", {"object": obj, "from": "input_bin"}),
        ("place", {"object": obj, "to": "own_output_bin"}),
    )


def _sample_sim(args: argparse.Namespace, poses: Poses) -> dict[str, dict[str, list[float]]] | None:
    objects = list(poses.objects)
    samples: dict[str, dict[str, list[float]]] = {"pick": {}, "place": {}}
    with SimTable(poses, speed=args.speed) as table:
        for arm in table.arm_ids:
            worker = SkillWorker(arm, table.driver(arm), poses, "sim", clock=table.sim_time)
            for c in range(args.cycles):
                table.reset()
                for skill, step_args in _steps(objects[c % len(objects)]):
                    res = worker.execute(Task(id=f"{arm}_{skill}_{c}", skill=skill, args=step_args))
                    if res.outcome != Outcome.OK:
                        print(f"{arm} {skill} cycle {c}: {res.outcome.value} {res.reason}")
                        return None
                    samples[skill].setdefault(arm, []).append(res.duration)
    return samples


def _sample_real(
    args: argparse.Namespace, poses: Poses
) -> dict[str, dict[str, list[float]]] | None:
    from hfteama.arms.real import from_config

    arms = [a for a in args.arms.split(",") if a]
    print("This MOVES REAL ARMS:", ", ".join(arms))
    print(
        "Run only after the read-only and single-joint checks passed; keep a hand near the power."
    )
    if input("Type 'yes' to continue: ").strip() != "yes":
        return None
    objects = list(poses.objects)
    samples: dict[str, dict[str, list[float]]] = {"pick": {}, "place": {}}
    for arm in arms:
        worker = SkillWorker(arm, from_config(arm, args.real_config), poses, "real", time.time)
        for c in range(args.cycles):
            obj = objects[c % len(objects)]
            input(f"{arm}: put {obj} at {poses.objects[obj]} and press Enter ")
            for skill, step_args in _steps(obj):
                res = worker.execute(Task(id=f"{arm}_{skill}_{c}", skill=skill, args=step_args))
                if res.outcome != Outcome.OK:
                    print(f"{arm} {skill} cycle {c}: {res.outcome.value} {res.reason}")
                    worker.park()
                    return None
                samples[skill].setdefault(arm, []).append(res.duration)
        worker.park()
    return samples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", choices=("sim", "real"), default="sim")
    ap.add_argument("--arms", default="", help="real backend: comma-separated arm ids")
    ap.add_argument("--real-config", default=str(ROOT / "configs" / "real_arms.json"))
    ap.add_argument("--poses", default=str(ROOT / "tasks" / "sort" / "poses.json"))
    ap.add_argument("--cycles", type=int, default=5, help="pick+place cycles per arm")
    ap.add_argument("--speed", type=float, default=4.0, help="sim speed; durations are sim seconds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.backend == "real" and not args.arms:
        ap.error("--backend real needs --arms")
    out = Path(args.out or ROOT / "runs" / "durations" / f"sort_{args.backend}.json")

    poses = Poses.load(args.poses)
    samples = (_sample_sim if args.backend == "sim" else _sample_real)(args, poses)
    if samples is None:
        return 1

    skills = {
        skill: {
            arm: {
                "n": len(ds),
                "p50": round(float(np.percentile(ds, 50)), 3),
                "p80": round(float(np.percentile(ds, 80)), 3),
            }
            for arm, ds in by_arm.items()
        }
        for skill, by_arm in samples.items()
    }
    unit = "sim seconds" if args.backend == "sim" else "seconds"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": args.backend, "unit": unit, "skills": skills}, indent=1))
    print(f"{'skill':<8}{'arm':<8}{'n':>3}{'p50 s':>9}{'p80 s':>9}")
    for skill, by_arm in skills.items():
        for arm, s in by_arm.items():
            print(f"{skill:<8}{arm:<8}{s['n']:>3}{s['p50']:>9.2f}{s['p80']:>9.2f}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
