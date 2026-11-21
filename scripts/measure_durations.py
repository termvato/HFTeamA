"""Measure how long pick and place take on each arm: p50 / p80 per (skill, arm).

Feeds Operations' duration estimates (#13): they plan with the 80th percentile. Times are in
simulated seconds (the sim's clock), so they are comparable to a real arm's wall-clock durations.

    python scripts/measure_durations.py --cycles 5 --out runs/durations/sort_sim.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hfteama.arms import generic_skills  # noqa: F401  (registers the skills)
from hfteama.arms.base import Outcome
from hfteama.arms.poses import Poses
from hfteama.arms.sim import SimTable
from hfteama.arms.worker import SkillWorker
from hfteama.taskgraph import Task

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--poses", default=str(ROOT / "tasks" / "sort" / "poses.json"))
    ap.add_argument("--cycles", type=int, default=5, help="pick+place cycles per arm")
    ap.add_argument("--speed", type=float, default=4.0, help="sim speed; durations are sim seconds")
    ap.add_argument("--out", default=str(ROOT / "runs" / "durations" / "sort_sim.json"))
    args = ap.parse_args(argv)

    poses = Poses.load(args.poses)
    objects = list(poses.objects)
    samples: dict[str, dict[str, list[float]]] = {"pick": {}, "place": {}}
    with SimTable(poses, speed=args.speed) as table:
        for arm in table.arm_ids:
            worker = SkillWorker(arm, table.driver(arm), poses, "sim", clock=table.sim_time)
            for c in range(args.cycles):
                table.reset()
                obj = objects[c % len(objects)]
                steps = (
                    ("pick", {"object": obj, "from": "input_bin"}),
                    ("place", {"object": obj, "to": "own_output_bin"}),
                )
                for skill, step_args in steps:
                    res = worker.execute(Task(id=f"{arm}_{skill}_{c}", skill=skill, args=step_args))
                    if res.outcome != Outcome.OK:
                        print(f"{arm} {skill} cycle {c}: {res.outcome.value} {res.reason}")
                        return 1
                    samples[skill].setdefault(arm, []).append(res.duration)

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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"backend": "sim", "unit": "sim seconds", "skills": skills}, indent=1)
    )
    print(f"{'skill':<8}{'arm':<8}{'n':>3}{'p50 s':>9}{'p80 s':>9}")
    for skill, by_arm in skills.items():
        for arm, s in by_arm.items():
            print(f"{skill:<8}{arm:<8}{s['n']:>3}{s['p50']:>9.2f}{s['p80']:>9.2f}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
