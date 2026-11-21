"""Planner benchmark (backlog #7): solve time and finish time for greedy, HEFT and CP-SAT.

Sorting batches with one shared input bin (pick 4 s in the bin, place 6 s), at 10, 30 and 50 steps
and 2, 4 and 8 arms. Run: python scripts/bench_planners.py [--limit 5]
"""

from __future__ import annotations

import argparse
import time

from hfteama.scheduler.problem import Problem
from hfteama.scheduler.solve import solve
from hfteama.taskgraph import Task, TaskGraph


def sorting(steps: int) -> TaskGraph:
    tasks = {}
    for i in range(steps // 2):
        p, q = f"pick_{i}", f"place_{i}"
        tasks[p] = Task(p, "pick", resources=("input_bin",), duration=4.0)
        tasks[q] = Task(q, "place", deps=(p,), same_arm_as=p, duration=6.0)
    return TaskGraph(f"sort{steps}", tasks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=5.0, help="CP-SAT time limit (s)")
    args = ap.parse_args()
    print("| steps | arms | greedy | HEFT | CP-SAT | CP-SAT gap | CP-SAT time | bound |")
    print("|---|---|---|---|---|---|---|---|")
    for steps in (10, 30, 50):
        g = sorting(steps)
        for n in (2, 4, 8):
            p = Problem(g, [f"arm_{i}" for i in range(n)])
            gr = solve(p, "greedy")
            he = solve(p, "heft")
            t = time.perf_counter()
            cp = solve(p, "cpsat", time_limit=args.limit)
            dt = time.perf_counter() - t
            gap = cp.info.get("gap", float("nan"))
            print(
                f"| {steps} | {n} | {gr.makespan:.1f} | {he.makespan:.1f} | {cp.makespan:.1f} "
                f"| {gap:.1%} | {dt:.2f} s | {he.lower_bound:.1f} ({he.binding}) |"
            )


if __name__ == "__main__":
    main()
