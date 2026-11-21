# Plan: plug and produce

Finish a batch of work faster by plugging in more SO-101 arms. When an arm is added, removed or
fails, the orchestrator re-splits the remaining work live and shows the new predicted finish time
and what limits further speed-up (arm count, step order, or a shared zone).

Works for any task that splits into steps (see `tasks/README.md`); the demo picks one or two.
Background and sources: `research/`. Checklist: GitHub issue #1.

## Layout and owners

| Path | Part | Owner |
|---|---|---|
| `src/hfteama/taskgraph/` | task graph JSON, loader, validator (interface 1, frozen) | P |
| `src/hfteama/scheduler/` | HEFT, CP-SAT, plan checker, lower bound | P |
| `src/hfteama/executor/` | dispatch, zone locks, re-plan on add / remove / fail | E |
| `src/hfteama/arms/` | arm contract (interface 2, frozen), fake / MuJoCo / real backends, skills, discovery | H1, H2, S |
| `src/hfteama/events.py` | event log (interface 3, frozen) | E |
| `src/hfteama/view/` | live Gantt, finish time, limit, scaling plot | S |
| `tasks/` | one folder per task: graph, poses, extra skills | whoever builds that task |
| `configs/` | arm registry (copy `arms.example.json` to `arms.json`, git-ignored) | H1 |
| `scripts/` | benchmarks and one-off tools | anyone |

Roles: P planner, E executor and glue, H1/H2 hardware, S sim and view. Names: to be agreed.

## Phases and gates

1. **Interfaces (H0-H2).** Task graph, arm contract, event log. Gate: frozen (done in this commit;
   change only with team agreement).
2. **Software loop, no hardware (H2-H10).** Fake arms, HEFT, plan checker, executor, Gantt.
   Gate: "add arm" / "remove arm" in the UI re-splits work and the finish time moves.
3. **CP-SAT and MuJoCo arms (H10-H20).** Gate: CP-SAT under 5 s at demo size, else HEFT live.
4. **Real arms, one at a time (from H2, in parallel).** Registry, plug detection, skills, grasp
   check, zone locks before motion. Gate: plugging arm 2 mid-run triggers a re-plan.
5. **Demo (H30+).** Scaling sweep 1-8 sim arms with real 1-3 points; one arm vs HEFT vs CP-SAT,
   3 trials each; backup video; code freeze H40.

## Later extensions (not in the core build)

- Two-hand steps unlocked by a second arm: a `needs_arms: 2` step (hold and insert, handoff) stays
  blocked with one arm; plugging in arm 2 makes it runnable, so a new arm adds capability as well
  as speed.

## Real-arm safety

- Before any motion: read pose, temperatures, torque; target inside joint limits and above the table.
- Shared zones are locks in the scheduler and are checked again right before motion.
- Removing an arm is a slow stop: "remove arm N" -> finish or abort skill -> park -> then unplug.
  No torque cut with the arm raised; no kill signals to a process driving an arm.
- New motion types: tell the team before the first real run.

## Run

```bash
pip install -e ".[dev,sched,view]"   # add sim for MuJoCo; LeRobot from source for real arms
pytest
```
