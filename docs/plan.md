# Plan: plug and produce

Finish a batch of work faster by plugging in more SO-101 arms. When an arm is added, removed or
fails, the system re-splits the remaining work live and shows the new predicted finish time and
what limits further speed-up (arm count, step order, or a shared zone).

Works for any task that splits into steps (see `tasks/README.md`); the demo picks one.
Background and sources: `research/`. Visual map: `docs/README.md` (diagram) and `docs/map.html`. Umbrella issue: #1.

## Teams

Each team owns a problem. How to solve it, and how to split it inside the team, is the team's call.

| Team | People | Problem | Issue |
|---|---|---|---|
| Hardware | Diego, Om | Three real arms on one table, safe to command, detected when plugged in or out | #19 |
| Action | Vivek, Sahil | Real and simulated arms carry out the steps of a task and report how long they take | #20 |
| Operations | Vato, Aravind | Decide who does what, run it, re-split live, show finish time and bottleneck | #21 (backlog #2 to #18) |

## Branches

One branch per team. Work on your team's branch; open a PR to `main` when CI is green.

| Branch | Team | Merged in so far |
|---|---|---|
| `orchestrator` | Operations (Vato, Aravind) | `termvato`, `aravinds1610` |
| `hardware` | Hardware (Diego, Om) | `lemillion789` (empty) |
| `action` | Action (Vivek, Sahil) | `VivekSai07`, `sahilrajpurkar03` |

The per-person branches are kept for now; new work goes on the team branch:
`git fetch && git switch <team> && git pull`.

## Where teams meet

These are the only things teams must agree on. Change them together, not alone.

| Between | What | Where |
|---|---|---|
| Action -> Operations | each step callable by name, returns ok or the kind of failure, with timing | `src/hfteama/arms/base.py` (#4) |
| Hardware -> Operations | event when an arm becomes available or goes away | `src/hfteama/events.py` (#3, #5) |
| Hardware -> Action | a safe way to move each real arm, all arms in one table frame | agreed between the two teams |
| Operations -> everyone | the task format | `src/hfteama/taskgraph/` (#2) |

## Checkpoints (whole team)

| When | Target |
|---|---|
| Saturday evening | Live screen running on fake arms; one real arm does a step; arms calibrated |
| Sunday midday | Adding an arm (fake or real) re-splits the work on screen |
| Sunday evening | Full demo run: 3 real arms plus sim arms, a failure, the speed-up curve |
| Sunday night | Code freeze, rehearsal, backup video |

## Demo

1. One arm works through a batch; predicted finish time on screen.
2. Plug in a second arm mid-run: it is detected, the work re-splits, the finish time drops.
3. Plug in the third: a smaller drop, and the screen names the bottleneck.
4. Remove or fail an arm: its work flows back to the others.
5. Simulated arms extend the curve past 3 to show where it flattens.

## Later extensions (not in the core build)

- Two-hand steps unlocked by a second arm: a step that needs two arms stays blocked with one arm;
  plugging in arm 2 makes it runnable, so a new arm adds capability as well as speed.

## Real-arm safety (everyone)

- Before any motion: check the arm's state and that the target is inside joint limits and above
  the table.
- Two arms never in the same shared area at once.
- Stopping or removing an arm is a slow stop (freeze, lower, then torque off). No torque cut with
  the arm raised; no kill signals to a process driving an arm.
- New motion types: tell the team before the first real run.

## Repo layout

| Path | What |
|---|---|
| `src/hfteama/taskgraph/` | task format: loader, validator |
| `src/hfteama/scheduler/` | planning, lower bound and bottleneck |
| `src/hfteama/executor/` | running the plan, re-splitting |
| `src/hfteama/arms/` | arm contract, skill registry, arm backends (real, sim, fake) |
| `src/hfteama/events.py` | event log |
| `src/hfteama/view/` | live screen |
| `tasks/` | one folder per task |
| `configs/` | arm registry template (`arms.json` is git-ignored) |
| `scripts/` | benchmarks and tools |
| `research/` | background report and notes |

## Run

```bash
pip install -e ".[dev,sched,view]"   # add sim for MuJoCo; LeRobot from source for real arms
pytest
```
