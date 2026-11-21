# Plan: plug and produce

Finish a batch of work faster by plugging in more SO-101 arms. When an arm is added, removed or
fails, the system re-splits the remaining work live and shows the new predicted finish time and
what limits further speed-up (arm count, step order, or a shared zone).

Works for any task that splits into steps (see `tasks/README.md`); the demo picks one.
Background and sources: `research/`. Visual map: `docs/README.md` (diagram) and `docs/map.html`. Umbrella issue: #1. Progress: [`STATUS.md`](../STATUS.md) (everyone updates their own block).

## Teams

Each team owns a problem. How to solve it, and how to split it inside the team, is the team's call.

| Team | People | Problem | Issue |
|---|---|---|---|
| Hardware | Diego, Om | Three real arms on one table, safe to command, detected when plugged in or out | #19 |
| Action | Vivek, Sahil | Real and simulated arms carry out the steps of a task and report how long they take | #20 |
| Operations | Vato, Aravind | Decide who does what, run it, re-split live, show finish time and bottleneck | #21 (backlog #2 to #18) |

## Branches

One branch per team. Work on your team's branch. `main` is kept in sync automatically: each team
branch has a "Sync <branch> into main" PR that is merged when CI is green (a red CI keeps it open).

| Branch | Team | Merged in so far |
|---|---|---|
| `orchestrator` | Operations (Vato, Aravind) | `termvato`, `aravinds1610` |
| `hardware` | Hardware (Diego, Om) | `lemillion789` (empty) |
| `action` | Action (Vivek, Sahil) | `VivekSai07`, `sahilrajpurkar03` |

The per-person branches are kept for now; new work goes on the team branch:
`git fetch && git switch <team> && git pull`.

## Architecture (locked Sat 21 Nov, Vato and Aravind)

- **VLM front end** (camera + Go): decides the categories, writes the task graph in the format on
  `main`, checks each step from the camera, re-scans when the scene changes.
- **Reachability graph**: which arm can reach which object and bin, from each arm's base pose and reach;
  fills `allowed_arms`, adds handovers when no single arm reaches both ends.
- **Planner**: a full schedule on every re-plan (needed for the countdown and the what-if estimate);
  greedy as the never-fails fallback; CP-SAT when it fits the time limit.
- **Re-plan rule**: running steps never move; switch plans only if over 10% better, unless arms changed
  or something failed.
- **Arm contract**: one worker per arm, `execute(task) -> Result`. The worker reads only the task's id,
  skill and args (object id, target bin, table coordinates). Outcomes `ok`, `failed` + reason,
  `timeout`, `arm_down`, with start and end times. A dispatcher-level `execute(arm_id, ...)` may wrap it.
- **Screen**: live countdown, what-if estimate, the binding limit in words, plus steps unblocked by a new arm.
- **Code** in `src/hfteama/`. Task format, arm contract and event log change only with all teams.
- **Step timeout**: 3x the estimate while durations are guesses, then 1.5x the arm's measured
  80th-percentile time.

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

## Product (demo): a visual sorting system

Decided 21 Nov. The demo task is sorting; the core stays task-agnostic.

| Feature | What the user sees |
|---|---|
| Go button | Objects on the table, press Go. No other input needed. |
| VLM as the brain | A camera VLM looks at the table, decides the categories itself, and assigns each object to a bin. |
| Estimate before Go | Approximate time for the arms plugged in now, plus a what-if row for 1 / 2 / 3 / 4 arms. |
| Live countdown | Predicted finish time, updated whenever an arm joins, leaves or a step fails. |
| Why it stops speeding up | The binding limit in words (arms, step order, or a shared area), plus steps that only became possible because a new arm can reach them. |
| Failures | Says what happened ("grasp failed, retrying", "arm lost, moving its work", "too slow") and recovers; a camera check confirms every place step (decided 21 Nov: one GPU, ~2.8 s per check; a failed pick shows up as a grasp failure). |
| Scene changes | New or moved objects mid-run: the camera notices, the VLM re-reads the table, the plan and countdown update. |
| Simulated arms | Only in the what-if estimate; they do not execute in the live run. |

## Pitch (Hugging Face track: "Desk Hero")

Track brief: solve a simple everyday problem on your desk using LeRobot SO-101.

- **The desk problem:** a messy desk. Point a camera at it and the arms tidy it: the VLM names what
  is there and decides where each thing goes. No training per object; a judge can add their own item.
- **The moment:** in a hurry? Plug in another arm. The work is re-split in seconds, the predicted
  time drops, and the screen says what limits further speed-up.
- **Beyond the desk:** the same system fits small-batch work in industry (sorting, kitting,
  packing): add or remove arms without re-programming, and know the finish time. Say this last, in
  one sentence; the desk is the story.
- **Words:** say "tidy the desk" and "plug in another arm" in the pitch. Keep "plug and produce",
  scheduling terms and the plan chart for questions.
- **Demo objects:** real desk things (pens, tape, lip balm, a snack bar) into a bowl or tray.

## Demo (filmed on the real setup)

1. **The VLM splits the task.** The top camera sees the table; the VLM names the objects and
   decides where each goes. The screen shows a top view: every object in the colour of the arm
   that will take it, each arm's reach, and the estimated time.
2. **One real arm is sorting**; the countdown runs.
3. **Someone plugs in the second arm.**
4. **The system re-plans, visibly:** "Arm joined: re-planning…", then the plans it compared (reach,
   free arms, shared areas) and the time before and after. Narration: once the plan settles on the
   best split for reach and availability, the arms move.
5. **The predicted finish time comes down** and **both arms work together** until the batch is done.

For the time to come down in step 5, the first arm must be able to reach every object; objects only
the second arm can reach add work when it joins, and the time goes up instead.

Try it without hardware: `python -m hfteama.demo --backend fake --table configs/hacklab_table.json
--task tasks/sort/hacklab6.json --plugged rakesh_follower`.

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
| `src/hfteama/registry/` | arm catalog and reach: which arm can reach which step |
| `src/hfteama/perception/` | camera calibration, VLM object positions, sorting plan, step check, re-scan |
| `src/hfteama/discovery.py` | plug-in detection: arms join and leave by USB serial |
| `src/hfteama/demo.py` | the demo runner: `python -m hfteama.demo` |
| `src/hfteama/events.py` | event log |
| `src/hfteama/view/` | live screen |
| `tasks/` | one folder per task |
| `configs/` | pairs and calibrations (Hardware), table layout, arm positions (`real_arms.json`, Action's format) |
| `scripts/` | benchmarks and tools |
| `research/` | background report and notes |

## State (Sat 21 Nov, 18:05)

| Piece | State |
|---|---|
| Planner, executor with live re-split, live screen, plug-in detection, demo runner | done, tested on fake arms |
| Camera to plan (calibration, VLM positions, sorting, reach) | done as a dry run on the real top camera |
| Scaling benchmark | done: about 2.2x with 3 arms, then a shared area or step order limits it |
| Real-arm skills and driver | written and tested in MuJoCo (branch `action-arm-skills`); not yet run on a real arm |
| Real run (camera + VLM + executor + real arms) | waiting for the first real arm motion |

## Run

```bash
pip install -e ".[dev,sched,view,arms]"   # add sim for MuJoCo, vlm and camera for the VLM
pytest
python -m hfteama.demo --backend fake --plugged rakesh_follower   # then http://127.0.0.1:8050
```
