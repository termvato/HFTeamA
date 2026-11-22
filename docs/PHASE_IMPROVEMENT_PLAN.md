# Phase improvement plan: two SO-101 arms working at the same time

Repo: `HFTeamA` (branch `main`). Read `CLAUDE.md` first: team ownership, frozen interfaces and
the real-arm safety rules all apply to every item below.

**Goal:** when a second arm joins mid-run, the predicted finish time drops visibly and the batch
really finishes sooner, with both arms working at the same time instead of waiting for each
other.

**State on 22 Nov 2026:** both arms pick, place and re-split live. Two problems remain:
- they lose 35 to 50 s per 5-box batch to unbalanced work and turns at the bowl;
- some runs slowed down 3 to 4 times for a reason not yet pinned down.

---

## 1. How a run works today

| Step | Command | What it does |
|---|---|---|
| 1. Look at the table | `hfteama sort export` | Camera: AprilTag boxes → `tasks/sorter_boxes/boxes.json` + `poses_real.json`. Qwen only labels each box with a bin; with one bowl that's always "box → bowl". `--no-vlm` skips Qwen |
| 2. Run the arms | `python -m hfteama.runner --backend sorter --no-vlm --view [--watch] --arms ... --task tasks/sorter_boxes/boxes.json --poses tasks/sorter_boxes/poses_real.json` | Planner (HEFT) + executor + both arms in **one Python process**. Dashboard on http://127.0.0.1:8050 (`--host 0.0.0.0` lets other laptops watch, view only) |
| 3. Join mid-run | `--watch` + plug a follower's USB in | `ArmWatcher` adds the arm; the executor re-plans the remaining steps at once |

How the export writes each box:
- **pick:** locks `half_<arm>`, the half of the table the box is on;
- **place:** locks `bowl`, on the same arm (`same_arm_as`);
- **allowed arms:** only those that reach the box;
- **cross-half penalty:** an arm on the other half is planned `FAR_HALF_FACTOR = 1.5` times slower (`src/hfteama/sorter/run.py`).

The Qwen-in-the-loop path (`python -m hfteama.demo --planner hfteama.perception.live:...`) is
not used for the demo: every run with it failed (see section 3).

## 2. Logs

Every run writes `runs/<date>-<time>/`. The folder is git-ignored and local to the robot laptop.
- **`events.jsonl`:** one JSON line per event:
  - `arm_added`, `arm_removed`, `arm_failed`, `arm_parked`;
  - `plan` and `replan`: predicted finish (`makespan`), the limit (`binding`), the `blocked` steps, the reason;
  - `task_start` and `task_end`: outcome, reason, duration;
  - re-scan events.
- **`commands.jsonl`:** dashboard button presses (Go, Pause, Stop all, Plug).
- **Not saved:** console output (prints, warnings, the export's table).

## 3. What the runs showed (22 Nov)

| Run | Setup | Boxes placed | Typical ok step | Batch | Problem |
|---|---|---|---|---|---|
| 07:38, 07:45 | Qwen live path | 0 | – | – | estimate 4.6 s per step → watchdog (3× estimate) cut every pick at ~14 s. **Fixed** |
| 08:06 | Qwen live path | 0 | – | – | old 30 s step cap. **Fixed by Hardware** (cap 120 s, IK cached) |
| 08:46 | runner, both arms | 4 of 5 | 14 s | 111 s | box ~5 cm from an arm base: "unreachable" after 18 s |
| 08:52 | joint speed 60 °/s | 4 of 5 | 17 s | 164 s | 7 of 11 grasps missed. **Reverted to 45 °/s** |
| 08:57 | runner, both arms | 5 of 5 | 14 s | 120 s | clean |
| 09:03 | runner, both arms | 3 | 13 s | – | rakesh `heartbeat lost` mid-place: bus froze, not parked |
| 09:20 | runner, both arms | 5 of 5 | 14 s | **106 s** | best run |
| **09:25** | **vato alone, rakesh joins at 32 s** | 4 of 5 | 15 s | 124 s | **the join demo worked** |
| 09:29 | vato alone, rakesh joins at 103 s | 0 | 43 s | stopped | everything 3 to 4 times slower; 2 timeouts |
| 09:43 | vato alone, rakesh joins at 120 s | 4 of 5 | 28 s | 401 s | same slowdown; 3 timeouts; both re-scans timed out at 120 s |

### 3.1 The join (09:25)

| Time | Event | Predicted finish |
|---|---|---|
| 0 s | vato alone; rakesh's boxes blocked | 120 s |
| 30 s | first box in the bowl | 110 s |
| 32 s | rakesh's USB in → joins in under 1 s, everything unblocked | 190 s |
| 124 s | finished, as predicted | |

The prediction **rose** at the join because every box was reachable by only one arm: the
newcomer added its own boxes instead of taking a share of the existing work.

### 3.2 Where the time goes with two arms

Picks already run in parallel. Only the bowl makes the arms take turns: a `place` step locks it
for its whole ~15 s (travel, lower, release, travel back).

| Run | Batch | Early-finishing arm idle at the end | Waiting at the bowl holding a box | Bowl locked |
|---|---|---|---|---|
| 08:57 | 120 s | 36 s | 14 s | 66% |
| 09:20 | 106 s | 38 s | 21 s | 71% |
| 09:25 | 124 s | 36 s | 0 s | 52% |

Bounds for 5 boxes (pick ≈ 14 s, place ≈ 15 s):

| | Time |
|---|---|
| One arm alone | about 145 s |
| Two arms, perfect split | about 72 s |
| Bowl alone (5 places × 15 s, one at a time) | 75 s |

**Losses, biggest first:**
1. Unbalanced work: one arm finishes about 36 s early.
2. Turns at the bowl: 0 to 21 s per run.
3. Missed grasps and their retries.

### 3.3 The slowdown (09:29, 09:43)

- **Step times:** 10 to 23 s in all six runs from 08:46 to 09:25; then 43 to 66 s, on both arms, including one arm's very first pick before anything else ran.
- **No code change:** the same code ran at full speed at 09:20 and 09:25.
- **Not the laptop:** i9, 32 GB, on mains, nothing hogging CPU; Windows logged no throttling. The machine isn't short of power.
- **Known, measured cause in our design:** both arms' 50 Hz motion loops (`move_deg` / `move_path` in `src/hfteama/sorter/robot.py`), both arms' IK, the executor, the USB watcher and the dashboard share **one Python process** and its interpreter lock.
  - A move doesn't skip ticks when a tick runs late. A loop held back by other Python work stretches every move.
  - Measured: a re-scan running in that process made the 50 Hz loop run 21 ms late on average (126 ms worst).
- **Unexplained:** in 09:43 the re-scan ran in its own process (fix 14f6e77) and still timed out at 120 s. The same command takes 15 s offline, and with a saved image it touches neither the camera nor the serial ports. It probably got **stuck** rather than slow.

### 3.4 Smaller findings
- **Box placement:**
  - boxes within ~10 cm of an arm base fail as "unreachable" after a long IK search;
  - the export is a snapshot: boxes moved after it are picked at the old spot until a missed grasp triggers a re-scan.
- **Orientation:** the box angle flows correctly from camera to wrist. Measured offline at 0° to 165°: 0.0° jaw error on both arms. Remaining grasp misses are physical: tag not square to the box, wrist offset, or aim fitted before the angle was used.
- **A dropped arm isn't parked:** after `heartbeat lost` the executor drops the arm without parking it, so Stop all can't reach it. The operator must support it and cut its 12 V.

## 4. Already fixed (on `main`)

| Commit | Change |
|---|---|
| a88b151 | Live planner: 15 s step floor, nearest-arm preference. Stop all cancels the running step, then home + torque off (confirmed on the real arms). `arm_parked` event. Dashboard camera panel |
| 14538b2 | Camera feed off by default (`HFTEAMA_FEED=1` turns it on) |
| 81d3021, a5e141f | Hardware: step cap 120 s, IK cached once per spot, stop checked between IK solves; export plans 40 s per box |
| 14f6e77 | Re-scan in a child process, one at a time |
| ccf9df6, afb576a | `runner --host`: dashboard for other laptops (view only without the token); `docs/VIEW_DASHBOARD_REMOTE.md` |

## 5. Improvement plan

Order: **E first** (measure), then **A** (free), then **C** or **B**, then **F**.

### A. Balance the work
- **Change:** put most boxes in the middle band both arms reach (within ~10 cm left or right of the bowl, ≥ 10 cm in front of the bases).
- **Optional:** lower `FAR_HALF_FACTOR` (1.5, `src/hfteama/sorter/run.py`) to 1.0 to 1.2 so an arm takes boxes from the other half. Ask Hardware: far-half grasps are less well tuned.
- **Expected:** batch ~120 s → ~85 s. At the join, the prediction drops instead of rising.
- **Owner:** whoever sets up the table; the factor is Hardware's.
- **Done when:** a 5-box run with the join shows the predicted finish falling at the join, and neither arm idles more than ~10 s at the end.

### B. Lock the bowl only for the drop
- **Change:** split each place into three steps chained with `same_arm_as`:
  1. move to a waiting pose beside the bowl (no lock);
  2. drop: lower, open, lift (lock `bowl`, ~5 s);
  3. back to the carry pose (no lock).
- **Files:** the new skills in `src/hfteama/arms/generic_skills.py` (Action); the export writing the three steps in `src/hfteama/sorter/run.py` (Hardware). The task format already supports it; no executor change.
- **Expected:** bowl bound 75 s → ~30 s; no more waiting at the bowl holding a box.
- **Constraint:** a new motion type. Tell the team before its first real run.
- **Done when:** the logs show the two arms' drops back to back, with no waits over 2 s.

### C. A drop spot per arm (alternative to B)
- **Change:** two bowls, one in front of each arm (or a split tray), each its own resource, so places run fully in parallel.
- **Bonus:** gives Qwen a real sorting decision (which bin) in the export.
- **Files:** the export's one-bowl config and code (`configs/sorter.json` `box`, `sort export`). Hardware.
- **Expected:** ~72 s for 5 boxes with two arms, against ~145 s with one.
- **Done when:** both arms place at the same time without collision, and Qwen's bin choice matches the boxes.

### D. Honest estimates
- **Change:** plan with the measured times (pick ≈ 14 s, place ≈ 15 s) instead of 20 s per step, so the predicted finish is right before and after the join.
- **Note:** the executor's `DurationModel` already switches to measured 80th-percentile times once it has runs.
- **Owner:** Operations; the export's `--duration` belongs to Hardware.

### E. Measure the motion loop (do this first)
- **Change:** in `move_deg` / `move_path` / `gripper` (`src/hfteama/sorter/robot.py`), or around them in the bridge, record per move: the number of ticks, the actual time per tick, the time spent in `send_action` and `get_observation`, and the step's IK time. Print a one-line summary per move to the console, and add it to `task_end` data.
- **Also:** in the runner, log process CPU and thread count every few seconds.
- **No motion change.**
- **Done when:** the next slow run shows directly whether ticks run over 20 ms because of servo bus I/O (arm side) or Python scheduling (our process).

### F. One process per arm
- **Change:** run each arm's driver (motion loop + IK) in its own process. The `SkillWorker` / executor in the runner talks to it through a small command queue: execute, cancel, park, heartbeat.
- **Why:** no arm can slow the other, and the dashboard and re-scans can't slow either one. This is the structural fix for "both arms truly at the same time".
- **Frozen interface:** the `ArmWorker` contract (`src/hfteama/arms/base.py`) stays as it is; only the worker's inside changes.
- **Owners:** Action (worker), Hardware (bridge/driver).
- **Done when:** with both arms running and a re-scan in progress, E's tick time stays at ~20 ms.

### G. Smaller follow-ups
- **The stuck re-scan:** after E, check why the export child stalled >120 s in 09:43 (log its stderr; add a progress print).
- **Park an arm dropped for `heartbeat lost`** once its bus answers again. That's a new motion: agree with Hardware first.
- **Re-scan right before Go** (or at Go), so boxes moved after the export are found.
- **Grasp misses:** check that the tags are square to the boxes; Hardware to re-fit the aim corrections now that the wrist turns to each box's angle.

## 6. Rules that apply

- Only the person at the rig runs commands that move an arm: the runner, `hfteama sort goto|jog|touch|run`, `arm_check` motion. Read-only is fine: `sort export`, `sort preview`, the logs, offline tests.
- Never calibrate or write calibration to the servos.
- Stopping is a slow stop: freeze, home, then torque off. Never kill a process driving an arm.
- New motion types (B, G) need the team's agreement before their first real run.
- The core stays task-agnostic. The frozen interfaces (task format, `ArmWorker` contract, event log) change only with the other teams.
- Tests: `pytest` (on Windows deselect `tests/test_runner.py::test_ctrl_c_is_a_slow_stop_not_a_kill`, which kills pytest itself); `ruff check .`.
