# Status

Where everyone is, in one place. **Update your own block** when you finish something, start something,
or get stuck (a few lines, not a diary). Change the "Updated" time when you do.

How to update: edit this file directly on `main`, in the GitHub web editor (pencil icon) or with a
small commit (`Status: <your name>`). Status edits are the one thing that goes straight to `main`;
code still goes through your team branch. Only edit your own block, so edits don't collide.

Plan and product: [`docs/plan.md`](docs/plan.md). Umbrella issue: #1.

## Messages to teams (read me)

- **Everyone: the demo is filmed on our real setup** (plan on #1 and in `docs/plan.md`, Demo): the
  VLM splits the task on screen, one arm sorts, the second arm is plugged in, the screen shows the
  re-planning, the predicted time comes down, both arms work. No Isaac Sim. Layout rule: every
  object within the first arm's reach (0.14-0.42 m from its shoulder).
- **Everyone:** plug-in detection is on `main` (`src/hfteama/discovery.py`): a follower is noticed
  by its USB serial; plugged in = joins and the work re-splits, unplugged = dropped at once. Arm ids
  everywhere must be the follower ids from `configs/teleop_pairs.json` (`rakesh_follower`,
  `vato_follower`). Arm positions: one file, `configs/real_arms.json` (Action's format).
- **Action (Vivek, Sahil):** CI on `action` fails on lint only (`vla/cameras.py`, `vla/detect_arms.py`).
  Fix: `ruff format vla/`, then `ruff check vla/` (B023 in `detect_arms.py` is a false positive:
  `# noqa: B023`). Details on #20 and PR #24. Keep working on `action`, not personal branches.
- **Hardware (Diego, Om) and Action:** the real-arm path is Action's (Vivek's `SkillWorker` and
  `LeRobotDriver`, branch `action-arm-skills`); Operations is not building an adapter. Hardware:
  please give Vivek the park and home pose per follower, the table height and grasp thresholds
  (#19, #20). LeRobot safety notes for the driver are on #20.

## Next checkpoint: Sunday midday

| Target | Team | State |
|---|---|---|
| Adding an arm (fake or real) re-splits the work on screen | all | in progress: plug-in detection on `main`; real-arm path running (`demo --backend sorter`, Diego and Aravind overnight); a real arm joining mid-run not reported yet |

Saturday evening checkpoint (read from the overnight commits, Vato, Sun 10:00):

| Target | Team | State |
|---|---|---|
| Live screen running on fake arms | Operations | done |
| One real arm does a step (pick and place) | Action + Hardware | done per commits: real picks and places take 15-25 s (Diego, `a5e141f`, 08:24); two-arm picks hit the old 30 s cap, now 120 s |
| Arms calibrated | Hardware | 2 arms (rakesh, hacklab): bases calibrated on the hacklab table, C922 overhead camera (Diego, `d8e4908`, 03:15); third arm not reported |

States: `not started`, `in progress`, `done`, `blocked: <on what>`.

## Open decisions

| Decision | Who | Where |
|---|---|---|
| Who owns arm detection (`vla/detect_arms.py` on `action` overlaps Hardware's scope) | Hardware, Action | #19, #20 |
| Top-down camera: mounting and calibration to the table | Hardware, Operations | #19 |

Architecture is locked (21 Nov): see [`docs/plan.md`](docs/plan.md#architecture-locked-sat-21-nov-vato-and-aravind).

---

## Operations (`orchestrator`, #21)

Team detail: [`OPERATIONS_STATUS.md`](https://github.com/termvato/HFTeamA/blob/orchestrator/OPERATIONS_STATUS.md) on `orchestrator`.

### Vato
- Updated: Sun 22 Nov 10:00
- Done: planner, executor with live re-split, live screen (countdown, re-planning moment, top-view
  map), plug-in detection, demo runner, scaling benchmark (#15), chain validated in MuJoCo; Desk
  Hero pitch framing and deck; 15 cm clearance between arms (`--clearance`).
- Doing: the demo film and the pitch for Sunday evening.
- Next: rehearsal on the real setup with the overnight sorter path.
- Blocked on: nothing.

### Aravind
- Updated: Sat 21 Nov 16:15 (from the repo)
- Done: Operations backlog #2-#18; design doc; arm registry and reachability (#3); VLM decomposer,
  step check and re-scan; object positions from the camera (Qwen pointing onto the table plane).
  First real Qwen3-VL-4B run on the RTX 4060 passed (synthetic image: 6/6 objects, 0.6 cm mean
  error, 8.5 s per scan; step check 2.8 s per call).
- Doing: not reported yet
- Next: not reported yet
- Blocked on: nothing reported.

## Hardware (`hardware`, #19)

### Diego
- Updated: not reported yet
- Done:
- Doing:
- Next:
- Blocked on:

### Om
- Updated: not reported yet
- Done:
- Doing:
- Next:
- Blocked on:

## Action (`action`, #20)

### Vivek
- Updated: Sat 21 Nov 13:59 (from the repo)
- Done: `CLAUDE.md` (commands, ownership, interfaces, safety) on `action`.
- Doing: not reported yet
- Next: not reported yet
- Blocked on:

### Sahil
- Updated: Sun 22 Nov 13:45
- Goal: replace the scripted IK pick-and-place step with a learned policy (ACT / SmolVLA / X-VLA),
  one copy per arm behind the same step interface; the orchestrator splits the objects.
- Done: recording studio and first dataset
  ([`so101_stack_single_rakesh`](https://huggingface.co/datasets/sahilrajpurkar03/so101_stack_single_rakesh),
  20 episodes). Trained SmolVLA with early stopping (best at step 2000) and X-VLA (3000 steps,
  vision and language frozen to fit 8 GB). Isaac Sim view of the MuJoCo sim: 3 arms build a
  14-box pyramid, arm 2 joins after 3 boxes, arm 3 after 7, run by the Executor; no contact in 3
  runs. Video [`docs/media/pyramid_isaac_sim.mp4`](docs/media/pyramid_isaac_sim.mp4), code in
  PR #49 (`action-sim`).
- Doing: merging PR #49.
- Next: evaluate SmolVLA and X-VLA on arm A (attempts so far cut short by USB dropouts: arm and
  cameras disconnect; needs a powered hub). Fix CI lint on `vla/`.
- Blocked on: a faulty gripper motor on the new follower board (no answer on the bus), and the
  USB dropouts above.
