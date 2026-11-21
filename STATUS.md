# Status

Where everyone is, in one place. **Update your own block** when you finish something, start something,
or get stuck (a few lines, not a diary). Change the "Updated" time when you do.

How to update: edit this file directly on `main`, in the GitHub web editor (pencil icon) or with a
small commit (`Status: <your name>`). Status edits are the one thing that goes straight to `main`;
code still goes through your team branch. Only edit your own block, so edits don't collide.

Plan and product: [`docs/plan.md`](docs/plan.md). Umbrella issue: #1.

## Messages to teams (read me)

- **Action (Vivek, Sahil):** CI on `action` fails on lint only (`vla/cameras.py`, `vla/detect_arms.py`).
  Fix: `ruff format vla/`, then `ruff check vla/` (B023 in `detect_arms.py` is a false positive:
  `# noqa: B023`). Details on #20 and PR #24. Keep working on `action`, not personal branches.
- **Hardware (Diego, Om) and Action:** the real-arm path is Action's (Vivek's `SkillWorker` and
  `LeRobotDriver`, branch `action-arm-skills`); Operations is not building an adapter. Hardware:
  please give Vivek the park and home pose per follower, the table height and grasp thresholds
  (#19, #20). LeRobot safety notes for the driver are on #20.

## Next checkpoint: Saturday evening

| Target | Team | State |
|---|---|---|
| Live screen running on fake arms | Operations | done (on `orchestrator`) |
| One real arm does a step (pick and place) | Action + Hardware | not reported yet |
| All 3 arms calibrated | Hardware | in progress: 1 leader-follower pair working; Vato's arm being set up (reported in person, ~14:35) |

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
- Updated: Sat 21 Nov 16:20
- Done: research; repo, plan, teams, issues; architecture locked with Aravind; **core loop on fake
  arms**: planner (greedy, HEFT, CP-SAT), executor with live re-split, durations, live screen
  (`python -m hfteama.view --fake-arms 1`) on `orchestrator`.
- Doing: supporting Action's real-arm driver (review notes on #20); next Operations task to pick.
- Next: first run with a real arm once Hardware and Action have one step working.
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
- Updated: Sat 21 Nov 17:44
- Goal: replace the scripted IK pick-and-place step with a learned policy (ACT / SmolVLA), one
  copy per arm behind the same step interface; the orchestrator splits the objects.
- Done: recording studio in `vla/` on `action` (live top + wrist cameras, start/stop/upload),
  using Hardware's calibrated pairs. First dataset, public:
  [`so101_stack_single_rakesh`](https://huggingface.co/datasets/sahilrajpurkar03/so101_stack_single_rakesh)
  (20 episodes, rakesh pair, "Put the objects in the box"). Research note: one policy on any number
  of arms ([`vla/multi_arm_policy.md`](https://github.com/termvato/HFTeamA/blob/action/vla/multi_arm_policy.md)).
- Doing: choosing how arms pick objects (by table side or by name), then more episodes.
- Next: train ACT (~3-4 h on my laptop), test on arm A, then the same model on arm B; dual-arm
  data. Fix CI lint on `vla/`.
- Blocked on: nothing.
