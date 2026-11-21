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
- **Hardware (Diego, Om) and Action:** Operations is building the real-arm adapter
  (`src/hfteama/arms/real.py`). It opens followers from `configs/teleop_pairs.json`, checks state and
  limits before any motion, moves and parks slowly, and runs Action's skills by name. Needed from
  Hardware: park and home pose per follower, table limit, grasp thresholds. Proposed skill interface
  for Action on #20. Reply on #19 / #20.

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
| Step timeout (1.5x, 2x or 3x the estimate) | Vato, Aravind | #21 |

Architecture is locked (21 Nov): see [`docs/plan.md`](docs/plan.md#architecture-locked-sat-21-nov-vato-and-aravind).

---

## Operations (`orchestrator`, #21)

Team detail: [`OPERATIONS_STATUS.md`](https://github.com/termvato/HFTeamA/blob/orchestrator/OPERATIONS_STATUS.md) on `orchestrator`.

### Vato
- Updated: Sat 21 Nov 15:35
- Done: research; repo, plan, teams, issues; architecture locked with Aravind; **core loop on fake
  arms**: planner (greedy, HEFT, CP-SAT), executor with live re-split, durations, live screen
  (`python -m hfteama.view --fake-arms 1`) on `orchestrator`.
- Doing: real-arm adapter (`arms/real.py`): a real follower as an arm worker the executor can plan
  with.
- Next: first run with a real arm once Hardware and Action have one step working.
- Blocked on: nothing.

### Aravind
- Updated: Sat 21 Nov 14:02 (from the repo)
- Done: Operations backlog #2-#18; task allocation design doc (`docs/TASK_ALLOCATION_ARCHITECTURE.md` on `orchestrator`).
- Doing: not reported yet
- Next: not reported yet
- Blocked on: arm contract decision with Vato (#21).

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
- Updated: Sat 21 Nov 14:40
- Goal: replace the scripted IK pick-and-place step with a learned policy (SmolVLA first), trained
  on teleop data for the sort task, behind the same step interface.
- Done: data collection in `vla/` on `action` (arm detection, calibration, teleop, recording to the
  HF Hub). Local recording web page (live cameras, start/save/discard, upload), tested without
  arms, not pushed yet.
- Doing: connecting one arm and the cameras to record the single-arm dataset (~50 episodes).
- Next: second arm, dual-arm dataset, train SmolVLA, plug it in as an arm worker. Fix CI lint on
  `vla/detect_arms.py`.
- Blocked on: nothing.
