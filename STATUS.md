# Status

Where everyone is, in one place. **Update your own block** when you finish something, start something,
or get stuck (a few lines, not a diary). Change the "Updated" time when you do.

How to update: edit this file directly on `main`, in the GitHub web editor (pencil icon) or with a
small commit (`Status: <your name>`). Status edits are the one thing that goes straight to `main`;
code still goes through your team branch. Only edit your own block, so edits don't collide.

Plan and product: [`docs/plan.md`](docs/plan.md). Umbrella issue: #1.

## Next checkpoint: Saturday evening

| Target | Team | State |
|---|---|---|
| Live screen running on fake arms | Operations | not started |
| One real arm does a step (pick and place) | Action + Hardware | not reported yet |
| All 3 arms calibrated | Hardware | in progress: 1 leader-follower pair working; Vato's arm being set up (reported in person, ~14:35) |

States: `not started`, `in progress`, `done`, `blocked: <on what>`.

## Open decisions

| Decision | Who | Where |
|---|---|---|
| Arm contract: `execute(task)` vs `execute(arm_id, skill, params)`; distinct failure types | Vato, Aravind (then Action, Hardware agree) | #21, #4 |
| Who owns arm detection (`vla/detect_arms.py` on `action` overlaps Hardware's scope) | Hardware, Action | #19, #20 |
| Top-down camera: mounting and calibration to the table | Hardware, Operations | #19 |
| Step timeout: 1.5x or 3x the estimate | Vato, Aravind | #21 |

---

## Operations (`orchestrator`, #21)

Team detail: [`OPERATIONS_STATUS.md`](https://github.com/termvato/HFTeamA/blob/orchestrator/OPERATIONS_STATUS.md) on `orchestrator`.

### Vato
- Updated: Sat 21 Nov 14:35
- Done: research; repo structure; task format, arm contract, event log; lower bound with named limit;
  product spec (visual sorting, Go button, VLM brain); team branches and issues.
- Doing: team setup and product spec (this file, issues, branches).
- Next: core loop on fake arms (full-schedule planner, executor, countdown screen with what-if),
  then the VLM decomposer on top.
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
- Updated: Sat 21 Nov 14:03 (from the repo)
- Done: VLA data-collection setup in `vla/` on `action` (arm detection by USB serial, calibration,
  teleop, recording single- and dual-arm datasets).
- Doing: not reported yet
- Next: not reported yet
- Blocked on: CI lint fails on `vla/detect_arms.py` (see #20).
