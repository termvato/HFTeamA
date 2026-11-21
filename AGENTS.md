# AGENTS.md

Instructions for coding agents (Codex, Copilot, Claude). Same rules as `CLAUDE.md`.

## What this is

HFTeamA, "plug and produce": finish a batch of work faster by plugging in more SO-101 arms. When an
arm is added, removed or fails, the orchestrator re-splits the remaining work live and shows the new
predicted finish time and what limits further speed-up (`arms`, `order`, or `resource:<name>`).
Hackathon project (48 h, weekend of 21-22 Nov 2026). Umbrella issue: #1.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # + sched (OR-Tools), view (Dash/Plotly), sim (MuJoCo) as needed
pytest                               # all tests (fast, no hardware)
pytest tests/test_interfaces.py -k bound   # one test
ruff check . && ruff format --check .      # CI runs `ruff check .` and `pytest` on every push
```

Real arms need LeRobot installed from source (not pinned in `pyproject.toml`). The MuJoCo SO-101
model comes from `mujoco_menagerie/robotstudio_so101` (not vendored).

## Teams and ownership

Each team owns a problem; the method is the team's call. Do not build inside another team's area
without asking them.

| Team | People (GitHub) | Owns | Issue |
|---|---|---|---|
| Hardware | Diego (`lemillion789`), Om (`infinityengi`) | rig, calibration, safe motion, arm detection | #19 |
| Action | Vivek (`VivekSai07`), Sahil (`sahilrajpurkar03`) | skills on real + sim arms, measured durations | #20 |
| Operations | Vato (`termvato`), Aravind (`aravinds1610`) | task format, scheduler, executor, view | #21, backlog #2-#18 |

## Architecture

```
tasks/<name>/*.json --> taskgraph (load, validate) --> scheduler (plan + lower bound)
                                                          |  ^ re-plan
                                                          v  |
            view <-- events.jsonl <-- executor --execute(task)--> ArmWorker (real / sim / fake)
                                                                     |
                                                       arms.skills.REGISTRY[task.skill]
```

| Path | What | Owner |
|---|---|---|
| `src/hfteama/taskgraph/` | Task / TaskGraph, JSON loader, validator | Ops (#2) |
| `src/hfteama/arms/base.py` | `ArmWorker` contract, `Outcome`, `Result`, `ArmState` | Ops, agreed with Action + Hardware (#4) |
| `src/hfteama/arms/skills.py` | `@skill("name")` registry | Action fills it |
| `src/hfteama/arms/kinematics.py` | LeRobot <-> MuJoCo Menagerie joint angles (`ArmFrame`) | Action / Hardware |
| `src/hfteama/scheduler/` | `Plan`, `lower_bound` (done); HEFT, CP-SAT, checker (todo) | Ops (#6-#9, #12, #16) |
| `src/hfteama/executor/` | dispatch, resource locks, re-split (stub) | Ops (#10, #11) |
| `src/hfteama/events.py` | append-only JSONL event log | Ops (#5) |
| `src/hfteama/view/` | live Gantt, finish time, bottleneck (stub) | Ops (#17) |
| `tasks/` | one folder per task (`sort/sort12.json`, `assemble/assemble3.json`) | whoever adds the task |
| `research/` | background report and sourced notes (read before choosing an approach) | - |

### Frozen interfaces (change only with the other teams' agreement)

1. **Task format**: `src/hfteama/taskgraph/`. Field names: `resources` (not `zones`), `capability`,
   `allowed_arms`, `duration`/`durations`, `deps`, `same_arm_as`, `needs_arms`.
2. **Arm contract**: `src/hfteama/arms/base.py`. `execute(task) -> Result` blocks and never raises for
   a task failure; outcomes are `ok | failed | timeout | arm_down`, with free-text `Result.reason`
   (e.g. `"grasp"`). Also `state()`, `heartbeat()`, `cancel(task_id)`, `park()`.
3. **Event log**: `src/hfteama/events.py`. JSON lines; the view reads only this.

If a change to one of these seems needed, stop and say so instead of editing it; propose it on the
relevant issue (#2, #4, #5). Tests in `tests/test_interfaces.py` pin the current behaviour.

## Rules

- **Task-agnostic core.** Nothing in `src/hfteama/` may know about a specific task (sort, assemble,
  block ids, bins). Task-specific skills go in `tasks/<name>/skills.py` via `@skill("name")`. Skill
  names are global; registering a duplicate raises.
- **Every backend, one contract.** Real, MuJoCo and fake arms all implement `ArmWorker`; the
  executor must not branch on `backend` (it is only a label for the Gantt: REAL / SIM).
- **Durations in the task JSONs are placeholders** until measured on the real arms (#13).
- Python >= 3.10, `from __future__ import annotations`, dataclasses (frozen where shared), line
  length 100, ruff rules `E F W I B UP`. Add a test for any new behaviour; keep CI green.
- `configs/arms.json` (serials, calibration ids) is git-ignored: copy `configs/arms.example.json`.
  `runs/` (event logs) is git-ignored too.

## Real-arm safety (non-negotiable)

Claude must **not** run anything that moves a real arm without the human at the rig explicitly
confirming in that moment. Prefer fake or MuJoCo arms for any development and testing.

- Before any motion: check the arm's state, and that the target is inside joint limits
  (`kinematics.LIMITS_RAD`) and above the table.
- Two arms never in the same shared resource at once (runtime lock, not just the plan).
- Stopping or removing an arm is a slow stop: freeze, lower, then torque off. Never cut torque with
  the arm raised. Never `kill -9` / Ctrl-C-spam a process that is driving an arm.
- New motion types: tell the team before their first real run.

## Workflow

- Each person works on a branch named after their GitHub login (e.g. `VivekSai07`) and merges to
  `main` via PR. Reference the issue number in commit messages (e.g. `Sim arm backend (#20)`).
- `gh issue view N` fails on this repo (Projects classic deprecation). Use:
  `gh issue view N --json title,body,comments --jq '.title, .body, (.comments[] | .body)'`
- Issues #2-#18 are the Operations backlog (label `orchestrator`; P0 = first end-to-end run).
  Open design questions on re-split policy live in #18.
- Checkpoints: Sat evening (live screen on fake arms, one real arm does a step), Sun midday (adding
  an arm re-splits on screen), Sun evening (full demo), Sun night (code freeze + backup video).

## Action branch (`action-arm-skills`, issue #20)

- Read `ACTION_STATUS.md` and `docs/handoff/claude-to-copilot.md` (next steps) first; update the
  status file when a task changes state, and add a dated line to its log. Team-wide `STATUS.md` (on `main`) holds one block per person; update Vivek's block there.
- Design: `docs/superpowers/specs/2026-11-21-action-arm-skills-design.md`. Follow it; if the code
  needs to differ, update the design doc in the same commit and say why.
- Our scope: skills, primitives, IK, sim arms, `SkillWorker`, durations, real backend + handoff.
  Not ours: executor, scheduler, view, fake arm (#14), arm detection.
- Skills call only `ArmDriver` primitives and never branch on sim vs real.
- Commits: author `VivekSai07 <viveksaisurya07@gmail.com>` (or Sahil's own identity), no
  AI attribution or `Co-Authored-By` trailers. Do not push; a human pushes.
- Keep `CLAUDE.md` and `AGENTS.md` in sync: same rules, different tools read each.
