# Handoff: Action branch, Claude -> Copilot / Codex (Sat 21 Nov, ~20:40)

Read `AGENTS.md` (rules, safety, commit identity) and `ACTION_STATUS.md` (log) first. This file is
the "where we are and what is next" for whoever continues on `action-arm-skills`.

## Where things stand

- Branch `action-arm-skills`, 8 commits ahead of `origin/action-arm-skills` (Vivek pushes:
  `git push origin action-arm-skills`). Merged with `origin/main` at 2828b1a (PR #44).
- 351 tests green (`.venv/bin/pytest -q`, about 4 min). `ruff check src tests scripts` clean;
  `ruff check .` fails only in `vla/` (Sahil's).
- Local `main` has one unpushed `Status: Vivek` commit and is 14 behind `origin/main`: rebase it
  on `origin/main` before pushing, and refresh Vivek's block in `STATUS.md` while there.

### Built (ours, `src/hfteama/`)

| What | Where |
|---|---|
| IK on the vendored SO-101 MuJoCo model (deterministic, 0.05 rad limit margin) | `arms/ik.py`, `arms/model.py` |
| Primitives (the Action <-> Hardware interface), `clip_to_limits` | `arms/primitives.py` |
| Skills `pick` `place` `pick_place` `inspect` `park`; pick/place stage through a per-arm `carry` pose and leave the shared area before returning | `arms/generic_skills.py` |
| `SkillWorker` (contract, timeout, cancel, lost arm) | `arms/worker.py` |
| MuJoCo table, per-arm rest pose, bin markers (`radii`) | `arms/sim.py`, `arms/poses.py` |
| LeRobot driver: bus lock, stop latch, `prepare_port` (goal := present before torque-on) | `arms/real.py` |
| Raw Feetech probe (reads only) and goal hold | `arms/feetech.py` |
| Runner: Operations' Executor + ArmWatcher + our workers; typed `yes`/`go`; Ctrl-C = slow stop; `--inject`, `--timeout-factor` (default 2.5) | `runner.py` |
| Staged real-arm tool: `check-config probe read joint point descend step park` | `arm_check.py` |
| Real table layout in sim (2 arms, 0.40 m apart, both facing -y, shared bowl) | `tasks/hacklab/`, `configs/real_arms.hacklab.json` |
| Video recorder (`--layout hacklab` default, `--layout sort`) | `scripts/record_sim_video.py` |

Sim results: `tasks/hacklab/hacklab12.json` 24/24 ok on 2 arms; sort12 24/24 on 3 arms. Videos
(git-ignored): `runs/sim_hacklab.mp4`, `runs/sim_sort12.mp4`.

### Real arm (vato_follower), done with Vivek's explicit go

- Windows sees COM7 = `vato_follower` (pair `hacklab`), COM8 = `hacklab_leader`. WSL sees no USB.
- Passive probe, connect (torque on, arm held still), `joint wrist_roll +-0.1` (landed within 1
  deg), `point 0.2 0 0.12` (8 mm error), `point 0.2 0 0.07` (2.3 mm), `park` (slow, torque off).
- **Last state: parked, torque OFF.**
- Not done: `descend` to the table (Vivek must confirm by eye that the table is at z = 0 in
  `configs/real_arms.json`, which is the bench config in the arm's own base frame), then `step`
  (one real pick).

## Next, in order

1. **Integrate Diego's `origin/hardware-real-arm-check` (5b90e6b).** He ran our driver on
   `rakesh_follower` and changed `arms/real.py` and `configs/real_arms.json`. A trial merge
   conflicts in exactly those two files. What he found and fixed:
   - **Gripper commands are clipped by LeRobot's `max_relative_target` (10)**: "open to 70" opens
     by 10 units. This breaks our `open_gripper` / `close_gripper` on any real arm. His fix ramps
     in 5 % steps and re-sends the goal until reached or stalled on the part. Take it.
   - Per-arm `limits_deg` (measured range minus 3 deg, never wider than the model), checked by
     `_check(q, ref)`, which lets a joint resting past a limit move back inward. We solved the same
     problem differently (`clip_to_limits` in `_base`). Keep one approach, keep both tests green.
   - rakesh: `wrist_roll` offset +90 deg, `grip_holding_min_pct` 3.5, folded `park_q`
     (shoulder -95, elbow 88). Check whether vato_follower needs the same wrist_roll offset
     before any `pick_yaw_deg` grasp.
   - Keep our `prepare_port` and logging. Merge both arms into one `configs/real_arms.json`.
   - His scripts `scripts/real_stage1.py`, `real_stage2.py`, `real_mirror.py` overlap `arm_check`;
     do not delete them (his), just note the overlap.
2. **Talk to Diego about `origin/hardware-ik`** (a "Sorter" with its own IK, jog mode, vision,
   per-arm grasp offsets). It overlaps our skills. Do not build into it; ask Vivek how to split.
3. **Real arm, one step at a time, each with Vivek's go:** `descend` then `step` on vato_follower;
   measure its `grip_holding_min_pct` (empty close vs smallest part, like Diego did).
4. **Measured values still missing:** arm base poses in the table frame (Operations blocker #2,
   marker touch; `configs/real_arms.hacklab.json` uses camera estimates +-3-5 cm), rakesh `park_q`
   in that file is a copy of vato's (use Diego's), real step durations (`scripts/measure_durations.py
   --backend real`).
5. **Runner overlap:** Vato's `python -m hfteama.demo` (screen, plug-in detection, `--clearance
   0.15`) already uses our `SkillWorker` via `action_factory`. Our `hfteama.runner` duplicates the
   wiring. Suggest: demo = Vato's runner; ours stays as the bench/CI tool. Confirm that
   `hfteama.demo --backend sim` runs `tasks/hacklab` with our poses file.
6. **Tell Operations:** Blocker #7 (place must leave the bowl) is fixed in 9421181; we changed the
   glob in their `tests/test_interfaces.py` to skip `poses*.json`; their `test_fake_arm.py` timing
   asserts flake under load; executor `timeout_measured` 1.5 cancels the occasional slow place
   (we pass 2.5); a step `resolve` leaves with no allowed arm is never run and the batch never
   ends (seen with a bowl spot 2 mm past the 0.42 m reach): it should be reported, not waited on.
   Their `hacklab6.json` objects are not in any poses file, so in sim every grasp fails: the sim
   only spawns the objects listed in `poses.json`.
7. Post the primitives comment on #19 and #20 (draft at the end of `docs/handoff/action-to-hardware.md`).
   Ask Vivek first: posting is visible to the team.

Deferred minors (from the review): malformed coordinate args raise a raw `ValueError` (arm does
not move); `place` resolves `target`/`to` less forgivingly than `pick`; `pick_xyz` + `object` not
cross-checked; tiny check-then-act gap in `cancel()`; table check covers the tool centre only;
`inspect` sleeps in wall-clock time. Sim has no arm-arm collision avoidance during concurrent joint
sweeps: safety comes from carry staging, the bowl lock and Operations' clearance locks.

## How to run things

```bash
.venv/bin/pytest -q                                   # ~4 min; -k to narrow
.venv/bin/python -m hfteama.runner --backend sim --task tasks/hacklab/hacklab12.json \
  --poses tasks/hacklab/poses.json --arms rakesh_follower,vato_follower --yes --log /tmp/hl.jsonl
.venv/bin/python scripts/record_sim_video.py          # -> runs/sim_hacklab.mp4
.venv/bin/python -m hfteama.arm_check check-config --config configs/real_arms.hacklab.json \
  --poses tasks/hacklab/poses.json --pairs configs/teleop_pairs.json
```

Real arm (from WSL, runs on Windows in the isolated venv `C:\Users\utxowqr\hfteama-win`, LeRobot
0.4.4; the wrapper `hf.ps1` imports `src/` from the WSL path and forwards piped input):

```bash
cd /tmp && powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  'C:\Users\utxowqr\hfteama-win\hf.ps1' hfteama.arm_check probe --port COM7      # reads only
cd /tmp && powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
  "'yes' | & 'C:\Users\utxowqr\hfteama-win\hf.ps1' hfteama.arm_check park --arm vato_follower"
```

## Gotchas

- **Never move a real arm without Vivek confirming in that moment.** Probe first (reads only).
  Never kill a process driving an arm; Ctrl-C in the runner is a slow stop.
- Offscreen MuJoCo in WSL: the D3D12 GPU path crashes with a threaded renderer; the recorder sets
  `LIBGL_ALWAYS_SOFTWARE=1` and turns shadows off (about 70 ms per frame).
- The runner deletes its `--log` file at start (one log = one run).
- Several sim tests take 20-60 s each; run `pytest tests/test_skills.py tests/test_real.py` etc.
  for quick loops, the full suite before committing.
- Commits: `VivekSai07 <viveksaisurya07@gmail.com>`, no AI trailers, do not push.
