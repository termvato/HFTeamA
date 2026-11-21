# Action status: scripted skills on MuJoCo (#20)

Branch `action-arm-skills` (merges into team branch `action`). Owner: Vivek (`VivekSai07`).
Sahil works on the learned policy separately (`vla/`, see his block in `STATUS.md` on `main`).
Design: `docs/superpowers/specs/2026-11-21-action-arm-skills-design.md`.
Update this file whenever a task changes state (agents: Claude, Codex, Copilot alike).

## Now

Plan done, wired to Operations' executor, real hacklab layout runs 24/24 in sim, vato_follower
moved on the bench (parked, torque off). Next steps: `docs/handoff/claude-to-copilot.md` (first:
integrate Diego's `origin/hardware-real-arm-check` gripper ramp and per-arm limits).

## Workstreams

| # | Work | Owner | Status | Notes |
|---|---|---|---|---|
| 0 | Design doc | Vivek | done | updated after 14:50 sync |
| 1 | Implementation plan | Vivek | done | `docs/superpowers/plans/2026-11-21-action-arm-skills.md` |
| 2 | Vendor SO-101 model; MuJoCo 4.0.1 vs 3.17 check | Vivek | done | vendored so101-nexus model, MuJoCo 4.0.1 works |
| 3 | IK (`arms/ik.py`) | Vivek | done | `arms/ik.py`, orientation weight 0.01 |
| 4 | Primitives protocol (`arms/primitives.py`) | Vivek | done | `arms/primitives.py`; draft to post on #19 is in Task 9 |
| 5 | Sim table + driver (`arms/sim.py`) | Vivek | done | `arms/sim.py`, option B grasp |
| 6 | Generic skills + `poses.json` | Vivek | done | `arms/generic_skills.py`, `arms/poses.py`, `tasks/sort/poses.json` |
| 7 | `SkillWorker` (`arms/worker.py`) | Vivek | done | `arms/worker.py` |
| 8 | Sim demo on sort12, 3 arms | Vivek | done | `scripts/sim_demo.py`: 24/24 steps ok on 3 arms, 40.5 s sim time; grasp failure recovers |
| 9 | Duration measurement script | Vivek | done | `scripts/measure_durations.py`; table below |
| 10 | Real backend (`arms/real.py`) | Vivek | done | `arms/real.py`; unrun on hardware |
| 11 | Handoff doc to Hardware | Vivek | done | `docs/handoff/action-to-hardware.md` |
| 12 | Option A: contact grasping in sim | Vivek | later | follow-up |

## To agree with other teams

- [ ] Hardware (#19): primitives list
- [ ] Operations (#21): skills raise `SkillFailed`; `sim` extra in CI; `poses.json`; coordinate arg
  names `pick_xyz` / `place_xyz`; fake arm (#14); step timeout factor (their open decision)
- [ ] Sahil: `set_joint_target` primitive is there for the VLA worker; scripted skill as fallback

## Log

- 2026-11-21: Onboarded; CLAUDE.md on main; branch renamed `VivekSai07` -> `action-arm-skills`;
  picked sim option B (A later); design drafted.
- 2026-11-21 14:50: Synced with remote. Team branches now `orchestrator` / `hardware` / `action`;
  ours renamed `action-arm-skills` (git forbids `action/...` next to `action`), merged `origin/action`
  (Sahil's `vla/`) and `origin/main`. Ops locked the architecture: sim arms only in the what-if
  estimate, task args carry table coordinates; design updated. `vla/detect_arms.py` fails ruff
  (Sahil's, on his list).
- 2026-11-21 ~16:00: Tasks 1-7 of the plan done (model, IK, primitives, skills, worker, sim, sort12 demo,
  durations). Merged origin/main (Operations' planner, executor, fake arm, view) first; suite 160 green.

## Measured sim durations (sort poses, 5 cycles per arm, sim seconds)

| skill | arm_a p50 / p80 | arm_b p50 / p80 | arm_c p50 / p80 |
|---|---|---|---|
| pick | 3.28 / 3.30 | 3.34 / 3.42 | 3.54 / 3.57 |
| place | 2.24 / 2.27 | 2.19 / 2.27 | 2.44 / 2.45 |

`tasks/sort/sort12.json` still has placeholders (pick 4.0, place 6.0); real-arm numbers will differ.
Regenerate: `python scripts/measure_durations.py` (writes `runs/durations/sort_sim.json`).
- 2026-11-21 ~16:30: Tasks 8-9 done: `arms/real.py` (mocked-bus tests), `configs/real_arms.example.json`,
  `--backend real` in `scripts/measure_durations.py`, handoff `docs/handoff/action-to-hardware.md`
  (staged checklist; draft comment for #19/#20 at its end, not posted). Merged origin/main and
  origin/action again (Hardware's multi-pair CLI, Sahil's studio and cameras).
- 2026-11-21 ~16:15: Independent whole-branch review done; its 2 critical and 8 important findings are fixed
  with tests (stop latch, commands on the last command, bus lock, idle heartbeat, validate before moving,
  park/execute races, handoff doc). 8 minors deferred (list in the ledger / final report). 199 tests green.
  Not run on hardware.
- 2026-11-21 ~17:50: Merged latest main (Aravind: registry, perception, markers, reach; Vato: planner, executor,
  Isaac spike) and origin/action. Operations' real Executor now drives our SkillWorker on 3 sim arms: sort12
  24/24 ok (`tests/test_executor_with_sim_arms.py`). That run found two real bugs, both fixed: pick left the arm
  in the shared area when the executor released its lock (now ends at a per-arm `carry` pose), and the sim rest
  pose pointed each arm at its neighbour's bin (mirrored). Handoff rewritten with a status section; real arm
  ids must be catalog ids (`rakesh_follower`, `vato_follower`). 248 tests green. No real arm run yet.
- 2026-11-21 ~19:35: Wiring done: `hfteama.runner` (Executor + ArmWatcher + our SkillWorker factory, typed `yes`/`go`
  gates, Ctrl-C = slow stop), `hfteama.arm_check` (check-config, probe, read, joint, point, descend, step, park),
  grasp yaw (`pick_yaw_deg`), carry-pose staging, IK limit margin, bin markers, `scripts/record_sim_video.py`
  (runs/sim_sort12.mp4). 322 tests green. Hardware: vato_follower is on COM7 (hacklab pair; leader on COM8).
  Passive probe: 6/6 servos answer, 12 V, torque OFF, resting folded. Findings: (1) LeRobot's connect() enables
  torque WITHOUT goal := present, so `from_config` now writes goal := present first (`real.prepare_port`);
  (2) at rest shoulder_lift reads -1.947 rad, 0.20 past the model limit -1.745 (calibration range wider than the
  model): commands are clipped into the model limits and the first command moves it inside. Windows side: isolated
  venv C:\Users\utxowqr\hfteama-win (LeRobot 0.4.4, pyserial), run via hf.ps1. No motion command sent yet.
- 2026-11-21 ~20:40: vato_follower on the bench (Vivek's go): connect held still, joint +-0.1 rad within 1 deg,
  point at 12 cm 8 mm error, at 7 cm 2.3 mm; parked, torque off. Sim rebuilt for the real table
  (`tasks/hacklab`: 2 arms 0.40 m apart facing -y, shared bowl, one spot per block): 24/24 ok after `place`
  learned to leave the bowl (Operations' Blocker #7). Runner: `--timeout-factor` (2.5), fresh log per run;
  check-config: each object needs one arm. Merged origin/main (PR #44); 351 tests green. Diego ran our driver on
  rakesh_follower on `origin/hardware-real-arm-check` (gripper ramp, per-arm limits): not merged yet, conflicts
  in `arms/real.py` and `configs/real_arms.json`. Handoff for the next agent: `docs/handoff/claude-to-copilot.md`.
