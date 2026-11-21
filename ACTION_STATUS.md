# Action status: scripted skills on MuJoCo (#20)

Branch `action-arm-skills` (merges into team branch `action`). Owner: Vivek (`VivekSai07`).
Sahil works on the learned policy separately (`vla/`, see his block in `STATUS.md` on `main`).
Design: `docs/superpowers/specs/2026-11-21-action-arm-skills-design.md`.
Update this file whenever a task changes state (agents: Claude, Codex, Copilot alike).

## Now

Design approved and updated after the team sync. Next: implementation plan.

## Workstreams

| # | Work | Owner | Status | Notes |
|---|---|---|---|---|
| 0 | Design doc | Vivek | done | updated after 14:50 sync |
| 1 | Implementation plan | Vivek | doing | |
| 2 | Vendor SO-101 model; MuJoCo 4.0.1 vs 3.17 check | Vivek | todo | |
| 3 | IK (`arms/ik.py`) | Vivek | todo | |
| 4 | Primitives protocol (`arms/primitives.py`) | Vivek | todo | post draft to Hardware (#19) |
| 5 | Sim table + driver (`arms/sim.py`) | Vivek | todo | option B grasp |
| 6 | Generic skills + `poses.json` | Vivek | todo | |
| 7 | `SkillWorker` (`arms/worker.py`) | Vivek | todo | |
| 8 | Sim demo on sort12, 3 arms | Vivek | todo | proof on MuJoCo |
| 9 | Duration measurement script | Vivek | todo | feeds Ops #13 |
| 10 | Real backend (`arms/real.py`) | Vivek | todo | mocked-bus tests only |
| 11 | Handoff doc to Hardware | Vivek | todo | `docs/handoff/action-to-hardware.md` |
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
