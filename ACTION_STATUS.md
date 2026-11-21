# Action status: scripted skills on MuJoCo (#20)

Branch `action/arm-skills`. Team: Vivek (`VivekSai07`), Sahil (`sahilrajpurkar03`).
Design: `docs/superpowers/specs/2026-11-21-action-arm-skills-design.md`.
Update this file whenever a task changes state (agents: Claude, Codex, Copilot alike).

## Now

Design written, awaiting approval. Next: implementation plan.

## Workstreams

| # | Work | Owner | Status | Notes |
|---|---|---|---|---|
| 0 | Design doc | Vivek | in review | |
| 1 | Implementation plan | Vivek | todo | after design approval |
| 2 | Vendor SO-101 model; MuJoCo 4.0.1 vs 3.17 check | - | todo | |
| 3 | IK (`arms/ik.py`) | - | todo | |
| 4 | Primitives protocol (`arms/primitives.py`) | - | todo | post draft to Hardware (#19) |
| 5 | Sim table + driver (`arms/sim.py`) | - | todo | option B grasp |
| 6 | Generic skills + `poses.json` | - | todo | |
| 7 | `SkillWorker` (`arms/worker.py`) | - | todo | |
| 8 | Sim demo on sort12, 3 arms | - | todo | proof on MuJoCo |
| 9 | Duration measurement script | - | todo | feeds Ops #13 |
| 10 | Real backend (`arms/real.py`) | - | todo | mocked-bus tests only |
| 11 | Handoff doc to Hardware | - | todo | `docs/handoff/action-to-hardware.md` |
| 12 | Option A: contact grasping in sim | - | later | follow-up |

## To agree with other teams

- [ ] Hardware (#19): primitives list
- [ ] Operations (#21): skills raise `SkillFailed`; `sim` extra in CI; `poses.json`; fake arm (#14)

## Log

- 2026-11-21: Onboarded; CLAUDE.md on main; branch renamed `VivekSai07` -> `action/arm-skills`;
  picked sim option B (A later); design drafted.
