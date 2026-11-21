# Action team design: arm skills on MuJoCo, then real arms

Owner: Vivek (`VivekSai07`), Action team, issue #20. Branch `action-arm-skills` (merges into the
team branch `action`). Sahil works separately on a learned policy (SmolVLA, `vla/`); this design is
the scripted path, which is also his baseline and fallback.
Status: approved 21 Nov 2026; updated after the 14:50 team sync (see "Changes after team sync").

## Changes after team sync (21 Nov, 14:50)

Operations locked the architecture and product in `docs/plan.md` on `main`. Effects here:

- **Sim arms do not run in the live demo** ("only in the what-if estimate"). The sim is our
  development and proof environment and the source of duration numbers, not a demo feature. Dropped:
  plugging sim arms in at runtime. Kept: several arms in one scene to prove parallel runs.
- **Task args carry table coordinates.** The worker reads only `task.id`, `task.skill`, `task.args`;
  args hold object id, target bin and table coordinates (the VLM and camera fill them). Skills take
  coordinates from args first and fall back to named poses in `tasks/<name>/poses.json`. Proposed
  arg names to agree with Operations: `object`, `target`, `pick_xyz`, `place_xyz` (metres).
  VLM output uses one `pick_place` step with `object` / `target`; sort12 uses separate `pick` /
  `place` with `from` / `to`. Skills accept both.
- **Step timeout is still open** (1.5x or 3x the estimate, Operations deciding): the worker takes the
  factor as a parameter.
- **Real arms are configured by Hardware** in `configs/teleop_pairs.json` (serials, calibration ids,
  calibration files in `configs/calibration/`), loaded by `hfteama.hardware.pairs`. The real driver
  reuses that loader instead of `configs/arms.json`.
- **Learned policy (Sahil):** plans a `VLAArmWorker` behind the same contract. To make a policy
  pluggable as just another skill, primitives gain a non-blocking `set_joint_target(q)` and
  `read_joints()` at control rate.
- One person on this track: no team split.

## Intent

The executor (Operations, #10) hands an arm one step at a time: `worker.execute(task) -> Result`.
Our job is everything behind that call: turning a step like `pick block_01 from input_bin` into
arm motion, saying whether it worked, and how long it took. It has to work the same way on
simulated arms and on the real SO-101s.

We prove it end to end on MuJoCo first, then hand a real-arm backend to Hardware (Diego, Om) to
test on the rig with a written checklist.

## Goals

1. Generic skills `pick`, `place`, `pick_place`, `inspect`, `park`, written once, running unchanged
   on sim and real arms.
2. A small set of arm primitives (the Action <-> Hardware interface) that every backend implements.
3. IK: table-frame target point -> joint angles, shared by sim and real.
4. A MuJoCo table with several SO-101 arms (default 3) in one scene, real-time paced, viewable.
5. One generic worker class that implements the frozen `ArmWorker` contract over any backend:
   outcomes, timeouts, heartbeat, cancel, park.
6. Measured per-skill, per-arm durations (p50 / p80) that Operations can plan with (#13).
7. A real-arm backend (LeRobot) plus a handoff document and test checklist for Hardware.

## Non-goals (other teams own these)

- The executor, scheduler, re-split, live view (Operations, #6-#17). Our proof scripts call
  `worker.execute()` directly in a simple loop; that is a test harness, not an executor.
- The fake arm for orchestrator testing (Operations, #14). Our worker plus a no-motion driver could
  serve as one; we offer it, they decide.
- Arm detection, calibration, the shared table frame, the physical rig (Hardware, #19).
- Learned policies. Possible later on one skill, never on the critical path.

## Anti-goals (would be a failure even if it "works")

- Skill code that branches on sim vs real. If a skill needs to know, a primitive is missing.
- Changing a frozen interface (`taskgraph/`, `arms/base.py`, `events.py`) without the other teams.
- Anything in `src/hfteama/` that knows about sorting, blocks or bins (task-agnostic core).
- Code that moves a real arm without having run the same path in sim first.

## Constraints

- Frozen contract `arms/base.py`: `execute` blocks, never raises for a task failure, returns
  `Result(outcome in ok|failed|timeout|arm_down, reason, started_at, finished_at)`.
- Python >= 3.10 for the core (CI on 3.11). The MuJoCo extra may require newer Python.
- Checkpoints: Sat evening one real arm does a step; Sun midday adding an arm re-splits; Sun
  evening full demo; Sun night code freeze.
- Real-arm safety rules in `CLAUDE.md` (limits, above the table, slow stop, one arm per shared area).

## Decisions taken

- **Sim fidelity: B now, A later** (decided by Vivek). B: the arm really moves through IK in
  MuJoCo, so motion and timing are real; a grasp attaches the block to the gripper when the
  gripper closes near it; failures are injected on command. A (follow-up): real contact
  grasping and a contact-based grasp check on the same scene.
- **SO-101 model: vendor the so101-nexus copy of the Menagerie model** (Apache-2.0, pinned commit,
  with its PROVENANCE.md). Better than the local copy: corrected wrist-roll limit, a tool-centre
  site (`gripperframe`), and jaw collision and contact tuning already validated for grasping,
  which is what option A needs. We vendor only the model folder (~18 MB of meshes), not the
  package: the package needs Python >= 3.12 and pins `mujoco < 4`, and pulls gymnasium and
  Hugging Face Hub, none of which we use. (Our call; alternative is a fetch script.)
- **IK: our own damped-least-squares solver on the MuJoCo model**, adapted from
  so101-nexus `kinematics.py` (Apache-2.0, attributed). Reasons: it uses the exact model the sim
  uses, so sim and real agree; no new dependency; SO-101 has 5 arm joints, so position is solved
  exactly and orientation is best effort (weighted), which is what pick and place need.
  - ikpy: works (numpy + scipy, reads MJCF), but generic optimiser IK on a second copy of the
    kinematics; slower and can disagree with the sim model. Kept as a fallback.
  - PyRoki: strong (collision-aware, JAX), but not on PyPI, git dependencies, JAX on the
    critical path. Overkill for a 5-joint arm in 48 h.
  - LeRobot `RobotKinematics` (placo): fine on the real arm but adds placo and a URDF path, and
    does not help the sim.
- **MuJoCo version: try 4.0.1 first, fall back to 3.17.** MuJoCo 4.0.1 is on PyPI (released
  11 Nov 2026; MuJoCo now ships as `mujoco` + `mujoco-core` + `mujoco-studio`). It is 10 days
  old and the vendored model was tuned on 3.17, so the first plan task loads the model and runs
  the sim tests on 4.0.1; if anything breaks, pin `mujoco>=3.17,<4`. Code uses only stable APIs
  (`MjSpec`, `mj_step`, `mj_kinematics`, `mj_jacSite`, `mujoco.viewer`).
- **One thread per arm, not one process.** Each real arm is its own USB bus and runs at ~30 Hz,
  so threads are enough; MuJoCo releases the GIL while stepping. Process isolation can be added
  behind the same worker later if one arm's crash takes others down.

## Design

### Layers

```
executor (Ops) ── execute(task) ──> SkillWorker (ArmWorker contract)        arms/worker.py
                                         │ looks up REGISTRY[task.skill]
                                         v
                                   skill(arm, task, poses)                   arms/skills.py
                                         │ calls primitives only
                                         v
                                   ArmDriver (primitives)                    arms/primitives.py
                                  /                    \
                     MujocoDriver (sim)            LeRobotDriver (real)      arms/sim.py, arms/real.py
                        │                               │
                     SimTable: one scene, N arms     SOFollower + ArmFrame (kinematics.py)
                        └──────────── IK (ik.py) ───────┘  same model, same solver
```

### Primitives (`arms/primitives.py`), the Action <-> Hardware interface

An `ArmDriver` protocol. Positions are metres in the shared table frame; joint angles are radians
in the model's convention (real backends convert with `kinematics.ArmFrame`).

- `read_joints() -> np.ndarray` (6: five arm joints + gripper)
- `move_joints(q, speed=...)`: blocking, interpolated, rejects targets outside joint limits.
- `move_to(xyz, yaw=None, speed=...)`: IK then `move_joints`; rejects targets below table height
  or out of reach.
- `open_gripper()`, `close_gripper() -> Grasp` (`holding: bool`, gripper position, effort).
- `stop()`: freeze in place. `park()`: slow stop to the parked pose, then torque off.
- `alive() -> float`: time of the last good read (feeds `heartbeat()`).
- `set_joint_target(q)`: non-blocking, limit-checked, rate-limited target for streaming control
  (a learned policy at 30 Hz); `move_joints` is built on the same path.

Failures are exceptions: `MotionError(reason)` (unreachable, limit, blocked) and
`ArmDown(reason)` (bus error, lost). Safety checks live in the driver, so no skill can bypass them.

### Skills (`arms/skills.py` + `arms/generic_skills.py`)

- Signature `skill(arm: ArmDriver, task: Task, poses: Poses) -> None`. Success = returns;
  failure = raise `SkillFailed(reason)`. This replaces the current docstring's "returns Outcome",
  because the contract now needs a `reason` with each failure. `skills.py` is ours, not frozen.
- `pick`: approach above object, descend, close, check grasp, lift, re-check grasp.
  `place`: approach above target, descend, open, retreat. `pick_place`: both. `inspect`: move to
  a look pose and hold (placeholder). `park`: go to the parked pose.
- Skills check `cancelled()` between primitive calls, so `cancel(task_id)` stops cleanly.

### Poses (`tasks/<name>/poses.json`)

Named locations and objects in the table frame, so the core stays task-agnostic.

```json
{"table_z": 0.0,
 "locations": {"input_bin": [0.30, 0.20, 0.02]},
 "per_arm": {"arm_a": {"own_output_bin": [0.10, -0.15, 0.02], "park": [...]}},
 "objects": {"block_01": [0.31, 0.21, 0.015]}}
```

`own_*` names resolve per arm. The sim spawns objects from this file; the real arm uses the same
positions (blocks placed by hand at marked spots).

### Sim (`arms/sim.py`)

- `SimTable`: builds one MuJoCo scene with `MjSpec.attach_body`: a table, N SO-101s at base poses
  given in the scene config (default: 0.45 m apart, as in `configs/arms.example.json`), and the objects from `poses.json`. One physics thread
  steps at real time (speed factor option for fast tests). Optional passive viewer.
- `MujocoDriver(table, arm_id)`: implements the primitives by setting that arm's actuator targets
  and waiting until reached (with a stall timeout).
- Option B grasp: on `close_gripper`, if an object's centre is within a radius of the tool site,
  it is attached and carried each step; on `open_gripper` it is released.
- Failure injection: `table.inject(arm_id, "grasp" | "slow" | "down")`, for the demo and tests.

### Worker (`arms/worker.py`)

`SkillWorker(arm_id, driver, poses, backend="sim"|"real")` implements `ArmWorker`:
- `execute(task)`: runs the skill with a deadline of 1.5x `task.duration_on(arm_id)`
  (configurable); maps return -> `OK`, `SkillFailed` -> `FAILED`, deadline -> `stop()` + `TIMEOUT`,
  `ArmDown` -> `ARM_DOWN`; never raises. Unknown skill -> `FAILED("unknown skill")`.
- `state()` follows STARTING -> READY <-> BUSY -> PARKING -> DOWN.
- `heartbeat()` = `driver.alive()`. `cancel(task_id)`: sets the cancel flag, then `stop()`.
  `park()`: `driver.park()`.

### Durations (`scripts/measure_durations.py`)

Runs each skill N times per arm (sim now, real later), prints p50 / p80 per (skill, arm), and writes
a JSON that Operations' estimator (#13) can read. The event log already carries real timings.

### Real backend (`arms/real.py`), for the handoff

`LeRobotDriver`: wraps LeRobot `SOFollower` (one per arm, its own port), converts with
`ArmFrame`, solves IK with the same `ik.py`, interpolates moves with a max joint step per tick,
checks limits and table height before every move, slow park. Unit-tested against a mocked bus;
we do not run it on hardware ourselves. Handoff: `docs/handoff/action-to-hardware.md` with what it
does, what Hardware must supply (ports, calibration ids, base poses in the table frame, park pose,
gripper thresholds) and a staged test checklist (read-only -> one joint, small, slow -> one skill).

### Proof on MuJoCo

`scripts/sim_demo.py`: loads `tasks/sort/sort12.json` and its poses, starts 3 sim arms, and runs
the steps through `SkillWorker.execute` with a trivial round-robin harness (respecting
`same_arm_as`), printing each `Result`. Viewer optional. Done when all 24 steps return `OK`, an
injected grasp failure returns `FAILED("grasp")`, and `measure_durations.py` produces p80 numbers.

### Tests

- IK: random reachable targets (from FK of random joint angles) solve to < 2 mm.
- Worker: outcome mapping, timeout, cancel, unknown skill, against a scripted in-memory driver
  (no MuJoCo, runs in CI).
- Skills: call order and failure paths against the in-memory driver.
- Sim (marked `sim`, skipped if MuJoCo is missing): one arm picks and places one block; 3 arms in
  parallel; injected failure.
- Real driver: limit and table checks, ArmFrame conversion, with a mocked bus.

## To agree with other teams (we propose, they decide)

- Hardware (#19): the primitives list above, posted today, before the handoff.
- Operations (#21): skill convention change (raise `SkillFailed`), adding the `sim` extra to CI,
  `poses.json` per task, the coordinate arg names (`pick_xyz`, `place_xyz`), and whether they want our worker + a no-motion driver as the fake arm (#14).

## Left to the builder

Interpolation profile and speeds, grasp radius, approach heights, stall timeouts, the scene's
table and bin geometry, the exact `poses.json` numbers for sort12, test thresholds.
