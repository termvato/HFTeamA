# Action Arm Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scripted pick-and-place behind the frozen `ArmWorker.execute(task) -> Result` contract, proven with 3 SO-101 arms in MuJoCo, plus a real-arm (LeRobot) backend and a handoff to Hardware.

**Architecture:** Four layers in `src/hfteama/arms/`: `SkillWorker` (contract, timeouts, cancel) → skills (`pick`, `place`, `pick_place`, `inspect`, `park`) → `ArmDriver` primitives (safety checks live here) → backends (`MujocoDriver` on a shared `SimTable`, `LeRobotDriver`). One damped-least-squares IK on the vendored SO-101 MuJoCo model serves both backends.

**Tech Stack:** Python ≥3.10 (core), numpy, MuJoCo (4.0.1, fallback 3.17), pytest, ruff. LeRobot only for the real backend (lazy import, never needed in tests).

**Spec:** `docs/superpowers/specs/2026-11-21-action-arm-skills-design.md`

## Global Constraints

- Frozen, do not edit: `src/hfteama/taskgraph/`, `src/hfteama/arms/base.py`, `src/hfteama/events.py`.
- The worker reads only `task.id`, `task.skill`, `task.args`, plus `task.duration_on(arm_id)` for the deadline.
- Coordinate args (proposed to Operations): `pick_xyz`, `place_xyz`, metres, table frame. Named fallbacks: `object`, `target` (VLM format), `from`, `to` (sort12).
- Nothing in `src/hfteama/` names a task, block or bin. Task data lives in `tasks/<name>/poses.json`.
- Skills call only `ArmDriver` methods; no `isinstance` or backend checks in skills.
- Joint limits: `hfteama.arms.kinematics.LIMITS_RAD`. No target below `table_z + 0.01` m.
- Tests needing MuJoCo use `pytest.importorskip("mujoco")` and `@pytest.mark.sim`; the base CI install must stay green without MuJoCo.
- Line length 100, ruff `E F W I B UP`, `from __future__ import annotations`.
- Before each task: `git fetch --all --prune`, merge `origin/action` and `origin/main` into `action-arm-skills`, and note teammates' changes in `ACTION_STATUS.md`'s log.
- After each task: update `ACTION_STATUS.md`. Commit as `VivekSai07 <viveksaisurya07@gmail.com>`, with no AI trailers. Never push.
- Never run anything that moves a real arm.

## Review Focus

1. **A target outside reach or below the table** (a VLM coordinate off by a bin's width): `move_to` must raise `MotionError("unreachable")` or `MotionError("below table")` without moving. Tested in Tasks 2, 3 and 7.
2. **A skill that hangs** (stalled joint, gripper never settles): `execute` must return `TIMEOUT` near the deadline and leave the arm stopped, never block forever. Tested in Task 5.
3. **Args with neither coordinates nor a known name** (`{"object": "mystery"}`): the step must return `FAILED("unknown location: mystery")`, not crash the worker thread. Tested in Task 4.
4. **The arm is lost mid-step**: `ArmDown` inside a primitive must give `ARM_DOWN`, and every later `execute` must return `ARM_DOWN` at once (state `DOWN`). Tested in Task 5.
5. **`cancel()` on a different task id, or when idle**: it must do nothing, not stop the arm's current work. Tested in Task 5.

---

### Task 1: Vendor the SO-101 model and choose the MuJoCo version

**Files:**
- Create: `src/hfteama/arms/assets/so101/` (copy of so101-nexus `src/so101_nexus/assets/SO101_menagerie/`, from GitHub `johnsutor/so101-nexus` at commit `6c7ac1c`: `so101.xml`, `assets/*.stl`, `LICENSE`, `PROVENANCE.md`; drop `scene*.xml`)
- Create: `src/hfteama/arms/assets/so101/NOTICE.md` (source repo, commit, Apache-2.0, "vendored for hfteama")
- Create: `src/hfteama/arms/model.py`
- Modify: `pyproject.toml` (`sim` extra; pytest marker `sim`)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `MODEL_XML: pathlib.Path` (vendored `so101.xml`); `TCP_SITE = "gripperframe"`; `ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")`; `GRIPPER = "gripper"`.

- [ ] **Step 1: Copy the model files** from a fresh clone into `src/hfteama/arms/assets/so101/`, and write `NOTICE.md`.
- [ ] **Step 2: Install MuJoCo 4.0.1.** Run `.venv/bin/pip install "mujoco==4.0.1"`. If it does not install on Python 3.14, try `mujoco==3.17.0`. If neither works, recreate the venv with `uv venv -p 3.13 .venv` and install `-e ".[dev]"` again.
- [ ] **Step 3: Write the failing test**

```python
mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.sim

def test_vendored_model_loads():
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    assert m.nu == 6
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE) >= 0
    for j in ARM_JOINTS + (GRIPPER,):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0

def test_model_limits_match_kinematics():  # our LIMITS_RAD and the model agree to 1e-3
    ...
```

- [ ] **Step 4: Run `pytest tests/test_model.py -v`**. Expected: FAIL (`hfteama.arms.model` missing).
- [ ] **Step 5: Implement `model.py`** with the constants above.
- [ ] **Step 6: Run `pytest -v`.** Expected: all pass. If 4.0.1 fails to load the model, switch to 3.17.0 and rerun.
- [ ] **Step 7: Pin the result in `pyproject.toml`**: `sim = ["mujoco>=4.0.1"]`, or `["mujoco>=3.17,<4"]` on fallback. Add `markers = ["sim: needs MuJoCo"]` under `[tool.pytest.ini_options]`. Record the chosen version and the reason in the design doc's MuJoCo decision line.
- [ ] **Step 8: Commit** `Action (#20): vendor SO-101 MuJoCo model (so101-nexus, Apache-2.0); sim extra`

### Task 2: IK on the MuJoCo model

**Files:**
- Create: `src/hfteama/arms/ik.py`
- Test: `tests/test_ik.py`

**Interfaces:**
- Consumes: `MODEL_XML`, `TCP_SITE`, `ARM_JOINTS` (Task 1).
- Produces:
  - `ArmBase(xyz: tuple[float, float, float], yaw: float = 0.0)` (frozen dataclass): the arm base in the table frame. `to_base(p_table) -> np.ndarray`, `to_table(p_base) -> np.ndarray`.
  - `IKResult(q: np.ndarray, error_m: float, ok: bool)` (frozen dataclass; `q` has 5 entries).
  - `Kinematics(base: ArmBase = ArmBase((0, 0, 0)))` with:
    - `fk(q5) -> tuple[np.ndarray, np.ndarray]`: TCP position (table frame) and the TCP z axis (table frame).
    - `solve(xyz_table, seed: np.ndarray | None = None, down: bool = True, tol: float = 0.002) -> IKResult`.

Approach: the model is loaded once per `Kinematics`, with its own `MjData`. The solver uses only `mj_kinematics`, `mj_comPos` and `mj_jacSite`, never `mj_step`. It is damped least squares: damping 0.05, position error plus, when `down`, the alignment of the TCP z axis with (0, 0, -1) weighted 0.1. Joints are clamped to `LIMITS_RAD` after every iteration, with at most 200 iterations. Restart from the seed, the zero pose and 4 random seeds (fixed `np.random.default_rng(0)`), and keep the best. `ok = error_m <= tol`. Credit so101-nexus `kinematics.py` (Apache-2.0) in the module docstring.

- [ ] **Step 1: Write the failing tests**

```python
def test_fk_ik_roundtrip():  # 50 random q in 80% of LIMITS_RAD, rng(1); down=False
    for q in qs:
        p, _ = kin.fk(q)
        r = kin.solve(p, down=False)
        assert r.ok and r.error_m < 0.002

def test_down_pose_points_down():  # table targets x in [0.15, 0.28], y in [-0.1, 0.1], z = 0.03
    r = kin.solve(p, down=True)
    _, z_axis = kin.fk(r.q)
    assert r.ok and z_axis @ np.array([0, 0, -1]) > 0.9

def test_unreachable_is_not_ok():
    assert not kin.solve(np.array([2.0, 0, 0.1])).ok

def test_base_transform():  # ArmBase((0.3, -0.2, 0), yaw=pi/2): to_table(to_base(p)) == p; fk shifts by base
```

- [ ] **Step 2: Run `pytest tests/test_ik.py -v`**. Expected: FAIL (import error).
- [ ] **Step 3: Implement `ik.py`.** If `test_down_pose_points_down` fails only at the edge of the x range, shrink the range in the test to what the arm reaches and note the reachable band in the docstring. That band drives the poses in Task 7.
- [ ] **Step 4: Run `pytest tests/test_ik.py -v`.** Expected: PASS.
- [ ] **Step 5: Commit** `Action (#20): damped-least-squares IK on the SO-101 model`

### Task 3: Primitives protocol and a test driver

**Files:**
- Create: `src/hfteama/arms/primitives.py`
- Create: `tests/fake_driver.py` (in-memory driver for skill and worker tests; no MuJoCo)
- Test: `tests/test_primitives.py`

**Interfaces:**
- Produces, in `primitives.py`:
  - `class MotionError(Exception)`, `class ArmDown(Exception)`, `class SkillFailed(Exception)`. Each takes `reason: str` and keeps it as `.reason`.
  - `@dataclass(frozen=True) class Grasp: holding: bool; position: float; effort: float = 0.0`
  - `class ArmDriver(Protocol)`, with:
    - `arm_id: str`, `table_z: float`
    - `read_joints() -> np.ndarray` (6 entries)
    - `set_joint_target(q: np.ndarray) -> None`
    - `move_joints(q: np.ndarray, speed: float = 1.0) -> None`
    - `move_to(xyz: np.ndarray, speed: float = 1.0) -> None` (gripper pointing down)
    - `open_gripper() -> None`, `close_gripper() -> Grasp`, `holding() -> bool`
    - `stop() -> None`, `park() -> None`, `alive() -> float`
  - `check_joints(q: np.ndarray) -> None`: raises `MotionError("joint limit: <name>")` for the first of the 5 arm joints outside `LIMITS_RAD`. The gripper is checked against its own limit.
  - `check_point(xyz, table_z: float, clearance: float = 0.01) -> None`: raises `MotionError("below table")`.
- Produces, in `tests/fake_driver.py`: `FakeDriver(arm_id="arm_a", move_s=0.0)`. It records `calls: list[tuple[str, tuple]]`. `fail_grasp`, `drop_after_lift`, `down_on: str | None` (method name that raises `ArmDown`) and `hang_on: str | None` (method that blocks until `stop()`). `stop()` makes a hanging call raise `MotionError("stopped")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_check_joints_rejects_each_limit(): ...  # each joint at hi + 0.01 -> MotionError naming it
def test_check_joints_accepts_zero(): check_joints(np.zeros(6))
def test_check_point_below_table(): pytest.raises(MotionError, check_point, [0.2, 0, 0.005], 0.0)
def test_fake_driver_satisfies_protocol(): assert isinstance(FakeDriver(), ArmDriver)  # runtime_checkable
```

- [ ] **Step 2: Run, expect FAIL. Step 3: implement. Step 4: run `pytest tests/test_primitives.py -v`, expect PASS.**
- [ ] **Step 5: Commit** `Action (#20): ArmDriver primitives (Action <-> Hardware interface)`

### Task 4: Poses and generic skills

**Files:**
- Create: `src/hfteama/arms/poses.py`
- Create: `src/hfteama/arms/generic_skills.py`
- Modify: `src/hfteama/arms/skills.py` (docstring only: a skill is `(arm, task, poses) -> None` and raises `SkillFailed(reason)`)
- Test: `tests/test_skills.py`

**Interfaces:**
- Consumes: `ArmDriver`, `SkillFailed`, `MotionError` (Task 3); `skill`, `REGISTRY` (existing).
- Produces:
  - `Poses(locations: dict[str, xyz], per_arm: dict[str, dict[str, xyz]], objects: dict[str, xyz], arms: dict[str, dict], table_z: float = 0.0)`:
    - `Poses.load(path) -> Poses`, `Poses.empty() -> Poses`
    - `point(name: str, arm_id: str) -> np.ndarray`: lookup order `per_arm[arm_id]`, then `locations`, then `objects`. Raises `SkillFailed(f"unknown location: {name}")`.
  - `has(name: str, arm_id: str) -> bool`: the same lookup, without raising.
    - The `arms` field holds sim base poses: `{"arm_a": {"base": [x, y, z], "yaw": rad}}`.
  - `pick_point(task, poses, arm_id)`: `args["pick_xyz"]`, else `point(args["object"])`.
  - `place_point(task, poses, arm_id)`: `args["place_xyz"]`, else `point(args.get("target") or args["to"])`.
  - Constants: `APPROACH_DZ = 0.08`, `LIFT_DZ = 0.10`, `GRASP_DZ = 0.0`.
  - Registered skills:
    - `pick`: `open_gripper`; `move_to` above the pick point; `move_to` down to it; `close_gripper`, then `SkillFailed("grasp")` if not holding; `move_to` up; `holding()`, then `SkillFailed("dropped")` if not.
    - `place`: `move_to` above the place point; `move_to` down to it; `open_gripper`; `move_to` back up.
    - `pick_place`: pick, then place.
    - `inspect`: `move_to(point("inspect"))` if that name exists, else above the pick point; hold 0.5 s.
    - `park`: `arm.park()`.

  Importing `generic_skills` registers the skills. Re-importing must not raise: guard with `if name not in REGISTRY`.

- [ ] **Step 1: Write the failing tests** (all on `FakeDriver`, no MuJoCo)

```python
def test_pick_call_order(): ...      # open, move_to(above), move_to(at), close, move_to(above), holding
def test_pick_uses_pick_xyz_over_object(): ...   # args has both; move_to targets pick_xyz
def test_place_resolves_own_bin_per_arm(): ...   # per_arm["arm_b"]["own_output_bin"] used for arm_b
def test_place_accepts_target_or_to(): ...
def test_grasp_failure_raises(): ...  # FakeDriver(fail_grasp=True) -> SkillFailed("grasp")
def test_drop_after_lift_raises(): ...  # -> SkillFailed("dropped")
def test_unknown_location(): ...      # {"object": "mystery"} -> SkillFailed("unknown location: mystery")
def test_registered(): assert {"pick", "place", "pick_place", "inspect", "park"} <= set(REGISTRY)
```

- [ ] **Step 2: Run, expect FAIL. Step 3: implement. Step 4: run `pytest tests/test_skills.py -v`, expect PASS.**
- [ ] **Step 5: Commit** `Action (#20): poses and generic skills (pick, place, pick_place, inspect, park)`

### Task 5: SkillWorker

**Files:**
- Create: `src/hfteama/arms/worker.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `ArmWorker`, `ArmState`, `Outcome`, `Result` (frozen `base.py`); `REGISTRY`; `ArmDriver`, `SkillFailed`, `MotionError`, `ArmDown`; `Poses`; imports `generic_skills`.
- Produces: `SkillWorker(arm_id: str, driver: ArmDriver, poses: Poses, backend: str, timeout_factor: float = 1.5, min_timeout_s: float = 5.0, clock=time.time)`, implementing `ArmWorker`.
  - The deadline is `max(min_timeout_s, timeout_factor * task.duration_on(arm_id))`.
  - `execute` runs the skill in a helper thread and joins until the deadline:
    - return → `OK`
    - `SkillFailed` / `MotionError` → `FAILED` with the exception's reason
    - `ArmDown` → `ARM_DOWN`, and the state becomes `DOWN`
    - unknown skill → `FAILED("unknown skill: <name>")`
    - any other exception → `FAILED(repr(e))`
    - deadline passed → `driver.stop()`, join for up to 2 s, then `TIMEOUT`
  - `state()`: `READY` after construction; `BUSY` during `execute`; `PARKING` during `park()`; `DOWN` after `ArmDown`.
  - `cancel(task_id)`: if `task_id` is the running task, set a flag and call `driver.stop()`; that `execute` returns `FAILED("cancelled")`. Otherwise do nothing.
  - `heartbeat()` returns `driver.alive()`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ok_result_has_times(): ...       # outcome OK, started_at <= finished_at, task_id/arm_id set
def test_skill_failed_maps_reason(): ...  # fail_grasp -> FAILED, reason "grasp"
def test_unknown_skill(): ...
def test_timeout_stops_arm(): ...         # FakeDriver(hang_on="move_to"), min_timeout_s=0.2, duration 0.1
                                          # -> TIMEOUT within 3 s; "stop" in calls
def test_arm_down_is_sticky(): ...        # down_on="close_gripper" -> ARM_DOWN; next execute ARM_DOWN, no calls
def test_cancel_other_task_is_noop(): ... # cancel("other") while idle and while running: no stop()
def test_cancel_running_task(): ...       # hang_on="move_to"; cancel(task.id) from a thread -> FAILED "cancelled"
def test_never_raises(): ...              # skill that raises ZeroDivisionError -> FAILED
def test_satisfies_contract(): assert isinstance(worker, ArmWorker)
```

- [ ] **Step 2: Run, expect FAIL. Step 3: implement. Step 4: run `pytest tests/test_worker.py -v`, expect PASS.**
- [ ] **Step 5: Commit** `Action (#20): SkillWorker implements the ArmWorker contract`

### Task 6: SimTable and MujocoDriver (option B grasp)

**Files:**
- Create: `src/hfteama/arms/sim.py`
- Test: `tests/test_sim.py` (`sim` marker)

**Interfaces:**
- Consumes: `MODEL_XML`, `TCP_SITE` (Task 1); `Kinematics`, `ArmBase` (Task 2); `ArmDriver`, `Grasp`, `MotionError`, `ArmDown`, `check_joints`, `check_point` (Task 3); `Poses` (Task 4).
- Produces:
  - `SimTable(poses: Poses, arm_ids: list[str] | None = None, speed: float = 1.0, viewer: bool = False)`.
    - Arm ids default to the keys of `poses.arms`. If `poses.arms` is empty, use 3 arms `arm_a`..`arm_c`, 0.45 m apart along x, yaw π/2.
    - Context manager: `__enter__` starts the physics thread; `__exit__` stops it and closes the viewer.
    - `driver(arm_id) -> MujocoDriver`
    - `inject(arm_id, kind: Literal["grasp", "down", "slow", "clear"])`
    - `object_pos(name) -> np.ndarray`
    - `reset()`: objects back to their `poses.json` positions, attachments cleared, arms to the zero pose.
  - `MujocoDriver`: implements `ArmDriver` for one arm of the table, with `table_z = poses.table_z`.

Scene: built with `MjSpec`. It has a floor plane at `table_z`. Each arm is attached with `frame.attach_body(..., prefix=f"{arm_id}/")` at its base pose. Each object in `poses.objects` is a free-joint box with 0.015 m half-size.

The physics thread holds one `threading.Lock` per `mj_step`. It steps at `speed` × real time: `timestep` from the model, sleeping off any slack. Each driver has its own `Kinematics(ArmBase(...))`.

Driver behaviour:
- `move_joints` interpolates in joint space at `1.5 * speed_arg` rad/s, at a 50 Hz command rate. It returns when every joint is within 0.02 rad of the target. It raises `MotionError("stall")` if not settled 3 s after the interpolation ends, and `MotionError("stopped")` after `stop()`.
- `move_to` = `check_point`, then `solve(down=True)` (`MotionError("unreachable")` if not ok), then `move_joints`.
- Gripper targets: open 1.2 rad, closed -0.1 rad. `close_gripper` waits 0.4 s, then attaches the nearest object within 0.03 m of the TCP, unless `grasp` was injected (consumed by one close).
- An attached object is carried kinematically: after each step, its free-joint `qpos` is set to the TCP position with zero velocity. `open_gripper` releases it.
- `down` makes every call raise `ArmDown("injected")`. `slow` scales joint speed by 0.3.
- `alive()` returns the wall time of the last physics step.

- [ ] **Step 1: Write the failing tests** (`speed=4.0`)

```python
def test_move_to_reaches(): ...           # tcp within 0.005 m of a reachable point
def test_move_to_rejects_unreachable_and_below_table(): ...  # MotionError, joints unchanged
def test_pick_place_moves_block(): ...    # run generic pick then place via the registry on one arm;
                                          # block ends within 0.03 m (x, y) of the place point
def test_injected_grasp_failure(): ...    # close_gripper().holding is False
def test_injected_down(): ...             # ArmDown from move_to
def test_three_arms_move_in_parallel(): ...  # 3 threads move_to at once; wall time < 1.6x a single move
def test_set_joint_target_nonblocking(): ...  # returns in < 0.05 s; joints converge within 2 s
def test_reset_restores_objects(): ...
```

- [ ] **Step 2: Run `pytest tests/test_sim.py -v`.** Expected: FAIL.
- [ ] **Step 3: Implement `sim.py`.**
- [ ] **Step 4: Run `pytest -v`.** Expected: all pass. The sim tests take under 60 s.
- [ ] **Step 5: Commit** `Action (#20): MuJoCo SimTable and MujocoDriver (option B grasp)`

### Task 7: sort12 on 3 sim arms, and measured durations

**Files:**
- Create: `tasks/sort/poses.json`
- Create: `scripts/sim_demo.py`
- Create: `scripts/measure_durations.py`
- Test: `tests/test_sort_poses.py` (`sim` marker), `tests/test_sim_demo.py` (`sim` marker)

**Interfaces:**
- Consumes: `SimTable`, `SkillWorker`, `Poses`, `Kinematics`, `ArmBase`; `taskgraph.load`; `events.EventLog`, `Event`.
- Produces:
  - `tasks/sort/poses.json` layout:
    - `input_bin` at the origin; 12 objects `block_01`..`block_12` in a 4×3 grid, 0.035 m pitch, centred on it, z = 0.015.
    - 3 arms: base 0.26 m from the bin centre at angles 210°, 270° and 330°, yaw facing the bin.
    - `per_arm.<arm>.own_output_bin` 0.2 m from that arm's base, rotated 70° away from the bin.
    - Tune the numbers until `test_sort_poses.py` passes.
  - `scripts/sim_demo.py` exposes `main(argv: list[str] | None = None) -> int`, with options `[--task tasks/sort/sort12.json] [--speed 1] [--viewer] [--fail arm_b:grasp] [--events runs/sim_demo/events.jsonl]`. It is a test harness, not an executor: one thread per arm takes the next unstarted `pick` from a shared list and then runs the matching `place` (`same_arm_as`). A lock is held for each task's `resources` (the `input_bin`). It writes `task_start` / `task_end` events with the outcome and prints a summary table.
    - A failed pick goes back to the list once.
    - Exit code is 0 only if every task ends `OK`.
  - `scripts/measure_durations.py` exposes `main(argv) -> int`, with options `[--cycles 5] [--out runs/durations/sort_sim.json]`: on each arm, runs `cycles` pick/place cycles on the sort poses, resetting the scene between cycles. It writes `{"backend": "sim", "skills": {"pick": {"arm_a": {"n": 5, "p50": s, "p80": s}}, ...}}` and prints the table.

- [ ] **Step 1: Write the failing tests**

```python
def test_sort_poses_reachable():  # every object, input_bin, and own_output_bin of each arm:
    # Kinematics(ArmBase(...)).solve(p + [0, 0, dz], down=True).ok for dz in (0.0, APPROACH_DZ, LIFT_DZ)

def test_sim_demo_runs_sort12(tmp_path):  # main(["--speed", "4", "--events", str(p)]) == 0;
    # 24 task_end events with outcome "ok"; at least 2 distinct arms in them

def test_sim_demo_recovers_from_grasp_failure(tmp_path):  # --fail arm_b:grasp -> still exit 0;
    # one task_end "failed" with reason "grasp" in data, then the same task "ok"
```

- [ ] **Step 2: Run, expect FAIL. Step 3: write `poses.json` and both scripts. Step 4: run `pytest -v`, expect PASS.**
- [ ] **Step 5: Run `python scripts/sim_demo.py --viewer`.** Watch the 3 arms sort the 12 blocks. Expected: exit 0, summary table printed.
- [ ] **Step 6: Run `python scripts/measure_durations.py`.** Expected: a JSON file with p50/p80 for `pick` and `place` per arm. Paste the table into `ACTION_STATUS.md`.
- [ ] **Step 7: Commit** `Action (#20): sort12 on 3 MuJoCo arms; measured sim durations`

### Task 8: Real-arm backend (LeRobot)

**Files:**
- Create: `src/hfteama/arms/real.py`
- Create: `configs/real_arms.example.json`
- Test: `tests/test_real.py` (fake robot object, no LeRobot)

**Interfaces:**
- Consumes: `ArmFrame` (`kinematics.py`); `Kinematics`, `ArmBase`; primitives; `hfteama.hardware.pairs.load_pair` / `PairConfig` (Hardware's loader).
- Produces:
  - `LeRobotDriver(arm_id: str, robot, frame: ArmFrame, kin: Kinematics, table_z: float, park_q: np.ndarray, grip_holding_min_pct: float, max_step_rad: float = 0.05, rate_hz: float = 30.0)`, implementing `ArmDriver`.
    - `robot` is any object with `get_observation() -> dict`, `send_action(dict)`, `disconnect()` and `bus.disable_torque()`. In practice that's a LeRobot SO follower.
    - Every command passes `check_joints` (and `check_point` for `move_to`).
    - Moves are interpolated so no joint changes more than `max_step_rad` per tick.
    - `close_gripper` reads the gripper after 0.3 s: `holding = gripper.pos >= grip_holding_min_pct`. The gripper stalled on an object means holding; nearly closed means empty.
    - `park()` = `move_joints(park_q, speed=0.3)`, then `bus.disable_torque()`.
    - Any exception from `get_observation` / `send_action` raises `ArmDown("bus: <exc>")`.
  - `from_config(arm_id: str, path="configs/real_arms.json") -> LeRobotDriver`. The JSON holds `pair`, base `xyz`/`yaw`, `table_z`, `park_q`, `grip_holding_min_pct` and optional `offsets_deg`. It gets the port and calibration from `load_pair(pair=...)`. LeRobot is imported inside this function: `from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig`. Check those names against the installed LeRobot (Sahil's `vla/setup.sh`, 0.7.x) and adjust if they differ. This function is not unit-tested.

- [ ] **Step 1: Write the failing tests** (against a `FakeRobot` that records actions and returns a settable observation)

```python
def test_move_joints_steps_never_exceed_max_step(): ...
def test_rejects_limit_and_below_table_before_sending(): ...  # no send_action recorded
def test_commands_are_converted_with_armframe(): ...  # elbow command includes the 6.9 deg offset
def test_grasp_threshold(): ...      # gripper.pos 30 -> holding (min 15); 2 -> not holding
def test_bus_error_is_arm_down(): ...
def test_park_ends_with_torque_off(): ...  # last call is disable_torque, after reaching park_q
```

- [ ] **Step 2: Run, expect FAIL. Step 3: implement. Step 4: run `pytest tests/test_real.py -v`, expect PASS.**
- [ ] **Step 5: Add `--backend real --arms arm_a[,arm_b]` to `scripts/measure_durations.py`.** It builds drivers with `from_config`, never runs in tests, and prints a warning plus a 5 s countdown before the first motion. This is the command Hardware runs in the handoff checklist.
- [ ] **Step 6: Commit** `Action (#20): LeRobot real-arm driver (mocked-bus tests only)`

### Task 9: Handoff to Hardware, close-out

**Files:**
- Create: `docs/handoff/action-to-hardware.md`
- Modify: `ACTION_STATUS.md`; `STATUS.md` on `main` (Vivek's block, `Status: Vivek` commit)

- [ ] **Step 1: Write the handoff doc.** It covers:
  - what `LeRobotDriver` does and the safety checks it applies;
  - what Hardware fills in `configs/real_arms.json`: the pair name, base pose in the table frame, `table_z`, a safe `park_q`, and `grip_holding_min_pct`, measured with 10 empty and 10 holding closes per arm;
  - a staged checklist: (1) torque off, read joints only and compare with the sim pose; (2) one joint, ±0.1 rad, `speed=0.3`; (3) `move_to` above the input bin with nothing grasped; (4) one `pick` / `place`; (5) `measure_durations.py --backend real`;
  - and the slow-stop rule.
- [ ] **Step 2: Draft the #19 / #20 comment** proposing the primitives list (the `ArmDriver` signatures) and the arg names `pick_xyz` / `place_xyz`, in the handoff doc's last section. **Do not post it.** Vivek posts it.
- [ ] **Step 3: Run `pytest -v && ruff check src tests scripts`.** Expected: all pass, no lint errors (the `vla/` errors are Sahil's).
- [ ] **Step 4: Update `ACTION_STATUS.md` and Vivek's block in `STATUS.md` on `main`. Commit both.**
