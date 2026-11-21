# Action -> Hardware handoff: running the scripted skills on a real SO-101

For Diego and Om (#19). From Vivek (Action, #20). Branch `action-arm-skills`; status in
`ACTION_STATUS.md`. **Nothing here has moved a real arm yet.** Everything below was proven on
MuJoCo arms (24/24 sort steps on 3 arms, `scripts/sim_demo.py`) and on a fake robot in unit tests.
The first real motion is yours, in the order of the checklist.

## Read this first (status, Sat 21 Nov, 17:50)

**Ready:** stage 0 and 1 below (config, conversion, read-only). **Not yet proven:** any motion on a real
arm. The code path is proven only on MuJoCo and a fake robot, so the first motion is a supervised,
staged one: Diego or Om at the rig, one arm, `speed_scale` 0.3, hand at the power.

What is proven (199 -> 248 tests green, all without hardware):
- Operations' real `Executor` runs sort12 on 3 simulated arms through our `SkillWorker`: 24/24 steps ok
  (`tests/test_executor_with_sim_arms.py`). Contract, timeouts, resource locks and retries all work end to end.
- An independent review found real-arm problems (stop forgotten, sagging commands, bus races, idle
  heartbeat); all fixed with tests. The sim run also exposed that `pick` must leave the shared area
  before it returns (the executor releases the lock then): `pick` now ends at a per-arm `carry` pose.

What we still need from you before stage 2 (nothing here is guessable):
1. `configs/real_arms.json`, **keyed by catalog id** (`rakesh_follower`, `vato_follower`), not `arm_a`:
   Aravind's reach graph and the executor use that id. Template: `configs/real_arms.example.json`.
2. A real poses file for the table: copy `tasks/sort/poses_real.example.json` (its numbers are the SIM
   layout and are wrong): input bin, each arm's `own_output_bin` and `carry` (above its own bin, clear of
   the shared input area), object spots. Pass it as `Poses.load(...)` in stage 4 and `--poses` in stage 5.
3. Base poses from Aravind's marker touch (touch two table markers with each gripper) in the same frame.
4. A parked-pose and gripper threshold per arm (table below).

Known gaps: the grasp angle is not controlled (the tool points straight down, wrist roll is not solved), so
cubes are fine and long objects are not (Operations' blocker #4: `pick_yaw_deg`, not built). Windows
laptop: the driver uses no Linux paths itself; ports come from `hfteama.hardware.pairs`. LeRobot import
names are checked against 0.6.2 source only.

## What you get

`hfteama.arms.real.LeRobotDriver`: the same primitives the simulator implements
(`hfteama.arms.primitives.ArmDriver`), on top of a LeRobot `SOFollower`. Skills (`pick`, `place`,
`pick_place`, `inspect`, `park`) call only these primitives, so a step that worked in sim runs
unchanged on the arm.

| Primitive | What it does on the real arm |
|---|---|
| `read_joints()` | 6 values (5 joints + gripper), radians, model convention (`kinematics.ArmFrame`) |
| `move_joints(q, speed)` | interpolated; no joint moves more than 0.05 rad per 30 Hz tick; waits to settle |
| `move_to(xyz, speed)` | IK (same MuJoCo model as the sim), gripper pointing down, then `move_joints` |
| `open_gripper()` / `close_gripper()` | close returns `Grasp(holding=...)`: stalled above `grip_holding_min_pct` = holding |
| `holding()` | closed on the last command and still above the threshold |
| `set_joint_target(q)` | one rate-limited step, non-blocking (for a learned policy at 30 Hz) |
| `stop()` | holds the last commanded pose (grip included) and latches: a blocked move, and every later one, raises `MotionError("stopped")` until `resume()` (the worker calls it at the start of each step) |
| `reachable(xyz)` | whether `move_to(xyz)` would be accepted; never moves. Skills check every waypoint of a step with it before the first motion |
| `park()` | slow move (0.3 speed) to `park_q`, then torque off. Never a torque cut mid-air. **The arm is then limp and the driver refuses motion (`torque off`) until it is reconnected** |
| `alive()` | time of the last good bus traffic; an idle arm is polled so the executor's 1.5 s heartbeat check does not declare it lost |

Safety checks run in the driver, so no skill can skip them: joint limits (`kinematics.LIMITS_RAD`),
target at least 1 cm above `table_z`, unreachable targets refused before anything is sent, a bus
error becomes `ArmDown` (the worker then latches the arm as DOWN), and `max_relative_target_deg`
(default 10) is set on the LeRobot config as a second limit. Commands are built on the last
*commanded* pose, not the measured one, so the arm does not sag lower or lose its grip when a gripper
command or a stop is sent. All bus access is behind one lock.

## Install

```bash
pip install -e ".[dev,sim]"          # sim = MuJoCo, needed for IK even on the real arm
# LeRobot from source (see vla/setup.sh); then:
python -c "from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig"
```

`from_config` imports those two names. I checked them against LeRobot 0.6.2 source, not against the
0.7.x install: **if the import fails, that line in `arms/real.py::from_config` is the only thing to fix.**

## What we need from you: `configs/real_arms.json`

Copy `configs/real_arms.example.json`. One entry per arm:

| Field | Meaning | How to get it |
|---|---|---|
| `pair` | name in `configs/teleop_pairs.json`; port, id and calibration come from there | existing |
| `base` | `[x, y, z]` m of the arm's base in the shared table frame | measure from the table marks / ChArUco; **same frame for all arms** |
| `yaw` | rotation (rad) of the arm's forward axis about z, from table +x | the direction the arm points at zero pan |
| `table_z` | table height in the table frame (m) | 0.0 if z is measured from the table |
| `park_q` | 5 joint angles (rad) of a safe, low, folded pose that stays out of every shared area | move the arm there by hand (torque off), read `read_joints()`; keep inside `kinematics.LIMITS_RAD` |
| `grip_holding_min_pct` | gripper reading above which a closed gripper counts as holding | 10 empty closes and 10 closes on the block: read `gripper.pos` 0.3 s after closing; pick a value between the two groups, per arm |
| `speed_scale` | global slow-down of every move (`0.3` = 30 %). **Keep 0.3 until stage 4 is clean**; skills otherwise run at about 1.5 rad/s | start at 0.3, raise it by hand later |
| `settle_tol_deg` | how far from its goal a joint may sit when a move ends before it counts as `stall` (default 5.7) | STS3215s typically sit 2-4 deg off under load; if moves end in `stall`, check the arm, then raise this |
| `offsets_deg` | per-joint zero offsets (LeRobot degrees vs model) | defaults assume the elbow offset measured on one arm (6.9 deg, `kinematics.py`); check each arm: command a known pose and compare |

The sim's folded pose (`arms/sim.py::REST_Q`) is only a starting point for `park_q`; the real one
must be checked by hand on each arm.

Frame conventions: the table frame is metres, z up. An arm's own frame has +x straight ahead at
zero pan and z up. With the tool pointing down the arm reaches 0.10-0.30 m from its base at table
height, and 0.18-0.32 m at the 11.5 cm lift height the skills use (nothing behind the base, pan within
the limits). Put blocks and bins inside that band for every arm that must reach them.
`tasks/sort/poses.json` is the simulated layout; replace `arms` and the locations with measured values.

## Staged checklist (stop at the first thing that surprises you)

Stage 0 has no motion at all. Do each stage on one arm, with the area clear and a hand at the power.
Connecting enables torque and the arm holds its current pose: **support it as you connect.**

0. **Config and conversion, no arm.** `pytest tests/test_real.py tests/test_ik.py`. Fill
   `configs/real_arms.json`.
1. **Read only.** Connect, print, compare against what you see:
   ```python
   from hfteama.arms.real import from_config
   d = from_config("rakesh_follower")
   print(d.read_joints(), d.alive())
   ```
   To compare against the real arm, move it by hand: **support the arm with one hand, then** switch torque
   off with `d._robot.bus.disable_torque()` (an arm that is raised drops when torque goes off), move it, and
   read again. Does each joint match what you expect in sign and size? Does the gripper reading go up when
   open? Reconnect (`from_config` again) before any commanded motion. If an offset is off, set
   `offsets_deg`; if a sign is flipped, tell us (`ArmFrame.signs` is not settable from the JSON yet).
2. **One joint, small and slow.** `q = d.read_joints()[:5]`, then `d.move_joints(q + [0.1, 0, 0, 0, 0],
   speed=0.3)` and back, for each joint in turn. (`read_joints()` returns 6 values; `move_joints` takes the 5
   body joints, or 6 with the gripper.)
3. **A point in the air, then a staged descent.** `d.move_to([x, y, 0.12], speed=0.3)` above the input bin
   with nothing in the gripper. Then lower one stage at a time: z = 0.10, 0.05, 0.03 m. **At each stage measure
   the real height of the fingertips against the commanded z before going on** (the elbow offset was
   measured on one arm only: 1 deg of error at 0.25 m reach is about 4 mm). `z = 0` is refused as below the
   table. Stop at the first stage that is off by more than a few mm and fix `offsets_deg` or `table_z`.
4. **One step.** One block at its marked spot:
   ```python
   from hfteama.arms import generic_skills
   from hfteama.arms.poses import Poses
   from hfteama.arms.worker import SkillWorker
   from hfteama.taskgraph import Task
   w = SkillWorker("rakesh_follower", d, Poses.load("tasks/sort/poses_real.json"), "real")
   print(w.execute(Task(id="t1", skill="pick", args={"object": "block_01", "from": "input_bin"})))
   print(w.execute(Task(id="t2", skill="place", args={"to": "own_output_bin"})))
   ```
   Expect `outcome=ok`. A miss returns `failed` with reason `grasp` and the arm lifts clear; any other
   motion failure also lifts back to approach height before returning. Keep `speed_scale` at 0.3 for this stage.
   After a `park` step or `worker.park()` the arm is limp: reconnect it before the next step.
5. **Durations.** `python scripts/measure_durations.py --backend real --arms rakesh_follower --poses tasks/sort/poses_real.json --cycles 3`
   asks you to type `yes`, then to put the block at its spot before every cycle. Send us
   `runs/durations/sort_real.json`; Operations plans with the p80 numbers.

## Stopping and removing an arm

- `driver.stop()` freezes the arm where it is; the running step ends with `failed` / `cancelled`.
- `worker.park()` aborts the running step, moves to `park_q` slowly, then torque off. Use it before
  unplugging an arm.
- Do not kill the process that is driving an arm (no Ctrl-C or `kill` mid-motion): the arm stops where
  it is, possibly raised, with torque on.

## Not verified

- No real arm has run any of this. IK, interpolation and skills are proven on MuJoCo only.
- The gripper map (LeRobot 0 to 100 onto the model's gripper range) is linear by assumption
  (`kinematics.py`); the threshold in `real_arms.json` is what actually decides `holding`.
- Sim grasping attaches the block when the gripper closes within 3 cm of it. A real grasp can miss
  or slip in ways the sim does not show; expect to tune approach heights (`APPROACH_DZ`, `LIFT_DZ`,
  `GRASP_DZ` in `arms/generic_skills.py`).
- Two arms sharing the bin is Operations' job (resource lock), not the driver's. What the worker
  guarantees for that lock: `execute` does not return TIMEOUT or `cancelled` until the stop is latched in
  the driver, and `worker.park()` only parks once the step's thread is confirmed dead (otherwise it leaves
  the arm frozen and sets `park_error`).
- `park_q`, `table_z` and the base pose are only as good as your measurements: the table check covers the
  tool centre only, not the jaw tips or the links, and not the path between two joint targets.

## Draft comment for #19 and #20 (Vivek posts it)

> **Primitives for the real arm (Action -> Hardware).** The skills only call these methods, so this
> is the whole interface Hardware has to make safe: `read_joints`, `move_joints`, `move_to` (tool
> down), `reachable`, `open_gripper`, `close_gripper -> Grasp(holding)`, `holding`,
> `set_joint_target`, `stop`, `resume`, `park`, `alive` (`src/hfteama/arms/primitives.py`). A real implementation is on branch
> `action-arm-skills` (`arms/real.py`), unrun on hardware. Handoff and a staged checklist:
> `docs/handoff/action-to-hardware.md`. What we need back: `configs/real_arms.json` (base poses in the
> shared table frame, a safe park pose, gripper threshold per arm) and a first real step.
>
> **Step arguments (Action -> Operations).** Skills take table coordinates in metres:
> `pick_xyz` and `place_xyz`; or names resolved in `tasks/<name>/poses.json`: `object` / `from` to
> pick, `target` / `to` to place. A failed step returns `failed` with a reason (`grasp`,
> `dropped`, `unreachable`, `below table`, `unknown location: <name>`).
