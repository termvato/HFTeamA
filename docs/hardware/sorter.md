# Vision-guided IK sorter

An overhead camera finds the small pieces and the round box on the table. Each SO-101 follower
picks the pieces on its side with inverse kinematics (no learned policy) and drops them into
the box. The arms take turns, so they never meet over the box.

```
hfteama sort preview            # camera only: what it sees and what it would do
hfteama sort goto rakesh --tag 2  # hover rakesh's tool tip 3 cm over tag 2's centre (its back-right tag)
hfteama sort goto rakesh --box    # hover over the box (the drop pose)
hfteama sort jog rakesh           # hover over a piece, correct by hand, fit where rakesh stands
hfteama sort touch rakesh         # fit where rakesh stands (torque off, touch tags)
hfteama sort run                # sort, both arms
```

## Setup

| What | Where |
|---|---|
| Overhead camera: Logitech C922, found by name, 1920x1080 MJPG, focus/exposure/white balance pinned | `configs/sorter.json` → `camera` |
| Camera intrinsics (from group-6 KapplaBuild, 41 ChArUco images, rms 0.54 px) | `configs/camera/overhead_c922_1920x1080.yml` |
| Reference tags: ArUco `DICT_4X4_50`, ids 0-3, centres on a 73.4 x 33.5 cm rectangle | `table.tags` |
| Box: open cylinder, 18 cm diameter, 5.2 cm high | `box` |
| Arms, named after their pair: `rakesh` (black), `hacklab` (green); base position and heading | `arms`, then `configs/sorter_placements.json` |

**Arm names, not left/right:** seen from behind the arms (facing the camera) rakesh is on
the right and hacklab on the left; seen from the camera it is the other way round.

**Table frame, seen from behind the arms (facing the camera):** origin at the centre of
tag 1 (back-left), x to the right along the back tags, y forward towards the camera, z up,
metres. Tag centres are 73.4 cm apart along x and 33.5 cm along y. The arms' base_link origins
are 1.5 cm behind the back tags (y = -0.015) and face +y (`yaw_deg` 90).

```
            hacklab (green)                     rakesh (black)
            base (0.163, -0.015)                base (0.571, -0.015)
   tag 1 (0, 0) ---------------------------------------- tag 2 (0.734, 0)
        |              pieces ...      ... pieces               |
        |                         (box)                         |
   tag 3 (0, 0.335) ------------------------------------ tag 0 (0.734, 0.335)
                                 camera
```

The camera pose comes from the tag centres (3 visible tags are enough; with 4 the residual is a
check). It does not depend on the tag size.

## Reach

The vertical-gripper workspace of an SO-101 at table height is about 5-30 cm from the base;
higher up it shrinks (the wrist joint runs out at about ±100°). So:

- Pieces up to about 30 cm from a base are picked with the gripper vertical.
- The drop is made with the gripper tilted. With the box centred between the arms
  (x ≈ 0.365), both arms reach it when its centre is at **y ≤ 0.24 m**, about 25 cm in front
  of the arms. `hfteama sort preview` prints whether each arm can reach it.

## First time on the real arms

Leave both arms folded at rest before every command: that pose is "home", and the arm goes
back to it before torque is released. The calibration is never touched. If the calibration
file and the servos differ, the command stops instead of letting LeRobot prompt.

1. `hfteama sort preview` (or `--live`, which streams to Rerun: camera, top-down view, piece mask and the pose table). Check that the box circle sits on the rim, that every
   piece is outlined and coloured by arm (blue = rakesh, green = hacklab, red = unreachable), and
   that the white jaw lines cross each piece's short side. The image is saved in
   `outputs/sorter/`.
2. `hfteama -n sort goto rakesh --tag 2` plans only. Then without `-n`: the tip should hover
   3 cm over the centre of tag 2. Measure the offset. Do the same with `hacklab --tag 1` (back-left).
3. If it is off by more than ~1 cm: `hfteama sort touch rakesh`. Torque goes off (hold the arm).
   Touch 3 or more tag centres with the tip of the closed gripper: tags 2 and 0 for rakesh (on its side),
   plus any extra `DICT_4X4_50` tag (id ≥ 4) you lay on the table, spread over the work
   area. The fit is saved to `configs/sorter_placements.json`.
   Or, with torque on: `hfteama sort jog rakesh`. Rerun opens with the camera: pieces
   numbered, the aim point (red circle) and the tool tip from the encoders (yellow cross),
   both drawn on the table. Type a piece number to hover 5 cm over it; nudge with f/b/l/r/u/d
   (seen from behind the arm) until the tip is over the piece's centre; `ok` records the
   camera-vs-arm offset; `g` grasps there, `box` drops it, `scan` goes home and renumbers.
   `q` fits the placement from the `ok` points (2-3 spread out) and offers to save it.
4. `hfteama -n sort run` prints the plan from one frame. Then `hfteama sort run --arms rakesh
   --max-picks 1`, then both arms.

## Tuning (`configs/sorter.json`)

| Key | Effect |
|---|---|
| `pieces.colour_threshold` | Lab distance from the table colour that counts as a piece; raise it if wood grain shows up |
| `pieces.min_area_cm2` / `max_area_cm2` | Blob size filter |
| `pieces.grasp_z`, `approach_z` | TCP height at the grasp and above the piece |
| `box.drop_above` | Height above the rim for the drop |
| `motion.max_joint_speed_deg_s` | Joint speed |
| `motion.gripper_open` / `gripper_closed` | Gripper (LeRobot 0..100) |
| `motion.grasp_min_open` | A grasp counts as missed if the gripper closes to within this of `gripper_closed` |
| `arms.<side>.offsets_deg` | Per-arm joint zero offsets for `ArmFrame` (default: elbow 6.9°) |

## Code

| Module | |
|---|---|
| `hfteama.arms.so101_ik` | FK/IK in pure numpy (chain from `so101_new_calib.urdf`): position first, then the gripper as vertical as possible; jaw yaw; limits from the calibration files |
| `hfteama.vision.table` | Camera pose from the tags, pixel ↔ table plane |
| `hfteama.vision.detect` | Box (Hough circle at rim height), pieces (colour vs. table, tags/arms/box masked) |
| `hfteama.vision.camera` | Opens the camera by name with pinned controls |
| `hfteama.sorter.geometry` | Arm placement on the table and fitting it from touched points |
| `hfteama.sorter.plan` | Which arm picks which piece, grasp and drop poses |
| `hfteama.sorter.robot` | LeRobot follower driven by joint targets, smooth and speed-limited |
| `hfteama.sorter.run` | The `hfteama sort` commands |

## With main's orchestration (executor, planner, skills)

The sorter can drive the arms for Operations' executor and Action's skills (`pick`, `place`,
`pick_place`) instead of `arms/real.py`. Nothing in the executor or the skills changes:

| Piece | What it is |
|---|---|
| `hfteama.sorter.bridge.SorterArmDriver` | Action's `ArmDriver` primitives (`move_to`, `open_gripper`, `close_gripper`, `reachable`, `stop`/`resume`, `park`, `alive`, ...) over a sorter arm: calibrated numpy IK and joint limits, the per-arm aim corrections from `configs/sorter_placements.json`, closed-loop gripper, one smooth vertical line when only the height changes, stop latched every control tick, home pose before torque off |
| `MainFrame` | main's table frame (`configs/hacklab_table.json`: origin marker 0, x towards 3, y towards 2) <-> the sorter's; the same four tags, so the conversion is exact |
| `hfteama sort export` | camera -> `poses_real.json` (boxes by tag, the bowl, each arm's drop point and `carry` pose) in main's frame + `boxes.json`: per box a `pick` (lock: `half_<arm>` of the table it is in) then a `place` in the bowl (lock: `bowl`, same arm). `allowed_arms` = every arm that reaches the box, so main's planner re-splits when an arm comes or goes; the other half's arm is planned 1.5x slower, so each arm keeps its own half while both are up |
| `python -m hfteama.runner --backend sorter` | main's runner with these drivers (same `yes` / `go` confirmations as `--backend real`) |

```
env -u PYTHONPATH .venv/bin/hfteama sort export                     # look, write tasks/sorter_boxes/
env -u PYTHONPATH .venv/bin/python -m hfteama.runner --backend sorter \
    --arms rakesh_follower,vato_follower \
    --task tasks/sorter_boxes/boxes.json --poses tasks/sorter_boxes/poses_real.json
```

`move_to(p)` takes the point where the gripped piece's centre should be; the arm's aim correction
turns it into the tool command, exactly as `hfteama sort run` does.

Over the bowl: main's `pick_place` goes to the place point + 8 cm, then down to it, opens and goes
back up. From the arms' bases the bowl centre is too far for the gripper to point down, so:

- `sort export` writes a per-arm `bowl` (`per_arm` in `poses_real.json`) at the sorter's drop
  point: moved from the bowl centre towards the arm (up to 3 cm inside the rim) until the gripper
  reaches nearly straight down, `--place-above` (default 3 cm) over the rim. Measured offline with
  the current calibration: 12 deg tilt for rakesh, 5 deg for hacklab at the place point.
- With a box in the jaws, a move to a new spot is solved as the sorter's drop
  (`plan.drop_solve`): gripper down within `max_tilt_deg`, else within `drop_max_tilt_deg`, else
  any orientation. The approach 8 cm higher tilts 50-60 deg (in the air, above the rim).
- A straight vertical line is used only between two gripper-down poses; to or from a tilted pose
  the move is a joint move (the line's IK keeps the gripper down and would fail on the way).

Two arms at once:

- Picks run in parallel (one arm per half of the table: the `half_*` locks); places take turns
  at the bowl. The arm leaves the half / the bowl for its `carry` pose (15 cm in front of its
  base, 15 cm up) before the step returns and the executor frees the lock.
- The lock keeps two arms out of the same half, not one arm out of the space over the other's
  `carry` pose: watch the first runs where an arm takes a box on the other half.
- The aim corrections were fitted from grasps on each arm's own half; on the other half they are
  extrapolated.
- A box only one arm reaches stays undone if that arm leaves: the batch then waits (Ctrl-C once
  for the slow stop).

Heartbeat: the executor drops an arm whose heartbeat is 1.5 s old. The bridge's heartbeat is the
last bus read/write that worked (every control tick during a move), and the IK of a straight line
(~1 s) is solved before taking the bus.
