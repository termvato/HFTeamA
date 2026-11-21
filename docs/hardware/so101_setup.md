# SO-101 leader/follower pairs: setup, calibration, teleop, recording

Hardware team (#19). **For day-to-day use (install, teleoperate one or both pairs,
bimanual, record) follow the step-by-step section of the [README](../../README.md).**
This page has the details: calibrating new arms, matching a leader to its follower,
cameras and troubleshooting. Run every command from the repo root with the virtual
environment active.

Files:

- `configs/teleop_pairs.json`: the pairs (arm IDs, USB serials, cameras) and recording settings.
- `configs/calibration/`: one LeRobot calibration file per physical arm; see its README.
- `src/hfteama/hardware/`: the `hfteama` subcommands used below.

## Install

LeRobot is only needed for real arms, so it is not in the base dependencies:

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate   # LeRobot needs Python 3.12+
uv pip install -e ".[dev]" "lerobot[feetech,core-scripts]>=0.7"
```

Tested with LeRobot 0.7.0. The serial ports need the `dialout` group:
`sudo usermod -aG dialout $USER`, then log out and back in.

## Pairs

| Pair | Follower | Leader | Cameras | Status |
|---|---|---|---|---|
| `rakesh` (default) | `rakesh_follower` (`5B79014975`) | `rakesh_leader` (`5B8E115290`) | `wrist` | Calibrated, teleop verified 21 Nov |
| `hacklab` | `vato_follower` (`5A7A015861`) | `hacklab_leader` (`5A7A015820`) | `wrist` | Calibrated, teleop verified 21 Nov |
| bimanual | left `rakesh`, right `hacklab` | | `left_wrist`, `right_wrist` | Teleop and recording verified 21 Nov |

```bash
hfteama pairs            # which arms are connected, and on which port
```

Ports are found from each arm's USB serial, so `/dev/ttyACM*` order does not matter and a
pair works on any computer. Pick a pair with `--pair <name>` (default: `rakesh`).
Override a port with `HFTEAM_FOLLOWER_PORT=/dev/ttyACM0` / `HFTEAM_LEADER_PORT=...`.
Add `-n` to any command to print the underlying `lerobot-*` command without running it.

## Calibrate

```bash
hfteama --pair <pair> calibrate follower
hfteama --pair <pair> calibrate leader
```

The follower's power supply must be on. For each arm:

1. **Before the first Enter**, put every joint in the middle of its range (halfway between
   its two hard stops), not folded in its rest pose. This sets each joint's zero to the
   middle of the encoder (2047).
2. Press Enter, then move **each joint from one hard stop to the other**, including the
   base to both sides. In the live table values should move around 2047 without touching
   0 or 4095; `wrist_roll` reading 0-4095 is normal (full rotation).
3. Press Enter to save to `configs/calibration/.../<id>.json`. Commit it and add a row to
   `configs/calibration/README.md`.

If a joint other than `wrist_roll` jumps between 0 and 4095 in step 2, the arm was not
centered at step 1. Stop with `Ctrl+C` and start over: a range of 0-4095 on another joint
can make the follower take the long way around and hit something.

**Already calibrated arms:** don't recalibrate them. If LeRobot says "Press ENTER to use
provided calibration file", the file does not match the servos: do not press Enter (it
would write that file into the servos), type `c` and ask the team. Only exception:
`rakesh_leader`, whose file holds a base adjustment that may not be in its servos yet;
for it, press Enter.

## Teleoperate

```bash
hfteama --pair <pair> teleop               # arms only
hfteama --pair <pair> teleop --cameras     # also show cameras in Rerun
```

Put the leader in roughly the follower's pose first: the follower jumps to the leader's
pose. Move slowly at first. Stop with `Ctrl+C`.

If one joint is consistently offset, put both arms in the same pose, compare their
readings, and shift that joint's `range_min`/`range_max` **on the leader** by the
difference (steps = degrees x 4095 / 360). Shift the leader, not the follower: the
follower's range is its real hard-stop limit, and moving it past the hard stop makes the
servo push against it. For `wrist_roll` (full turn, range 0-4095) shift the leader's
`homing_offset` instead, by the same number of steps (Feetech: a larger offset lowers the
reading). Then write the file into the leader's servos (`bus.write_calibration`), or
LeRobot will ask to recalibrate on connect.

## Cameras

Cameras belong to a pair (`"cameras"` in `configs/teleop_pairs.json`). Each one becomes a
stream (`wrist`, `front`, ...) in the dataset. Prefer a stable
`/dev/v4l/by-id/...-video-index0` path over a numeric index: indices change between
computers and reboots. List them with `ls -l /dev/v4l/by-id/` (`-index1` is a metadata
node, not video) or `hfteama find-cameras`.

```bash
hfteama --pair <pair> camera           # all cameras of the pair, live in Rerun
hfteama --pair <pair> camera wrist     # just one
```

A black image (brightness near 0) means the lens is covered.

## Record a dataset

See the README ("Record a dataset"): per-pair datasets, commands for one or both pairs,
episode flow and keyboard controls. Each pair's `record` section in
`configs/teleop_pairs.json` overrides the shared one (at least `repo_id`, so two pairs
never write into the same dataset).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `hfteama pairs` shows "not connected" | USB unplugged, or follower without power. |
| Follower moves erratically or doesn't follow | Check the follower's power supply first. |
| Some motor IDs don't answer | Loose 3-pin cable in the daisy chain (often the base or gripper end). |
| `Permission denied` on the port | Join the `dialout` group. |
| `Device or resource busy` | Another process has the port open. |
| Calibration prompt every time | The pair's `id` doesn't match a file in `configs/calibration/`. |
