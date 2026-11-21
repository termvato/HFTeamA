# SO-101 leader/follower setup

Step-by-step guide to get a leader/follower pair of SO-101 arms working with this repo.
Run every command from the repo root with the virtual environment active.

## Registered pairs

### Rakesh pair (`rakesh`, default)

| Arm | ID | USB serial | Calibration file |
|---|---|---|---|
| Follower | `rakesh_follower` | `5B79014975` | `calibration/robots/so_follower/rakesh_follower.json` |
| Leader | `rakesh_leader` | `5B8E115290` | `calibration/teleoperators/so_leader/rakesh_leader.json` |

Calibrated and verified with teleoperation on 2026-11-21. Notes:

- The leader's `shoulder_pan` range was shifted by +176 steps (from `805–3367` to
  `981–3543`) so that both bases agree: with both arms pointing straight ahead, the
  leader and follower read within 1° of each other. Because of this, "straight ahead"
  reads about −12° on both arms, not 0°. To get 0°, recalibrate the follower base,
  pushing it fully to both hard stops.
- If the follower moves erratically during teleoperation, check its power supply first.
  That was the cause of the first failed test.

## Use a registered pair on another computer

The calibration files and USB serials are in the repo, so a pair needs no calibration
on a new computer:

```bash
git clone https://github.com/termvato/HFTeamA.git && cd HFTeamA
git switch lemillion789                     # or whichever branch has the pair
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

hfteama pairs          # both arms must show "connected at /dev/ttyACM*"
hfteama find-cameras   # camera indices differ per computer: update [cameras.*]
hfteama teleop         # quick check before recording
hfteama record --episodes 5 --task "..."
```

If `hfteama pairs` shows an arm as "not connected", check its USB cable and, for the
follower, its power supply. Serial port access needs the `dialout` group:
`sudo usermod -aG dialout $USER`, then log out and back in.

**Do not run `hfteama calibrate` for a registered pair** unless you mean to replace its
calibration; it overwrites the committed file.

## Add a new pair

1. Add a `[pairs.<name>.follower]` and `[pairs.<name>.leader]` section to
   `config/arms.toml` with new IDs (e.g. `<name>_follower`).
2. Get each arm's USB serial: connect only that arm and run `hfteama find-port`, or
   connect both and run:
   ```bash
   python -c "from serial.tools import list_ports; [print(p.device, p.serial_number) for p in list_ports.comports()]"
   ```
3. Set up motors (new arms only), calibrate, and test, as below, with `--pair <name>`.
4. Commit the new calibration files and add the pair to the table above.

## 1. Set up motors (new arms only)

Skip this if the arms were already assembled and configured. For a fresh arm, each
motor must get its ID (1-6) and baudrate:

```bash
hfteama --pair <name> setup-motors follower
hfteama --pair <name> setup-motors leader
```

Follow the prompts: connect **one motor at a time** to the controller board, starting
from the gripper (ID 6) down to the base (ID 1).

## 2. Calibrate

```bash
hfteama --pair <name> calibrate follower
hfteama --pair <name> calibrate leader
```

For each arm:

1. Move the arm to the **middle of its range** (every joint halfway between its two
   hard stops) **before** pressing the first Enter. This sets the homing offset to the
   middle of the encoder (2047).
2. Move **each joint through its full range** and press Enter. This records the min/max
   limits. In the live table, values should move around 2047 without touching 0 or
   4095. `wrist_roll` reading 0-4095 is normal (full rotation).

If a joint other than `wrist_roll` jumps from 0 to 4095 in step 2, the arm was not
centered at step 1. Stop with `Ctrl+C` and start over; a calibration with
`range_min: 0` / `range_max: 4095` on another joint can make the follower take the
long way around and hit something.

Recalibrating with the same ID overwrites the file.

## 3. Teleoperate

```bash
hfteama teleop               # arms only
hfteama teleop --cameras     # also show cameras in Rerun
```

Put the leader in roughly the same pose as the follower before starting: the follower
jumps to the leader's pose. Move the leader slowly at first. Stop with `Ctrl+C`.

If the follower moves too abruptly, uncomment `max_relative_target` in the
follower section of the config.

## 4. Cameras

```bash
hfteama find-cameras
```

Put the indices you want in the `[cameras.*]` sections of `config/arms.toml`. Each
section becomes one camera stream (`front`, `wrist`, ...) in the dataset.

## 5. Record a dataset

Set `repo_id` and `single_task` in the `[record]` section, then:

```bash
hf auth login                              # only needed if push_to_hub = true
hfteama record --episodes 5 --task "Pick up the cube and place it in the box"
hfteama record --resume --episodes 5       # append more episodes later
```

Keyboard controls while recording: `→` ends the current episode early, `←` re-records
it, `Esc` stops the session.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `hfteama pairs` shows "not connected" | USB cable unplugged, or the follower has no power. |
| `Could not connect on port ...` | Same as above, or another process has the port open. |
| `Permission denied` on the port | Join the `dialout` group (see above). |
| Follower moves erratically or doesn't follow | Check the follower's power supply first. |
| Follower offset on one joint | Put both arms in the same pose and compare readings; shift the leader's range for that joint (see the Rakesh pair notes). |
| `Device or resource busy` | Another process (another script, a second `hfteama`) already has the port open. |
| Calibration prompt every time you connect | The `id` in the config doesn't match a file in `calibration/`. |

To inspect the exact LeRobot invocation, add `-n`: `hfteama -n teleop`.
