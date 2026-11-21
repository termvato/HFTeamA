# Arm calibrations (LeRobot format)

One file per physical arm, named after its LeRobot id. Layout matches LeRobot's own cache, so pass the folder
directly:

```
--robot.type=so101_follower --robot.id=<id> --robot.calibration_dir=configs/calibration/robots/so_follower
--teleop.type=so101_leader --teleop.id=<id> --teleop.calibration_dir=configs/calibration/teleoperators/so_leader
```

`hfteama calibrate|teleop|record` passes these for you from `configs/teleop_pairs.json`
(see `docs/hardware/so101_setup.md`).

In `configs/arms.json`, set the arm's `calibration_id` to the same `<id>`.

| id | Arm | USB serial | Notes |
|---|---|---|---|
| `vato_follower` | Vato's SO-101 follower (12 V STS3215) | `5A7A015861` | LeRobot calibration from 2 Nov 2026. Checked 21 Nov 2026: identical to the values stored in the servos. Follower of the `hacklab` pair. |
| `rakesh_follower` | Rakesh's SO-101 follower | `5B79014975` | 21 Nov 2026, verified with teleoperation. Follower of the `rakesh` pair. |
| `rakesh_leader` (teleoperators/so_leader) | Rakesh's SO-101 leader | `5B8E115290` | 21 Nov 2026. `shoulder_pan` range shifted +176 steps (805-3367 to 981-3543) to match the follower's base within 1 deg; straight ahead reads about -12 deg on both arms. |

## Read before using

- A calibration belongs to one physical arm. The homing offsets and ranges are stored in that arm's servos.
- Never load another arm's file. If the file does not match the servos, LeRobot asks
  "Press ENTER to use provided calibration file". Pressing ENTER writes that file into the servos' memory and
  gives the arm a wrong zero. Type `c` instead and calibrate that arm.
- New arm: `lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=<id>
  --robot.calibration_dir=configs/calibration/robots/so_follower` (or `hfteama --pair <pair> calibrate
  follower|leader`), then commit the new `<id>.json` and add a row above.
- The USB serial identifies the arm's servo board, not the arm. Keep each board on its arm, or the serial
  will point at the wrong calibration.
- LeRobot puts each joint's zero in the middle of its calibrated range, not at the kinematic zero of the
  SO-101 model. `src/hfteama/arms/kinematics.py` converts between the two (the elbow is about 6.9 deg off).
