# Arm calibrations (LeRobot format)

One file per physical arm, named after its LeRobot id. Layout matches LeRobot's own cache, so pass the folder
directly:

```
--robot.type=so101_follower --robot.id=<id> --robot.calibration_dir=configs/calibration/robots/so_follower
```

In `configs/arms.json`, set the arm's `calibration_id` to the same `<id>`.

| id | Arm | Notes |
|---|---|---|
| `vato_follower` | Vato's SO-101 follower (12 V STS3215) | LeRobot calibration from 2 Nov 2026 |

## Read before using

- A calibration belongs to one physical arm. The homing offsets and ranges are stored in that arm's servos.
- Never load another arm's file. If the file does not match the servos, LeRobot asks
  "Press ENTER to use provided calibration file". Pressing ENTER writes that file into the servos' memory and
  gives the arm a wrong zero. Type `c` instead and calibrate that arm.
- New arm: `lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=<id>
  --robot.calibration_dir=configs/calibration/robots/so_follower`, then commit the new `<id>.json` and add a row
  above.
- LeRobot puts each joint's zero in the middle of its calibrated range, not at the kinematic zero of the
  SO-101 model. `src/hfteama/arms/kinematics.py` converts between the two (the elbow is about 6.9 deg off).
