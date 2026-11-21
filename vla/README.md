# VLA arm skills: data collection

Goal: replace the scripted IK skills behind `ArmWorker.execute(task)` (issue #20, `termvato` branch
`src/hfteama/arms/base.py`) with a learned policy, first a standard VLA (SmolVLA) and later our own.
The task: stack boxes, one box on top of another.

The speed-up story stays the orchestrator's: one arm builds the stacks in a few minutes, two arms
split the stacks and finish sooner. The policy only has to do **one step, one arm, on request**.
Nothing else changes when an arm is added.

## Datasets recorded

| Dataset (Hugging Face, public) | Arms | Episodes | Frames | Task text | Cameras |
|---|---|---|---|---|---|
| [`sahilrajpurkar03/so101_stack_single_rakesh`](https://huggingface.co/datasets/sahilrajpurkar03/so101_stack_single_rakesh) | single: rakesh pair | 20 | 23,915 (13.3 min at 30 fps) | "Put the objects in the box" | `top`, `wrist_a` |

- **Recorded:** 21 Nov 2026 with the studio. The leader drove the follower, with
  `max_relative_target` 10 deg per step. LeRobot dataset format v3.0, 640x480 at 30 fps, AV1 video.
- **Episodes:** each one puts the objects in the box, 24-64 s long.
- **Cleaned before upload:**
  - The recording had 21 episodes. Episode 0 (101 s, a warm-up) was deleted with
    `lerobot-edit-dataset --operation.type=delete_episodes`.
  - The next episode, recorded before the text box was changed, was relabelled from "Stack the
    box on top of the other box" with `--operation.type=modify_tasks`.
  - The untouched 21-episode recording is kept locally as `so101_stack_single_rakesh_raw21_backup`.
- **Name vs content:** the repo name says `stack` because that was the planned task. The episodes
  are pick-and-place into the box. Stacking data should go to a new dataset name.
- **Load it:** `LeRobotDataset("sahilrajpurkar03/so101_stack_single_rakesh")`, or view an
  episode with `lerobot-dataset-viz --repo-id sahilrajpurkar03/so101_stack_single_rakesh --episode-index 0`.

## Episodes: one stacking step each

Record each episode as **one box picked up and placed on another**, not a whole session in one
episode. Episodes have no time limit by default: press **Stop & save** when the box is placed.
- Short, single-step episodes train much better. LeRobot recommends 50+ of them.
- One episode maps to one orchestrator task (`stack box_2 on box_1`). A batch is just N of
  these, scheduled across however many arms are plugged in.
- Name the boxes in the instruction (`TASK="Stack the small box on the big box"`) so the
  orchestrator can later say which box goes where. Vary where the boxes start each time.

## Files

| File | What it does |
|---|---|
| `setup.sh` | one-time: conda env `lerobot`, LeRobot from source, ffmpeg |
| `config.sh` | **edit this**: which arm, cameras, task text, episode lengths, dataset names |
| `arms.json` | the team's four arms: id, USB serial, pair, side |
| `sync_calibration.sh` | copy the Hardware team's calibrations (never recalibrates) |
| `detect_arms.py` | which arms are plugged in, motor check, watch plug/unplug |
| `teleop.sh single\|dual` | leader drives follower, cameras on screen, nothing recorded |
| `record.sh single\|dual` | record episodes, push the dataset to the Hub |
| `studio.sh single\|dual` | the same recording from a local web page: live cameras, buttons, upload |
| `cameras.sh list\|check` | list the cameras; check the configured ones together before recording |
| `multi_arm_policy.md` | research: one trained policy on any number of arms, with the orchestrator splitting the work |

## Cameras

Three cameras, one line each in `CAMERA_TABLE` (`config.sh`):

| Name | Camera | Single-arm data | Dual-arm data |
|---|---|---|---|
| `top` | sees both arms, the objects and the box (C920) | yes | yes |
| `wrist_a` | on arm A's wrist (rakesh follower, Sonix USB2.0_CAM1) | if `SINGLE_ARM=a` | yes |
| `wrist_b` | on arm B's wrist (vato follower, Autodarts DIY Cam) | if `SINGLE_ARM=b` | yes |

- **Names are permanent.** Each one becomes `observation.images.<name>` in the dataset. Every
  camera is a top-level robot camera, also in dual mode, so a camera keeps the same name in single
  and dual data. LeRobot would otherwise rename per-arm cameras `left_*` / `right_*`.
- **The camera set is permanent per dataset.** Record only with all of a mode's cameras in place.
  The studio refuses to add episodes to a dataset recorded with other cameras or sizes. Changing
  cameras means a new dataset name.
- **Format:** every camera runs at 640x480, 30 fps (= dataset FPS), MJPG. LeRobot stops if a
  camera cannot give the size or fps, but it only warns if it falls back from MJPG to raw video.
  Raw video at 3 cameras overloads USB, so `cameras.sh check` treats that as a failure.
- **Mounted sideways or upside down:** set `rot` (90, 180, -90). A 90/-90 camera is stored as
  480x640 portrait; `lib.sh` swaps the size for you.
- **USB:** use `/dev/v4l/by-id/...-video-index0` paths. `index1` is a metadata node with no
  video. Spread the cameras over different USB ports; `check` warns when they share one.
  Make sure every camera gets the full 30 fps in its real lighting: a dim scene slows the
  exposure, and the frame rate drops with it.
- **Framing:** `check` saves `vla/camera_check.jpg` with every view. The wrist cameras should
  see the gripper fingers and the object being grasped. `top` sees the whole workspace,
  without people or the screen in the background. Keep every mount fixed between sessions.

## Arms

The Hardware team's two pairs, calibrated and committed on the `hardware` branch
(`configs/calibration/`, `configs/teleop_pairs.json`). **Never recalibrate them.**

| Side | Pair | Follower (USB serial) | Leader (USB serial) | Wrist camera |
|---|---|---|---|---|
| A, left | `rakesh` | `rakesh_follower` (`5B79014975`) | `rakesh_leader` (`5B8E115290`) | `wrist_a` |
| B, right | `hacklab` | `vato_follower` (`5A7A015861`) | `hacklab_leader` (`5A7A015820`) | `wrist_b` |

- Left/right are as the operator sees them, the same as Hardware's bimanual setup. In the
  `top` view the black arm is on the left and the green arm on the right.
- `SINGLE_ARM` (`config.sh`) picks the pair for single-arm recording. Each pair gets its own
  single-arm dataset (`so101_stack_single_rakesh` / `so101_stack_single_hacklab`), because the wrist cameras differ.
- Ports are found from each arm's USB serial every time, also by the studio on each Connect, so
  `/dev/ttyACM*` numbers don't matter and no udev rules are needed.
- `./vla/sync_calibration.sh` copies the four calibration files from `origin/hardware` to
  `~/.cache/hfteama/vla_calibration`. It also links the names `bi_follower_left/right` and
  `bi_leader_left/right` that LeRobot's dual mode looks for. Run it again when Hardware updates
  a calibration.
- **If LeRobot asks "Press ENTER to use provided calibration file":** press ENTER only for
  `rakesh_leader`, which is expected once. For any other arm, stop with Ctrl+C and ask Hardware.
  Never type `c`: it starts a new calibration.

---

## Recording from the browser (studio)

```bash
./vla/studio.sh single          # or dual; then open http://localhost:8765
```
- **Connect arms** starts teleop: the leader drives the follower, and the cameras and joint
  values show live. **Cameras only** previews the feeds before the arms are plugged in.
- **Start** (Space) records an episode with the instruction in the text box. **Stop & save** (→)
  and **Discard** (←) end it. There is no time limit by default (`EPISODE_TIME_S=0`), so the
  clock counts up until you stop. Set a number of seconds to make episodes save themselves.
  Change the text between episodes to record another task.
- **Auto-start** begins the next episode `RESET_TIME_S` after a save, for long sessions alone.
- **Upload to Hugging Face** pushes everything recorded so far. You can keep recording into the
  same dataset afterwards, and episodes already on disk (from earlier sessions or `record.sh`)
  are added to.
- **Disconnect** turns the follower's torque off, so put the arm in its rest pose first.
- It writes the same dataset as `record.sh` (same features, video and Hub format). It only binds
  to localhost. The page never calibrates. If an arm's calibration doesn't match its servos, the
  page shows an error. For `rakesh_leader` that is expected once: run `./vla/teleop.sh single`
  with `SINGLE_ARM=a` and press ENTER at the prompt.

## Part 0: Machine setup (once)

```bash
./vla/setup.sh                       # conda env "lerobot" + LeRobot 0.7.x from ~/lerobot
sudo usermod -aG dialout $USER       # serial access; log out and back in
hf auth login                        # write token, for pushing datasets
```

## Part 1: Detect robot 1

1. Set `SINGLE_ARM` in `config.sh`: `a` for the rakesh pair, `b` for the hacklab pair.
2. Get the calibrations (once, and when Hardware updates them):
   ```bash
   ./vla/sync_calibration.sh
   ```
3. Switch on the follower's 12 V supply and plug in both USB cables, then check:
   ```bash
   python vla/detect_arms.py list            # both arms of the pair: "/dev/ttyACM*  <id>"
   python vla/detect_arms.py probe vato_follower    # all 6 motors answer (use your arms' ids)
   python vla/detect_arms.py probe hacklab_leader
   ```
4. Cameras: see [Cameras](#cameras). Plug in `top` and this arm's wrist camera, then check that
   they run together (stop the studio first, as it holds the cameras):
   ```bash
   ./vla/cameras.sh list            # devices, USB ports, MJPG support, which name uses each
   ./vla/cameras.sh check single    # opens them together; saves vla/camera_check.jpg
   ```
5. Teleoperate to check everything before recording:
   ```bash
   ./vla/teleop.sh single
   ```

Done when: `teleop.sh single` moves the follower smoothly and both camera views show the arm, the
objects and the box.

## Part 2: Single-arm dataset to the Hub

```bash
TASK="Pick up the red cube and put it in the box" ./vla/record.sh single
TASK="Pick up the blue cube and put it in the box" ./vla/record.sh single --resume --dataset.num_episodes=10
```
- Recording keys: `→` end the episode early, `←` discard and redo it, `Esc` stop and save.
- The dataset goes to `https://huggingface.co/datasets/<HF_USER>/so101_stack_single_<pair>`
  (private until `PRIVATE=false`). Check it:
  `lerobot-dataset-viz --repo-id <HF_USER>/so101_stack_single_<pair> --episode-index 0`.
- Target: about 50 episodes. Vary the object position each time, and include a few where the object
  starts near the box and a few at the edge of reach. Keep the cameras fixed between sessions.
- Also time a few full sorts of the batch with this one arm. That is the 1-arm baseline (about 3 min).

## Part 3: Connect robot 2

Plug in the other pair and its wrist camera. Run `list` and `probe` as in Part 1 (all four
arms should show), then `./vla/cameras.sh check dual`. Then:
```bash
python vla/detect_arms.py watch      # unplug/replug arm B: you should see arm_left / arm_joined
./vla/teleop.sh dual                 # both leaders drive both followers
```
The `watch` events have the same shape as the orchestrator's arm registry events (#3). Hardware
(#19) owns discovery for the real run; this is only for checking the rig.

## Part 4: Dual-arm dataset to the Hub

```bash
TASK="Pick up the red cube and put it in the box" ./vla/record.sh dual
```
- Both arms work at the same time, each on objects on its own side, into the shared box. This is
  the 2-arm case of the same batch, so the cameras also see the other arm moving.
- Pushed to `<HF_USER>/so101_stack_dual`. Each frame holds both arms: `left_*` (arm A) and
  `right_*` (arm B) joints, plus the `top`, `wrist_a` and `wrist_b` cameras.
- Before the first dual session, plug in `wrist_b`, add it to `CAMERA_TABLE`, and run
  `./vla/cameras.sh check dual`.
- Time a few full 2-arm sorts against the Part 2 baseline.

**Decide before recording much (it changes Part 5):**
- **Per-arm policy (recommended).** One model drives one arm. That fits the orchestrator as-is: one
  `ArmWorker` per arm, any number of arms. The dual dataset is then split into two single-arm
  datasets (left half, right half). The value of the dual recording is that the policy learns
  to work while another arm moves in view.
- **Bimanual policy.** One model drives both arms with a 12-D action. The pair becomes one worker
  for the orchestrator, and a third arm needs a new model. It is only worth it for steps that
  really need two arms (handoff, hold-and-insert).

Recording with `bi_so_follower` keeps both options open.

---

## Part 5 (later): Train

SmolVLA fine-tuned from `lerobot/smolvla_base`. Our camera names are mapped to the ones the base
model was trained with:
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<HF_USER>/so101_stack_single_rakesh \
  --policy.repo_id=<HF_USER>/smolvla_so101_stack \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist_a": "observation.images.camera2"}' \
  --batch_size=64 --steps=20000 \
  --output_dir=outputs/train/smolvla_so101_stack --job_name=smolvla_so101_stack
```
- This laptop has an RTX 3070 with 8 GB. That is enough to run SmolVLA, but tight for training it
  at batch 64, so train on a bigger GPU (cloud or HF Jobs) or lower the batch size.
- Train ACT on the same data as a baseline. It trains in a few hours, and it is the comparison the
  research notes recommend.
- Measure success over 10-20 trials per object position before wiring it in.

## Part 6 (later): Use the policy in our architecture

The team-facing summary of this alternate method is [`docs/alternate_method_vla.md`](../docs/alternate_method_vla.md).
How one policy trained on one arm serves 2+ arms (who splits the work, what transfers to an
untrained arm, what to record): [`multi_arm_policy.md`](multi_arm_policy.md).

A `VLAArmWorker` implements the same `ArmWorker` contract as the scripted arm, so the orchestrator
does not change:
- `execute(task)`: map the task to an instruction (`pick_place obj_3` → "Pick up the red cube
  and put it in the box"), run the policy on this arm's observations at 30 Hz until done or
  timed out, and return `Result(OK | GRASP_FAIL | TIMEOUT | ARM_DOWN)`.
- Deciding "done": the arm is back near its rest pose with the gripper open, or a time limit.
  Check the grasp the same way the scripted skills do (gripper position + current).
- Durations: each `Result` already carries start and end times, so measured policy durations
  feed the finish-time prediction (#13).
- Several arms: one policy instance per arm. SmolVLA on 2 arms should fit one GPU (not yet
  measured); 3+ may need a smaller model or batching.
- Keep the scripted IK skill as the fallback for a task the policy keeps failing.
