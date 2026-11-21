# VLA arm skills: data collection

Goal: replace the scripted IK skills behind `ArmWorker.execute(task)` (issue #20, `termvato` branch
`src/hfteama/arms/base.py`) with a learned policy, first a standard VLA (SmolVLA) and later our own.
The task is the team's sort task: pick each object from the table and put it in the box.

The speed-up story stays the orchestrator's: one arm sorts the batch in about 3 min, two arms split
the same objects and finish sooner. The policy only has to do **one object, one arm, on request**.
Nothing else changes when an arm is added.

## Episodes: one object each, not one whole sort

Record each episode as **one object picked and placed in the box** (about 10-20 s), not a 3-min
run over every object.
- Short episodes train much better. LeRobot recommends 50+ of them, and a 3-min episode is
  mostly the same move repeated.
- One episode maps to one orchestrator task (`pick_place obj_3`). The 3-min batch is just N of
  these, scheduled across however many arms are plugged in.
- Name the object in the instruction (`TASK="Pick up the red cube and put it in the box"`) so the
  orchestrator can later say which object to take. Record each object type as its own short run
  with `--resume`.

## Files

| File | What it does |
|---|---|
| `setup.sh` | one-time: conda env `lerobot`, LeRobot from source, ffmpeg |
| `config.sh` | **edit this**: ports, cameras, task text, episode lengths, dataset names |
| `detect_arms.py` | find arms by USB serial, name them, udev rules, watch plug/unplug |
| `calibrate.sh <arm>` | LeRobot calibration for one arm, reused by dual mode |
| `teleop.sh single\|dual` | leader drives follower, cameras on screen, nothing recorded |
| `record.sh single\|dual` | record episodes, push the dataset to the Hub |

Arm names: `follower_a` + `leader_a` = robot 1, `follower_b` + `leader_b` = robot 2. In dual
mode, A is left and B is right.

---

## Part 0: Machine setup (once)

```bash
./vla/setup.sh                       # conda env "lerobot" + LeRobot 0.7.x from ~/lerobot
sudo usermod -aG dialout $USER       # serial access; log out and back in
hf auth login                        # write token, for pushing datasets
```

## Part 1: Detect robot 1

1. Power the follower (12 V) and plug both USB cables in.
2. Name each board by its USB serial (it asks you to unplug and replug it):
   ```bash
   python vla/detect_arms.py add follower_a
   python vla/detect_arms.py add leader_a
   python vla/detect_arms.py list            # both show up with their names
   ```
3. Give them stable names (`/dev/so101/follower_a`) so `ttyACM0`/`ttyACM1` swaps don't matter:
   ```bash
   python vla/detect_arms.py udev | sudo tee /etc/udev/rules.d/99-so101.rules
   sudo udevadm control --reload && sudo udevadm trigger
   ls -l /dev/so101/
   ```
4. Check all 6 motors answer on each arm:
   ```bash
   python vla/detect_arms.py probe follower_a
   python vla/detect_arms.py probe leader_a
   ```
5. Calibrate. If the team already calibrated this arm, copy their file into
   `~/.cache/huggingface/lerobot/calibration/` instead; a new calibration changes the joint zeros.
   ```bash
   ./vla/calibrate.sh follower_a
   ./vla/calibrate.sh leader_a
   ```
6. Cameras: find them, then set `CAM_TOP` / `CAM_FRONT` in `config.sh`. Prefer
   `/dev/v4l/by-id/...` paths so they survive replugging. Keep the same two camera names for
   single and dual data, so one policy can read both.
   ```bash
   lerobot-find-cameras opencv               # saves a test frame from each camera
   ```
7. Teleoperate to check everything before recording:
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
- The dataset goes to `https://huggingface.co/datasets/<HF_USER>/so101_sort_single` (private
  until `PRIVATE=false`). Check it: `lerobot-dataset-viz --repo-id <HF_USER>/so101_sort_single --episode-index 0`.
- Target: about 50 episodes. Vary the object position each time, and include a few where the object
  starts near the box and a few at the edge of reach. Keep the cameras fixed between sessions.
- Also time a few full sorts of the batch with this one arm. That is the 1-arm baseline (about 3 min).

## Part 3: Connect robot 2

Same as Part 1 with `follower_b` and `leader_b`: `add`, `udev` again (it prints all four arms),
`probe`, `calibrate.sh`. Then:
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
- Pushed to `<HF_USER>/so101_sort_dual`. Each frame holds both arms: `left_*` (arm A) and
  `right_*` (arm B) joints, plus the shared `top` and `front` cameras.
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
  --dataset.repo_id=<HF_USER>/so101_sort_single \
  --policy.repo_id=<HF_USER>/smolvla_so101_sort \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.front": "observation.images.camera2"}' \
  --batch_size=64 --steps=20000 \
  --output_dir=outputs/train/smolvla_so101_sort --job_name=smolvla_so101_sort
```
- This laptop has an RTX 3070 with 8 GB. That is enough to run SmolVLA, but tight for training it
  at batch 64, so train on a bigger GPU (cloud or HF Jobs) or lower the batch size.
- Train ACT on the same data as a baseline. It trains in a few hours, and it is the comparison the
  research notes recommend.
- Measure success over 10-20 trials per object position before wiring it in.

## Part 6 (later): Use the policy in our architecture

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
