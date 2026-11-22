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
| `train.sh smolvla\|act` | train with held-out episodes and early stopping (`train_watch.py`) |
| `train_xvla.sh` | fine-tune X-VLA on the dual-arm dataset in a detached container; `resume <run dir>` continues a run |
| `docker_train.sh <name> <args>` | run any `lerobot-train` command in a detached container, with the GPU guard and checkpoint pruning (`container_entry.sh`) |
| `gpu_guard.sh <pid>` | pause a training process when the GPU gets too hot, resume it when it has cooled |
| `make_bundle.sh <run dir> <hub repo>` | a folder to copy by SFTP and publish to the Hub from another machine |
| `check_policy.py <checkpoint>` | open-loop check: a checkpoint's predicted actions against the recorded ones |
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

## Part 5: Train (with early stopping)

```bash
./vla/train.sh smolvla          # or: ./vla/train.sh act
tail -f ~/vla_runs/<run>.watch.log
```
- **Held-out episodes:** the last 10% of the episodes (`EVAL_SPLIT`) are never trained on.
  Every 1000 steps (`EVAL_STEPS`) the loss is computed on them, and a checkpoint is saved.
- **Early stopping:** `train_watch.py` keeps the best checkpoint and deletes the others. It
  stops training after 3 validations in a row (`PATIENCE`) without at least a 2% improvement
  (`MIN_DELTA`), and never before step 3000 (`MIN_STEPS`). `MAX_STEPS` (20000) is only a cap.
  When training ends, it uploads the best checkpoint to the Hub as a private model.
- **SmolVLA** starts from `lerobot/smolvla_base`. The dataset's cameras are mapped to the base
  model's `camera1`, `camera2`, ... in the dataset's order.
- **Runs in the background:** both processes survive closing the terminal. `kill <pid>` stops
  training early; the best checkpoint stays on disk.
- **Speed on the RTX 3070 laptop:** at batch size 8, SmolVLA takes 0.36 s per step and about 3 GB
  of GPU memory; ACT takes 0.32 s per step and about 3.7 GB.
- **The loss is not the success rate.** Test the best checkpoint on the arm, 10-20 trials.

### Models trained

| Model (Hugging Face, private) | Data | Stopped | Best validation loss |
|---|---|---|---|
| `sahilrajpurkar03/smolvla_so101_stack_single_rakesh_20261121_1903` | `so101_stack_single_rakesh`: 18 train / 2 held-out episodes | step 5000 (37 min) | 0.404 at step 2000 |

Validation loss by step: 0.501 (1000), **0.404 (2000)**, 0.397 (3000), 0.424 (4000), 0.423 (5000).
Over the same steps the training loss kept falling, from 0.36 to 0.18. That is overfitting: 18
episodes are too few, so the next gain comes from more data, not from longer training.

## Part 5b: X-VLA on the dual-arm dataset

One model drives both arms (12 joint targets) for a step that needs two arms: lifting a box
together.

| Dataset (Hugging Face, public) | Arms | Episodes | Frames | Task text | Cameras |
|---|---|---|---|---|---|
| [`diegoloayza789/hfteam_box_middle`](https://huggingface.co/datasets/diegoloayza789/hfteam_box_middle) | dual: both followers | 41 | 20,556 (11.4 min at 30 fps) | "Lift the box with both arms and place it in the middle" | `front`, `left_wrist`, `right_wrist` |

Episodes are 12-33 s long (17 s on average). State and action are the 12 joint positions,
`left_*` then `right_*`.

### Machine setup (RTX 6000 Ada workstation)

```bash
mamba create -y -n lerobot -c conda-forge python=3.12 ffmpeg=7
git clone https://github.com/huggingface/lerobot.git ~/lerobot && git -C ~/lerobot checkout v0.7.0
uv pip install --python ~/miniforge3/envs/lerobot/bin/python -e "$HOME/lerobot[xvla,training,dataset]"
```
- **PyTorch must match the driver.** The default wheel is built for CUDA 13.2, and this
  machine's driver (570) only goes up to CUDA 12.8: `torch.cuda.is_available()` is then false.
  Install `torch 2.14.1+cu126` and `torchvision 0.29.1+cu126` instead.
- **Behind the proxy** `uv`/`pip` cannot use the PyTorch index (it redirects to a host the proxy
  refuses). Download the two wheels with `curl` from `https://download.pytorch.org/whl/cu126/`
  and install the files.

### Train

```bash
./vla/train_xvla.sh                  # STEPS=7000 BATCH=16 by default
docker logs -f xvla_box              # or: tail -f ~/vla_runs/<run>.log
tail -f ~/vla_runs/<run>.gpu.log     # temperature and pauses
```
The training runs in a detached Docker container (`docker_train.sh`), so it goes on after you
disconnect. The container has no software of its own: it mounts the host's `lerobot` conda
env, `~/lerobot` and the Hugging Face cache at the same paths. A checkpoint therefore resumes
in a container or on the host alike. `docker stop xvla_box` stops it.

**One training at a time.** X-VLA and ACT side by side each ran at half speed, filled the
48 GB, and the card reached 89 °C. To queue a second one:
`docker wait xvla_box; ./vla/docker_train.sh act_box ...` under `setsid nohup`.

What the script sets, and why:
- **Start:** `lerobot/xvla-base`, everything trainable, as the LeRobot X-VLA guide recommends.
  The language model part learns at 1/10 of the learning rate (LeRobot's default for X-VLA).
- **Cameras:** `front` is mapped to the base model's main view (`image`), which goes through
  the language model with the instruction. The wrist cameras are the extra views
  (`image2`, `image3`). Images are resized to 224x224 with padding.
- **Actions:** `action_mode=auto`. The 12 joint targets are padded to the base model's 20, and
  the loss counts only the 12. The `so101_bimanual` mode was not used: it trains the grippers
  as plain numbers but squashes them to 0-1 at inference.
- **Normalisation:** state and action by mean and standard deviation. The base model leaves the
  state raw, but our state is in degrees (up to ±130).
- **Precision:** float32 weights with bf16 autocast (`--accelerator.mixed_precision=bf16`).
  The guide's `--policy.dtype=bfloat16` stores the weights themselves in bf16, where most
  small updates round away.
- **Batch size 16:** 35 GB of GPU memory. 32 does not fit in 48 GB.
- **All 41 episodes are trained on.** No episodes are held out: the time allowed only about 6
  passes over the data, so too little training is the risk, not overfitting.
- **GPU temperature:** at full load (300 W) the card reaches 86 °C within a minute, with its
  fan at only 60%. `gpu_guard.sh` reads the temperature 4 times a second, pauses the training
  at `MAX_C` and resumes it at `RESUME_C`. This run used 87 °C and 85 °C: the card's own
  slowdown starts at 88 °C. The pauses took a quarter to a third of the time.

### Model trained

| Model (Hugging Face, public) | Data | Steps | Time | Training loss | Open-loop error |
|---|---|---|---|---|---|
| [`VivekSai07/xvla_hfteam_box_middle`](https://huggingface.co/VivekSai07/xvla_hfteam_box_middle) | all 41 episodes | 7000 at batch 16 (5.5 passes) | 1 h 42 min (12:14-13:56) | 0.003 | 0.9 |

- Training loss by step: 0.79 (100), 0.041 (1000), 0.022 (2000), 0.012 (3000), 0.007 (4000),
  0.004 (5000-6000), 0.003 (7000).
- Open-loop error: the mean error of the predicted 1 s chunks on 64 recorded frames, in degrees
  (gripper 0-100). It was 5.4 at step 1000 and 0.9 at step 7000, at most 1.65 for any joint.
  These frames were trained on: the numbers show the model fits the demonstrations, not that
  it generalises. The success rate on the arms is not measured yet.
- One chunk takes about 1 s to compute with another training sharing the GPU.
- Steps 0-2000 ran on the host, steps 2000-7000 in a container resumed from the step-2000
  checkpoint. The checkpoints kept are steps 6000 (weights only) and 7000 (resumable).

### Checkpoints and disk space

The system disk is nearly full, and a full disk locks everyone out of the machine. One X-VLA
checkpoint is 9.9 GB: 3.3 GB of weights and 6.6 GB of optimizer state, which only resuming needs.
- A checkpoint is saved every 2000 steps (`SAVE_FREQ`) and at the end.
- The container prunes after each save: only the newest checkpoint keeps its optimizer state,
  the one before it keeps its weights, and older ones are deleted (`KEEP_OLDER=1`). A run
  never holds more than about 23 GB.
- `docker system df -v` shows what an image holds on its own ("unique"). Removing an image
  that shares its layers with another one frees almost nothing.

### Resume

```bash
./vla/train_xvla.sh resume ~/vla_runs/<run>           # a run that was stopped: on to its planned steps
./vla/train_xvla.sh resume ~/vla_runs/<run> 10000     # a finished run: train on, to 10000 steps in total
```
It continues from `checkpoints/last`: weights, optimizer, learning-rate schedule and data order.
- This run was moved from the host into a container that way, at step 2000.
- Training a finished run on stretches the cosine schedule to the new total. The learning rate
  therefore goes up again: at step 7000 of 10000 to about a fifth of its peak.
- Resuming needs the newest checkpoint's `training_state` folder. Do not delete it.

### ACT on the same data

A second model for the same step: ACT (52M parameters, no language input). It trains from
scratch in under an hour, its checkpoint is 200 MB, and it answers in milliseconds.

```bash
OUT=~/vla_runs/act_<name> ./vla/docker_train.sh act_box --policy.type=act --policy.device=cuda \
    --policy.push_to_hub=false --dataset.repo_id=diegoloayza789/hfteam_box_middle \
    --batch_size=32 --num_workers=20 --steps=10000 --save_freq=2500 --log_freq=200 \
    --output_dir=$HOME/vla_runs/act_<name> --job_name=act_box
```
No camera renaming: ACT takes the dataset's own names, at the rig too (no `--rename_map`).

Reinforcement learning was not an option here: it needs a simulation of this two-arm task
with a reward, or hours of supervised trials on the real arms.

### Publish from another machine

This workstation cannot upload to the Hub. Make a bundle, copy it by SFTP, publish from there:

```bash
CARD=model_card.md ./vla/make_bundle.sh ~/vla_runs/<run> VivekSai07/xvla_hfteam_box_middle
```
The result is `~/vla_bundles/<name>/`: `model/` (what gets published) and `PUBLISH.md` with the
SFTP and `hf upload` commands to run on the other machine. The weights are hard links to the
checkpoint, so the bundle takes no extra disk space.

### Check a checkpoint without the arms

```bash
python vla/check_policy.py ~/vla_runs/<run>/checkpoints/last/pretrained_model
```
It predicts action chunks for recorded frames and prints the error per joint. A low error means
the checkpoint loads and reproduces the demonstrations. It is not a success rate: that needs
10-20 trials on the arms.

### Run it on the arms

A person at the rig runs this, never an agent (see "Real-arm safety" in `AGENTS.md`):

```bash
lerobot-rollout --strategy.type=base \
    --policy.path=VivekSai07/xvla_hfteam_box_middle \
    --robot.type=bi_so_follower <the other --robot.* flags the dataset was recorded with> \
    --rename_map='{"observation.images.front":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}' \
    --task="Lift the box with both arms and place it in the middle" --duration=30
```
- **Same robot setup as the recording:** the cameras must arrive as `front`, `left_wrist` and
  `right_wrist` at 640x480, and the joints in the dataset's order (`left_*`, then `right_*`).
  `--rename_map` is the same camera mapping as in training, and `lerobot-rollout` needs it
  again. The checkpoint normalises state and actions by itself.
- **Same task text** as in the dataset. The model has seen no other.
- **Start as the episodes start:** arms in the recorded start pose, the box where it was during
  recording. Keep `max_relative_target` set on the followers and a hand on the power switch.
- The model plans 30 steps (1 s) at a time and runs them all before it looks again
  (`n_action_steps=30`). If the motion is too blind, pass `--policy.n_action_steps=15`.

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
