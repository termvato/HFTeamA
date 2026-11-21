# SO-101 multi-arm orchestration: software stack and demo design for a 48-hour hackathon

Researched 21 Nov 2026. LeRobot API names were checked against a local clone of huggingface/lerobot at commit
`d3793b7` (23 Oct 2026, `version = "0.6.2"`), referred to below as "LeRobot 0.6.2 (local)". GitHub links point at that commit.
`LR` = https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb
Some numbers come from local runs on this machine (MuJoCo 3.17.0, rerun-sdk 0.33.1). They are marked "(local run)".

## 1. Running several SO-101 arms from one computer with LeRobot

### Takeaway
LeRobot has no N-arm robot class. `bi_so_follower` is a fixed left/right pair. To run N arms, create N `SOFollower` objects, each with its own serial port and its own thread or process. That is practical because each arm is its own USB-serial Feetech bus. The real limits are camera USB bandwidth, per-bus read glitches, and GPU contention if every arm runs a learned policy. The async/remote inference server serves only one robot session per process.

### Cited Findings
- **Single-arm config.** `SOFollowerConfig` fields are `port`, `disable_torque_on_disconnect`, `max_relative_target`, `cameras`, `use_degrees`, the position PID gains (`position_p/i/d_coefficient`, defaults 16/0/32) and `num_read_retries=2`. The config is registered as both `so101_follower` and `so100_follower`. — [LR/src/lerobot/robots/so_follower/config_so_follower.py](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/robots/so_follower/config_so_follower.py)
- **Read retries.** In the code's own words, Feetech buses "can occasionally return a corrupted status packet ('Incorrect status packet!'), especially when several joints move at once". `num_read_retries` retries immediately, with no sleep. — same file. A user issue reports a sync_read crash on Present_Position with this error, with no USB disconnects in the kernel log. — [lerobot issue #3131 (mirror)](https://github-users.yhajoa.workers.dev/huggingface/lerobot/issues/3131)
- **Bimanual class.** The bimanual robot is `bi_so_follower` (the old name `bi_so100_follower` is gone). `BiSOFollowerConfig` takes `left_arm_config: SOFollowerConfig`, `right_arm_config: SOFollowerConfig` and optional shared `cameras`. Per-arm cameras get `left_`/`right_` prefixes. — [LR/src/lerobot/robots/bi_so_follower/config_bi_so_follower.py](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/robots/bi_so_follower/config_bi_so_follower.py)
- **Registered robots.** The robots registered in 0.6.2 are so100/so101_follower, bi_so_follower, koch, omx, lekiwi, hope_jr, reachy2, unitree_g1, openarm, bi_openarm, rebot_b601, bi_rebot_b601 and earthrover_mini_plus. None is an N-arm or multi-robot class. — `LR/src/lerobot/robots/*/config*.py` (local grep)
- **One control cycle.** `get_observation()` does one `bus.sync_read("Present_Position")`. `send_action()` does one `sync_write("Goal_Position")`, plus another read when `max_relative_target` is set. — [LR/src/lerobot/robots/so_follower/so_follower.py](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/robots/so_follower/so_follower.py)
- **Gripper limits set at connect.** `configure()` sets the gripper to `Max_Torque_Limit=500`, `Protection_Current=250` and `Overload_Torque=25`. The comments say this is "to avoid burnout". — same file
- **Motor driver.** In 0.6.2, Feetech motors are driven through **rustypot** (`serial-motors = [rustypot, pyserial]`), not the older scservo_sdk. — [LR/pyproject.toml](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/pyproject.toml)
- **Baud rate.** The Feetech bus is typically 1,000,000 baud. — [LR/docs/source/feetech.mdx](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/docs/source/feetech.mdx)
- **Naming several arms.** The SO-101 docs say to give each robot a unique `--robot.id`. That id names its calibration file. — [LR/docs/source/so101.mdx](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/docs/source/so101.mdx)
- **Ports for a two-arm setup.** A Seeed dual-arm guide gives each arm its own `/dev/ttyACM*` device: four devices for 2 leaders and 2 followers. — [Seeed bimanual SO-ARM guide](https://wiki.seeedstudio.com/ja/lerobot_double_arm_so_arm_training/)
- **Control rate.** LeRobot's recording loop targets 30 Hz (a 33.3 ms budget per tick) and prints a "cadence summary" of missed ticks. Its camera examples use 640x480 @ 30 fps. — [LR/docs/source/il_robots.mdx](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/docs/source/il_robots.mdx)
- **Async inference has a one-robot limit.** "One server process owns one deployment/model and admits one session... Concurrent model sharing, batching... remain outside this implementation." The backends are `sync`, `rtc` (local asynchronous real-time chunking) and `remote` (over Zenoh). It is "not a hard-real-time controller". — [LR/src/lerobot/remote_inference/README.md](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/remote_inference/README.md). Optional extra: `remote = ["eclipse-zenoh>=1.9.0,<1.11.0", "msgpack"]` — pyproject.toml
- **RTC.** RTC "asynchronously generat[es] the next chunk while the robot continues executing the current one". — [LR/docs/source/rtc.mdx](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/docs/source/rtc.mdx)
- **Built-in VLM planner.** `lerobot.rollout.planner.VlmPlanner` sends an OpenAI-compatible VLM the request plus past (images, command) pairs. The VLM replies with one JSON field, following arXiv 2602.13193, "Steerable VLA Policies". It steers one robot's language-conditioned policy, not several arms. — [LR/src/lerobot/rollout/planner.py](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/rollout/planner.py)
- **USB latency timer.** On USB-serial adapters the default latency timer is 16 ms, and it can be lowered. This figure is from the Dynamixel SDK FAQ; I did not confirm it for Feetech or rustypot. — [ROBOTIS DYNAMIXEL SDK FAQ](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/faq/)

### Inferences
- **Suggested architecture.** Run one Python process per arm: `SOFollower(SOFollowerRobotConfig(port=..., id=f"arm{i}"))`, a skill executor and a heartbeat. A central orchestrator talks to the arm processes over a queue (ZMQ, Zenoh or multiprocessing). Separate processes avoid the GIL and contain a bus crash to one arm. (inference, not checked on hardware)
- **USB bandwidth.** Serial traffic is tiny: 1 Mbaud is roughly 100 kB/s per bus, against USB 2.0's 480 Mbit/s. Cameras are what saturate a USB controller. Uncompressed 640x480 YUYV @ 30 fps is about 18 MB/s per camera, so use MJPEG, one overhead camera, or spread cameras across controllers. (inference from arithmetic, not checked)
- **Stable udev names.** Each follower should get a stable udev symlink by serial number (the lab already uses `/dev/servo_bench`), because `/dev/ttyACM*` order changes between replugs. (inference)
- **Learned policies on N arms.** Each arm needs its own policy server or local engine. One GPU serving N ACT models is probably fine because ACT is small, but N SmolVLA/pi0 instances will contend. (inference)

### Gaps
- I found no published measurement of LeRobot control-loop rate with 3 or more SO-101s on one host, or of the rustypot latency-timer default.
- I found no LeRobot "multi-robot" feature beyond bimanual. I checked the robot registry and the docs list, not Discord.

## 2. Scripted skills vs learned policies for pick/place on SO-101

### Takeaway
For a 48-hour orchestration demo, use scripted IK pick/place skills (LeRobot ships placo-based IK processors), with fixed object positions or AprilTag/colour-detected ones. Small-data learned policies on SO-101 pick/place reach about 30-40% success in independent tests, which is too low for a throughput demo where one failure stalls the schedule. Hackathon winners mostly showed learned policies, but their projects were single long-horizon tasks, not throughput.

### Cited Findings
- **IK in LeRobot.** LeRobot 0.6.2 ships `lerobot.model.RobotKinematics` and SO-follower processor steps: `InverseKinematicsEEToJoints`, `ForwardKinematicsJointsToEEObservation`, `EEBoundsAndSafety` and `GripperVelocityToJoint`. Install them with the `kinematics` extra (placo 0.9.6-0.9.15). — [LR/src/lerobot/robots/so_follower/robot_kinematic_processor.py](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/robots/so_follower/robot_kinematic_processor.py), pyproject.toml
- **ACT training time (official).** The docs say ACT "trains in a few hours on a single GPU" and to "expect a few hours for 100k training steps on a single GPU". — [LR/docs/source/act.mdx](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/docs/source/act.mdx)
- **SmolVLA training time (official).** The docs recommend ~50 episodes (e.g. 5 positions x 10). They say 25 episodes "was not enough", and that 20k steps take "~4 hrs on a single A100". — [LR/docs/source/smolvla.mdx](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/docs/source/smolvla.mdx)
- **Data advice for the base pick/place task.** Record at least 50 episodes, 10 per location, with fixed cameras and consistent grasps. — [LR/docs/source/il_robots.mdx](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/docs/source/il_robots.mdx)
- **Independent SO-101 test (apple into bowl, 10 trials each).** SmolVLA reached ~0% with 20 demos and ~30% with 50 demos (40k steps). pi0.5 reached ~30% with 50 demos. ACT reached ~40% with 50 demos and 10k steps, and moved most smoothly. "No policy was reliable", and most failures were at the grasp. — [Tiensmith, "Are VLAs overhyped?", Apr 2026](https://justintiensmith.github.io/vla-research/2026/04/10/are-vlas-overhyped-first-results-on-a-real-robot.html)
- **Paper claim.** The SmolVLA paper reports real-world success of 78.3% for SmolVLA, 48.3% for ACT and 61.7% for pi0 on the authors' SO-100/101 tasks. — [SmolVLA blog](https://huggingface.co/blog/smolvla); [arXiv 2506.01844](https://arxiv.org/abs/2506.01844). This is the authors' own setup and is not matched by the independent result above.
- **Seeed/NVIDIA/HF Embodied AI Hackathons (Oct 2025, Shenzhen and Mountain View).** Every winner that documented its method used a learned policy:
  - GR00T N1.5 for matcha-making on XLeRobot, and for desk tidying on SO-ARM101 (300 episodes, 90% real and 10% sim).
  - SmolVLA on 170 episodes, split into four per-action models.
  - ACT for dual-arm SO-ARM101 cloth folding.

  The Sprinkle team found that modular per-action models were smoother and easier to coordinate than one long-horizon model. The matcha team needed distinct start/end states and visual cues because GR00T has no memory. — [Seeed recap](https://seeed.cc/post/2025-embodied-ai-hackathon-recap)
- **LeRobot Worldwide Hackathon (14-15 June 2025).** It ran in more than 100 locations. Prizes were Hope Jr + LeKiwi for 1st-3rd place and an SO-101 for 25th-30th. Teams uploaded datasets to the `LeRobot-worldwide-hackathon` org. — [HF all-winners Space](https://huggingface.co/spaces/LeRobot-worldwide-hackathon/all-winners); [Univ. Lille news](https://cristal.univ-lille.fr/en/news/2025-06-19-982/). Example local results: 2nd overall in London went to plastic/paper segregation, and 1st in Bilbao to an AI bartender. — [Kulkarni project page](https://om-kulkarni.github.io/projects/waste-segregation-hackathon); [UdG MIRS page](https://www.udg.edu/en/masters-en-tecnologia/mirs/petjada-mirs/aslanli)
- **LLM planning over scripted skills (Intel/lablab AI Infra Summit challenge, simulated SO-101s).** These entries are the closest precedent for "LLM planner + scripted skills". The project pages now redirect, so I verified them only through search snippets.
  - An LLM turns an instruction into a structured plan. Scripted IK skills then coordinate two MuJoCo SO-101 arms through a shared staging zone. — [lablab profile](https://lablab.ai/u/@ari517)
  - A VLM picks scripted primitives for a cup handoff and reports 100% success over 10 seeds. — [lablab entry](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon/inzuppato/bimanual-pick-handoff-place-with-vlm-planning)

### Inferences
- **Score on reliability.** Judges at the hackathons above scored learned long-horizon tasks. A throughput-and-orchestration demo is judged on whether the schedule completes, so reliability matters more than learned-ness. Use scripted skills for the arms and put the "AI" in the planner (LLM task graph) and in perception (detection). (inference)
- **Optional learned policy.** If a learned policy is wanted for credibility, use ACT on one skill: about 50 episodes, roughly 1-3 h on one GPU, with a scripted fallback. Do not put SmolVLA/pi0 on the critical path in 48 h. (inference from the timings above)

### Gaps
- I could not read the full ranked list of LeRobot Worldwide Hackathon global winners. The Space loads it via JavaScript, so I don't know whether any winner was multi-arm or orchestration.
- I found no source that directly compares scripted IK against ACT reliability on SO-101.

## 3. Simulating N virtual SO-101 arms alongside 1 real one

### Takeaway
MuJoCo plus the Menagerie `robotstudio_so101` model is the fastest path. `MjSpec.attach_body` puts N prefixed copies in one scene. Locally, 16 arms stepped at about 41x real time on one CPU thread. Mirror the real arm by writing its read joint positions into one copy's `qpos` (or its position actuators). Isaac Lab has an SO-101 asset and LeIsaac tasks, but it is heavier to set up. gym-hil is a Franka Panda environment, not SO-101.

### Cited Findings
- **Menagerie model.** `robotstudio_so101` is in google-deepmind/mujoco_menagerie. It needs MuJoCo ≥3.1.3 and is derived from TheRobotStudio's `so101_new_calib.xml`. Menagerie's changes: implicitfast integrator, default forcerange, primitive collision geoms for the gripper and arm, gripper contact parameters tuned for manipulation, and a camera mount. Apache-2.0. — [Menagerie robotstudio_so101 README](https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/robotstudio_so101/README.md); upstream MJCF in [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- **(local run) Many arms in one scene.** With MuJoCo 3.17.0, each arm was attached via `MjSpec.from_file(so101.xml)`, `frame.attach_body(body, f"arm{i}_", "")`, then `compile()`. Timings on free arms (no objects, no rendering, 2 ms timestep):

  | Arms | Steps/s | Real-time factor |
  |---|---|---|
  | 1 | 251k | 502x |
  | 4 | 77k | 155x |
  | 8 | 42k | 83x |
  | 16 | 21k | 41x |

  Compiling printed a harmless "cone: parent has 0, child has 1, keeping parent value" warning. To keep elliptic cones, set `world.option.cone` to elliptic. — script at `research/scripts/multi_arm_bench.py`
- **Isaac Lab.** NVIDIA's "Getting started with Isaac Lab" course uses a built-in `SO101_CFG` (reference: `isaaclab_assets/robots/so101.py`, Isaac Lab release 3.0.0). It runs Newton physics with an MJWarp solver, has a tutorial task `IsaacTutorial-Inspect-SO101` and a place-vial task, and includes a wrist camera and a `sim2real` base-height preset. The page shows only `--num_envs 1`. — [NVIDIA DLI SO-101 page](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-a-robot-arm/02-so101-robot.html)
- **LeIsaac.** LeIsaac (LightwheelAI) integrates IsaacLab with SO101 leader/follower teleop. Its tasks include `LeIsaac-SO101-PickOrange-v0`, `-LiftCube-v0`, `-CleanToyTable-v0`, `-CleanToyTable-BiArm-v0` and `-FoldCloth-BiArm-v0`, and they load into LeRobot through EnvHub. — [LR/docs/source/envhub_leisaac.mdx](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/docs/source/envhub_leisaac.mdx)
- **gym-hil.** It is "a Franka Panda robot simulation based on MuJoCo" (`PandaPickCube*-v0`), so it is not SO-101. — [LR/docs/source/hilserl_sim.mdx](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/docs/source/hilserl_sim.mdx)
- **Genesis.** Genesis-Intelligence hosts an SO101 `scene.xml` that includes `so101_new_calib.xml`. — [HF Genesis-Intelligence/assets](https://huggingface.co/datasets/Genesis-Intelligence/assets/blob/main/SO101/scene.xml)
- **Other sim envs in LeRobot.** LeRobot 0.6.2's `envs/` contains LIBERO, MetaWorld, RoboCasa, RoboMME, RoboTwin and VLABench. None is SO-101-specific. — `LR/src/lerobot/envs/` (local listing)

### Inferences
- **Hybrid demo.** One real arm plus N-1 MuJoCo arms share one orchestrator and one Gantt. Each "arm worker" exposes the same `execute(skill) -> result` interface whether its backend is the real SOFollower or a MuJoCo copy. The real arm's joint readings drive a "ghost" copy in the scene, which gives the digital-twin view. The lab's fitted twin makes the simulated arms' timing more honest than stock Menagerie, which matters when claiming throughput. (inference)
- **Scaling in sim.** Run sim arms at 1x wall-clock so their task durations are comparable to the real arm's. A faster-than-real-time sweep (N=1..16) can generate the throughput-vs-N curve offline. (inference)

### Gaps
- I did not test Isaac Lab multi-arm scenes or Genesis performance.
- The MuJoCo timing did not include contact-heavy pick/place or rendering. Rendering N arms at 30 fps will cost more than the physics.

## 4. Live visualisation: Gantt of tasks per arm, plan vs actual, replanning

### Takeaway
The Rerun version LeRobot pins (rerun-sdk ≥0.24,<0.34; 0.33.1 installed here) has `StateTimelineView` with the `StateChange` archetype, which draws horizontal coloured lanes per entity over time. That is a native Gantt for "actual" task state per arm, in the same viewer LeRobot already uses. A Plotly `px.timeline` in Dash, or a small web UI, is better for a polished plan-vs-actual board that judges look at.

### Cited Findings
- **Rerun pin.** LeRobot 0.6.2 depends on `rerun-sdk>=0.24.0,<0.34.0`. — [LR/pyproject.toml](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/pyproject.toml)
- **(local check) rerun-sdk 0.33.1.**
  - `rerun.blueprint` has `StateTimelineView`, `TimeSeriesView`, `TextLogView`, `DataframeView`, `GraphView`, `Spatial3DView` and others.
  - `rr.StateChange(state=...)` "marks a new state at the given time… The state timeline view displays these as horizontal colored lanes over time". A null state is ignored.
  - `rr.StateConfiguration` maps state values to labels, colours and visibility, and should be logged as static.
  - Both are flagged "unstable and may change significantly".

- **Current Rerun docs (0.43.2).** They list 13 view types, including StateTimelineView, with no dedicated Gantt/interval view. — [Rerun view types](https://rerun.io/docs/reference/types/views)
- **GraphView.** It "visualizes time-varying directed or undirected graphs". — same page

### Inferences
- **Rerun recipe.** For each arm, log `rr.StateChange(state="pick A3")` on `actual/arm{i}` at each skill start and `state=None` at idle. Log the planned schedule on `plan/arm{i}` with planned start times. On a replan, re-log the plan lanes from "now". The GraphView can show the LLM's task DAG coloured by status, and Spatial3DView can show the real arm's URDF next to the sim arms. Whether lanes logged at future timestamps render well before "now" is untested. (inference, not checked)
- **Version risk.** The archetype is marked unstable, so pin the Rerun version for the hackathon. (inference)
- **Demo screen.** A Plotly/Dash timeline with plan bars ghosted, actual bars solid, and red failure markers reads instantly for judges. Use Rerun for the engineering views: camera, 3D and joint traces. (inference)

### Gaps
- I did not test Plotly/Dash live-update performance or the rendering of future-dated StateChange lanes.

## 5. Failure detection and recovery on cheap arms

### Takeaway
Detect grasp success from the gripper itself. After commanding closure, the gripper stalls short of its closed position with elevated load or current if an object is present. If it reaches near-closed, the grasp is empty. Add per-skill timeouts and a bus-read watchdog. Recover by re-queuing the task to the same or another arm (the orchestrator's job), not by in-skill cleverness. LeRobot already caps gripper torque and current and can read `Present_Current`.

### Cited Findings
- **Gripper caps at connect.** LeRobot writes `Max_Torque_Limit=500`, `Protection_Current=250` and `Overload_Torque=25` to the SO-101 gripper at connect time "to avoid burnout". A stalled grasp is therefore a normal, protected state. — [LR/src/lerobot/robots/so_follower/so_follower.py](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/robots/so_follower/so_follower.py)
- **Reading current.** LeRobot's RL code reads motor currents with `robot.bus.sync_read("Present_Current")` and appends them to the state. — [LR/src/lerobot/rl/joint_observations_processor.py](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/rl/joint_observations_processor.py)
- **Bus glitches.** Corrupted status packets happen "especially when several joints move at once". LeRobot retries reads (`num_read_retries`). — config_so_follower.py
- **General gripper practice (Robotiq).** Object detection comes from current exceeding a limit and the fingers stopping. No detection occurs if the gripper completes its motion. A pickup check only shows the object was held when it was checked, so re-check during transport. — [Robotiq: object detection](https://blog.robotiq.com/how-to-use-the-object-detection-feature-of-the-robotiq-grippers); [Robotiq: verify contact](https://blog.robotiq.com/bid/56591/how-to-verify-gripper-contact)
- **Soft objects.** Stock SO-101 grippers slip on fabric. One team added fingertip FSRs to sense a real hold. — search summary of a community project; I did not open the primary page
- **Per-skill models.** Hackathon teams found that splitting tasks into per-action models with distinct start and end states made coordination and failure handling easier. — [Seeed recap](https://seeed.cc/post/2025-embodied-ai-hackathon-recap)

### Inferences
- **Grasp check.** After `close`, wait about 300 ms. If gripper position is more than X ticks from closed and Present_Current/Load is above a threshold, mark the grasp OK. Otherwise mark `GRASP_FAIL`. Re-check gripper position after lift, to catch a drop.
  - Pick thresholds from about 10 empty and 10 holding closes per object type, and apply them per arm, because servos differ.
  - The lab's "two still-pose readings" rule fits this: read once after close, once after lift.

  (inference, not checked on hardware)
- **Timeouts.** Each skill gets a deadline of about 1.5x its sim-predicted duration. Missing it raises `TIMEOUT`, and the arm goes to a known safe pose. A missed heartbeat or bus exception marks the arm `DOWN`, and its tasks return to the pool. (inference)
- **Replanning.** Treat the plan as a list-scheduling problem over the task DAG. On any failure or arm loss, rerun the greedy scheduler (earliest-finish-time) over the remaining tasks and arms. That takes milliseconds and shows up as a visible replan on the Gantt. (inference)

### Gaps
- I found no SO-101/STS3215-specific published grasp-detection thresholds, or units/sign conventions for Present_Load vs Present_Current. Measure them on the rig.

## 6. Demo design that hackathon judges find compelling

### Takeaway
Winning embodied-AI demos showed long-horizon completion, generalisation, dual-arm coordination and multi-robot collaboration. A strong 48-hour demo has six parts:
1. Natural-language order, turned by an LLM into a visible task DAG.
2. A live Gantt across 1 real plus N simulated arms.
3. A measured throughput-vs-N curve that bends, honestly, because of dependencies and shared zones.
4. A live failure injection (knock an object out of the gripper, or unplug an arm), with automatic detection and a replan on screen.
5. Makespan before and after the failure.
6. A one-slide "why it doesn't scale linearly", with Amdahl-style numbers from the run.

### Cited Findings
- **What winning demos emphasised.** The Seeed/NVIDIA/HF Embodied AI Hackathon winners emphasised "long-horizon completion, generalization, dual-arm coordination, and multi-robot collaboration". The page lists no judging rubric. — [Seeed recap](https://seeed.cc/post/2025-embodied-ai-hackathon-recap)
- **Simulated-arm precedents.** The lablab entries with simulated SO-101s used an LLM or VLM to turn natural language into structured plans over scripted primitives, used shared staging zones for handoff, and reported fallbacks when the LLM API hit rate limits. — [lablab profile](https://lablab.ai/u/@ari517); [lablab handoff entry](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon/inzuppato/bimanual-pick-handoff-place-with-vlm-planning) (verified only through search snippets)
- **Language planning in LeRobot.** LeRobot 0.6.2 itself ships an in-context VLM planner pattern, OpenAI-compatible, replying in one JSON field, which can be borrowed for the planner prompt. — [LR/src/lerobot/rollout/planner.py](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/rollout/planner.py)

### Inferences
- **Scaling plot.** Plot makespan and items/min against N = 1, 2, 4, 8 (sim sweep), with the real-arm point overlaid. Draw the ideal 1/N line and the measured curve. Annotate the gap with its causes: critical-path length (dependencies), shared-zone mutex wait time and orchestrator overhead. The honest bend is more convincing than a straight line. (inference)
- **Shared zones.** Model shared zones (bin, handoff tray) as mutex resources in the scheduler, and show waits as hatched bars on the Gantt. (inference)
- **LLM safety and reliability.** The LLM output must be a validated JSON DAG: skills from a fixed library, object IDs from perception. Keep a cached fallback plan in case the API fails on stage. (inference)
- **Physical-risk budget.** Run one real arm. The other arms run in sim unless more hardware is in hand. A second real arm adds a lot of risk (port, calibration, cabling) for little extra judge value. (inference; matches the lab's "minimise rig operations" rule)
- **48-hour plan.** (inference)
  - Day 1 AM: arm-worker interface plus MuJoCo N-arm scene.
  - Day 1 PM: scripted pick/place on the real arm, plus grasp check.
  - Day 1 night: scheduler, DAG and Gantt.
  - Day 2 AM: LLM decomposition, failure injection, the N sweep.
  - Day 2 PM: rehearse, record a backup video, and freeze the code.

### Gaps
- I found no official LeRobot hackathon judging rubric.
- I found no SO-101 project that specifically demonstrated throughput scaling across N arms, so this angle appears novel. That is based on search absence only.

## 7. Collaborative arms: handoffs, two arms on one object, shared frame (scope added by the user)

### Takeaway
In 48 hours, do collaboration on SO-101s as scripted, time-synchronised primitives in one calibrated table frame, not as a learned bimanual policy. Three primitives are within reach:
- **Handoff at a fixed pose.** One arm presents the object and holds; the other approaches, closes, and gets a grasp check; then the first opens.
- **Hold plus act.** One arm fixtures the part while the other inserts or presses.
- **Shared carry.** Only for a light, rigid object over a short path, because two position-controlled arms holding one rigid object fight each other (inference, not checked).

LeRobot's `bi_so_follower` gives two arms one combined action and observation dict. It reads and writes the two arms one after the other in the same thread, not in parallel. Gripper payload is a few hundred grams at best, so use light parts.

### Cited Findings
- **Bimanual reads and writes are sequential.** `BiSOFollower.get_observation()` calls `left_arm.get_observation()`, then `right_arm.get_observation()`. `send_action()` splits actions by the `left_`/`right_` prefix and sends left, then right. So each arm's command goes out at a slightly different time within one tick. — [LR/src/lerobot/robots/bi_so_follower/bi_so_follower.py](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/robots/bi_so_follower/bi_so_follower.py)
- **Per-arm settings.** Each arm keeps its own `port`, PID gains, `max_relative_target`, `num_read_retries` and cameras. — same file
- **Seeed dual-arm guide.** It uses `bi_so_follower` / `bi_so_leader` with separate left/right ports for teleop, recording and policy deployment. It does not cover handoff tasks. — [Seeed dual-arm guide](https://wiki.seeedstudio.com/lerobot_double_arm_so_arm_training/)
- **Community bimanual packages.** SIGRobotics-UIUC publishes community packages `lerobot-robot-bi-so101-follower` and `lerobot-teleoperator-bi-so101-leader` on PyPI. — [PyPI](https://pypi.org/project/lerobot-robot-bi-so101-follower/)
- **Bimanual reference tasks in sim.** LeRobot's `aloha` extra (`gym-aloha>=0.1.5,<0.2.0`) provides the canonical sim bimanual tasks. The default task in `envs/configs.py` is `AlohaInsertion-v0`. These are ALOHA arms, not SO-101. — [LR/pyproject.toml](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/pyproject.toml), `LR/src/lerobot/envs/configs.py`
- **Prior bimanual SO-101 work at hackathons.**
  - Cloth folding with dual SO-ARM101s and ACT.
  - Towel folding with dual SO-ARM101s.
  - A matcha robot whose team needed distinct start/end states per subtask.

  — [Seeed recap](https://seeed.cc/post/2025-embodied-ai-hackathon-recap)
- **Simulated SO-101 precedents.** These were verified only through search snippets because the pages redirect.
  - Scripted IK handoff through a "shared staging zone". — [lablab profile](https://lablab.ai/u/@ari517)
  - Arm A hands a cup to Arm B, with VLM-selected scripted primitives and 100% success over 10 seeds. — [lablab entry](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon/inzuppato/bimanual-pick-handoff-place-with-vlm-planning)
- **Payload and torque figures.**
  - Seeed's SO-ARM101 Pro spec sheet lists a 500 g payload. — [Seeed PDF](https://files.seeedstudio.com/Bazaar/product_pdf/100046482.pdf)
  - Another source says ~500 g, varying with pose and temperature. — [roboticscenter.ai](https://www.roboticscenter.ai/hardware/so-101/specs)
  - Follower servo variants differ: 7.4 V around 16.5-19.5 kg.cm stall, 12 V C047 at 30 kg.cm. — [Waveshare wiki](https://www.waveshare.com/wiki/SO-ARM100/101)
  - LeRobot caps the gripper at `Max_Torque_Limit=500` (50%) and `Protection_Current=250`, so grip force is deliberately limited. — so_follower.py
- **Two-arm calibration.** Standard practice is to calibrate each arm separately and then fuse the frames in a second step. Bi-JCR solves both hand-eye transforms and the relative base pose jointly, without markers, from wrist cameras. — [Bi-JCR, arXiv 2505.24819](https://arxiv.org/html/2505.24819v1). Touch-based dual-arm calibration (self-touch or touching a plane) is another method, shown on industrial arms. — [arXiv 2012.07548](https://arxiv.org/abs/2012.07548)

### Inferences
- **Shared frame in about 1 hour.** Bolt both bases to one board at measured offsets (CAD or a printed jig). Then touch the gripper tip of each arm to the same 3-4 printed points and solve the 2D rigid transform plus a z offset. That keeps handoff errors to a few mm.
  - The lab's AprilTag/ChArUco setup can cross-check it with an overhead camera.
  - Calibrate the handoff pose directly as well. Jog arm A to the handoff pose, then teach arm B's grasp pose against the real object there. Store both joint vectors, so the handoff does not depend on frame accuracy at all.

  (inference, not checked)
- **Reliable handoff sequence (scripted state machine).**
  1. A moves to `HANDOFF_PRESENT` and holds.
  2. B moves to `HANDOFF_PREGRASP`, then to `GRASP`.
  3. B closes, and the grasp check runs on B's gripper (stall short of closed, plus load).
  4. Only then does A open and retract.
  5. B gets a timeout; on failure, A keeps holding and B retries once.

  Both arms block on each other's state over the orchestrator, which acts as a barrier and is a visible "sync point" on the Gantt. Use a cube or dowel that both grippers can grip from orthogonal sides, so the fingers do not collide. (inference)
- **Hold plus act.** Arm A grips a peg block steady, or presses it against a fixture. Arm B inserts a peg with a chamfered or loose hole of about 2-3 mm clearance, because SO-101 repeatability will not do tight fits. Use the lab twin's compensation if it is available. (inference)
- **Shared carry.** Position control on both arms holding one rigid object creates internal forces. Use a compliant object (foam bar) or a loose grip on one side, keep the path short, and command both arms from one process so their goals go out in the same tick (sequential within about 1 ms, inference). Avoid it on the critical path. (inference)
- **Collaboration-visible demo ideas.** (inference)
  - **Assembly line with handoff.** Arm 1 picks from the bin and hands off to arm 2, which places into the kit tray. The Gantt shows handoff bars aligned across two lanes, and the LLM DAG shows the cross-arm edge.
  - **Fixture and insert.** One arm holds a part while the other inserts a peg. Inject a failure: B misses, A keeps holding, B retries.
  - **Reach-limited relay.** The goal sits outside arm 1's workspace, so the planner must route the task through a handoff. This shows that collaboration expands reach, not only throughput.
  - **Throughput twist.** Show that with N arms and a shared zone, adding a dedicated handoff "runner" arm can beat N independent pickers when the workspace is split. That is a nontrivial scheduling result worth one plot.
- **Gripper limits.** Keep collaborative parts under about 100-150 g, light and rigid: 3D-printed cubes, dowels, foam. 200 g has been run on the lab arm, but grip under LeRobot's 50% torque cap was not measured. Check the follower servo variant (7.4 V vs 12 V) per arm, because two arms with different variants will have different holding strength. (inference, not checked)

### Gaps
- I found no SO-101 handoff success-rate data on real hardware. The only numbers are from simulation.
- I found no published grip-force figure for the SO-101 gripper under LeRobot's caps.
- I found no LeRobot API for synchronised (parallel) dual-bus writes. The bimanual class is sequential.
