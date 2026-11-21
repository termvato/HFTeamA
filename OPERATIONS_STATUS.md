# Operations status (Vato, Aravind)

Working file for the Operations team, on the `orchestrator` branch. Team-wide progress stays in
[`STATUS.md`](https://github.com/termvato/HFTeamA/blob/main/STATUS.md) on `main`; this file is for
what we two need to coordinate. Issue: #21. Backlog: #2 to #18.

Update it when you start, finish or decide something. Notes for each other go under **Notes**.

---

## Now (Sat 21 Nov, 18:05)

1. **Architecture locked** (table below), including the step timeout.
2. **Core loop on fake arms: done.** Planner (greedy, HEFT, CP-SAT), executor with live re-split,
   live screen. Run: `python -m hfteama.view --fake-arms 1`, open http://127.0.0.1:8050.
3. **VLM on top of the loop: done** (Aravind): object detection by Qwen pointing, decomposer
   (camera + Go -> categories -> steps), step check (`Executor(verifier=...)`, place steps only),
   re-scan (`Executor.update_graph`). Reachability fills `allowed_arms` (SO-101 reach). Real Qwen
   run on the RTX 4060 passed on a synthetic image; a real camera photo is still to test.
4. **Real top camera, end to end (dry run): done** (Aravind). C920 -> ArUco calibration -> Qwen
   object positions in metres -> reach -> executor split over the two arms (fake workers, no
   motion): `scripts/real_table_plan.py --device "HD Pro Webcam C920"`. 1 arm 53 s, 2 arms 37 s.
5. **Real arms:** see Blockers. Aravind's part for Vato to verify: last section of this file.

## Blockers: real arms

| # | Item | Who | State |
|---|---|---|---|
| 1 | Real-arm worker (skills, IK, driver behind `execute(task)`) | Action | being built separately; integrated later through `Executor.add_arm(worker)`. Not an Operations blocker |
| 2 | Arm base poses in the table frame (marker 0). Nobody knows where the markers sit relative to the arms, so the arms measure it themselves: jog the gripper tip onto two marker centres, read the joints, FK (`arms/kinematics.py`) gives base x, y, yaw. ~2 min per arm. Until then: estimated from the image (+-3-5 cm; spacing checked against the measured 30 cm gap) | Aravind (tool), Hardware (jogging) | open |
| 3 | Machines: decided (Aravind, 17:55): once everything is merged on `main` and verified, the real run happens on Aravind's laptop (RTX 4060): arms, cameras, Qwen and the executor in one process, so no network service is needed. Catch: the laptop is Windows, and Hardware's and Sahil's code assumes Linux (`/dev/ttyACM*`, `/dev/v4l/by-id/...`, `/sys`, `/proc`). Needs Windows entries (COM ports, camera names; the C920 already works through ffmpeg/DirectShow) or WSL with USB passthrough | Hardware, Aravind | open: settle before integration |
| 4 | Grasp angle: pointing gives position only. IK cannot know the angle (it needs one as input). Cheap fix: Qwen gives two points along each object's long axis -> yaw in task args (`pick_yaw_deg`). A VLA can learn it from the wrist camera if its data has varied orientations (Sahil) | Aravind (yaw), Action (uses it) | open |
| 5 | Arg names: `object`, `target`, `pick_xyz`, `place_xyz` (metres, table frame) | Action, Operations | agreed in both designs; to confirm on #20 |
| 6 | Third robot for the plug-in test (an arm joins mid-run, work re-splits): needs its catalog id (its follower calibration id) and base pose (same touch tool as #2). Placed where it reaches objects the other two cannot, it also unblocks steps | Hardware, Aravind | open |

## Who does what (proposed, to confirm)

| Area | Owner | Backlog |
|---|---|---|
| VLM: object detection (pointing), decomposer, step check, re-scan | Aravind | new (to add) |
| VLM pipeline into the live screen: Go -> camera -> detect -> sort -> reachability -> executor; periodic re-scan via `update_graph` | Aravind | new (to add) |
| Reachability graph and arm registry | Aravind | #3 |
| Planner: greedy fallback, full schedule (HEFT, then CP-SAT), plan checker | Vato | #6, #7, #8, #16 |
| Executor and re-split | Vato | #10, #11 |
| Fake arms, durations | Vato | #13, #14 |
| Countdown screen, what-if estimate, named limit | Vato | #12, #17 |
| Task format, arm contract, event log | both | #2, #4, #5 |

## Architecture: locked Sat 21 Nov (Vato and Aravind)

| Topic | Decision | State |
|---|---|---|
| Arm contract | one worker per arm, `execute(task) -> Result`; worker reads only the task's id, skill and args (args carry object id, target bin, table coordinates); outcomes `ok`, `failed` + reason, `timeout`, `arm_down`, with start and end times. `execute(arm_id, skill, params)` can wrap it at the dispatcher | locked |
| Planner | full schedule on every re-plan (needed for the countdown and what-if); greedy as fallback | locked |
| Named limit | keep it on screen (arms, order, shared area), plus steps unblocked by a new arm | locked |
| Task format | the VLM writes the format on `main` (`depends_on` -> `deps`, `object`/`target` -> `args`) | locked |
| Code and interfaces | code in `src/hfteama/`; task format, arm contract and event log change only with all teams | locked |
| Re-plan rule | adopt the 10% hysteresis, except when arms change or something fails | locked |
| Step timeout | adaptive: 3x the estimate while durations are guesses, 1.5x the arm's measured 80th percentile once measured (Vato, 21 Nov) | locked |
| `same_arm_as` | keep: the arm holding a part also places it | locked |

## Backlog state

| # | Item | State |
|---|---|---|
| #2 | Task model and task graph | first version on `main` (no JSON schema file yet) |
| #3 | Arm registry | done: `registry/` catalog (`configs/arms.json`), SO-101 reach (0.44 m fully extended, hip +-110 deg, mounting yaw), `resolve()` fills `allowed_arms`; live arm state stays in the executor |
| #4 | Worker interface | locked; in `arms/base.py` |
| #5 | Event log | first version on `main` |
| #6 | Greedy allocator | done: greedy fallback + HEFT full schedule (`scheduler/greedy.py`, `heft.py`) |
| #7 | CP-SAT | done (`scheduler/cpsat.py`); benchmark in `scripts/bench_planners.py`: matches HEFT, proves optimal < 2 s at 4-8 arms, hits the limit at 2 arms; HEFT stays the live default |
| #8 | Shared resources | done: in every planner and as runtime locks in the executor |
| #9 | Two-arm tasks | later extension |
| #10 | Executor | done (`executor/executor.py`) |
| #11 | Re-split on change | done: re-plan on every event, running steps fixed, 10% hysteresis, retries, heartbeat, timeout, held parts, slow-stop removal |
| #12 | Finish time and bottleneck | done: predicted finish, lower bound, limit in words, steps unblocked by a new arm, what-if |
| #13 | Duration estimates | done: task file first, then per-arm 80th percentile (`scheduler/durations.py`) |
| #14 | Mock worker | done: fake arm with failure injection (`arms/fake.py`) |
| #15 | Scaling benchmark | done (Vato): `scripts/scaling_benchmark.py`, sort12 and assemble3 on 1-8 arms: planned (HEFT), lower bound and binding term, ideal 1/N, measured on fake arms (1-4); CSV + interactive chart under `runs/scaling/` |
| #16 | Plan validator | done (`scheduler/check.py`), run after every re-plan |
| #17 | Plan vs actual view | done: live screen `python -m hfteama.view` |
| #18 | Re-split policy questions | open |
| new | VLM decomposer, step check, re-scan | done; step check wired as `Executor(verifier=...)`. Real run on the RTX 4060 (Qwen3-VL-4B, 4-bit, synthetic table image, `scripts/vlm_smoke.py`): sorting by colour right on the first try, 12 s (once at Go); step check correct, 2.8 s per call; peak VRAM 3.1 GB. Not yet on a real camera photo |
| new | VLM pipeline into the live screen | next (Aravind) |
| new | Camera calibration from the table's ArUco markers | done (Aravind): `perception/markers.py`, 4x4 tags, 7 cm; focal length + camera pose fitted on all corners (0.7 px). Real C920: f 1412-1431 px, camera 0.74 m above the table, stable between frames within 1 cm. Table config `configs/hacklab_table.json` |
| new | Distance-aware step times | done (Aravind): `resolve(..., travel_s_per_m=...)` gives each arm its own duration by distance, so the planner picks the nearer arm; measured durations take over once recorded |
| new | Object detection (camera -> table positions) | done: Qwen points, pixel ray meets the table plane (`perception/pointing.py`). Real run: 6/6 objects, error mean 0.6 cm, max 0.9 cm, 8.5 s per scan (synthetic image). Position only, no yaw. OnePose dropped |

## Notes

- 18:48 (Vato): **Isaac Sim is dropped** (#33 closed; scripts and the joint streamer removed).
  **The demo is filmed on the real setup:** one arm sorting, the second arm is plugged in, the
  screen shows the re-planning moment, the predicted finish time comes down, both arms work
  together.

- 18:27 (Vato): **RTX rehearsal bridge works (#33):** `view/joint_stream.py` streams each simulated
  arm's joints over UDP; `python -m hfteama.demo --backend sim --isaac HOST:PORT` turns it on.
  Checked: Isaac Sim (headless, 60 fps, 2.5 GB) followed Action's MuJoCo arms run by our executor
  for 4179 arm-frames. Still open: the Isaac scene uses its own 3-arm layout (not the table
  config) and does not show carried cubes. Seen in that run: with sort12 on 3 sim arms under our
  executor, `pick_11` ended in `stall` on two arms in a row (2 steps done in 25 s); Action's own
  harness completes 24/24, so the difference (which arm takes which block, or lock timing around
  the shared bin) needs a look together with Vivek before a sim rehearsal is shown.

- 18:24 (Vato): Three executor / planner changes on `orchestrator`:
  1. **Lost part waits for a re-scan:** `Executor(rescan=fn)` where `fn() -> TaskGraph | None`
     (camera -> re-scan -> resolved graph). After a failed place, or an arm lost while holding a
     part, that object's pick and place are held until `fn` returns, then `update_graph` applies it.
     Without the hook the old behaviour stays (retry at the old position). Aravind: this is where
     `rescan(...)` + `resolve(...)` plugs in.
  2. **Lower bound is tight for shared areas:** earliest start + total time + shortest work that
     must follow. sort12 with 3+ arms: 54 s (was 48), equal to the best plan; assemble3: 41 s (was
     31) and the limit is named correctly as the output tray, not "order".
  3. **`method="auto"`** (demo default): CP-SAT when arms change or something fails (up to 40
     pending steps, 1 s limit), HEFT for the frequent re-plans after each finished step.

- 18:09 (Vato): **VLM -> reach -> executor -> IK validated in MuJoCo** on the hacklab layout, with
  `orchestrator` + `action-arm-skills` merged locally (scripts in `scripts/validation/`; they need
  Action's arm code). Results:
  1. Interfaces fit with no glue code: `sort_steps` tasks (`pick_xyz` at z = 0.02, `place_xyz` =
     bowl) run unchanged through `resolve` -> `Executor` -> `SkillWorker` -> IK. 4 of 5 objects
     sorted by two arms; each alone sorts the 5th too. Bowl placement: IK error 0.4 mm, both arms.
  2. **Reach is too generous (for Aravind):** `SO101_REACH` says 0 to 0.44 m; the pick skill's IK
     (tool down, hover + grasp + 10 cm lift) solves only from about 0.14 m to 0.42 m from the pan
     axis. On a 4 cm grid, 33 of 171 points the planner calls reachable fail in IK (36 of 173 for
     the other arm); none the other way round. Proposed: `r_min=0.14, r_max=0.42`. I tried it:
     `test_so101_reach_sector` and `test_vlm_plan_runs_on_executor_with_step_check` encode the old
     numbers and need their points moved, so the change is left to you.
  3. **Arms can block each other outside the bowl (likely, not confirmed by contact inspection):**
     with both arms working, an object between them (0.30, 0.30) ended in `stall` twice, while
     either arm alone sorts it. Only the bowl is a shared resource today. Proposed: `resolve` marks
     every pick point both arms can reach as a shared resource (the overlap zone of the design doc).
  4. **After a failed place the part is not where it was:** the executor re-queues the pick at the
     old `pick_xyz`, which then fails. With the camera this needs a re-scan + `update_graph` before
     the retry (Operations, mine).

- 18:05 (Vato): **Demo runner built** (`python -m hfteama.demo`, `src/hfteama/demo.py`): executor +
  plug-in detection + live screen; `--backend fake|sim|real` (sim and real load Action's
  `SkillWorker` when that code is in the checkout); manual plug / remove / pull-cable controls on
  the screen as the stage fallback. Checked first: nothing like it existed (Vivek's
  `scripts/sim_demo.py` is a harness with its own mini-scheduler, no executor or screen).
  **Hooks for Aravind:** `--planner module:function` must return a resolved `TaskGraph`
  (camera -> calibration -> Qwen -> decompose -> `registry.resolve`); `--verifier module:function`
  must return `StepChecker(...).as_verifier(grab)`. Re-scan can call `executor.update_graph`.
  Docs updated: README (Run the demo), `docs/plan.md` (layout, state), `docs/README.md`.

- 17:59 (Vato): **Plug-in detection built** (`src/hfteama/discovery.py`, extra `arms` = pyserial):
  `ArmWatcher(executor, followers_from_pairs("configs/teleop_pairs.json"), factory)` polls USB
  serial numbers; a known follower appearing -> `factory(arm_id)` -> `executor.add_arm(worker)`;
  disappearing -> new `executor.arm_lost(arm_id)` (drops it at once). Debounced; a failed scan or a
  worker that refuses to start is survived. `plug(arm_id)` / `unplug(arm_id)` are the manual stage
  fallback. Tested on fake arms; the real factory is Action's worker, wired once Hardware has run it.
- 17:59 (Vato): **One file for arm positions, Action's format:** `registry.load_real_arms(
  "configs/real_arms.json")` builds the planner's catalog from the file Vivek's driver reads
  (`base` metres, `yaw` radians, same convention as `base_yaw_deg`). Use it instead of the `arms`
  list in `hacklab_table.json` for real runs, with arm ids = follower ids (`rakesh_follower`,
  `vato_follower`). Aravind: the marker-touch poses should be written into `real_arms.json`.
- 17:59 (Vato): **Proposal for blocker 3 (real run on the Windows laptop):**
  1. 15-minute test first, before anything else is moved: on the laptop, `pip install -e ".[arms]"`,
     `python -c "from hfteama.discovery import connected_serials as c; print(c())"` with one
     follower plugged in (must print its serial), then Hardware's `hfteama --pair rakesh teleop`.
  2. What is already Windows-safe: plug-in detection and Hardware's port lookup both go by USB
     serial number through pyserial (COM ports, no `/dev` paths, no udev, no dialout group); the
     top camera through ffmpeg/DirectShow as Aravind runs it.
  3. What is not: `teleop_pairs.json` names the wrist cameras by `/dev/v4l/by-id/...` (Linux only;
     the orchestrated run does not need wrist cameras unless a VLA skill is used), and LeRobot on
     Windows is untested by us.
  4. If the test fails, fallback without new hardware: arms, top camera and executor on a
     Hardware Linux machine; Qwen stays on the laptop as a small HTTP service (frame in, JSON out;
     about 200 KB per frame), as in Aravind's earlier option.

- 17:20 (Aravind): Real top camera tested end to end. The C920 is on Aravind's laptop (Windows:
  frames through ffmpeg/DirectShow; OpenCV could not open it by index). Calibration comes from the
  ArUco markers already on the table, every run, so moving the camera only needs a new frame.
  Qwen at 1920 px finds all objects (1024 px missed one small object); positions agree with the
  image. Table frame = marker 0, y towards the arms. Arm bases still estimated (Blockers #2).

- 16:41 (Aravind): Taking "VLM pipeline into the live screen" from Vato's 16:32 list. It touches
  `view/__main__.py` (Go): I'll add the VLM path next to the fake-arm rig and keep the rig as is.
  Thanks for `update_graph`, the catalog what-if and the place-only check. Camera: no frame of the
  real table yet (none committed; Sahil's `top` C920 is only live on his machine).
  `scripts/vlm_smoke.py --image http://<host>:8765/stream/top --out points.png` tests it once
  the studio runs.

- 16:32 (Vato): **Step check: only place steps** (answer to Aravind's GPU-load note). The demo passes
  `Executor(verifier=..., verify_skills=PLACE_SKILLS)`; the executor default stays "every step", so
  existing tests keep their meaning. Next for Operations (all three): wire the VLM pipeline into the
  live screen, scaling benchmark (#15), RTX rehearsal (#33).

- 16:16 (Vato), answers to Aravind's three asks, on `orchestrator`:
  (1) `Executor.update_graph(graph)`: pass `rescan(...).graph` after `registry.resolve`; done and
  running steps are kept even if left out, new steps join, dropped pending steps go, changed args
  apply, then a forced re-plan (also works after the batch finished).
  (2) `Executor(catalog=[...all arm ids...])`: the what-if row adds the catalog's unplugged arms
  (with their reach) before any virtual arm.
  (3) Fake-arm tests now run at 10x, so Windows' ~15.6 ms timer tick no longer breaks them.

- 16:14 (Vato): Demo adds an RTX rehearsal: the plan plays in Isaac Sim first, then on the real arms. Operations builds it; Isaac Sim runs on a second machine. Arm motions in the sim should come from Action's skills (MuJoCo track) so sim and real move the same way.

- 16:09 (Vato): Real-arm adapter withdrawn: Action already plans it (Vivek's `SkillWorker` + `LeRobotDriver` on `action-arm-skills`). Operations posted LeRobot safety notes and offered to take his Task 8 (#20).

- 21 Nov (Aravind): First real Qwen run on the RTX 4060 passed (numbers in the backlog). Watch
  the step check: one GPU, one call at a time, 2.8 s each. With 3 arms finishing a step every
  ~5 s, checking every pick and place asks ~1.7 s of GPU per second and the checks queue up.
  Checking only places (or only when the camera sees a change) halves that.

- 21 Nov (Aravind): Built and on `orchestrator`: `src/hfteama/registry/` (#3) and
  `src/hfteama/perception/` (VLM). Objects: Qwen points at each one (0..1000 coords), the camera
  ray through that pixel meets the table plane at the objects' top height. Needs from Hardware's
  calibration: camera pose in an arm's base frame plus intrinsics K at the image size
  (`configs/camera.example.json`). Position only; grasps are from the top (no yaw).
- 21 Nov (Aravind): Pose estimator decision: OnePose (and SAM-6D) dropped. OnePose needs an iOS
  scan per object, CUDA 10.2 (does not run on the RTX 4060) and Linux; the VLM already sees the
  objects.
- 21 Nov (Aravind), for Vato, wiring:
  - `scene = build_scene(VLMPointDetector(vlm, cal).detect(image), cal)`, then
    `Decomposer(vlm, load_bins(...)).decompose(image, scene)` -> task graph (pick + place per
    object, `same_arm_as`, `args.pick_xyz` / `place_xyz`, place holds the bin as a resource).
  - `registry.resolve(graph, load_catalog("configs/arms.json").values()).graph` -> `Executor(graph)`.
    Pass the whole catalog: `allowed_arms` lists every arm that could reach a step, and your
    `blocked_tasks` / `unblocked_by` do the rest. Worker `arm_id`s must match catalog ids.
  - `Executor(verifier=StepChecker(vlm, bins).as_verifier(camera_grab))`: one camera check per
    step; camera or VLM errors count as not confirmed.
  - Needs you (all three answered by Vato at 16:16): (1) the executor cannot take new or changed tasks mid-run, so `rescan()` output
    (new objects, moved, vanished) has nowhere to go yet: an `Executor.update_graph(graph)` that
    keeps done/running steps would do it. (2) `whatif` adds `virtual-k` arms that only get steps
    open to every arm; with reach-filled `allowed_arms` the rows above the connected count drop
    out. Using the catalog's unplugged arms as the extra arms would fix it.
    (3) `tests/test_fake_arm.py` fails 2 tests on Windows (1000x clock: 4 ms steps vs ~15.6 ms
    timer tick); fine on Linux CI.
- 21 Nov (Aravind): VLM = Qwen3-VL-4B-Instruct from Hugging Face on the RTX 4060 Laptop (8 GB):
  auto 4-bit, VRAM preflight and 85% cap, no CPU offload, image/token/time caps, OOM retry,
  one call at a time, thermal guard (85 C). Smoke test: `python scripts/vlm_smoke.py`.

- 15:31 (Vato): Operations builds the real-arm adapter (`arms/real.py`), on top of Hardware's `configs/teleop_pairs.json` and calibrations. Hardware and Action notified on #19 and #20.

- 15:06 (Vato): Vato's backlog items are built and tested on fake arms (#6-#8, #10-#14, #16, #17).
  Step timeout is a setting (`Executor(timeout_factor=...)`), default 3x until we decide.
  The VLM step check plugs in as `Executor(verifier=...)`: `verifier(task, result) -> (ok, why)`.
  Real arms join through `Executor.add_arm(worker)` (Hardware's discovery calls it).

- (Vato): Architecture locked with Aravind: the merged proposal as written. Timeout still open.

- 14:36 (Vato): Product decided (see `docs/plan.md`, Product): visual sorting, Go button, VLM decides
  the categories, estimate with what-if before Go, live countdown, camera check after each step,
  re-scan on scene change, sim arms only in the estimate. The VLM is ours.
- 14:36 (Vato): Hardware has one leader-follower pair working and is setting up Vato's arm.

## Log

- 17:55 (Aravind): real run on Aravind's laptop after the merge to `main` (no Qwen service needed); Windows paths flagged; third robot for the plug-in test.
- 17:20 (Aravind): real camera end to end (markers, Qwen positions, reach, executor split); blockers table for the real arms.
- 16:41 (Aravind): status tidied (#15 done by Vato, asks answered); Aravind takes the VLM -> live screen wiring.
- 21 Nov (Aravind): registry + reachability (#3), VLM decomposer / step check / re-scan, Qwen GPU safeguards, executor verifier hook.
- 14:31 Product section in `docs/plan.md`.
- 14:20 Team branches; `orchestrator` = `termvato` + Aravind's design doc.
- 13:56 Names in the code aligned with the backlog (#2, #4, #5, #12).

## Aravind's side: for Vato to verify (21 Nov, 17:40)

Rig for the real run (17:55): 2 follower arms (`rakesh_follower` black/left, `vato_follower`
green/right) as arm workers; 2 leader arms (teleop and recording only, not used by the
orchestrator); 2 wrist cameras (for the grasping skill / VLA); 1 top camera (C920: calibration,
objects, step check); a third robot to plug in mid-run. All on Aravind's laptop after the merge to
`main`.

All on `orchestrator`. Verified on Aravind's laptop: `ruff check .` clean, `pytest` 125 passed
(2 skipped: no ortools / dash locally). Please run the checks below and tick them.

### What is built

| Piece | Where | What it does |
|---|---|---|
| Arm registry, reach (#3) | `registry/` | catalog from config; SO-101 reach 0.44 m from the hip axis, hip +-110 deg, mounting yaw; `resolve(graph, arms, travel_s_per_m)` fills `allowed_arms` and per-arm durations by distance (nearer arm is faster) |
| Camera calibration | `perception/markers.py` | camera pose + focal length from the ArUco tags on the table (4x4, 7 cm), every run; table frame = marker 0 |
| Object detection | `perception/pointing.py` | Qwen points at every object; pixel ray meets the table plane -> metres |
| Decomposer | `perception/decomposer.py` | Qwen picks categories and bins; pick + place per object (`same_arm_as`, `pick_xyz`, `place_xyz`, bin as resource); validator + 3 retries |
| Step check | `perception/check.py` | `StepChecker(...).as_verifier(grab)` for `Executor(verifier=..., verify_skills=PLACE_SKILLS)`; errors = not confirmed |
| Re-scan | `perception/rescan.py` | new / moved / vanished / dropped objects -> graph for `Executor.update_graph` |
| Qwen client | `perception/vlm.py` | Qwen3-VL-4B, local, 4-bit on 8 GB; VRAM check and 85% cap, no CPU offload, time and size caps, OOM retry, one call at a time, 85 C guard |
| Table config | `configs/hacklab_table.json` | both arms (bases estimated), bowl, marker settings |
| Dry run | `scripts/real_table_plan.py` | top camera -> calibration -> Qwen -> reach -> your executor on fake workers, 1..N arms, plus a picture |

### Results on the real table (C920, 1920x1080, RTX 4060)

| Check | Result |
|---|---|
| Calibration | fit error 0.7 px; focal 1412-1431 px; camera 0.74 m above the table; same within 1 cm on two frames |
| Arm spacing from the camera vs measured | 0.411 m vs 0.40 +-0.02 m (30 cm gap + ~10 cm base) |
| Objects found (Qwen at 1920 px) | 5/5 on frame 1 (4/5 at 1024 px), 4/4 on frame 2; points on the objects; labels loose |
| Sorting / step check | one try; check correct ("not in the bowl yet") |
| Time per Qwen call | find objects 11 s, sort 6-12 s, step check 2.7 s; first call +12 s to load |
| Split (placeholder step times) | 1 arm 52.9 s; 2 arms 36.5 s, each object to the nearer arm |

### Checks for Vato

- [x] `pytest` and `ruff check .` pass in CI (CI has ortools and dash; my laptop does not).
- [x] `pytest tests/test_perception.py -k executor`: VLM plan -> reach -> your executor -> camera
      check, one check per step.
- [x] `pytest tests/test_registry.py -k travel`: HEFT gives each object to the nearer arm.
- [x] Interfaces read right: `resolve(..., catalog arms)` for `allowed_arms`; `as_verifier` matches
      `Verifier`; `rescan(...).graph` for `update_graph`; worker `arm_id`s = catalog ids
      (`rakesh_follower` black/left, `vato_follower` green/right).
- [ ] Optional, needs the camera and a GPU (`pip install -e ".[vlm,camera]"`):
      `python scripts/real_table_plan.py --device "HD Pro Webcam C920"`.

Verified by Vato 17:15: ruff clean, 135 tests pass with ortools and dash; CI green on `74d6210`;
the executor and travel tests pass; `resolve()`, `as_verifier` (returns `(ok, why)`), `rescan().graph`
and catalog ids (`rakesh_follower`, `vato_follower`) all fit the executor. Watch: Action's handoff
example names the real worker `arm_a`; it must use the catalog id or reach and planning will not
match (flagged on #20). The camera run (last item) needs your laptop.

### Open on Aravind's side

1. Arm base poses by touching two markers with each gripper, all three arms (Blockers #2, #6).
2. Windows on the laptop: camera and arm access without Linux paths (Blockers #3, with Hardware).
3. Grasp yaw from two points along each object (Blockers #4).
4. VLM pipeline into the live screen (Go -> camera -> Qwen -> executor; periodic re-scan).
