# Operations status (Vato, Aravind)

Working file for the Operations team, on the `orchestrator` branch. Team-wide progress stays in
[`STATUS.md`](https://github.com/termvato/HFTeamA/blob/main/STATUS.md) on `main`; this file is for
what we two need to coordinate. Issue: #21. Backlog: #2 to #18.

Update it when you start, finish or decide something. Notes for each other go under **Notes**.

---

## Now (Sat 21 Nov, 16:30)

1. **Architecture locked** (table below), including the step timeout.
2. **Core loop on fake arms: done.** Planner (greedy, HEFT, CP-SAT), executor with live re-split,
   live screen. Run: `python -m hfteama.view --fake-arms 1`, open http://127.0.0.1:8050.
3. **VLM on top of the loop: first version done** (Aravind): decomposer (camera + Go -> categories
   -> steps), step check plugged into `Executor(verifier=...)`, re-scan. Reachability fills
   `allowed_arms` (SO-101 reach). Objects found by Qwen pointing + camera calibration (no extra
   model). Open: feeding re-scan results into a running executor (Vato, see Notes).

## Who does what (proposed, to confirm)

| Area | Owner | Backlog |
|---|---|---|
| VLM: object detection (pointing), decomposer, step check, re-scan | Aravind | new (to add) |
| Reachability graph and arm registry | Aravind | #3 |
| Planner: greedy fallback, full schedule (HEFT, then CP-SAT), plan checker | Vato | #6, #7, #8, #16 |
| Executor and re-split | Vato | #10, #11 |
| Fake arms, durations | Vato | #13, #14 |
| Countdown screen, what-if estimate, named limit | Vato | #12, #17 |
| Real-arm adapter: real follower as an arm worker (from Hardware's pairs config), safety checks, slow moves and park, heartbeat, runs Action's skills | Vato | #4, #19, #20 |
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
| #15 | Scaling benchmark | not started |
| #16 | Plan validator | done (`scheduler/check.py`), run after every re-plan |
| #17 | Plan vs actual view | done: live screen `python -m hfteama.view` |
| #18 | Re-split policy questions | open |
| new | VLM decomposer, step check, re-scan | code done and tested on a scripted VLM (`perception/`); step check wired as `Executor(verifier=...)`; Qwen3-VL-4B on the RTX 4060 with safeguards: first real-GPU run (`scripts/vlm_smoke.py`) pending, model still downloading; no accuracy or latency numbers yet |
| new | Object detection (camera -> table positions) | code done, tested on a scripted VLM: Qwen points at objects, pixel ray meets the table plane (`perception/pointing.py`); position only, no yaw. OnePose dropped (per-object iOS scans, CUDA 10.2, Linux only); pointing accuracy on the real model not measured yet |

## Notes

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
  - Needs you: (1) the executor cannot take new or changed tasks mid-run, so `rescan()` output
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

- 21 Nov (Aravind): registry + reachability (#3), VLM decomposer / step check / re-scan, Qwen GPU safeguards, executor verifier hook.
- 14:31 Product section in `docs/plan.md`.
- 14:20 Team branches; `orchestrator` = `termvato` + Aravind's design doc.
- 13:56 Names in the code aligned with the backlog (#2, #4, #5, #12).
