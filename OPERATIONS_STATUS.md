# Operations status (Vato, Aravind)

Working file for the Operations team, on the `orchestrator` branch. Team-wide progress stays in
[`STATUS.md`](https://github.com/termvato/HFTeamA/blob/main/STATUS.md) on `main`; this file is for
what we two need to coordinate. Issue: #21. Backlog: #2 to #18.

Update it when you start, finish or decide something. Notes for each other go under **Notes**.

---

## Now (Sat 21 Nov, 15:06)

1. **Architecture locked** (table below). Only the step timeout is still open.
2. **Core loop on fake arms: done.** Planner (greedy, HEFT, CP-SAT), executor with live re-split,
   live screen. Run: `python -m hfteama.view --fake-arms 1`, open http://127.0.0.1:8050.
3. **VLM on top of the loop:** decomposer (camera + Go -> categories -> steps), check after each step,
   re-scan when the scene changes.

## Who does what (proposed, to confirm)

| Area | Owner | Backlog |
|---|---|---|
| VLM: decomposer, step check, re-scan | Aravind | new (to add) |
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
| Step timeout | 1.5x or 3x the estimate | open |
| `same_arm_as` | keep: the arm holding a part also places it | locked |

## Backlog state

| # | Item | State |
|---|---|---|
| #2 | Task model and task graph | first version on `main` (no JSON schema file yet) |
| #3 | Arm registry | not started |
| #4 | Worker interface | first version on `main`, pending the contract decision |
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

## Notes

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

- 14:31 Product section in `docs/plan.md`.
- 14:20 Team branches; `orchestrator` = `termvato` + Aravind's design doc.
- 13:56 Names in the code aligned with the backlog (#2, #4, #5, #12).
