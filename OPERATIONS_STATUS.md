# Operations status (Vato, Aravind)

Working file for the Operations team, on the `orchestrator` branch. Team-wide progress stays in
[`STATUS.md`](https://github.com/termvato/HFTeamA/blob/main/STATUS.md) on `main`; this file is for
what we two need to coordinate. Issue: #21. Backlog: #2 to #18.

Update it when you start, finish or decide something. Notes for each other go under **Notes**.

---

## Now (Sat 21 Nov, 14:36)

1. **Architecture locked** (table below). Only the step timeout is still open.
2. **Core loop on fake arms:** full-schedule planner, executor, countdown screen with the what-if
   estimate. Target: running by Saturday evening.
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
| #6 | Greedy allocator | not started |
| #7 | CP-SAT | not started |
| #8 | Shared resources | in the task format and lower bound; not in a planner yet |
| #9 | Two-arm tasks | later extension |
| #10 | Executor | not started |
| #11 | Re-split on change | not started |
| #12 | Finish time and bottleneck | lower bound and named limit done; predicted finish needs the planner |
| #13 | Duration estimates | not started |
| #14 | Mock worker | not started |
| #15 | Scaling benchmark | not started |
| #16 | Plan validator | not started |
| #17 | Plan vs actual view | not started |
| #18 | Re-split policy questions | open |

## Notes

- (Vato): Architecture locked with Aravind: the merged proposal as written. Timeout still open.

- 14:36 (Vato): Product decided (see `docs/plan.md`, Product): visual sorting, Go button, VLM decides
  the categories, estimate with what-if before Go, live countdown, camera check after each step,
  re-scan on scene change, sim arms only in the estimate. The VLM is ours.
- 14:36 (Vato): Hardware has one leader-follower pair working and is setting up Vato's arm.

## Log

- 14:31 Product section in `docs/plan.md`.
- 14:20 Team branches; `orchestrator` = `termvato` + Aravind's design doc.
- 13:56 Names in the code aligned with the backlog (#2, #4, #5, #12).
