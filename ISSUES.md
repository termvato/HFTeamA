# Issues: task-splitting orchestrator

Scope: the **orchestration layer only**. Given a batch of tasks and the arms that are
currently available, decide which arm does which task and in what order. When arms join,
leave or fail, re-split the remaining work and update the predicted finish time.

The orchestrator does not know or care *how* a task is done. Skills, motion, perception,
hardware drivers and policies sit behind the worker interface (#3) and are out of scope here.

Priority: **P0** = needed for a first end-to-end run, **P1** = needed for the demo,
**P2** = nice to have.

---

## Core model

### #1 Task model and task graph (P0)
Define the input format the orchestrator schedules.

- `Task`: `id`, `capability` (what kind of arm can do it), `duration` (default, optionally
  per arm), `needs_arms` (1 by default; 2+ for joint steps), `resources` (shared zones it
  occupies), `status`.
- `TaskGraph`: tasks plus precedence edges (`a` must finish before `b` starts).
- Load and save as JSON. Write a JSON schema for it.
- Validation: the graph is acyclic, ids are unique, every task has at least one capable
  arm in the current registry (otherwise it is reported as unassignable, not silently dropped).

Done when: a hand-written JSON batch loads, validates, and bad inputs give clear errors.

### #2 Arm registry (P0)
Track which arms exist and what state they are in.

- `Arm`: `id`, `capabilities`, `status` (`idle`, `busy`, `down`), optional per-arm speed
  factor or duration overrides.
- Events: `arm_joined`, `arm_left`, `arm_failed`, `arm_recovered`.
- Registry emits these events to the orchestrator; it does not decide anything itself.

Done when: arms can be added and removed at runtime and every change produces one event.

### #3 Worker interface (P0)
One contract between the orchestrator and every arm, real, simulated or mock.

- `execute(task) -> Result`, where `Result = {status, started_at, finished_at}` and
  `status` is one of `ok`, `failed`, `timeout`, `arm_down`.
- `heartbeat()` at a fixed rate; a missed heartbeat window marks the arm `down`.
- `cancel()` for a task that is no longer assigned to this arm.
- Transport between orchestrator and workers (multiprocessing queue, ZMQ, etc.) is hidden
  behind this interface.

Done when: the orchestrator runs unchanged against two different backends (see #13).

### #4 Event log (P0)
A single append-only log that everything else reads from.

- One record per event: timestamp, type, arm id, task id, planned start/end, actual
  start/end, outcome, plan version.
- JSON lines on disk.

Done when: a whole run can be replayed from the log alone.

---

## Splitting the work

### #5 Baseline allocator: greedy list scheduling (P0)
A simple allocator that never fails and runs in milliseconds.

- HEFT-style: rank tasks by longest remaining path to the end of the graph, then put each
  task on the capable arm that finishes it earliest.
- Returns a `Plan`: per arm, an ordered list of tasks with planned start and end times.
- Used as the live fallback, as the comparison baseline, and as the warm start for #6.

Done when: it produces a valid plan (checked by #15) for any valid graph and arm set.

### #6 Optimal allocator: CP-SAT (P1)
Minimise the finish time of the whole batch (makespan) with OR-Tools CP-SAT.

- One optional interval per (task, capable arm), exactly one chosen per task.
- No overlap per arm; precedence as `start_b >= end_a`.
- Warm start from the #5 plan; configurable time limit (start with 5 s).
- Report the solver's bound and gap alongside the plan.
- If the solver times out without a solution, fall back to #5.

Done when: on the same inputs it is never worse than #5, and its solve time is measured
for 2, 4 and 8 arms with 10, 30 and 50 tasks.

### #7 Shared resources (P1)
Some tasks use a shared area or tool that only one arm can use at a time.

- Each resource is capacity 1: no overlap between tasks that use it.
- Supported by both allocators (#5 checks availability when placing; #6 adds a no-overlap
  constraint per resource).
- The executor (#9) also holds a runtime lock, so drift in real durations cannot cause a clash.

Done when: no plan or run ever has two tasks in the same resource at once.

### #8 Tasks that need more than one arm (P2)
Support tasks with `needs_arms >= 2` (for example a handoff between two arms).

- All participating arms start and end the task together.
- Allocators choose a valid arm set; the executor waits until all chosen arms are ready.
- Check plans for circular waits (deadlock): the combined graph of per-arm orders plus
  precedence edges must stay acyclic.

Done when: a graph with joint tasks schedules and executes without deadlock.

---

## Running and re-splitting

### #9 Executor (P0)
Run a plan as a partial order, not as fixed timestamps.

- Keep only the decisions from the plan: which arm, and the order on each arm and resource.
- Dispatch a task as soon as its predecessors are done, its arm is free and its resources
  are free.
- Write every dispatch and result to the event log (#4).

Done when: a plan runs to completion on mock workers, and an arm that runs slow delays
only the tasks that depend on it.

### #10 Re-split on change (P0)
The core feature: when the situation changes, re-plan the remaining work.

Triggers:
- an arm joins (more capacity),
- an arm leaves or fails (its unfinished tasks go back to the pool),
- a task fails (re-queue it, on the same arm or another capable one, with a retry limit),
- a task overruns its predicted duration by more than a threshold.

Rules:
- Finished tasks stay finished; running tasks are frozen in place.
- Re-plan with #6 (short time limit, 1–2 s) warm-started from the old plan; fall back to #5.
- Bump the plan version and log old and new predicted finish times.

Done when: adding or removing an arm mid-run changes the plan without restarting anything,
and no task is lost or run twice.

### #11 Finish-time prediction and bottleneck (P1)
Tell the user when the batch will finish and what is limiting it.

- Predicted finish = end of the current plan, updated on every re-split and on each result.
- Lower bound = max of: total remaining work / number of arms, longest remaining chain of
  dependent tasks, busiest shared resource.
- Name the binding term: **arms** (more arms would help), **task order** (dependencies cap
  it), or **resource** (a shared zone is the bottleneck).

Done when: every plan version comes with a predicted finish, a lower bound and a named bottleneck.

### #12 Duration estimates (P1)
Plans are only as good as their durations.

- Start from defaults in the task graph.
- Update per (arm, capability) from measured durations in the event log; plan with a high
  percentile (p80) rather than the mean.
- Allow an external estimator (e.g. a digital twin) to supply durations through the same interface.

Done when: predicted vs actual durations are logged and the gap shrinks over a run.

---

## Testing and evaluation

### #13 Mock worker backend (P0)
A fake arm for developing the orchestrator without any hardware.

- Implements #3; sleeps for the task duration times a configurable noise factor.
- Can be told to fail a task, time out, or drop its heartbeat on command.
- Optional speed-up factor so long runs can be tested quickly.

Done when: the full loop (#5, #9, #10) runs on any number of mock arms.

### #14 Scaling benchmark (P1)
Show how finish time changes with the number of arms.

- Run the same batch with N = 1, 2, 3, 4, 8 mock arms using #5 and #6.
- Plot measured finish time against N, the ideal 1/N line and the #11 lower bound.
- Also compare against a single arm doing everything in sequence.

Done when: one script produces the plot and a table of numbers.

### #15 Plan validator (P0)
An independent checker used in tests and after every re-split.

- Every task assigned exactly once to a capable arm.
- No overlap per arm or per resource; precedence respected; joint tasks aligned.

Done when: it runs in CI on every allocator test.

### #16 Plan vs actual view (P2)
A live timeline per arm: planned bars faded, actual bars solid, failures marked, plus the
current predicted finish time and bottleneck. Built only from the event log (#4).

---

## Out of scope for this layer
- How tasks are executed: skills, IK, policies, grasp detection, hardware drivers.
- Turning natural language into a task graph. An LLM front end can be added later as a
  separate adapter that outputs the #1 format.
- Collision avoidance in motion; the orchestrator only prevents two arms from booking the
  same resource.

## Open questions
- Re-split policy: always re-plan everything that is not running, or only move the tasks
  that are affected?
- Should a newly joined arm wait for the next re-plan or take ready tasks straight away?
- Retry limit and what happens to a task that keeps failing on every arm.
