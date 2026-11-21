# Algorithms for splitting a manufacturing task across N robot arms (makespan minimisation)

Scope: problem formulation, shared-zone modelling, algorithm choice, online rescheduling, speed-up limits, code pointers. Emphasis: what a team can build correctly in 48 h. Researched 21 Nov 2026. Sources were read in full where marked; a few claims come only from search-result snippets and are flagged.

## 1. How is the problem formalised, and which variant fits "N arms, task graph of pick/place/assemble, different reach, shared zones"?

### Takeaway
In MRTA terms this is **ST-SR-TA with cross-schedule dependencies (XD)** in Korsah's iTax: each arm does one task at a time, each subtask needs one arm (unless a two-arm handover/hold), and the plan covers time. In OR terms that is a **flexible job-shop / multi-mode RCPSP with general precedence (a DAG, not just chains)**. Reach becomes the set of allowed arms per task, and shared zones become extra unary (capacity-1) resources. Minimise Cmax.

### Cited Findings
- Gerkey & Matarić (IJRR 23(9):939-954, 2004) classify MRTA on three axes: single-task vs multi-task robots (ST/MT), single-robot vs multi-robot tasks (SR/MR), and instantaneous vs time-extended assignment (IA/TA). The taxonomy assumes independent tasks. - [Sims et al. survey, arXiv 2207.09650](https://arxiv.org/pdf/2207.09650); [Gerkey & Matarić PDF](https://cse-robotics.engr.tamu.edu/dshell/cs689/papers/gerkey04formal.pdf)
- ST-SR-IA is the optimal assignment problem, solvable in polynomial time (Hungarian method). The rest of the space is strongly NP-hard, and ST-SR-TA is a machine-scheduling problem. (Secondhand via a copy of Korsah et al.; not checked against the original.) - [search summary of Korsah IJRR copy](https://www.scribd.com/document/599483070/Korsah-IJRR-Taxonomy); the Hungarian method as a centralised polynomial method for linear assignment is confirmed in [Sims et al.](https://arxiv.org/pdf/2207.09650)
- Korsah, Stentz & Dias (IJRR 2013, "iTax") extend Gerkey-Matarić with dependency classes: no dependency (ND), in-schedule (ID), cross-schedule (XD) and complex (CD). They also map categories to OR models. - [Sims et al.](https://arxiv.org/pdf/2207.09650); [CMU RI page](https://www.ri.cmu.edu/publications/a-comprehensive-taxonomy-for-multi-robot-task-allocation)
- Nunes, Manner, Mitiche & Gini (RAS 2017) extend the taxonomy with temporal constraints. Their two classes are time windows (TA:TW) and synchronisation/precedence (TA:SP). - [Nunes et al. PDF](https://www-users.cs.umn.edu/~gini/papers/NunesRAS.pdf); [Sims et al.](https://arxiv.org/pdf/2207.09650)
- The closest published analogue is Tercio (Gombolay, Wilcox & Shah, RSS 2013 / IEEE T-RO 34(1) 2018). It is centralised allocation plus scheduling for robot teams in aircraft final assembly, with upper/lower-bound temporal constraints and **spatial proximity constraints**. The authors say optimally scheduling n ≥ 3 tasks of this kind is strongly NP-hard. - [Gombolay T-RO 2018 PDF](https://people.csail.mit.edu/gombolay/Publications/Gombolay_TRO_2018.pdf)
- A dual-arm ABB YuMi model in MiniZinc (Constraints, 2023) integrates layout, task allocation, sequencing, scheduling, cyclic scheduling and collision avoidance in one CP model. The authors call it the first CP model to do so. - [Springer, Constraints 2023](https://link.springer.com/article/10.1007/s10601-023-09345-4) (abstract and search snippet only; full text blocked)
- The FJSP form used by OR-Tools: each task picks one machine from its alternatives, each with its own duration. Machines are no-overlap resources, precedence links tasks, and the objective is minimum makespan. - [OR-Tools flexible_job_shop_sat.py](https://github.com/google/or-tools/blob/stable/examples/python/flexible_job_shop_sat.py)
- PyJobShop supports FJSP, assembly scheduling, RCPSP, multi-mode RCPSP, RCPSP with generalised precedence, sequence-dependent setup times, machines and renewable/non-renewable resources. It does **not** support multi-resource tasks or preemption. - [PyJobShop, arXiv 2502.13483](https://arxiv.org/html/2502.13483v1)

### Inferences
- Mapping for the hackathon problem:
  - arm = machine
  - subtask = task with modes, one per arm that can reach it, each with its own duration
  - task-graph edge = precedence, from end to start, optionally with a minimum lag
  - shared zone = a capacity-1 resource that a task occupies while it is in the zone
  - handover or two-arm assembly = an MR task: two intervals forced to the same start (sync constraint)
  
  This is FJSP with a DAG, which is multi-mode RCPSP. It falls in iTax XD, and in iTax CD only if the decomposition itself is chosen jointly.
- Arm travel between task locations is a **sequence-dependent setup time**. It matters when arms move far relative to task duration. A cheap way to cover it is to add a fixed approach/retract time to each task's duration.
- Two-arm tasks move the problem to ST-MR-TA, which is much harder for auctions and greedy methods but trivial to express in CP-SAT.

### Gaps
- I did not read the original Gerkey-Matarić or Korsah full text. The polynomial/NP-hard classification is secondhand.
- Full text of the YuMi CP paper (solvers, instance sizes, solve times) was paywalled.

## 2. How are shared workspace / collision zones modelled, and how do multi-arm systems coordinate timing?

### Takeaway
There are two layers. **Scheduling layer:** zones are mutex (unary) resources booked over time intervals, i.e. a NoOverlap per zone or a reservation table. **Motion layer:** decoupled planning plus time coordination (insert waits, velocity tuning, or prioritised planning), or conflict-based search over arm trajectories. For a hackathon, a coarse zone partition plus zone-booking in the scheduler, executed with a runtime zone lock, is the reliable choice.

### Cited Findings
- YuMi CP model: the reachable workspace is partitioned into instance-specific zones. Collisions are avoided by "booking" zones over time intervals, which is equivalent to non-overlapping interval rectangles. For cyclic schedules, intervals [a, b) and [a+period, b+period) go into one global constraint. - [Constraints 2023 (search snippet)](https://link.springer.com/article/10.1007/s10601-023-09345-4)
- Tercio's MILP encodes spatial proximity with big-M disjunctive constraints. For each pair of subtasks whose locations are closer than the allowed separation, a binary variable says which goes first, which gives mutually exclusive access to "tools, utilities, and floor space". After allocation, agent and spatial-resource sequencing constraints are added to a Simple Temporal Problem that is compiled to dispatchable form. - [Gombolay T-RO 2018](https://people.csail.mit.edu/gombolay/Publications/Gombolay_TRO_2018.pdf)
- In CBS-style multi-robot motion planning, collisions occur over **time intervals**. A conflict is a robot pair plus an interval, resolved by constraining one robot. - [Kinodynamic CBS, arXiv 2207.00576](https://arxiv.org/pdf/2207.00576) (search snippet)
- **Stop-N-Go** (2024) plans each arm's path independently, then resolves conflicts only by inserting pauses, using A* over conflicts to minimise makespan. It is optimal for the given initial trajectories and probabilistically complete. With 4 UR5 arms, its makespan was 26-43 s versus 59-92 s for sequential execution (about 2-3x faster), with 90-93% success against 37-43% for coupled RRT-Connect and 17-33% for RL, under a 40 s compute limit. Conflict resolution took 0.43-1.88 s. - [Stop-N-Go, arXiv 2410.07606](https://arxiv.org/html/2410.07606v1)
- Prioritised planning: arms are planned in priority order, and later arms treat earlier ones as moving obstacles. This cuts the search to single-arm configuration space but is incomplete and sensitive to the ordering. Velocity tuning keeps the paths fixed and picks velocity profiles that avoid the conflict points. (Search-result snippets only, not read in full.) - [Decoupled MPPI multi-arm, arXiv 2602.10114](https://arxiv.org/pdf/2602.10114); [Randomized scheduling for manipulators, arXiv 2412.00366](https://arxiv.org/pdf/2412.00366)
- CBS for manipulators is expensive because of high-dimensional search and heavy collision checking. Recent work cuts the replanning (repulsive trajectory modification, 2025) or moves it to GPU (cuArms). - [arXiv 2509.13882](https://arxiv.org/pdf/2509.13882); [cuArms arXiv 2610.36250](https://arxiv.org/pdf/2610.36250); [ECBS for manipulation arXiv 2405.01772](https://ar5iv.labs.arxiv.org/html/2405.01772) (snippets)
- RoboBallet (DeepMind, Science Robotics 2025) solves allocation, scheduling and collision-free motion jointly with a GNN policy trained by RL. It notes that sampling planners in composite space suit only about 2 or fewer robots, and that prior state of the art handled "only five robots and 10 tasks". - [RoboBallet, arXiv 2509.05397](https://arxiv.org/html/2509.05397v1)

### Inferences
- Hackathon recipe:
  1. Split the table into a few zones: each arm's private area plus the overlap regions.
  2. For each task, list the zones its motion sweeps. A conservative choice is the zones of its start and goal plus anything between.
  3. Put a capacity-1 resource per shared zone into the CP-SAT model as a NoOverlap over the intervals that use it.
  4. At run time, enforce the same thing with a zone lock (mutex) as a safety net, so an arm waits rather than collides when durations drift.

  This is the scheduling-level analogue of Stop-N-Go's "insert waits".
- Coarse zones throw away some parallelism (false conflicts) but are easy to verify. Finer zones, or swept-volume checks between pairs of tasks, can be added later.

### Gaps
- I found no source comparing coarse zone-locking against trajectory-level coordination on makespan for small tabletop arms.

## 3. Algorithms and trade-offs; solve times for N = 2-8 arms, 10-50 tasks

### Takeaway
**CP-SAT, with interval variables, optional intervals per arm, NoOverlap per arm and per zone, and precedence, is the default.** It is open source, models the whole variant directly, and is competitive with commercial CP on FJSP. At 10-50 tasks it should find optimal or near-optimal schedules quickly (inference, not checked; see section 3 inferences). Build **HEFT or critical-path list scheduling** alongside it as a baseline, a warm start, and a fast online fallback. Auctions (CBBA, SSI) matter only for decentralisation, MILP is slower than CP for this structure, and RL/GNN schedulers cost days of training.

### Cited Findings
- **OR-Tools FJSP example**:
  - each task has a main interval `new_interval_var`, plus one `new_optional_interval_var` per alternative machine with a presence literal
  - `add_exactly_one(presences)` picks the machine
  - local start and end are tied to the main ones with `.only_enforce_if(l_presence)`
  - `add_no_overlap(intervals)` per machine, and `start >= previous_end` for precedence
  - `add_max_equality(makespan, job_ends)`, then `minimize(makespan)`
  - default solver parameters

  - [flexible_job_shop_sat.py](https://github.com/google/or-tools/blob/stable/examples/python/flexible_job_shop_sat.py)
- **PyJobShop benchmark** (9,280 instances, 8 cores, 900 s limit):
  - On FJSP, OR-Tools CP-SAT had 0.68% relative deviation from best-known against CP Optimizer's 0.99%, with an optimality gap of 1.04% against 27.67%.
  - On JSP the deviation was 1.98% against 1.80%, and on RCPSP 0.93% against 0.46%.
  - CP Optimizer was better on large permutation and hybrid flow shops.
  - **Modelling matters:** an earlier OR-Tools FJSP model from prior work gave 7.55% deviation, against 0.68% for their optimised model.

  - [PyJobShop arXiv 2502.13483](https://arxiv.org/html/2502.13483v1)
- CP vs MILP:
  - On small online-printing-shop instances, CP Optimizer beat CPLEX MILP by one to two orders of magnitude. At n=4, the MILP proved no instance optimal, while CP solved all of them in seconds. - [arXiv 2006.13064](https://arxiv.org/pdf/2006.13064) (snippet)
  - A job-shop variant with peak-power limits: CP and SAT proved 4-6-job instances optimal within seconds, while MILP hit time limits. - [arXiv 2608.25351](https://arxiv.org/pdf/2608.25351) (snippet)
- MILP complexity for this problem class (Tercio): allocation and sequencing binaries give O(2^(|A||τ|^3)) branch-and-bound in the worst case. There are O(|τ|^2) sequencing variables and O(|τ|!) sequencing permutations, so sequencing is the bottleneck when tasks far outnumber agents. - [Gombolay T-RO 2018](https://people.csail.mit.edu/gombolay/Publications/Gombolay_TRO_2018.pdf)
- **Tercio** (hybrid: MILP allocation plus a polynomial-time sequencer):
  - It handles up to 10 agents and 1,000 subtasks in about 1 minute on average, and up to 10 agents and 500 tasks in under 20 s (RSS 2013).
  - Schedules are empirically within 10% of optimal.
  - The allocation MILP minimises a weighted sum of the maximum work per agent and total agent-hours.
  - The sequencer simulates execution with Earliest-Deadline-First and checks temporospatial feasibility.
  - It runs as an anytime algorithm, excluding tried allocations each iteration.

  - [Gombolay T-RO 2018](https://people.csail.mit.edu/gombolay/Publications/Gombolay_TRO_2018.pdf); [RSS 2013](https://roboticsproceedings.org/rss09/p49.html)
- **HEFT** (Topcuoglu, Hariri & Wu, IEEE TPDS 13(3), 2002) is list scheduling for a DAG on heterogeneous workers.
  - Phase 1 ranks each task by its "upward rank", i.e. its average cost plus the maximum over successors of the edge cost plus the successor's rank. This is the critical-path-to-exit distance.
  - Phase 2 takes tasks in rank order and places each on the worker with the earliest finish time, using insertion into idle gaps.
  - It is greedy and can miss the optimum.

  - [Wikipedia: HEFT](https://en.wikipedia.org/wiki/Heterogeneous_earliest_finish_time)
- **Graham list scheduling** is a (2 − 1/m)-approximation. LPT ordering improves this to 4/3 − 1/(3m) for independent jobs. List scheduling has **anomalies**: in a 9-job, 3-machine example, removing dependencies, shortening every job, or adding a machine each made the makespan *worse* (12→16, 12→13, 12→15). - [Wikipedia: List scheduling](https://en.wikipedia.org/wiki/List_scheduling)
- **Auctions/market methods**:
  - Sequential single-item auctions are simple: one task per round.
  - Combinatorial auctions get closer to optimal, but computation and communication grow exponentially with the number of tasks.
  - CBBA (Choi et al.) combines auctions with consensus for decentralised bundle allocation and needs submodular bids to guarantee convergence. Extensions handle online task arrival (Buckman et al.) and timing/capability constraints (Chen et al.).
  - Deterministic/exact methods are "typically slow and only suitable for small problem sizes", which is why heuristics dominate in deployed MRTA.

  - [Sims et al. survey](https://arxiv.org/pdf/2207.09650)
- **Metaheuristics** (GA, simulated annealing, tabu, PSO) sample the solution space and accept worse intermediate solutions to escape local optima. - [Sims et al.](https://arxiv.org/pdf/2207.09650). A greedy constructive heuristic plus local improvement has been used for MRTA with cooperative tasks and precedence. - [Bischoff et al., arXiv 2005.03902](https://arxiv.org/abs/2005.03902) (abstract only)
- **Learned schedulers**:
  - Wang & Gombolay use heterogeneous graph attention networks (imitation learning plus deep Q-learning) for robot-team scheduling with temporal and proximity constraints in final assembly. - [arXiv 1912.02059](https://arxiv.org/pdf/1912.02059); [Sims et al.](https://arxiv.org/pdf/2207.09650)
  - DRL with heterogeneous GNNs for FJSP is reported to beat OR-Tools on larger instances with much lower run time. - [arXiv 2310.15706](https://arxiv.org/pdf/2310.15706) (snippet)
  - RoboBallet: 8 Franka arms, 40 tasks, 0.3 ms per planning step on an A100 and about 30 ms on one CPU core. It needs about 6-24 h of training to reach 95% task solution, with gains continuing over 2-6 days.

  - [RoboBallet](https://arxiv.org/html/2509.05397v1)
- At large scale, the Stanford ConstructionBots system (250 robots, 1,845-part LEGO Saturn V) used a "graph-repair" MIP plus a modified greedy algorithm for allocation, and planned in under 3 minutes on a laptop. - [arXiv 2311.00192](https://arxiv.org/abs/2311.00192)

### Inferences
- **Hackathon choice: CP-SAT through PyJobShop or raw OR-Tools, with a HEFT baseline.**
  - CP-SAT gives optimality proofs or gaps, which are useful in a pitch: "provably within X% of optimal".
  - It takes zones, two-arm sync and minimum lags without extra algorithm work.
  - HEFT is about 40 lines of Python, never fails, and is the online fallback.
- Solve times at N = 2-8 and 10-50 tasks: I found **no source with CP-SAT timings at exactly this size**. Benchmark FJSP instances (Brandimarte etc.) are larger, about 10-20 jobs × 5-15 machines with up to ~240 operations (sizes from memory, inference, not checked), and CP-SAT reaches ~1% gaps within 900 s. Since 10-50 tasks is far smaller, I expect optimal or near-optimal in about 1-10 s with a 5-10 s time limit (inference, not checked). Run a quick benchmark on day 1.
- Avoid: MILP (slower to model and solve here), CBBA (only worth it if the arms are on separate computers and decentralisation is a selling point), RL/GNN (training cost and a correctness risk inside 48 h).

### Gaps
- No direct CP-SAT vs MILP timing on small FJSP with zone resources.
- The original papers on CBBA convergence and SSI approximation bounds were not read; findings come via the Sims survey.

## 4. Online / dynamic rescheduling when a task fails or runs slow

### Takeaway
Use a three-layer scheme:
1. Execute the plan as a **partial order**, not fixed timestamps: each task fires when its predecessors, its arm and its zones are free. A dispatchable temporal network absorbs small delays with no replanning.
2. On larger deviations, re-solve CP-SAT on the remaining tasks from the current state, warm-started with the old solution as a hint and given a short time limit.
3. Fall back to HEFT if the solver times out.

### Cited Findings
- Tercio compiles the final Simple Temporal Problem into a **dispatchable form**. This guarantees that a consistent choice of any time point within its flexible window can be extended by one-step propagation. Dispatchable forms reduced re-computation time after disturbances by up to 75% on random structured problems (Wilcox et al., 2012). - [Gombolay T-RO 2018](https://people.csail.mit.edu/gombolay/Publications/Gombolay_TRO_2018.pdf)
- Tercio's motivation was a fast dynamic method to respond to "schedule disruptions such as those caused by process time variation, re-work, and inspection". - [Gombolay T-RO 2018](https://people.csail.mit.edu/gombolay/Publications/Gombolay_TRO_2018.pdf)
- CBBA extensions reset part of each robot's allocation and replan as new tasks arrive (Buckman et al.). MURDOCH used online greedy auctions in which auctioneers monitor task completion, which gives fault tolerance to robot failure. - [Sims et al.](https://arxiv.org/pdf/2207.09650)
- CP-SAT warm start: users clear the old hint and `AddHint` each variable from the previous solution. A `repair_hint` parameter exists, but its semantics were not verified against official docs. - [or-tools-discuss thread](https://groups.google.com/g/or-tools-discuss/c/8SemJDNHcTY); [thread 2](https://groups.google.com/g/or-tools-discuss/c/ffNwXRXpNvo)
- RoboBallet *proposes* fault-tolerant and perception-based replanning, since inference is fast. It reports **no** failure-recovery experiments. - [RoboBallet](https://arxiv.org/html/2509.05397v1)
- List scheduling anomalies mean that a greedy re-dispatch after a task finishes *early* can still lengthen the makespan. - [Wikipedia: List scheduling](https://en.wikipedia.org/wiki/List_scheduling)

### Inferences
- Implementation:
  - Keep the solved schedule as a **set of chosen arms, per-arm orders and per-zone orders**, and execute it event-driven.
  - On failure, re-insert a retry task, or re-route the task to another arm that can reach it.
  - Re-solve with fixed history: tasks already done are removed, and running tasks become fixed intervals.
- Robustness: pad durations to a high percentile, e.g. p80 of measured times, or add slack before zone-entry tasks. Spread repeated tasks so no single arm sits on the critical path.

### Gaps
- No authoritative CP-SAT documentation found on hint and repair semantics.
- No benchmark found on rescheduling latency for small arm cells.

## 5. Theoretical limits on speed-up with N

### Takeaway
Makespan ≥ max(total work / N, critical-path length, busiest-zone load, and for each task its fastest-arm duration). Greedy list scheduling guarantees ≤ W/N + (1 − 1/N)·L, so it is within 2 − 1/N of optimal. So speed-up saturates at W/L (the DAG's parallelism), which is the Amdahl-type ceiling. Shared zones add a further floor: the total time all tasks need a given zone. Empirically, multi-arm cells get roughly 2-3x from 4 arms over sequential, with diminishing returns as density grows.

### Cited Findings
- Lower bounds on the optimal makespan: the longest job and the average load (total work / m). - [Wikipedia: List scheduling](https://en.wikipedia.org/wiki/List_scheduling)
- With precedence, Graham-style list scheduling satisfies Cmax ≤ (1/m)·Σp_j + max over chains Q of Σ_{j∈Q} p_j. The idle time in a list schedule is at most (m − 1)·|CP|, which gives the 2 − 1/m ratio. - [Lemma 19, arXiv 2004.09682](https://arxiv.org/pdf/2004.09682); [Two-resource-types survey, arXiv 1909.11365](https://arxiv.org/pdf/1909.11365); [Vivien lecture notes](https://graal.ens-lyon.fr/~fvivien/Enseignement/Master2-2014-2015/01-parallel-scheduling.pdf) (via search summary)
- Adding a machine can *increase* a list-scheduling makespan (12→15 in Graham's example). The general anomaly bound is t2/t1 ≤ 1 + (m1 − 1)/m2. - [Wikipedia: List scheduling](https://en.wikipedia.org/wiki/List_scheduling)
- RoboBallet, 20 tasks:
  - Execution time went from about 7.5 s with 4 arms to about 4.5 s with 8 arms. The authors call this "approximately optimal scaling", since tasks per arm went from 5 to 2-3.
  - The paper's "60% reduction" is arithmetically 40%, an error in the paper.
  - The authors note a density limit "beyond which there would be no benefit".
  - Layout optimisation of robot placement improved execution time by up to 33%.

  - [RoboBallet](https://arxiv.org/html/2509.05397v1)
- Stop-N-Go with 4 UR5s: makespan about 2-3x lower than sequential execution, not 4x, because of conflicts in a packed workspace. - [Stop-N-Go](https://arxiv.org/html/2410.07606v1)

### Inferences
- Pitch metric: report the achieved makespan next to the lower bound LB = max(W/N, L, max-zone-load). This shows how close to "perfect" the schedule is, and whether the limit is the task graph (L), the arms (W/N) or a zone. CP-SAT's best bound gives the same information directly.
- Pick N where W/N ≈ L. More arms than about W/L give no gain, and with shared zones the plateau comes earlier.

### Gaps
- I found no closed-form result for list scheduling with resource (zone) constraints at small N beyond the classic RCPSP bounds. Not searched further.

## 6. Code-level pointers and Python libraries

### Takeaway
Start from OR-Tools `flexible_job_shop_sat.py`, or use **PyJobShop** (`pip install pyjobshop`, CP-SAT backend by default), which already has multi-mode tasks, resources, precedence with lags and Gantt plotting.

### Cited Findings
- OR-Tools examples: `examples/python/flexible_job_shop_sat.py` (FJSP with optional intervals, exactly-one machine choice, per-machine NoOverlap, makespan via `add_max_equality`) and the job shop tutorial. - [flexible_job_shop_sat.py](https://github.com/google/or-tools/blob/stable/examples/python/flexible_job_shop_sat.py); [OR-Tools Job Shop docs](https://developers.google.com/optimization/scheduling/job_shop?hl=en)
- PyJobShop: MIT licence, `Model` class (add machines, jobs, tasks, modes, constraints, then `model.solve(time_limit=...)`, `result.best`, `plot_machine_gantt`), OR-Tools by default and CP Optimizer optional. - [PyJobShop arXiv 2502.13483](https://arxiv.org/html/2502.13483v1)
- ConstructionBots.jl (Julia) is an open-source simulator and planner for large multi-robot assembly. - [arXiv 2311.00192](https://arxiv.org/abs/2311.00192)

### Inferences
- Minimal CP-SAT model for the cell (adding to the FJSP example):
  - Use a main interval per task, plus an optional interval per arm that can reach it (duration specific to that arm), with `add_exactly_one`.
  - Use `add_no_overlap` per arm, and per shared zone over the intervals of tasks that use the zone. Build zone intervals from the task's main interval, or from sub-intervals for the approach and retract phases.
  - Add DAG edges as `start[j] >= end[i] (+ lag)`.
  - For a two-arm task, make two required intervals on different arms with `start` equal.
  - Set `max_time_in_seconds` to 5-10 and `num_workers` to 8.
  - Feed in the HEFT solution as a hint.

  This is about 150 lines of code (inference, not checked).
- Validation: check the schedule with an independent simulator before running it on the arms (precedence, arm overlap, zone overlap), and compare against the lower bound in §5.

### Gaps
- Exact PyJobShop API names were not re-checked against current docs; only the paper's description is used.

## 7. Collaborative tasks: handoffs, two arms on one object, hold-and-insert (added on scope clarification)

### Takeaway
Collaborative steps are **multi-robot (MR) tasks** in Gerkey-Matarić terms. The problem becomes ST-MR-TA with cross-schedule dependencies, solved via coalitions or "alliances". In a scheduler each pattern is a small set of interval constraints:
- **co-temporal (two arms on one object):** one task occupies two arms over the same interval
- **handoff:** a short co-temporal task in a handoff zone, chained by precedence (pick by A, then handoff A+B, then place by B)
- **hold-and-insert:** the holding interval must contain the inserting interval

The core risk is **deadlock**. With cooperative tasks plus precedence, a plan can be infeasible even when every robot's sequence looks fine, and the test is acyclicity of the combined graph. Collaboration raises total arm-time and adds sync waiting, so it lowers the speed-up compared with independent splitting. It pays only when it makes otherwise impossible steps possible (reach, a heavy or large part, holding), or when a handoff shortens the critical path.

### Cited Findings
- The Gerkey-Matarić MR category covers tasks that need multiple robots. Nunes et al. classify synchronisation and precedence constraints as TA:SP. - [Sims et al.](https://arxiv.org/pdf/2207.09650); [Nunes et al.](https://www-users.cs.umn.edu/~gini/papers/NunesRAS.pdf)
- Bischoff et al. (arXiv 2005.03902) set up ST-MR-TA with cooperative tasks and precedence as follows:
  - A set of allowed robot **alliances** A = {a_j ⊆ R} lists the coalitions that can do each task.
  - Each robot's schedule is a directed path graph, and a cooperative task appears as **a vertex in the path graphs of all participating robots**. Precedence edges are added to give the extended graph M+.
  - **Lemma 2: the plan is feasible (executable in finite time) iff M+ is acyclic.** Otherwise there is cyclic waiting, i.e. deadlock.
  - Cross-schedule dependencies arise "from the consideration of cooperative tasks or from waiting times due to precedence constraints". So standard neighbourhood moves, e.g. vehicle-routing relocate, can produce infeasible plans unless they check this.
  - Their solver is a greedy constructive heuristic (MinStepSum-like, extended to precedence) plus relocate-based local improvement.

  - [Bischoff et al., arXiv 2005.03902](https://arxiv.org/pdf/2005.03902)
- Multi-arm rearrangement models picks, places and **handoffs** as edges of a "mode graph", whose nodes are object placements and object-in-hand transfer states. Vertex capacities apply: each arm carries one object. Integer programming solvers are used, guided by a heuristic, and actions are synchronised for subsets of arms to cut makespan. - [Synchronized multi-arm rearrangement, arXiv 2005.09127](https://arxiv.org/abs/2005.09127) (abstract only)
- Dual-arm tabletop rearrangement: Shome et al. give an optimal mixed-integer model and a scalable "Tour Over Matching" algorithm, assuming simultaneous pick-and-places on monotone instances. Gao & Yu use dependency-graph-guided search to coordinate handoffs and individual pick-and-places when some objects must be handed off. They report "significant time savings" over greedy methods and over naive parallelisation of single-robot plans, with no figures in the abstract. - [arXiv 1810.12202](https://arxiv.org/pdf/1810.12202) (snippet); [arXiv 2207.08078](https://arxiv.org/abs/2207.08078)
- Tercio and the YuMi CP model both handle cross-arm coupling with temporal constraints, i.e. upper/lower-bound waits and zone booking, inside one centralised model. - [Gombolay T-RO 2018](https://people.csail.mit.edu/gombolay/Publications/Gombolay_TRO_2018.pdf); [Constraints 2023](https://link.springer.com/article/10.1007/s10601-023-09345-4) (snippet)
- The PyJobShop paper lists "multi-resource/multi-skill" as unsupported, so two-arm tasks may need raw CP-SAT. - [PyJobShop](https://arxiv.org/html/2502.13483v1)

### Inferences (all inference, not checked unless cited above)
- **CP-SAT encodings:**
  - **Co-temporal task on an arm pair:** for each allowed pair (a, b), use one presence literal shared by two optional intervals, one on arm a and one on arm b. Constrain `start_a == start_b` and `end_a == end_b`, and put `add_exactly_one` over the pair literals. Each arm's NoOverlap then blocks both arms automatically.
  - **Handoff:** a co-temporal task on (giver, receiver) with a fixed handoff duration, using the handoff-zone resource. Add precedence `end(pick_A) <= start(handoff)` and `end(handoff) <= start(place_B)`. The receiver must already be at the zone, so its approach time goes into its own interval.
  - **Hold-and-insert:** `start_hold <= start_insert` and `end_insert <= end_hold`, an Allen "during" relation. Holding is long arm occupancy that does no other work.

  A CP solver does not deadlock: an infeasible cyclic wait simply has no assignment. Greedy or auction code must check that the combined graph is acyclic, as in Bischoff's Lemma 2.
- **Effect on speed-up:**
  - The lower bound becomes LB = max( Σ_i k_i·p_i / N, L ), with k_i = arms needed by task i. So a two-arm task costs double arm-time.
  - The critical path now includes handoff durations and the waits for the partner arm to arrive.
  - List schedulers such as HEFT must find a slot when **both** arms are free. This creates idle gaps, and the Graham 2 − 1/N bound for single-arm tasks no longer applies as stated (inference, not checked: I found no bound for rigid multi-arm tasks).
  - Rule of thumb: use collaboration where it is required (reach, weight, fixturing), and keep the rest as independent single-arm tasks.
- **Where it helps:** with arms of different reach, a handoff lets parts flow from the far side to the assembly point without moving the fixture. It can shorten L if the alternative is one arm doing a long pick-carry-place chain alone. For SO-101-class arms, a bimanual hold-and-insert likely replaces a jig. That is a capability gain, not a speed gain.
- **Execution:** sync points need a runtime barrier. Neither arm starts the co-temporal motion until both report "at pre-grasp/handoff pose". Durations of synchronised segments should come from the slower arm.

### Gaps
- No quantitative source found on how much makespan handoffs save versus single-arm plans. The abstracts claim savings with no numbers, and the full texts were not read.
- No known approximation bound for list scheduling with rigid multi-arm (k > 1) tasks plus precedence was located.
- PyJobShop's current API support for multi-machine modes was not checked against its docs.
