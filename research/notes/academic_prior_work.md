# Academic prior work (2022-2026): orchestrating multiple robot arms to finish a shared task faster

Research date: 21 Nov 2026. About 26 search/fetch calls. Numbers come from arXiv HTML/abstract pages unless the source is marked "secondary". Where only the abstract was readable, the note says so.

## 1. LLM/VLM-based multi-robot planners and task decomposition

### Takeaway
LLM multi-robot planners from 2023-2024 (RoCo, SMART-LLM, COHERENT, LaMMA-P, DART-LLM) mostly score task success in step-based simulators (AI2-THOR, BEHAVIOR-1K, MuJoCo), and timing is either ignored or crude: turn-based steps, Python threads, or "steps" as the efficiency metric. The 2025-2026 trend is explicit dependency DAGs plus a scheduler that minimizes makespan (RoboPARA, Plan-over-Graph, OSDAG), with reported wall-clock or makespan cuts of 30-50%. Almost all real-robot validation is qualitative.

### Cited Findings
**RoCo (Mandi, Jain, Song; arXiv Jul 2023; Columbia)**
- Each arm has its own LLM agent. The agents "discuss" strategy and output sub-task plans plus task-space waypoints, and a centralized multi-arm RRT-based motion planner turns them into joint trajectories. Collision-check and IK failures are fed back so the LLM can re-plan. - [RoCo arXiv](https://arxiv.org/abs/2307.04738); [RoCo HTML](https://arxiv.org/html/2307.04738)
- RoCoBench has 6 tasks with 2-3 arms (UR5e, Franka, humanoid): Sweep Floor, Make Sandwich, Sort Cubes (3 arms), Pack Grocery, Move Rope, Arrange Cabinet (3 arms). Tasks are labelled by decomposition (parallel or sequential) and workspace overlap (low, medium or high). It also includes a text-only agent-reasoning dataset. - [RoCo HTML](https://arxiv.org/html/2307.04738)
- Success over 20 runs, Dialog vs Central-Plan oracle (GPT-4): Pack Grocery 0.44 vs 0.82, Arrange Cabinet 0.75 vs 0.90, Sweep Floor 0.95 vs 1.00, Make Sandwich 0.80 vs 0.96, Sort Cubes 0.93 vs 0.70, Move Rope 0.65 vs 0.50. Dialog loses most on the high-overlap packing task. - [RoCo HTML](https://arxiv.org/html/2307.04738)
- Real world (human plus robot, 10 runs per setup): 9/10 and 8/10 with an oracle human, 7/10 and 6/10 with an imperfect human. Detection used OWL-ViT. - [RoCo HTML](https://arxiv.org/html/2307.04738)
- Stated limitations: assumes accurate perception and collision checks; open-loop trajectory execution; one LLM query per dialog turn adds latency that "hurts dynamic or speed-sensitive tasks". The paper does not report wall-clock time or makespan. - [RoCo HTML](https://arxiv.org/html/2307.04738)

**SMART-LLM (Kannan, Venkatesh, Min; arXiv Sep 2023, rev. Mar 2024; submitted to IROS 2024; Purdue)**
- Few-shot programmatic prompting runs 4 stages: task decomposition, coalition formation, task allocation, and code execution. Benchmark: 36 AI2-THOR instructions (6 elemental, 8 simple, 14 compound, 8 complex) with 1-4 robots. - [SMART-LLM HTML](https://arxiv.org/html/2309.10062)
- Success rate with GPT-4: 1.00 elemental, 0.62 simple, 0.69 compound, 0.71 complex. Claude-3 Opus scored 1.00, 0.87, 0.69 and 0.71. Across 5 GPT-4 runs, complex-task SR was 0.48 ± 0.40, so results vary a lot from run to run. - [SMART-LLM HTML](https://arxiv.org/html/2309.10062)
- Parallelism: tasks run "in parallel when feasible", and the prompt examples use Python threading. There is no timing metric; "Robot Utilization" compares transition counts to ground truth. Real-robot results are qualitative only (aerial and ground robots), and dynamic re-allocation is left as future work. - [SMART-LLM HTML](https://arxiv.org/html/2309.10062)

**COHERENT (Liu et al.; arXiv Sep 2024; ICRA 2025)**
- A centralized assigner runs Proposal-Execution-Feedback-Adjustment (PEFA) loops over heterogeneous robots (quadrotors, robot dogs, arms). The benchmark has 100 tasks in 5 BEHAVIOR-1K scenes, of which 40 were evaluated because of API cost. - [COHERENT HTML](https://arxiv.org/html/2409.15146v3)
- Success rate / average steps (ground truth = 10.3 steps): COHERENT 0.975 / 12.4; CMRS 0.625 / 16.5; CoELA variants 0.600 / 17.2 and 0.375 / 19.6; LLM-MCTS 0.200 / 20.5. The model was gpt-4-0125-preview. The real world was one qualitative demo (3 quadrotors, a dog with an arm, and a Franka). - [COHERENT HTML](https://arxiv.org/html/2409.15146v3)

**LaMMA-P (Zhang et al.; arXiv Sep 2024; ICRA 2025)**
- Combines an LM with a PDDL heuristic-search planner for subtask allocation in heterogeneous teams. Benchmark: MAT-THOR (AI2-THOR, 2 complexity levels). The abstract reports "105% higher success rate and 36% higher efficiency" than prior LM-based multi-agent planners. Only the abstract was checked, not how parallel timing is handled. - [LaMMA-P arXiv](https://arxiv.org/abs/2409.20560)

**DART-LLM (Univ. Tokyo; arXiv Nov 2024)**
- Builds a dependency DAG of subtasks and criticizes SMART-LLM and RoCo for "linear decomposition". The modules are a QA LLM, a breakdown/assignment function, an actuation module on the ROS Navigation stack, and a VLM detector. DeepSeek-r1-671B had the highest success, and Llama-3.1-8B the most reliable latency. Explicit dependencies help small models most. Only 2 real-robot experiments (construction-type robots). - [DART-LLM search summary / arXiv](https://arxiv.org/abs/2411.09022v2); [project](https://wyd0817.github.io/project-dart-llm)

**Dependency-DAG + scheduling generation (2025-2026)**
- RoboPARA (Beihang et al.; arXiv 2506.06683, v2 Mar 2026): an LLM builds a validated task DAG, then a graph re-traversal schedules it onto the left and right arms, with sync and deadlock rollback. It introduces the X-DAPT benchmark (>1,000 task packages, 10 scenes, a "first dataset for dual-arm task parallelism"). It claims a 30-50% cut in execution time, >=1.3-1.5x successful steps per unit time over 7 LLM baselines including RoCo, and about 1.3x more tokens. The paper's parallel-step multipliers are internally inconsistent (4.5x vs 2.8x/10x). Real-robot tests used Franka FR3, UR5e and a humanoid, with few numbers. - [RoboPARA HTML](https://arxiv.org/html/2506.06683v2)
- Plan-over-Graph (arXiv Feb 2025): an LLM decomposes a task into a dependency graph and outputs a parallel schedule that minimizes makespan or cost. It is a generic agent benchmark, not a physical robot one. Seen only in a search snippet. - [Plan-over-Graph](https://arxiv.org/abs/2502.14563)
- OSDAG (Canh et al.; arXiv 13 Jun 2026, v2 2 Sep 2026): one LLM call parses the instruction into a DAG with precedence, capability and resource constraints, and a lightweight online scheduler assigns ready tasks to whichever robot goes idle. Makespan is up to 38% lower than sequential baselines across 5 scenarios, and reasoning is 5-15x faster than dialogue-based methods. Robot count and type are not in the abstract. - [OSDAG arXiv](https://arxiv.org/abs/2606.15255)

**Survey context**
- Survey "LLMs for Multi-Robot Systems" (Li, An, Abrar, Zhou, Drexel; v5 May 2026). Of 49 surveyed works, 22 are general-purpose, 9 household, 1 manufacturing and 1 warehouse. Few systems are validated beyond about 10 robots. Its guidance: LLMs suit sparse, event-triggered calls, not dense per-step planning. - [Survey HTML](https://arxiv.org/html/2502.03814)

### Inferences
- The field moved from "LLMs talk to each other" (RoCo, 2023) to "an LLM writes a DAG once and a classical scheduler runs it" (OSDAG, RoboPARA, 2025-26). The second approach is faster and has fewer failure points.
- Before RoboPARA, almost no LLM planner reported wall-clock makespan on real arms. "Steps" and "robot utilization" are proxies, not time.
- GenSim (2023) is a single-arm task-generation paper, not a multi-arm planner (from prior knowledge, not checked this session). Co-NavGPT is about navigation, not arm manipulation (not checked). MetaGPT-style role agents appear in the survey's taxonomy, but I found no arm-specific results.

### Gaps
- LaMMA-P full results tables and parallel-execution mechanism (only the abstract was read).
- OSDAG robot types and whether any arms were used (PDF not readable).
- Real-world success or makespan numbers for any LLM multi-arm planner: none found.
- No dedicated multi-arm results were found for MetaGPT-style agents, GenSim or Co-NavGPT.

## 2. Learned multi-arm / bimanual policies and multi-arm benchmarks

### Takeaway
Learned policies with two arms work well when one policy controls both arms as a single body: ACT/ALOHA reaches 80-90% on fine tasks, and GR00T N1 reaches 76.8% on real GR-1 bimanual tasks. Benchmark success is still modest and depends heavily on protocol. None of these frameworks coordinates N independent arms; that is left to planners.

### Cited Findings
- ACT/ALOHA (Zhao et al., Apr 2023, RSS 2023): 80-90% success on 6 real fine-grained bimanual tasks (e.g. slotting a battery, opening a translucent cup) from about 10 minutes (50 demos) of data. - [ACT paper](https://arxiv.org/pdf/2304.13705.pdf)
- Mobile ALOHA (Fu, Zhao, Finn, Jan 2024): 50 demos per task. Co-training with static ALOHA data raised whole-task success by 45, 20, 80, 95 and 80 points on 5 of 7 tasks (mean +34 points), giving >80% success. - [Mobile ALOHA](https://arxiv.org/pdf/2401.02117)
- pi0 (Physical Intelligence, Oct 2024): a flow-matching VLA pre-trained on 7 robot configurations and 68 tasks, including bimanual and mobile setups, with up to 50 Hz control; demos include laundry folding. The blog says 8 robots, which differs from the paper. Laundry success rates were not found. - [pi0 arXiv](https://arxiv.org/abs/2410.24164v1); [PI blog](https://www.pi.website/blog/pi0)
- GR00T N1 (NVIDIA, Mar 2025): one set of weights covers single-arm, bimanual and humanoid. Simulation averaged 45.0% vs 33.4% for Diffusion Policy at 100 demos per task. Real GR-1 bimanual tasks reached 76.8% vs 46.4% for DP. Numbers are from NVIDIA's developer blog. - [GR00T N1 blog](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1); [arXiv](https://arxiv.org/abs/2503.14734)
- PerAct2 / RLBench2 (Grotz et al., Jul 2024): 13 bimanual tasks with 23 variations. The original baselines averaged only 16.8-17.5%. Later methods report up to 85.1% (3DFA) under different protocols. - [PerAct2](https://arxiv.org/pdf/2407.00278); [project](https://bimanual.github.io)
- RoboTwin 2.0 (Jun 2025): 50 dual-arm tasks, 5 embodiments, >100k trajectories. A VLA fine-tuned on its data reached 42.0% (+367% relative). Later papers report about 72% average on the Easy setting. The CVPR 2025 RoboTwin challenge (17 tasks) drew 64 teams. - [RoboTwin 2.0](https://arxiv.org/html/2506.18088v1); [LeRobot docs](https://huggingface.co/docs/lerobot/robotwin.md)
- LeRobot supports SO-ARM bimanual setups (`bi_so_follower` / `bi_so_leader`). Hackathon projects have shown 2 SO-100/101 arms trained together on collaborative pick-and-place, but no general orchestration layer for more than 2 arms or independent arm instances was found. - [Seeed double-arm guide](https://wiki.seeedstudio.com/lerobot_double_arm_so_arm_training/); [Hackster hackathon](https://www.hackster.io/news/embodied-ai-hackathon-winners-announced-2dc69c76942e)

### Inferences
- Bimanual learned policies trade orchestration for one joint action space. That does not scale to 3+ arms in separate cells or to adding arms on the fly. It is an opening for an orchestrator that dispatches per-arm learned skills.
- ManiSkill multi-agent tasks and multi-arm assembly benchmarks were not checked this session (gap).

### Gaps
- ManiSkill3 multi-agent task list and numbers; a pi0.5 multi-arm result; any benchmark that scores makespan for learned multi-arm policies (none found).

## 3. Multi-arm motion planning and coordination

### Takeaway
Learned decentralized planners scale to 8-10 arms (Ha et al. 2020: >90% success at 10 arms; RoboBallet 2025: 8 arms, 40 tasks, jointly doing allocation, scheduling and motion). GPU sampling (cuRobo, ScheduleStream) makes centralized multi-arm TAMP fast enough to use online.

### Cited Findings
- Ha, Xu, Song, "Learning a Decentralized Multi-arm Motion Planner" (CoRL 2020; just before the 2022 window). A per-arm policy trained with SAC plus BiRRT expert demos on 1-4 arms with static targets generalizes to 5-10 arms and moving targets, with >90% success for a 10-arm system. The authors say it scales sub-linearly, whereas centralized planners scale exponentially. - [Project page](https://multiarm.cs.columbia.edu/); [arXiv](https://arxiv.org/abs/2011.02608)
- RoboBallet (Google DeepMind et al.; arXiv 5 Sep 2025; Science Robotics 10, eads1204). A GNN policy trained with TD3 on about 1M simulated workspaces of 4 or 8 Franka arms jointly solves task allocation, scheduling and motion planning: 8 robots, 40 reaching tasks in a shared workspace, zero-shot to new layouts. Secondary source: time for 20 targets fell from 7.5 s with 4 arms to 4.3 s with 8 arms, and the 4-arm case matched hand-optimized baselines (8-11 s). The authors note it enables fault-tolerant and perception-based re-planning. - [RoboBallet arXiv](https://arxiv.org/html/2509.05397v1); [DeepMind page](https://deepmind.google/research/publications/roboballet-planning-for-multi-robot-reaching-with-graph-neural-networks-and-reinforcement-learning/); [The Batch, secondary](https://www.deeplearning.ai/the-batch/google-deepminds-roboballet-research-project-blends-graph-neural-networks-with-reinforcement-learning-to-drive-8-armed-robots)
- RoCo (above) uses waypoints from the LLM to seed a centralized multi-arm RRT. - [RoCo HTML](https://arxiv.org/html/2307.04738)

### Inferences
- RoboBallet's tasks are reaching only, with no grasping or object hand-offs. Precedence-constrained pick-and-place across arms still needs a task layer. The RoboBallet abstract confirms "reaching"; this conclusion is my inference.

### Gaps
- Per-difficulty tables for Ha et al. and RoboBallet's primary-source timing numbers (not read).

## 4. Throughput-oriented work: multi-arm cells, multi-robot TAMP, makespan

### Takeaway
Classical multi-arm TAMP explicitly minimizes makespan and shows about 2x shorter schedules than sequential execution. ScheduleStream (NVIDIA, 2025-26) does this on SO100 arms in simulation and on a bimanual Kinova arm in the real world. Mixed-integer methods (Chen et al. 2022) handle assemblies of up to 23 parts but are slow to compute.

### Cited Findings
- Chen et al., "Cooperative TAMP for Multi-Arm Assembly Systems" (arXiv Mar 2022; MIT/NVIDIA). A MILP allocates robots to tasks under precedence constraints, including collision constraints and movement durations, and a MAPF-based low level produces motions. Assemblies have up to 23 objects (Lego, bars, plates). A later paper describes the makespans as "high quality" but at "substantial computational cost". - [arXiv 2203.02475](https://arxiv.org/abs/2203.02475); [NVIDIA page](https://research.nvidia.com/publication/2022-03_cooperative-task-and-motion-planning-multi-arm-assembly-systems); [disassembly paper citing it](https://arxiv.org/pdf/2608.05830)
- ScheduleStream (Garrett and Ramos, NVIDIA; arXiv 2511.04758, v2 May 2026) uses hybrid durative actions with asynchronous starts and GPU samplers on cuRobo, with 2-4 arms in simulation (Franka and **SO100**).
  - Average makespan: Sequential 3.4 (first) / 3.1 (best) vs Ours+GPU 2.0 / 1.5, i.e. about half. Franka-Assigned-4 went from 2.8/2.4 to 0.8/0.6, and SO100-Any-4 from 7.8/7.6 to 3.6/2.4.
  - Success / time to first solution: Ours+GPU 99% / 1.9 s, Sequential 99% / 6.5 s, Hierarchical 64% / 28.3 s, Ours without GPU 91% / 31.4 s. Without GPU, SO100-Any-4 took 244 s at 18% success.
  - Real world: 21 qualitative demos on a 2x Kinova Gen3 robot, 10 tasks, with GroundingDINO perception and no success-rate numbers. - [ScheduleStream HTML](https://arxiv.org/html/2511.04758)
- Dual-arm tight assembly (arXiv 2607.17876) reports a large makespan cut over a single-arm baseline and plans to extend to three hands. Seen only in a snippet; the number was not checked. - [arXiv 2607.17876](https://arxiv.org/pdf/2607.17876)
- RoboPARA reports a 30-50% cut in dual-arm execution time and OSDAG up to 38% lower makespan (see Q1). - [RoboPARA](https://arxiv.org/html/2506.06683v2); [OSDAG](https://arxiv.org/abs/2606.15255)

### Inferences
- With 2-4 arms, published makespan gains cluster at 1.5-3x over sequential, well below N-times linear speed-up, because reach limits and pairwise collision checks constrain parallelism. A hackathon claim of about 2x with 2-4 arms is in line with the literature. A claim of N-times speed-up would need strong evidence.
- ScheduleStream already uses SO100 arms in simulation, so "first multi-arm makespan planning on SO-10x arms" is **not** a novel claim. Real-hardware SO-10x multi-arm makespan numbers do appear to be unpublished (inference, not exhaustively checked).

### Gaps
- Industrial multi-arm cell papers (e.g. automotive welding-cell scheduling, IKEA multi-arm assembly) were not covered within the call budget.
- No real-hardware makespan numbers were found for any of the 2025-26 LLM or TAMP schedulers.

## 5. Open problems, gaps and credible hackathon novelty

### Takeaway
The recurring gaps are: (a) real-hardware wall-clock or makespan numbers; (b) live failure handling during parallel execution; (c) heterogeneous or low-cost arms with imprecise control; (d) LLM latency inside the control loop; (e) no standard orchestration layer for open-source arms. Closed-loop multi-arm recovery papers only started appearing in 2026.

### Cited Findings
- The survey names these as open: closed-loop failure recovery ("open-loop planning consistently underperforms"), LLM latency (keep calls sparse and event-triggered), and benchmarks that assume atomic actions, perfect observability and negligible latency, so success rates "may overstate" real multi-robot performance. Also heterogeneity (some instructions are infeasible for some robots) and scalability beyond about 10 robots. - [Survey HTML](https://arxiv.org/html/2502.03814)
- RoCo itself lists open-loop execution, perception assumptions and LLM latency as limits. - [RoCo HTML](https://arxiv.org/html/2307.04738)
- SMART-LLM lists dynamic task allocation as future work and has no quantitative real-robot results. - [SMART-LLM HTML](https://arxiv.org/html/2309.10062)
- Live failure handling, emerging work:
  - CoMuRoS (arXiv 2511.22354, late 2025) claims to be the first to show runtime event-driven replanning on physical heterogeneous robots.
  - A closed-loop multi-agent multi-robot manipulation framework (arXiv 2607.06990, Jul 2026) adds a Verification Agent and hierarchical recovery: local retries for grasp slips, global replanning otherwise, over 6 tasks.
  - REPAIR (arXiv 2603.28156) asks a human operator for help when it cannot recover.
  - All three were seen only in search snippets; numbers not checked. - [CoMuRoS](https://arxiv.org/html/2511.22354v1); [Closed-loop framework](https://arxiv.org/pdf/2607.06990); [REPAIR](https://arxiv.org/abs/2603.28156)
- ScheduleStream's real-world results are qualitative, and hierarchical planning fails when only some arms can reach some objects. - [ScheduleStream HTML](https://arxiv.org/html/2511.04758)
- LeRobot coordinates more than one arm only through the bimanual config and joint policies, and no general N-arm orchestration layer was found. - [Seeed guide](https://wiki.seeedstudio.com/lerobot_double_arm_so_arm_training/)

### Inferences (what a hackathon project could credibly claim; all inference)
- **Not new:** LLM task decomposition across arms (RoCo, SMART-LLM); LLM to DAG to parallel schedule (DART-LLM, RoboPARA, OSDAG); makespan-optimal multi-arm TAMP, including on SO100 in simulation (ScheduleStream); learned decentralized N-arm motion (Ha et al., RoboBallet).
- **Plausibly new / defensible:**
  1. Measured **real-hardware wall-clock makespan** on 2+ low-cost SO-101 arms versus a single arm and a sequential baseline, with success rate. The literature mostly lacks this.
  2. A **twin-in-the-loop timing model**: per-arm durations predicted from a system-identified digital twin rather than assumed. Schedulers today assume known durations; heterogeneous wear on cheap arms makes durations differ per arm.
  3. **Live re-dispatch on failure** during parallel execution, where an arm's failed subtask moves to another idle arm, measured as makespan lost per failure. Recovery papers from 2025-26 are mostly qualitative.
  4. A thin, open **orchestration layer for LeRobot arms** (DAG plus online "next ready task to next idle arm" scheduler, per OSDAG) that wraps per-arm learned skills.
- A credible claim states its baseline (single arm, sequential, OSDAG-style greedy), the number of arms, the number of trials, and real wall-clock seconds.

### Gaps
- No exhaustive check that real SO-10x multi-arm makespan results are absent; a targeted search of arXiv and Hugging Face for "SO-101 multi-arm" papers is still needed.
- Numbers for CoMuRoS, the 2607.06990 closed-loop framework and REPAIR were not read from the primary sources.
