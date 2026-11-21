# Industry products and companies for multi-robot / cell orchestration in manufacturing

Researched 21 Nov 2026. 16 web searches/fetches. Most capability claims come from vendor press releases or trade press that repeats them; they are marked as claims. Anything not checked against a source is marked "(inference, not checked)".

## 1. What do the main products offer for coordinating several robots or arms?

### Takeaway
No product found sells automatic, dynamic task splitting across several arms in one manufacturing cell as a shipped feature. Today's products fall into four groups: (a) vendor controller features that coordinate the motion of 2-4 arms on one controller (ABB MultiMove, FANUC MultiArm), (b) cell or line software platforms that connect equipment and sequence work (Flexxbotics, Vention MachineLogic, Bright Machines, MujinOS, Intrinsic Flowstate), (c) simulation and digital-twin tools (NVIDIA Mega, DELMIA), and (d) mobile-fleet orchestrators (Open-RMF, InOrbit). Learned multi-arm coordination exists only as research (DeepMind/Intrinsic RoboBallet, simulation only).

### Cited Findings
**Intrinsic (Alphabet / now Google)**
- Feb 25, 2026: Intrinsic moved from Alphabet "Other Bet" into Google ("Intrinsic joins Google to accelerate the future of physical AI"). | [Intrinsic blog](https://www.intrinsic.ai/blog). A secondary source says Google took in Flowstate itself. | [Trend Hunter](https://www.trendhunter.com/trends/intrinsic-flowstate-platform)
- Flowstate is a web-based development environment and simulator for building robot apps from modular "skills". | [Trend Hunter](https://www.trendhunter.com/trends/intrinsic-flowstate-platform); [TechCrunch 2024](https://techcrunch.com/2024/05/06/alphabet-owned-intrinsic-incorporates-nvidia-tech-into-robotics-platform/amp)
- Sept 3, 2025 blog post "Specialized AI for scalable and adaptive multi-robot orchestration" (DeepMind Robotics with Intrinsic) describes research that "enables fully automated, collision-free coordination for multiple industrial robots performing tasks together". | [Intrinsic blog](https://www.intrinsic.ai/blog)
- That research is RoboBallet (DeepMind + Intrinsic + UCL, Science Robotics): a graph neural network trained with RL that plans tasks and motion for up to 8 arms. Claims: average task completion time fell 60% going from 4 to 8 robots, and it beat expert-designed solutions by about 25%. **These results are from simulation only, not production lines.** Limits: no ordered tasks (precedence), no robots with different capabilities. The code is open source. | [UCL News](https://www.ucl.ac.uk/news/2025/sep/robots-learn-work-together-well-choreographed-dance); [The Batch](https://www.deeplearning.ai/the-batch/google-deepminds-roboballet-research-project-blends-graph-neural-networks-with-reinforcement-learning-to-drive-8-armed-robots); [arXiv 2509.05397](https://arxiv.org/pdf/2509.05397); [The Decoder](https://the-decoder.com/roboballet-uses-ai-to-choreograph-multiple-industrial-robots-for-safe-and-efficient-teamwork/); [Control.com](https://control.com/news/google-deepminds-roboballet-cuts-robot-planning-time/)
- Nov 20, 2025: Foxconn and Intrinsic joint venture for electronics assembly. May 13, 2026: FANUC integrates with Intrinsic and Flowstate. | [Intrinsic blog](https://www.intrinsic.ai/blog)
- June 22, 2026 (Automate 2026): "Intrinsic Intelligence Cell", a modular reference design for a workcell. Integrators such as Trinity Automation can add AI skills and manage systems "without programming each robot". The demo was FANUC-based electronics assembly. | [SiliconANGLE](https://siliconangle.com/2026/06/22/intrinsic-unveils-next-gen-accessible-automated-industrial-ai-robotic-assembly/)
- Sept 22, 2026: "Intrinsic Core", an open-source set of ROS-compatible capabilities and reference solutions. | [Intrinsic blog](https://www.intrinsic.ai/blog)

**NVIDIA Isaac / Omniverse "Mega" blueprint**
- Mega is a reference workflow for testing heterogeneous robot fleets in a factory digital twin before deployment. A fleet manager talks to containerized "robot brains" over **VDA5050** (the AGV/AMR standard). A scheduler models latency and multi-rate execution. Sensor RTX streams are bridged to ROS 2 topics. Status: preview on build.nvidia.com. | [NVIDIA blog: Mega blueprint](https://blogs.nvidia.com/blog/mega-omniverse-blueprint); [NVIDIA developer blog](https://developer.nvidia.com/blog/simulating-robots-in-industrial-facility-digital-twins/); [build.nvidia.com card](https://build.nvidia.com/nvidia/mega-multi-robot-fleets-for-industrial-automation/blueprintcard)
- Named adopters: Accenture + Schaeffler (Hannover Messe demo), Foxconn (manipulators, humanoids, AMRs), KION (warehouses). | [NVIDIA blog: ecosystem adopts Mega](https://blogs.nvidia.com/blog/mega-omniverse-blueprint-industrial-digital-twins/)
- No independent evaluation of performance or deployment results was found. | same sources

**Bright Machines**
- Sells "software-defined" microfactories, with more than 130 deployed in over 10 countries (company claim). Now focused on AI-infrastructure (server) assembly in the US. Raised $126M in July 2024. | [The Robot Report 2024](https://www.therobotreport.com/bright-machines-plans-microfactory-innovations-126m-funding/); [Dealroom](https://dealroom.co/news/159992-teradyne-invests-in-bright-machines-to-speed-ai-hardware-manufacturing/)
- Oct 5, 2026: Teradyne invested in a late-stage round. Plan (not yet a deployment): run the Bright Machines platform, UR cobots and Teradyne board testers on one production floor. July 29, 2026: added a "hybrid BRC" robot cell. | [The Robot Report](https://www.therobotreport.com/teradyne-invests-ibright-machines-brings-robotics-ai-infrastructure-manufacturing/); [Manufacturing.net](https://www.manufacturing.net/software/news/22975719/teradyne-partners-with-bright-machines-to-put-robots-test-tech-in-ai-infrastructure-microfactories); [VMBlog](https://vmblog.com/news/bright-machines-expands-bright-factory-platform-with-hybrid-brc-for-resilient-ai-infrastructure-production/)

**Vention (MachineMotion / MachineLogic)**
- MachineLogic now supports several FANUC families (CRX, LR Mate, M-710iD, M-20iD and others) as well as UR. | [The Robot Report](https://www.therobotreport.com/vention-collaborates-with-fanuc-universal-robots-software-defined-automation/); [Newswire](https://www.newswire.ca/news-releases/vention-and-fanuc-america-join-forces-to-bring-industrial-robots-to-vention-s-ai-driven-hardware-and-software-platform-827509552.html)
- One MachineMotion AI controller runs up to 20 motors over EtherCAT and several applications at once. These are motors and conveyors, not several arms. | [Robotics247, PACK EXPO 2026](https://www.robotics247.com/article/pack-expo-2026-vention-introduces-24-7-video-based-remote-support/supply_chain)
- Rapid Operator AI (launched GTC 2026) is single-arm deep bin picking. Claims: CAD-to-pick in 15 minutes, up to 99% first-pick success. Shown at IMTS 2026 on a UR12e with NVIDIA Isaac models. | [Robotics247, IMTS 2026](https://www.robotics247.com/article/imts-2026-vention-facilitates-manufacturing-with-physical-ai-and-agentic-ai-in-one-platform)
- Vention's Robotmaster (offline programming) supports programming multiple robots in one cell. No source confirmed that MachineLogic coordinates two or more arms. | same search results; see Gaps

**Formic (robots as a service)**
- Business model: pay per hour "as low as $8/hour" (Formic marketing), with guaranteed uptime. Some deals start payment only once performance targets are met. A third-party estimate puts it at $8-15 per robot-hour, or about $2.8k-5.3k/month for a cobot running 16 h/day. Palletizing is priced per pallet. | [Robotics247](https://www.robotics247.com/article/formic_offers_robotics_as_a_service_and_financing_to_manufacturers); [FreightWaves](https://www.freightwaves.com/?p=527925); [Robotomated RaaS guide (Apr 2026)](https://robotomated.com/learn/market/robotics-as-a-service-raas-guide)
- Formic sells single-application cells (palletizing, machine tending). No multi-robot orchestration product was found. (inference, not checked beyond these sources)

**READY Robotics (ForgeOS)**: dead
- READY shut down in **August 2024** after raising $41.5M. Reported causes: a soft robotics market, trouble persuading robot makers to support ForgeOS, and funding that fell through. Assets were sold through an ABC (assignment for the benefit of creditors). | [The Robot Report](https://www.therobotreport.com/ready-robotics-maker-of-forgeos-shuts-down/); [CB Insights](https://www.cbinsights.com/company/ready-robotics)
- **Nov 2024: Standard Bots bought all of READY's IP** and planned to market READY's API with its own training software. | [The Robot Report](https://www.therobotreport.com/standard-bots-acquires-ready-robotics-ip-advance-ai-robot-training/); [Standard Bots blog](https://standardbots.com/blog/ready-robotics-acquisition)
- One aggregator (robotics.press) says READY was Maine-based and closed in 2026. This conflicts with all the primary sources and is likely wrong. | [robotics.press](https://www.robotics.press/news/ready-robotics-shuts-down-after-raising-41-5m-in/)

**Standard Bots**
- June 9, 2026: $200M Series C at a $1B valuation. Product line: Spark, Core, Thor and the bimanual Bolt. Core starts at about $37k (18 kg payload, 1.3 m reach). Revenue is not disclosed. | [SiliconANGLE](https://siliconangle.com/2026/06/09/standard-bots-raises-200m-1b-valuation-revolutionize-ai-native-industrial-robotics/); [The Robot Report](https://www.therobotreport.com/standard-bots-raises-200m-expand-u-s-manufacturing-footprint/); [Sacra](https://sacra.com/c/standard-bots)
- No source describes multi-robot orchestration on Standard Bots. | same

**Flexxbotics**
- FlexxCORE connects robots with CNCs, PLCs, inspection equipment and IT systems. Features: robot-to-machine interfacing, program control, alarms, automated changeover, digital-thread traceability. The March 2025 release added one robot tending multiple machines at once for "lights-out" production. | [Robotics Tomorrow 2025](https://www.roboticstomorrow.com/news/2025/03/26/flexxbotics-unveils-latest-release-of-flexxcore-with-powerful-new-capabilities-for-multi-machine-robotic-automation/24516); [UR+ listing](https://www.universal-robots.com/plus/products/flexxbotics/flexxtend)
- Open-source "Transformer" connector drivers (Apache 2.0) on GitHub. The FANUC connector (Feb 2026) claims standardized communication with "over 1000 makes and models" of equipment, plus "robot cluster orchestration". These are vendor claims. | [Automation World](https://www.automationworld.com/communication/news/55355056/flexxbotics-uploads-open-source-connector-drivers-to-github); [Automation.com](https://www.automation.com/article/flexxbotics-fanuc-industrial-robot-connector-driver-machine-interfacing-open-source-github-project)
- Dec 2025: IPR StackTrax dual-rail transfer unit offered with Flexxbotics. | [Automation.com](https://www.automation.com/article/flexxbotics-ipr-robotics-stacktrax-advanced-robotic-machine-tending)

**ABB / FANUC / DELMIA (vendor multi-arm motion)**
- ABB MultiMove: one controller coordinates up to 4 robots plus positioners (36 axes). "Independent" mode is 2-4 separate programs on one controller. "Coordinated" mode is for true cooperation, such as one robot holding a part while another works on it. Programs are hand-written in RAPID with SyncMove/SyncTask. | [ABB MultiMove PDF](https://library.e.abb.com/public/734fb908d1c8ee50c12576dd005b69d0/ABB%20MultiMove%20functionality.pdf); [Emerald Industrial Robot](https://www.emerald.com/ir/article/31/5/401/182110/ABB-s-MultiMove-fires-the-imagination)
- FANUC MultiArm / Coordinated Motion and Yaskawa Motoman coordinated control are the equivalent vendor features. | [arXiv 2404.06687](https://ar5iv.labs.arxiv.org/html/2404.06687)
- DELMIA R2025x added simulation and offline programming for synchronized and coordinated multi-robot motion (FANUC via MCRS, ABB via SyncMove; KUKA planned for Q4 2025). | [3DS community](https://3dswym.3dexperience.3ds.com/post/delmia-virtual-factory-robotics/new-in-r2025xfd01-support-for-abb-multimove-simulation-and-olp_qNEo9i_QTheZgFcwyuNkfQ); [3DS wiki](https://3dswym.3dexperience.3ds.com/wiki/delmia-virtual-factory-robotics/how-to-mcrs-and-synchronisation-modes-in-delmia-robotics_rR30t6pHSN6aZ16yIV8GAA)

**Mujin**
- MujinOS: a real-time digital twin that "perceives, plans and controls", with collision-free motion planning and an orchestration component that coordinates multiple robots and system components (company description). Mostly logistics and depalletizing. Raised a $233M Series D in Dec 2025. First integrator partners were named in Aug 2025 (Convergix, Honeywell Intelligrated, among others). | [Automation World](https://www.automationworld.com/55297500); [Control Design](https://www.controldesign.com/industry-news/news/55310088/mujin-launches-first-wave-of-automation-integrator-partnerships-with-mujinos-platform); [BusinessWire](https://www.businesswire.com/news/home/20250814187510/en)

**Covariant**
- Aug 30, 2024: Amazon hired co-founders Abbeel, Chen and Duan plus about a quarter of the staff, and took a non-exclusive license to RFM-1 (a "reverse acquihire"). Covariant continued under new leadership. Its focus is warehouse picking, not cell orchestration. | [GeekWire](https://www.geekwire.com/2024/amazon-hires-covariant-founders-inks-licensing-deal-with-robotics-ai-startup-in-latest-reverse-acquihire-deal/); [SiliconANGLE](https://siliconangle.com/2024/09/02/amazon-hires-founders-ai-robotics-startup-covariant)

**Fleet managers (analogues)**
- InOrbit.AI raised a $10M Series A (around Sept 2025) for its robot orchestration platform. | [Robotics247 orchestration tag](https://www.robotics247.com/topic/tag/Orchestration/P40)
- Mobile-fleet orchestrators coordinate by traffic and missions (VDA5050, Open-RMF), not by shared-workspace arm motion. This is also the interface NVIDIA Mega uses. | [NVIDIA dev blog](https://developer.nvidia.com/blog/simulating-robots-in-industrial-facility-digital-twins/)

### Inferences
- Commercial "orchestration" in manufacturing means **connectivity and sequencing** (Flexxbotics, MachineLogic, Bright Machines) or **one vendor's synchronized motion** (MultiMove). Learned dynamic allocation across arms is still research (RoboBallet), now being packaged by Intrinsic through Google.
- The biggest players (NVIDIA, Intrinsic/Google, Foxconn, FANUC) are converging on "skills + digital twin + open ROS-compatible layer". Intrinsic Core (open source) and Mega (VDA5050/ROS 2) both point that way.
- The cross-vendor middleware startup (READY) failed in 2024, partly because robot makers would not cooperate. This is evidence that a cross-vendor layer is hard to sell, not that nobody needs it.

### Gaps
- Siemens (SIMATIC Robot Integrator / Robot Library; Process Simulate) and Rockwell: no source retrieved. From memory: Siemens lets a PLC command robots from multiple vendors through standard function blocks, and Process Simulate does multi-robot cycle-time and interference simulation (inference, not checked).
- Formant and Freedom Robotics status in 2026: not searched (budget).
- Pricing for Intrinsic, Mujin, Flexxbotics, Bright Machines and NVIDIA Mega: none found public. Mega is free to developers as a blueprint (inference, not checked).
- Whether Vention MachineLogic can coordinate two or more arms in one cell: not confirmed.

## 2. How do they split work between robots today: fixed programming, line balancing, or dynamic allocation?

### Takeaway
In production the split is almost always **fixed at design time**: each station or arm gets a hand-written or offline-programmed program, and multi-arm cooperation is scripted synchronization (MultiMove SyncMove, FANUC MCRS). Dynamic, automatic task splitting across arms is **rare and research-stage**. It is more common for mobile fleets, where task auctions and traffic negotiation are standard (Open-RMF).

### Cited Findings
- MultiMove "Independent" = separate programs on a shared controller. "Coordinated" = explicitly programmed cooperation. | [ABB MultiMove PDF](https://library.e.abb.com/public/734fb908d1c8ee50c12576dd005b69d0/ABB%20MultiMove%20functionality.pdf)
- DELMIA's 2025 support is for simulating already-defined synchronized and coordinated programs, not for allocating them. | [3DS community](https://3dswym.3dexperience.3ds.com/post/delmia-virtual-factory-robotics/new-in-r2025xfd01-support-for-abb-multimove-simulation-and-olp_qNEo9i_QTheZgFcwyuNkfQ)
- RoboBallet, the leading published system for automatic task + motion allocation across 8 arms, is simulation-only and cannot handle ordered tasks or different robot capabilities. | [The Batch](https://www.deeplearning.ai/the-batch/google-deepminds-roboballet-research-project-blends-graph-neural-networks-with-reinforcement-learning-to-drive-8-armed-robots); [Control.com](https://control.com/news/google-deepminds-roboballet-cuts-robot-planning-time/)
- Flexxbotics' "multi-machine" feature is one robot serving many machines, i.e. sequencing, not splitting work across arms. | [Robotics Tomorrow](https://www.roboticstomorrow.com/news/2025/03/26/flexxbotics-unveils-latest-release-of-flexxcore-with-powerful-new-capabilities-for-multi-machine-robotic-automation/24516)
- Open-RMF has a task-allocation planner for fleets. Its task description structure is described as "narrow" and limited by design. | [Open Robotics discourse #169](https://discourse.openrobotics.org/t/generalized-task-composition-169/44714.md); [#147](https://discourse.openrobotics.org/t/clarifications-regarding-the-task-allocation-planner-147/44905.md)

### Inferences
- A demo where arms share one task list and re-split it live (for example when an arm is removed, slowed or added) would be ahead of what is shipped in manufacturing. It sits squarely in the gap RoboBallet leaves: precedence constraints and robots with different capabilities.

### Gaps
- No industry survey found that quantifies how often dynamic allocation is used in production lines.
- Line balancing (assigning tasks to stations to equalize cycle time) is done in Process Simulate, DELMIA or spreadsheets. No source on how automated it is today was retrieved.

## 3. What are the known pain points?

### Takeaway
The consistent pain points are: integration costs a multiple of the robot price; each vendor has its own language and tools, so plants standardize on one brand; reprogramming for new products is slow; and cross-vendor orchestration has no commercial winner (READY failed, and Flexxbotics and Intrinsic are trying).

### Cited Findings
- Each robot maker ships its own language and tools, which creates "strong vendor lock-in arising from its technical structure". | [Trend Micro](https://www.trendmicro.com/pl_pl/research/21/c/the-security-dilemma-of-smart-factories-part-1-specificity-of-the-programming-languages-used-to-move-industrial-robots.html)
- Integrators and most manufacturers try to stick to one robot brand because of language complexity (FANUC TP/KAREL, ABB RAPID). Languages persist because of backward compatibility and vendor revenue from spare parts and options. | [Robohub: future of offline programming](https://robohub.org/the-future-of-robot-off-line-programming)
- Example of a plant with five robot brands needing staff who know five languages. Switching brands is costly. | [Wevolver](https://www.wevolver.com/article/eliminating-industrial-robot-brand-lock-in-is-more-important-than-ever)
- ARIA (UK) survey: about three-quarters of developers and integrators say they design for interoperability, yet companies may resist openness to protect market share. | [ARIA report](https://www.aria.org.uk/media/skbgnqaq/the-future-of-robotics-modularity-and-interoperability.pdf)
- READY failed partly because robot makers would not work with ForgeOS. | [The Robot Report](https://www.therobotreport.com/ready-robotics-maker-of-forgeos-shuts-down/)
- Open-RMF's own maintainers list traffic negotiation and task behaviors as "not customizable enough". Resource contention over shared resources such as tool heads is still an open question. | [ROSCon 2024: State of Open-RMF](https://roscon.ros.org/2024/talks/The_State_of_Open-RMF.pdf); [ROSCon 2024: Resource contention](https://roscon.ros.org/2024/talks/OpenRMF_and_the_Challenge_of_Resource_Contention_in_Large-Scale_Robot_Fleets.pdf)
- Vendors' own pitches target set-up time: Vention claims 15-minute CAD-to-pick, and Intrinsic says integrators can manage cells "without programming each robot". This implies that programming each robot is the status-quo pain. | [Robotics247](https://www.robotics247.com/article/imts-2026-vention-facilitates-manufacturing-with-physical-ai-and-agentic-ai-in-one-platform); [SiliconANGLE](https://siliconangle.com/2026/06/22/intrinsic-unveils-next-gen-accessible-automated-industrial-ai-robotic-assembly/)

### Inferences
- The pain is strongest in high-mix, low-volume work, where reprogramming time dominates. Vendor pitches (Intrinsic skills, Vention, Flexxbotics automated changeover) all target that segment.

### Gaps
- No time study found on reprogramming or changeover hours per new product.
- Ramp-up time figures were not found.

## 4. Open-source orchestration layers (ROS 2) and their limits

### Takeaway
Open-RMF is the main open-source multi-robot orchestrator, but it was built for mobile fleets. Arms enter only as "workcells" (dispenser/ingestor) through custom nodes, with MoveIt doing the motion. New open layers from industry (Intrinsic Core, Flexxbotics Transformers, the RoboBallet code) are emerging in 2025-2026.

### Cited Findings
- Open-RMF was designed for high-level task planning of mobile robots. Users ask whether manipulators are possible at all. A static arm used as a "dispenser" needs a custom node written from scratch. Workcell motion planning is done by MoveIt, not RMF. | [Discourse #668](https://discourse.openrobotics.org/t/open-rmf-integration-with-manipulators-668/44555); [Discourse #136](https://discourse.openrobotics.org/t/mobile-manipulation-and-rmf-136/44709)
- The "next generation" of Open-RMF aims to generalize across mobile robots and workcells (status unknown as of Nov 2026). | [Discourse: conveyor/workcell](https://discourse.openrobotics.org/t/guidance-on-warehouse-automation-using-openrmf-and-conveyor-traffic-management/41668.md)
- Intrinsic Core (Sept 2026): open-source ROS-compatible capabilities and reference solutions. | [Intrinsic blog](https://www.intrinsic.ai/blog)
- Flexxbotics Transformers: Apache-2.0 connector drivers for machines, PLCs and robots. | [Automation World](https://www.automationworld.com/communication/news/55355056/flexxbotics-uploads-open-source-connector-drivers-to-github)
- RoboBallet code was open-sourced. | [The Decoder](https://the-decoder.com/roboballet-uses-ai-to-choreograph-multiple-industrial-robots-for-safe-and-efficient-teamwork/)

### Inferences
- MoveIt 2 can plan for multiple arms as one combined planning group (joint-space planning for a combined group, collision-checked against each other). It does not allocate tasks. BehaviorTree.CPP gives a way to sequence work but no allocator. ros2_control can run several arms under one controller manager. None of them decides which arm does what (inference, not checked; not searched this session).

### Gaps
- No current (2026) docs checked for MoveIt multi-arm limits or BehaviorTree.CPP multi-agent patterns.
- No source found on the Open-RMF next-gen release status in 2026.

## 5. Market numbers (integration cost, programming share)

### Takeaway
Industry rules of thumb put the arm at about 25-40% of installed cell cost, and integration plus programming at roughly 1x to 4-6x the arm price. All of these figures come from vendor or integrator marketing, not audited studies.

### Cited Findings
- The arm and controller make up about 30-40% of installed cost. End effector and tooling 10-20%, fixtures 10-15%, controls/PLC/HMI 10-15%, installation and commissioning 15-20%. | [grabarobot 2026 guide](https://grabarobot.com/blog/robot-integration-cost-guide-2026/) (marketing source)
- Integration including safety, tooling and programming is about 4-6x the robot cost (rule of thumb). | [Engineering.com](https://www.engineering.com/story/time-and-money-how-much-do-industrial-robots-cost)
- Integration is 50-100% of hardware cost for a first cell, per another guide. Integrators quote 1.5-3x the arm price. | [LANPDT](https://lanpdt.com/tech-talk-episode-19/); [Inbolt](https://www.inbolt.com/resources/what-are-integration-costs)
- KUKA: integration adds $10k-50k, or 2-6x the robot price for complex systems (Standard Bots marketing). | [Standard Bots](https://standardbots.com/articles/kuka-robot-pricing)
- "Programming and integration are 50-70% of the cost of a robot application", attributed to an industry official. I could not verify the original. | [GCN](https://gcn.com/factory-robot-arm-priced-000-carries/21611/)
- Formic RaaS: $8-15 per robot-hour. | [Robotomated](https://robotomated.com/learn/market/robotics-as-a-service-raas-guide)

### Gaps
- No IFR or analyst figure was retrieved for the size of the orchestration-software market or the programming share of cost (budget).

## 6. Gaps a hackathon prototype could credibly show

### Takeaway
The credible white space is **automatic, live re-splitting of one task across several low-cost arms, with precedence and different arm capabilities, checked in sim and run on real hardware**. That is exactly what RoboBallet leaves open and what commercial tools do not ship.

### Cited Findings
- RoboBallet's stated limits are no ordered tasks, no robots with different capabilities, and simulation only. | [The Batch](https://www.deeplearning.ai/the-batch/google-deepminds-roboballet-research-project-blends-graph-neural-networks-with-reinforcement-learning-to-drive-8-armed-robots); [Control.com](https://control.com/news/google-deepminds-roboballet-cuts-robot-planning-time/)
- Commercial multi-arm coordination is a fixed program per vendor (MultiMove, MCRS). | [ABB](https://library.e.abb.com/public/734fb908d1c8ee50c12576dd005b69d0/ABB%20MultiMove%20functionality.pdf)
- Open-RMF does not handle arms in a shared workcell well. | [Discourse #668](https://discourse.openrobotics.org/t/open-rmf-integration-with-manipulators-668/44555)
- Mega does twin-first fleet testing, but for AMR-style VDA5050 fleets, and is still in preview. | [NVIDIA](https://blogs.nvidia.com/blog/mega-omniverse-blueprint)

### Inferences
- Demo ideas, each grounded in a gap above:
  1. **Live re-allocation**: drop one arm mid-run (or slow it down) and the others absorb its share, showing throughput before and after.
  2. **Different capabilities + precedence**: arms with different reach or grippers, and a task graph with ordering (RoboBallet's admitted limits).
  3. **Twin-gated execution**: every allocation is collision-checked in a calibrated digital twin before the real arms move (Mega-style, but for arms).
  4. **Natural-language job to multi-arm plan**: no per-robot programming, the Intrinsic and Vention pitch, made cross-arm.
- What would not be credible: claims about cross-vendor industrial controllers, safety certification, or production throughput. Hobby arms cannot show these.
