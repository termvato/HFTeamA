# Orchestra: video script (target 2:30, limit 3:00)

One continuous real run gives shots 4 to 7: film the table and record the screen at the same
time, then cut between them. Shots 1, 2, 9 and 10 need no robots.

| # | Time | Shot | Voice-over | On screen |
|---|---|---|---|---|
| 1 | 0:00-0:12 | Messy desk, someone checking the time | "Your desk is a mess and your call starts in two minutes." | |
| 2 | 0:12-0:27 | Title card, then the problem | "This robot arm costs about a hundred dollars. Teaching it is the expensive part: one independent test got 30 to 40 percent success after 50 demonstrations. And a second arm means starting over." | `SO-101: about $100` then `30-40% after 50 demos` |
| 3 | 0:27-0:47 | Screen: camera view, objects named and coloured by arm, reach, estimated time | "Ours needs no training per object. The camera names what is there. Each object goes to an arm that can reach it. Every step has a time, so we know when it is done." | `0 training runs per new object` |
| 4 | 0:47-1:05 | One real arm tidying, countdown in a corner | "One arm starts. The clock runs on measured times, not guesses." | |
| 5 | 1:05-1:12 | Close-up: a hand plugs in the USB | (silence) | |
| 6 | 1:12-1:27 | Screen: "arm joined", old plan against new plan | "The second arm is found by its USB and joins in under a second. The rest of the work is re-split, live. Nothing is re-programmed." | `joined in under 1 s` |
| 7 | 1:27-1:55 | Both arms working, taking turns at the bowl, last box in | "Two arms never go into the same spot at once. Both finish the desk." | the run's real time, e.g. `5 boxes, 106 s` |
| 8 | 1:55-2:13 | Three-arm simulation, sped up | "Same code, three arms, another job. Arms join as it runs: 28 of 28 steps. This one is simulated." | `SIMULATED` and `28/28 steps, 14 boxes` |
| 9 | 2:13-2:25 | Result card | "Factories installed 603,000 robots last year, and setting one up is quoted at one and a half to three times its price. Add an arm, re-program nothing. Built in 48 hours by six people." | `603,000 robots installed in 2025` |
| 10 | 2:25-2:30 | End card | "Plug in an arm. Watch the clock drop." | Orchestra, the tagline |

About 190 words of voice-over: a steady pace for 2:30. If it runs long, cut the second sentence
of shot 2 or the first of shot 9.

## Before filming

- **Shot 6 depends on where the boxes are.** The predicted time only drops if both arms can reach
  the boxes (the middle band by the bowl). In the one recorded join (22 Nov 09:25) every box was
  reachable by one arm only and the prediction rose from 120 s to 190 s. If it rises in the take,
  show the re-split and leave the before/after numbers out.
- **Shot 7:** use the time of the run that is filmed. 106 s for 5 boxes is the best run so far
  (22 Nov 09:20). If the middle is sped up, put a small "2x" on screen.
- **Shot 8** is `docs/media/pyramid_isaac_sim.mp4` (1:52 long): three arms, 14 boxes, arm 2 joins
  after 3 boxes, arm 3 after 7, 110.5 s. Keep the "simulated" label on for the whole shot.

## Where the numbers come from

| Number | Source | How firm |
|---|---|---|
| SO-101 about $100 (parts $100-130) | Hugging Face launch, TechCrunch, 28 Apr 2025 | firm |
| 30-40% success after 50 demos | one independent blog test on SO-101 pick-and-place (Tiensmith, Apr 2026): ACT about 40%, SmolVLA and pi0.5 about 30%. SmolVLA's authors report 78% on their own setup | one test: say "one independent test" |
| Joined in under 1 s; 5 boxes in 106 s | our real runs, 22 Nov (`docs/PHASE_IMPROVEMENT_PLAN.md`) | measured by us |
| 28/28 steps, 14 boxes, 110.5 s | our simulation (MuJoCo, shown in Isaac Sim) | simulated |
| 603,000 robots installed in 2025 (+11%) | IFR World Robotics | firm |
| Set-up at 1.5-3x the robot's price | integrator and trade-press quotes (Inbolt, Engineering.com) | rough range: say "quoted at" |
