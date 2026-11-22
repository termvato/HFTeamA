# Alternate method: learned policy per arm (VLA)

**Status:** alternative to the scripted IK skills; being prepared on `action` (Sahil, #20). The
main plan ([`plan.md`](plan.md)) is unchanged. This only swaps how an arm carries out a step.

## The idea

Today a step such as "pick object 3, place it in bin B" runs as a scripted skill: inverse
kinematics to table coordinates. In this method the same step runs as a **learned policy**
(ACT or SmolVLA), trained from teleoperated demonstrations. It drives the arm from the
camera images and the arm's joint angles.

**The policy does not split the work.** It knows nothing about other arms. The orchestrator
splits the objects across the arms that are connected, as it does now. Then **the same trained
model runs once per arm**, each copy doing the step it is given:

```
4 objects ──► Orchestrator (team's scheduler)
                 "2 arms connected → A: obj1, obj2 · B: obj3, obj4"
                     │                            │
          "put the red cube in the box"   "put the blue cube in the box"
                     ▼                            ▼
          policy copy on arm A            policy copy on arm B
          (same trained weights)          (same trained weights)
```

Plug in a third arm and the orchestrator re-splits as usual. The new arm runs another copy of
the same weights, with **no new training**.

## What stays the same

| Piece | Change |
|---|---|
| VLM front end, task graph, planner, re-split, live screen | none |
| Arm contract `execute(task) -> Result` | none: a `VLAArmWorker` implements it |
| Outcomes `ok`, `failed` + reason, `timeout`, `arm_down`, with start and end times | none |
| Shared areas (the box or bin) | none: zone locks keep one arm there at a time |

## What the worker does

1. Turn the task's skill and args into an instruction, e.g. `pick_place(object=red_cube, bin=box)`
   becomes "Put the red cube in the box".
2. Run the policy on this arm's cameras (`top` plus its own wrist camera) and joints at 30 Hz.
3. Stop when the arm is back near rest with the gripper open, or at the step timeout.
4. Return the result. The grasp check (gripper position plus current) and the VLM camera check
   run as for scripted skills.

## What it needs

- **Instructions that name the object**, so each arm can be told which one to take. These go with
  SmolVLA, which reads the instruction; **ACT ignores it**. The simpler alternative is a split by
  table side: each arm handles the objects on its half, which works with ACT too.
- **Data from both arms if possible.** The wrist view carries over to an untrained arm better
  than the top view, but expect a drop. 10-20 episodes on the second arm, trained into the same
  model, should close most of that gap. It is still one model for every arm.
- **Recordings with both arms moving**, so a moving neighbour in view is normal for the policy.

## Collisions with 2+ arms

**What we already have.** The executor locks shared areas. A step that uses the box takes its
lock right before the motion and releases it when the step ends, so two arms are never in the box
at once (`src/hfteama/executor/executor.py`, and "Two arms never in the same shared area at once"
in the plan's real-arm safety rules). With scripted IK that is enough: every motion is planned,
so its path is known before it runs.

**What a learned policy changes:**
1. **Its path is not known in advance.** The policy decides each move as it goes, so the route
   cannot be checked before the step starts, and the arm can drift outside the region we expect.
2. **One long step holds the box lock for the whole step.** A pick-and-place episode lasts
   25-60 s, and the lock covers all of it, even while the arm is still picking far from the box.
   With 2 or more arms they mostly wait for each other, and the speed-up disappears.

**Four layers, cheapest first:**

| Layer | What | Prevents | Training? |
|---|---|---|---|
| 1. Table layout | Place the bases so their reach overlaps only at the box; each arm has its own pick area. With N arms in a row, only neighbours can meet. | most arm-to-arm contact | no |
| 2. Split the step | Record "pick the red cube" and "place it in the box" as separate episodes and instructions. The executor then locks the box only during the place step, with no change to the arm contract. | long box locks, lost parallelism | yes: record pick and place separately |
| 3. Safety supervisor | In each policy worker, between the policy's output and the motors, every tick (30 Hz). Compute the gripper's position from the joint angles. If it leaves the arm's area, hold. If it nears the box without the lock, hold until granted. If it comes within a safety margin of another arm, freeze the arm that started later. | anything the policy gets wrong | no |
| 4. Hardware limits | Already on: a 10-degree cap per step on follower motion, and LeRobot's gripper torque limits. | violent motion if the other layers fail | no |

Layer 3 is the key one for policies: whatever the model outputs, the motors only get commands
that stay inside safe limits. It uses what Hardware already provides: every arm's position in one
shared table frame, and the joint-angle conversion in `src/hfteama/arms/kinematics.py`.

**With 3 or more arms:**
- In a row layout an arm can only meet its neighbours, so more arms do not mean more ways to
  collide.
- The box becomes the bottleneck. The team's benchmark already shows about 2.2x with 3 arms,
  limited by a shared area. One box per arm, or per pair, removes most of that waiting.
- The executor's lock makes arms queue for the box, so they cannot deadlock over it.
- The supervisor can only stop an arm, not steer it around another. If two arms freeze facing
  each other, the later one goes back to rest and retries.

**Not guaranteed:** the policy will not learn to avoid the other arm by itself; demonstrations do
not teach that reliably. Safety comes from layers 1, 3 and 4, not from the model.

## Where it stands (Sat 21 Nov)

- **Recording studio:** live top and wrist cameras, start / stop / upload, using Hardware's
  calibrated pairs. Code and guide in [`vla/`](../vla/README.md).
- **First dataset:**
  [`sahilrajpurkar03/so101_stack_single_rakesh`](https://huggingface.co/datasets/sahilrajpurkar03/so101_stack_single_rakesh)
  (public). 20 episodes on the rakesh pair, "Put the objects in the box", `top` and `wrist_a`.
- **Training time** on the RTX 3070 laptop: ACT measured at 0.32 s per step, about 3-4.5 h for
  30-50k steps. SmolVLA measured at 0.36 s per step; with early stopping the first run stopped at step 5000 after 37 min (best at step 2000).
- **Next:**
  1. Choose split by side or by name.
  2. Record more episodes.
  3. Train ACT.
  4. Test on arm A, then the same model on arm B.
  5. Run it behind the orchestrator with 2 arms and 4 objects.

## When to use which

- **Scripted IK:** precise and predictable for known objects at known coordinates. Needs a
  reliable object pose from the camera.
- **Learned policy:** copes with varied objects and imprecise positions. Its success rate has to
  be measured, and it needs demonstrations.
- **Both behind the same contract:** an arm can try the policy and fall back to the scripted skill
  when a step fails, with no change to the orchestrator.

Details (why the model cannot split the work itself, risks, likelihood):
[`vla/multi_arm_policy.md`](../vla/multi_arm_policy.md).
