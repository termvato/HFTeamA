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

## Where it stands (Sat 21 Nov)

- **Recording studio:** live top and wrist cameras, start / stop / upload, using Hardware's
  calibrated pairs. Code and guide in [`vla/`](../vla/README.md).
- **First dataset:**
  [`sahilrajpurkar03/so101_stack_single_rakesh`](https://huggingface.co/datasets/sahilrajpurkar03/so101_stack_single_rakesh)
  (public). 20 episodes on the rakesh pair, "Put the objects in the box", `top` and `wrist_a`.
- **Training time** on the RTX 3070 laptop: ACT measured at 0.32 s per step, about 3-4.5 h for
  30-50k steps. SmolVLA estimated at about 6-8 h for 20k steps.
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
