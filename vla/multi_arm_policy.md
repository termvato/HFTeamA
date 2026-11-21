# One trained policy, any number of arms

Team summary of this as an alternate method: [`docs/alternate_method_vla.md`](../docs/alternate_method_vla.md).

**Question.** We trained a policy on one arm ("put the objects in the box"). When a second arm is
plugged in, can that same model split the work, e.g. 4 objects: arm A does two and arm B the
other two, without training anything on arm B?

**Answer.** Yes, but the model is not the part that splits. A policy trained on one arm maps that
arm's camera images and joint angles to that arm's next joint moves. It has no idea how many
arms exist, so it cannot divide work, and two copies would both reach for the nearest object.
The split belongs to the orchestrator (Operations, #21). The model only has to do the one
pick-and-place it is told, on whichever arm it runs.

## Design: the split is in software, the skill is in the model

```
 4 objects ──► Orchestrator: arms connected? split the objects
                 1 arm : A does obj1..obj4 one after another
                 2 arms: A does obj1, obj2 · B does obj3, obj4
                     │                              │
        "Put the red cube in the box"   "Put the blue cube in the box"
                     ▼                              ▼
          policy copy on arm A              policy copy on arm B
          (same trained weights)            (same trained weights)
```

1. **The orchestrator assigns objects.** It knows which arms are connected (Hardware's
   plug/unplug detection, #19) and re-splits the remaining objects when an arm joins or leaves.
   This is scheduling code; nothing is trained.
2. **The same model runs once per arm** as a `VLAArmWorker` behind the existing
   `ArmWorker.execute(task)` contract. It gets one instruction at a time and returns ok or the
   kind of failure (README, Part 6).
3. **Adding an arm needs no new training**, only another copy of the same model.

A model that coordinates arms by itself would need two-arm demonstrations (our dual-arm dataset).
Even then it learns "these two arms", not "any number of arms".

## What has to change

| Problem | Why | Fix |
|---|---|---|
| The model must be told which object | Our 20 episodes all say "Put the objects in the box". **ACT ignores the task text entirely.** | Either per-object instructions ("Put the red cube in the box") with **SmolVLA**, which reads the text, or a **split by side**: each arm only handles objects on its own half of the table, which needs no language. |
| Arm B was never trained | Same arm model and calibration method, so joint angles mean the same thing. But arm B stands elsewhere: it appears in a different place in `top`, and the objects are at different positions relative to its base. | Feed `wrist_b` into the model's wrist input: the wrist view is the same relative to the gripper, so it transfers best. Show each arm only its half of `top`, or rely mainly on the wrist. Expect a drop in success on arm B. |
| The other arm moves in view | During single-arm recording the second arm stood still. | Add the dual-arm recordings (README, Part 4) to training. |
| Both arms at the box at once | Collision risk at the shared box. | Orchestrator shared-zone lock (#8): one arm in the box at a time. |

## How likely is it to work?

- **Splitting the work:** very likely. It is plain software that Operations is building already.
- **Arm A doing its share:** likely, with more varied data. 20 episodes is a small start (LeRobot
  recommends 50 or more).
- **Arm B with arm A's model and no arm B data:** uncertain. The wrist camera may carry it
  partly, but do not plan on it. A cheap fix is 10-20 episodes on arm B and **one model trained
  on both arms' data**. That is still one model for every arm, and adding an arm still needs no
  training.

## Plan

1. Decide how arms choose objects: **by side** (simplest, works with ACT) or **by name** (SmolVLA
   plus per-object instructions in the recordings).
2. Record more arm A episodes: vary start positions and objects, with the instruction that
   matches step 1.
3. Train, test on arm A, then run the same model on arm B unchanged. That test answers the
   "without training that arm" question directly.
4. Wire it to the orchestrator: two arms, four objects. Measure the time against one arm.

Training time on the RTX 3070 laptop for the current 20 episodes (23,915 frames):
- ACT: measured at 0.32 s per step at batch size 8, so about 3-4.5 h for 30-50k steps.
- SmolVLA: measured at 0.36 s per step. The first run stopped early at step 5000 after 37 min (best at step 2000).
