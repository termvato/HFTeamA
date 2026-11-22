# Isaac Sim view of the MuJoCo simulation

Vivek's simulation (`scripts/sim_demo.py`: SO-101 arms sorting blocks, one worker per arm, locks
on the shared bins) runs in MuJoCo as before. `--isaac PORT` also streams it to Isaac Sim, which
draws the same arms, blocks and bins with RTX. **Isaac Sim only renders**: MuJoCo does the IK,
timing, grasps and locks, so what Isaac shows is exactly what was simulated.

```bash
scripts/isaac/run.sh                       # starts Isaac Sim, then the 3-arm sort12 run
scripts/isaac/run.sh --speed 2             # any sim_demo.py option passes through
RECORD=/tmp/frames scripts/isaac/run.sh    # headless: PNG frames of the camera view
```

Or in two terminals (either can start first; a new run with another task rebuilds the scene):

```bash
env -u PYTHONPATH ~/isaac-sim/python.sh scripts/isaac/mirror.py      # --view top|side, --headless --record DIR
.venv/bin/python scripts/sim_demo.py --isaac 9871
```

Setup once: Isaac Sim 6.1 in `~/isaac-sim` (or `ISAAC=<folder>`), and the repo's sim extra in a
venv: `python -m venv .venv && .venv/bin/pip install -e ".[sim]"`.

## Pyramid demo: arms joining as the work goes on

`tasks/pyramid/` (written by `make_pyramid.py`): 3 arms build a 3-layer pyramid (3x3, 2x2, 1) of
14 white 40 mm boxes, each with its own AprilTag (tag36h11) on top. Arm 1 starts alone; after 3
boxes are placed arm 2 is handed to the team's `Executor` with `add_arm()`, after 7 boxes arm 3,
as plug-and-produce does when a real arm is plugged in. From then on they work at the same time.

```bash
scripts/isaac/pyramid_live.sh        # Isaac Sim window, live (VIEW=close, ORBIT=0 for a still camera)
scripts/isaac/pyramid_video.sh       # -> runs/pyramid/pyramid.mp4: captions, slowly circling camera
.venv/bin/python scripts/isaac/pyramid_demo.py --join-after 2,6   # MuJoCo only, with checks
```

| Arm | Base (cm) | Faces | Colour |
|---|---|---|---|
| 1 | (0, 0) | south (+x) | black |
| 2 | (0, 59) | south (+x) | green |
| 3 | (51, 29) | north (-x) | white |

- **Boxes:** scattered at random (seeded) just outside the middle circle, in each arm's own wedge
  of the table, 25-48 degrees either side of its direction: 5 / 5 / 4. Each arm fills the pyramid
  slots nearest to it, never more than 40 cm away (a stretched-out arm stalls near 43 cm).
- **Two steps per box, run by the Executor:** the pick grips the box with the jaws square to it
  and lifts it straight up; the place carries it straight to its slot, sets it down square, lifts
  out and stands the arm upright over its base. No detour in between.
- **No collisions:** no arm enters another arm's wedge, so picks need no lock; the pyramid is the
  one shared area, and every place holds its lock. The Executor only starts a step once its lock
  is free, so no arm waits mid-motion: no timeouts, no deadlock. While one arm places, the others
  pick.
- **Gripper and boxes:** 28 mm between boxes in the pyramid; the jaws close to the box's 40 mm,
  open only to 52 mm, and are turned (of the four square headings) for the most room, since the
  SO-101's fixed jaw carries the bulky motor. Simulation only, in `pyramid_demo.py`.
- **Checked on every run** (`pyramid_demo.py` exits 1 otherwise): all steps ok, no contact
  between two arms (50 Hz), smallest gap between arms, smallest gap between a gripper and a box it
  is not moving, no arm dropped by the Executor's heartbeat check. Three runs on 22 Nov: 28/28
  steps each in 120-122 s, boxes 5 / 5 / 4, no contact, closest 6.1 cm, gripper within 1.7 mm of
  a neighbouring box, no arm dropped; every box within 2.9 mm of its slot.
- **Look** (`mirror.py`): Isaac Sim's default light rig, oak-plank table top (the texture that
  ships with Isaac Sim), glossy PBR materials, AprilTag images from `make_apriltags.py`.
- The video replays the recorded poses frame by frame (`mirror.py --replay`), so it plays at 30 fps
  in sim time however slowly Isaac Sim renders; `make_video.py` adds the captions.

## How it works

| Side | File | Does |
|---|---|---|
| MuJoCo | `src/hfteama/view/isaac_mirror.py` | On start, writes every visible shape of the compiled scene (geom groups 0-2: arm meshes, blocks, bin discs) to `~/.cache/hfteama_isaac/scene_<id>.npz`. Then sends each body's world pose over UDP, 30 times a second. |
| Isaac Sim | `scripts/isaac/mirror.py` | Builds one USD Xform per body with those meshes, moves them to the received poses every frame. Adds a table, Isaac Sim's default light rig, glossy PBR materials and a camera framed on the scene. No PhysX. Or replays a recording (`--replay`). |

- No MJCF import: Isaac Sim 6.1's MJCF converter rejects the SO-101 model (its jaw pads are inline
  meshes), and an articulation would need physics. Copying MuJoCo's compiled shapes avoids both.
- Each arm gets its own colour (`--same-colour` keeps the model's yellow).
- Both run on the same machine (the scene file goes by path). The Isaac Sim window used about
  1.5 GB of GPU memory on the RTX 3070 laptop (8 GB); X-VLA training took 5.4 GB there, so do not
  run both at once.
- Earlier attempt, for reference: the joint-only Isaac scene removed in 9a28447 (#33).
