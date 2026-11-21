# HFTeamA: plug and produce

Finish a batch of work faster by plugging in more SO-101 arms. When an arm is added, removed or
fails, the orchestrator re-splits the remaining work live and shows the new predicted finish time
and what limits further speed-up. Task-agnostic: any job that splits into steps, order and shared
zones (see `tasks/README.md`). Plan and teams: `docs/plan.md`. Who is doing what right now: [`STATUS.md`](STATUS.md). Visual map: `docs/README.md`.

## How it works

```mermaid
flowchart LR
    T["task folder<br/>steps, order, shared resources"] --> G["taskgraph<br/>load and validate"]
    G --> S["scheduler<br/>who does what, when<br/>+ lower bound"]
    S -- plan --> E["executor<br/>run, re-split on change"]
    E -- re-plan --> S
    E -- "execute(task)" --> W["arm workers<br/>real / sim / fake"]
    W -- Result --> E
    D["discovery<br/>arm plugged in / out"] -- arm events --> E
    E --> L["event log"]
    L --> V["live screen<br/>Gantt, finish time, bottleneck"]

    classDef ops fill:#e6f2ec,stroke:#2e7d4f,color:#16212b
    classDef act fill:#e7f0fb,stroke:#1d76db,color:#16212b
    classDef hw fill:#fbeaea,stroke:#b60205,color:#16212b
    class G,S,E,L,V ops
    class W act
    class D hw
```

Green: Operations (#21). Blue: Action (#20). Red: Hardware (#19). Full plan: [`docs/plan.md`](docs/plan.md).
More detail (what happens when an arm is plugged in, speed-up limits, who builds what):
[`docs/README.md`](docs/README.md), and the one-page map [`docs/map.html`](docs/map.html) (download and open in a browser).

## Requirements

- Python 3.10+

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,sched,view]"   # add sim for MuJoCo
```

## Usage

```bash
hfteama
# or
python -m hfteama
```

### Real arms: teleoperate a pair (Hardware, #19)

All arms are already calibrated; the calibration files are in the repo
([`configs/calibration/`](configs/calibration/README.md)). **Do not calibrate them again.**

**1. Once per computer**, from the repo root on the `hardware` branch:

```bash
git fetch && git switch hardware && git pull
uv venv --python 3.13 .venv && source .venv/bin/activate      # LeRobot needs Python 3.12+
uv pip install -e ".[dev]" "lerobot[feetech,core-scripts]>=0.7"
```

If you already have a venv with LeRobot, you still need `uv pip install -e ".[dev]"`
(or `pip install -e ".[dev]"`) inside this repo, or `hfteama: command not found`.

**2. Every time:** plug in both arms by USB and switch on the follower's 12 V supply, then:

```bash
source .venv/bin/activate
hfteama pairs                     # both arms of your pair must say "connected at /dev/ttyACM*"
hfteama --pair rakesh teleop      # or: --pair hacklab
```

Put the leader in roughly the follower's pose before starting (the follower jumps to the
leader's pose), move slowly at first, stop with `Ctrl+C`.

**If LeRobot asks** `Press ENTER to use provided calibration file associated with the id ...`:

| id in the message | What to do |
|---|---|
| `rakesh_leader` | Press **ENTER**. Expected once: it writes the base adjustment from the file into the leader. |
| anything else | Don't press anything; stop with `Ctrl+C` and ask the Hardware team. |

Never type `c`: it starts a new calibration and overwrites the one that works.

If it starts calibrating without asking (it shows the "middle of the range" diagram),
it did not find the files: stop with `Ctrl+C` and make sure you ran it with `hfteama` (not
`lerobot-teleoperate` by hand) on the `hardware` branch. `hfteama -n --pair rakesh teleop`
prints the exact LeRobot command; it must contain `--robot.calibration_dir=.../configs/calibration/...`.

| Pair | Follower (USB serial) | Leader (USB serial) | Cameras |
|---|---|---|---|
| `rakesh` (default) | `rakesh_follower` (`5B79014975`) | `rakesh_leader` (`5B8E115290`) | `wrist` |
| `hacklab` | `vato_follower` (`5A7A015861`) | `hacklab_leader` (`5A7A015820`) | `wrist` |

#### Ports change when you plug in: that's fine

`/dev/ttyACM0`, `ttyACM1`, ... are assigned in plug-in order, so they change every time.
`hfteama` finds each arm by its **USB serial** (table above) and each camera by its
`/dev/v4l/by-id/` name, so you never set ports by hand. Always check first:

```bash
hfteama pairs     # every arm you need must say "connected at /dev/ttyACM*", and show its cameras
```

If an arm says "not connected": its USB cable is out, or (follower) its 12 V supply is off.
Keep each servo board on its own arm: the serial belongs to the board, not the arm.

#### Run both pairs at once (with cameras)

One terminal per pair; each arm is its own USB bus, so they don't interfere.

```bash
# terminal 1                                   # terminal 2
source .venv/bin/activate                      source .venv/bin/activate
hfteama --pair rakesh teleop --cameras         hfteama --pair hacklab teleop --cameras
```

Both stream to the same Rerun window; pick each pair's recording in Rerun's left panel.
With 4 arms and 2 cameras on one laptop, plug the cameras into different USB ports (or a
powered hub); camera timeouts or dropped fps usually mean too much on one port.

#### Record a dataset

Each pair records to its own dataset (`record.repo_id` per pair in
[`configs/teleop_pairs.json`](configs/teleop_pairs.json)):

| Pair | Dataset |
|---|---|
| `rakesh` | `diegoloayza789/hfteam_rakesh_so101` |
| `hacklab` | `diegoloayza789/hfteam_hacklab_so101` |

Defaults (shared `record` section): 10 episodes, 60 s each, 30 s to reset between
episodes, 30 fps, not pushed to the Hub. Datasets are saved under
`~/.cache/huggingface/lerobot/<repo_id>`.

```bash
hf auth login     # once, only to push datasets to the Hub (push_to_hub: true)

# one pair
hfteama --pair rakesh record --episodes 10 --task "Pick up the cube and place it in the box"

# both pairs: one terminal each, same task description per pair
hfteama --pair rakesh  record --episodes 10 --task "..."     # terminal 1
hfteama --pair hacklab record --episodes 10 --task "..."     # terminal 2

# add more episodes to an existing dataset later
hfteama --pair rakesh record --resume --episodes 5 --task "..."
```

Flow of each episode: record (move the leader to do the task) → reset time (put the
objects back) → next episode. Keys while recording: `→` ends the episode early, `←`
discards it and records it again, `Esc` stops and saves.

**Two pairs recording at once:** on a normal desktop (X11) LeRobot listens to the keyboard
globally, so **a key press goes to both recordings** (`Esc` stops both, `→` ends the
episode in both). Start both terminals together and use the same `--episodes` so they stay
in step, or record one pair at a time if each needs its own control.

`-n` prints the LeRobot command without running it, e.g. `hfteama -n --pair hacklab record`.
Calibrating a new arm, camera setup and troubleshooting:
[`docs/hardware/so101_setup.md`](docs/hardware/so101_setup.md).

## Development

```bash
pytest          # run tests
ruff check .    # lint
ruff format .   # format
```

## Project structure

```
.
├── src/hfteama/
│   ├── taskgraph/   # task graph JSON: loader, validator (frozen interface)
│   ├── scheduler/   # HEFT, CP-SAT, plan checker, lower bound
│   ├── executor/    # dispatch, resource locks, re-plan on arm added / removed / failed
│   ├── arms/        # arm contract, skill registry, kinematics, real / sim / fake arms
│   ├── view/        # live Gantt, predicted finish, binding limit, scaling plot
│   └── events.py    # event log (frozen interface)
├── tasks/           # one folder per task: graph, poses, extra skills
├── configs/         # arm registry template
├── scripts/         # benchmarks, tools
├── research/        # background report and notes
├── docs/            # plan.md (main plan), map.html + diagram in docs/README.md
├── tests/           # run by CI on every push (.github/workflows/ci.yml)
└── pyproject.toml
```
