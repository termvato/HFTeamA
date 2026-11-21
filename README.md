# HFTeamA

SO-101 leader/follower setup on top of [LeRobot](https://github.com/huggingface/lerobot).
The `hfteama` CLI wraps the `lerobot-*` commands and fills in ports, arm IDs, the
calibration directory and cameras from [`config/arms.toml`](config/arms.toml), so the
whole team runs the arms the same way.

## Requirements

- Linux, Python 3.12+
- Two SO-101 arms: one **leader** (teleoperator) and one **follower** (robot), both over USB

## Setup

```bash
uv venv --python 3.13 .venv        # or: python3.13 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"         # or: pip install -e ".[dev]"
```

This installs `lerobot[feetech,core-scripts]` (pinned to 0.7.x).

## Arm pairs

| Pair | Arm | ID | USB serial | Calibrated |
|---|---|---|---|---|
| `rakesh` (default) | follower | `rakesh_follower` | `5B79014975` | 2026-11-21 |
| `rakesh` (default) | leader | `rakesh_leader` | `5B8E115290` | 2026-11-21 |

Ports are found from the USB serial, so a pair works on any computer without
mapping `/dev/ttyACM*`. Check with `hfteama pairs`. Select another pair with
`hfteama --pair <name> ...`.

## Quick start

```bash
hfteama pairs                    # check both arms are detected
hfteama teleop                   # move the leader, the follower follows
hfteama record --episodes 5      # record a dataset with cameras
hfteama calibrate follower       # only for new arms (or to redo a calibration)
```

Add `-n` to any command to print the underlying `lerobot-*` command without running it.
Full walkthrough: [docs/so101_setup.md](docs/so101_setup.md).

## Commands

| Command | Wraps | Purpose |
|---|---|---|
| `hfteama find-port` | `lerobot-find-port` | Find which `/dev/ttyACM*` an arm uses |
| `hfteama pairs` | — | List arm pairs and the port each connected arm is on |
| `hfteama find-cameras` | `lerobot-find-cameras opencv` | List camera indices |
| `hfteama setup-motors {leader,follower}` | `lerobot-setup-motors` | Assign motor IDs on a new arm |
| `hfteama calibrate {leader,follower}` | `lerobot-calibrate` | Calibrate an arm |
| `hfteama teleop [--cameras]` | `lerobot-teleoperate` | Leader drives follower |
| `hfteama record [--episodes N] [--task T] [--repo-id R] [--resume]` | `lerobot-record` | Record a dataset |

Override ports without editing the config: `HFTEAM_FOLLOWER_PORT=/dev/ttyACM0 hfteama teleop`.

## Development

```bash
pytest          # run tests
ruff check .    # lint
ruff format .   # format
```

If ROS is sourced in your shell, pytest may fail loading ROS plugins; run
`env -u PYTHONPATH pytest` instead.

## Project structure

```
.
├── calibration/     # Per-arm calibration JSON files (committed, shared by the team)
├── config/          # arms.toml: arm pairs, cameras, recording settings
├── docs/            # Documentation
├── src/hfteama/     # CLI and config loading
├── tests/           # Test suite
└── pyproject.toml   # Project metadata and tool configuration
```
