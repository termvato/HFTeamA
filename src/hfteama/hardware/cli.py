"""`hfteama` hardware subcommands: thin wrappers over the lerobot-* tools.

Each command fills in ports, arm IDs, calibration directory and cameras from
configs/teleop_pairs.json, so every teammate runs the arms the same way.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from hfteama.hardware.pairs import (
    find_port_by_serial,
    follower_args,
    leader_args,
    load_pair,
    read_config,
    record_args,
)


def lerobot_cmd(name: str) -> str:
    """Prefer the lerobot binary next to the running interpreter (same venv)."""
    local = Path(sys.executable).with_name(name)
    return str(local) if local.exists() else (shutil.which(name) or name)


def run(cmd: list[str], dry_run: bool) -> int:
    print("$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def build_command(args: argparse.Namespace) -> list[str]:
    if args.command == "find-port":
        return [lerobot_cmd("lerobot-find-port")]
    if args.command == "find-cameras":
        return [lerobot_cmd("lerobot-find-cameras"), "opencv"]

    cfg = load_pair(args.config, args.pair)
    if args.command in ("setup-motors", "calibrate"):
        tool = "lerobot-setup-motors" if args.command == "setup-motors" else "lerobot-calibrate"
        arm = follower_args(cfg) if args.arm == "follower" else leader_args(cfg)
        return [lerobot_cmd(tool), *arm]
    if args.command == "teleop":
        cmd = [
            lerobot_cmd("lerobot-teleoperate"),
            *follower_args(cfg, with_cameras=args.cameras),
            *leader_args(cfg),
        ]
        if args.cameras:
            cmd.append("--display_data=true")
        return cmd
    if args.command == "record":
        overrides = {}
        if args.episodes is not None:
            overrides["num_episodes"] = args.episodes
        if args.task:
            overrides["single_task"] = args.task
        if args.repo_id:
            overrides["repo_id"] = args.repo_id
        cmd = [
            lerobot_cmd("lerobot-record"),
            *follower_args(cfg, with_cameras=True),
            *leader_args(cfg),
            *record_args(cfg, overrides),
            "--display_data=true",
        ]
        if args.resume:
            cmd.append("--resume=true")
        return cmd
    raise ValueError(f"Unknown command: {args.command}")


def list_pairs(config: str | None) -> int:
    data = read_config(config)
    for name, pair in data["pairs"].items():
        default = " (default)" if name == data.get("default_pair") else ""
        print(f"{name}{default}")
        for role in ("follower", "leader"):
            arm = pair[role]
            port = find_port_by_serial(arm["serial"]) if arm.get("serial") else None
            status = f"connected at {port}" if port else "not connected"
            print(f"  {role:9} id={arm['id']:18} serial={arm.get('serial', '-')}  {status}")
        cams = ", ".join(pair.get("cameras", {})) or "none"
        print(f"  cameras   {cams}")
    return 0


def add_subcommands(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("find-port", help="Identify which USB port an arm uses")
    sub.add_parser("find-cameras", help="List OpenCV camera indices")
    sub.add_parser("pairs", help="List arm pairs and whether each arm is connected")
    c = sub.add_parser("camera", help="View the pair's cameras live in Rerun")
    c.add_argument("names", nargs="*", help="Camera names (default: all of the pair)")
    for name, help_ in (
        ("setup-motors", "Assign IDs/baudrate to new motors (one at a time)"),
        ("calibrate", "Calibrate an arm and save it under configs/calibration"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("arm", choices=["follower", "leader"])

    t = sub.add_parser("teleop", help="Drive the follower with the leader")
    t.add_argument("--cameras", action="store_true", help="Also stream cameras to Rerun")

    r = sub.add_parser("record", help="Record a teleoperated dataset")
    r.add_argument("--episodes", type=int)
    r.add_argument("--task")
    r.add_argument("--repo-id")
    r.add_argument("--resume", action="store_true", help="Append to an existing dataset")


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "pairs":
        return list_pairs(args.config)
    if args.command == "camera":
        from hfteama.hardware.camera_view import view

        return view(load_pair(args.config, args.pair), args.names)
    return run(build_command(args), args.dry_run)
