"""`hfteama` hardware subcommands: thin wrappers over the lerobot-* tools.

Each command fills in ports, arm IDs, calibration directory and cameras from
configs/teleop_pairs.json, so every teammate runs the arms the same way.
"""

from __future__ import annotations

import argparse
import copy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from hfteama.hardware.pairs import (
    _flag,
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
        return teleop_command(
            follower_args(cfg, with_cameras=args.cameras) + leader_args(cfg), args
        )
    if args.command == "record":
        arms = follower_args(cfg, with_cameras=True) + leader_args(cfg)
        return record_command(arms, record_args(cfg, record_overrides(args)), args)
    raise ValueError(f"Unknown command: {args.command}")


def teleop_command(arm_args: list[str], args: argparse.Namespace) -> list[str]:
    cmd = [lerobot_cmd("lerobot-teleoperate"), *arm_args]
    if args.cameras:
        cmd.append("--display_data=true")
    return cmd


def record_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    if args.episodes is not None:
        overrides["num_episodes"] = args.episodes
    if args.task:
        overrides["single_task"] = args.task
    if args.repo_id:
        overrides["repo_id"] = args.repo_id
    return overrides


def record_command(arm_args: list[str], dataset_args: list[str], args: argparse.Namespace):
    cmd = [lerobot_cmd("lerobot-record"), *arm_args, *dataset_args, "--display_data=true"]
    if args.resume:
        cmd.append("--resume=true")
    return cmd


def pair_names(args: argparse.Namespace) -> list[str]:
    return [n.strip() for n in (args.pair or "").split(",") if n.strip()]


def refuse_if_uncalibrated(arms, solo_hint: str) -> None:
    from hfteama.hardware.multi import check_calibrations

    bad = check_calibrations(arms)
    if bad:
        raise SystemExit(
            f"Calibration file and servos differ for: {', '.join(bad)}. LeRobot would ask "
            f"to confirm, which can't be done for several arms in one terminal. Run "
            f"{solo_hint} once and answer the prompt (see README), then retry."
        )


def run_multi(args: argparse.Namespace) -> int:
    """`--pair a,b teleop|record`: one lerobot process per pair, in this terminal."""
    from hfteama.hardware.multi import run_parallel

    names = pair_names(args)
    if len(set(names)) != len(names):
        raise SystemExit(f"Pair listed twice: {args.pair}")
    commands = {}
    for name in names:
        one = copy.copy(args)
        one.pair = name
        commands[name] = build_command(one)
    if not args.dry_run:
        for name in names:
            cfg = load_pair(args.config, name)
            refuse_if_uncalibrated([cfg.follower, cfg.leader], f"`hfteama --pair {name} teleop`")
    return run_parallel(commands, args.dry_run)


def build_bimanual(args: argparse.Namespace) -> list[str]:
    from hfteama.hardware.bimanual import bimanual_args, load_bimanual

    bi = load_bimanual(args.config)
    if args.action == "teleop":
        return teleop_command(bimanual_args(bi, with_cameras=args.cameras), args)
    rec = {**bi.record, **record_overrides(args)}
    dataset = [f"--dataset.{k}={_flag(v)}" for k, v in rec.items()]
    return record_command(bimanual_args(bi, with_cameras=True), dataset, args)


def run_bimanual(args: argparse.Namespace) -> int:
    from hfteama.hardware.bimanual import arms, load_bimanual

    cmd = build_bimanual(args)
    if not args.dry_run:
        bi = load_bimanual(args.config)
        refuse_if_uncalibrated(
            arms(bi), f"`hfteama --pair {bi.left.pair} teleop` / `--pair {bi.right.pair}`"
        )
    return run(cmd, args.dry_run)


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
    _record_flags(r)

    b = sub.add_parser(
        "bimanual",
        help="Both pairs of the 'bimanual' config as one two-arm robot (one dataset)",
    )
    b.add_argument("action", choices=["teleop", "record"])
    b.add_argument("--cameras", action="store_true", help="teleop: also stream cameras")
    _record_flags(b)


def _record_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--episodes", type=int)
    p.add_argument("--task")
    p.add_argument("--repo-id")
    p.add_argument("--resume", action="store_true", help="Append to an existing dataset")


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "bimanual":
        return run_bimanual(args)
    if len(pair_names(args)) > 1:
        if args.command not in ("teleop", "record"):
            raise SystemExit("Several pairs (--pair a,b) only work with teleop and record.")
        return run_multi(args)
    if args.command == "pairs":
        return list_pairs(args.config)
    if args.command == "camera":
        from hfteama.hardware.camera_view import view

        return view(load_pair(args.config, args.pair), args.names)
    return run(build_command(args), args.dry_run)
