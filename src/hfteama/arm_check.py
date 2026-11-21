"""Staged, supervised checks for a real arm, in this order (docs/handoff/action-to-hardware.md):

    python -m hfteama.arm_check check-config                     # no arm: is the config consistent?
    python -m hfteama.arm_check probe --port COM7                # passive: ping servos, read state
    python -m hfteama.arm_check read    --arm rakesh_follower    # read only (arm holds its pose)
    python -m hfteama.arm_check joint   --arm A --joint shoulder_pan --delta 0.1
    python -m hfteama.arm_check point   --arm A --xyz 0.2 0.0 0.12
    python -m hfteama.arm_check descend --arm A --xy 0.2 0.0     # 0.10, 0.05, 0.03 m, measure each
    python -m hfteama.arm_check step    --arm A --object block_01
    python -m hfteama.arm_check park    --arm A                  # lower slowly, then torque off

Every command that moves the arm asks for the typed word `yes` BEFORE it connects (connecting
enables torque: the arm holds the pose it is in, so support it). All motion is slow (speed 0.3 on
top of the config's `speed_scale`). Exit code 0 ok, 1 refused or failed, 2 not confirmed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from hfteama.arms.ik import ArmBase, Kinematics
from hfteama.arms.kinematics import JOINTS, LIMITS_RAD
from hfteama.arms.poses import Poses
from hfteama.arms.primitives import ArmDown, MotionError, check_joints, clip_to_limits

ROOT = Path(__file__).resolve().parents[2]
SLOW = 0.3
STAGES_M = (0.10, 0.05, 0.03)
WARNING = """\
This will CONNECT the arm and may MOVE it.
  - Connecting enables torque: the arm holds the pose it is in. Support a raised arm first.
  - Clear the table and keep a hand on the power. Do not kill the process while the arm moves.
"""


# --------------------------------------------------------------------------- config check (no arm)


def check_config(
    config_path: str | Path, poses_path: str | Path, pairs_path: str | Path | None = None
) -> list[str]:
    """Everything that can be checked without an arm. Returns a list of problems (empty = fine)."""
    from hfteama.discovery import followers_from_pairs

    problems: list[str] = []
    arms: dict[str, Any] = json.loads(Path(config_path).read_text()).get("arms", {})
    if not arms:
        return ["no arms in the config"]
    poses = Poses.load(poses_path)
    followers = set(followers_from_pairs(pairs_path).values()) if pairs_path else None
    reached: set[str] = set()
    for arm_id, a in arms.items():
        tag = f"{arm_id}:"
        if not a.get("pair"):
            problems.append(f"{tag} pair is missing (a name in teleop_pairs.json)")
        if followers is not None and arm_id not in followers:
            problems.append(
                f"{tag} id is not a follower id in teleop_pairs.json {sorted(followers)}"
            )
        pct = a.get("grip_holding_min_pct")
        if not isinstance(pct, int | float) or not 0 < pct < 100:
            problems.append(f"{tag} grip_holding_min_pct must be a number in (0, 100), got {pct!r}")
        base = a.get("base")
        if not (isinstance(base, list) and len(base) == 3):
            problems.append(f"{tag} base must be [x, y, z] metres in the table frame")
            continue
        kin = Kinematics(ArmBase(tuple(float(v) for v in base), float(a.get("yaw", 0.0))))
        table_z = float(a.get("table_z", 0.0))
        park = a.get("park_q")
        if not (isinstance(park, list) and len(park) == 5):
            problems.append(f"{tag} park_q must be 5 joint angles (rad)")
        else:
            try:
                check_joints(np.asarray(park, float))
            except MotionError as exc:
                problems.append(f"{tag} park_q out of limits ({exc.reason})")
            else:
                if kin.fk(np.asarray(park, float))[0][2] < table_z + 0.005:
                    problems.append(f"{tag} the park pose puts the tool centre under the table")
        found, can_reach = _reach_problems(arm_id, kin, poses, table_z)
        problems += found
        reached |= can_reach
    problems += [
        f"no arm can reach {name}: move it into the reach of an arm (or it will never be picked)"
        for name in poses.objects
        if name not in reached
    ]
    return problems


def _reach_problems(
    arm_id: str, kin: Kinematics, poses: Poses, table_z: float
) -> tuple[list[str], set[str]]:
    """What this arm cannot do (bowl, carry pose) and which objects it can pick.

    An object only has to be reachable by one arm, so unreachable objects are reported by the
    caller once every arm has been looked at.
    """
    from hfteama.arms.generic_skills import APPROACH_DZ, GRASP_DZ, LIFT_DZ

    if not poses.has("own_output_bin", arm_id):
        return [
            f"{arm_id}: no entry for arm '{arm_id}' in the poses file (per_arm.{arm_id})"
        ], set()

    def ok(p: np.ndarray, dzs: tuple[float, ...]) -> bool:
        return all(
            p[2] + dz >= table_z + 0.01 and kin.solve(p + np.array([0.0, 0.0, dz]), down=True).ok
            for dz in dzs
        )

    out: list[str] = []
    places = [("own_output_bin", poses.point("own_output_bin", arm_id), (APPROACH_DZ, GRASP_DZ))]
    if poses.has("input_bin", arm_id):
        places.append(("input_bin", poses.point("input_bin", arm_id), (APPROACH_DZ, GRASP_DZ)))
    if poses.has("carry", arm_id):
        places.append(("carry", poses.point("carry", arm_id), (0.0,)))
    else:
        out.append(f"{arm_id}: no 'carry' pose: pick would end hovering over the shared area")
    for name, p, dzs in places:
        if not ok(p, dzs):
            out.append(f"{arm_id}: cannot reach {name} from its base")
    can_reach = {
        n for n in poses.objects if ok(poses.point(n, arm_id), (APPROACH_DZ, GRASP_DZ, LIFT_DZ))
    }
    return out, can_reach


# --------------------------------------------------------------------------- passive probe

from hfteama.arms.feetech import probe  # noqa: E402  (re-exported: the tests and the CLI use it)


def _cmd_probe(args: argparse.Namespace) -> int:
    import serial  # pyserial, imported only here

    with serial.Serial(args.port, args.baud, timeout=0.05) as port:
        found = probe(port)
    if not found:
        print(f"no servo answered on {args.port} at {args.baud} baud (powered? right port?)")
        return 1
    names = (*JOINTS,)
    print(f"{args.port}: {len(found)} servo(s) answered (reads only, nothing was changed)")
    print(f"{'id':<4}{'joint':<15}{'raw':>6}{'torque':>8}{'volts':>7}{'temp C':>8}")
    for sid, v in sorted(found.items()):
        joint = names[sid - 1] if 1 <= sid <= len(names) else "?"
        torque = {True: "ON", False: "off", None: "?"}[v["torque"]]
        print(f"{sid:<4}{joint:<15}{v['position']:>6}{torque:>8}{v['volts']:>7}{v['temp_c']:>8}")
    if len(found) != 6:
        print("expected 6 servos (5 joints + gripper): check the cable chain and power")
        return 1
    return 0


# --------------------------------------------------------------------------- staged commands


def _fmt(v: np.ndarray) -> str:
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def _cmd_read(d: Any, args: argparse.Namespace) -> int:
    q = d.read_joints()
    print(f"{'joint':<14}{'rad':>8}{'deg':>9}   limits (rad)         margin")
    for name, v in zip(JOINTS[:5], q[:5], strict=True):
        lo, hi = LIMITS_RAD[name]
        margin = min(v - lo, hi - v)
        warn = "  <-- OUTSIDE the model limit" if margin < 0 else ""
        warn = warn or ("  <-- near a limit" if margin < 0.1 else "")
        print(
            f"{name:<14}{v:>+8.3f}{math.degrees(v):>+9.1f}"
            f"   [{lo:+.2f}, {hi:+.2f}]{margin:>9.2f}{warn}"
        )
    print(f"{'gripper':<14}{d.gripper_pct():>8.1f} %   (0 = closed, 100 = open)")
    pos, z_axis = d.tcp()
    print(f"TCP (table frame, m): {_fmt(pos)}   tool points down: {z_axis[2] < -0.5}")
    print(f"alive: {d.alive():.0f} (bus traffic just now)")
    print("Compare with what you SEE. Move it by hand only after supporting it and cutting torque.")
    return 0


def _cmd_joint(d: Any, args: argparse.Namespace) -> int:
    idx = JOINTS[:5].index(args.joint)
    q0 = clip_to_limits(d.read_joints())[:5]  # an arm resting on a stop is past the model limit
    q1 = q0.copy()
    q1[idx] += args.delta
    d.move_joints(q1, speed=SLOW)
    print(
        f"{args.joint}: commanded {args.delta:+.3f} rad, "
        f"measured {d.read_joints()[idx] - q0[idx]:+.3f}"
    )
    d.move_joints(q0, speed=SLOW)
    print(f"back: residual {d.read_joints()[idx] - q0[idx]:+.3f} rad")
    return 0


def _cmd_point(d: Any, args: argparse.Namespace) -> int:
    target = np.asarray(args.xyz, float)
    yaw = None if args.yaw is None else math.radians(args.yaw)
    d.move_to(target, speed=SLOW, yaw=yaw)
    pos, _ = d.tcp()
    err_mm = np.linalg.norm(pos - target) * 1000
    print(f"commanded {_fmt(target)}  measured TCP {_fmt(pos)}  error {err_mm:.1f} mm")
    return 0


def _cmd_descend(d: Any, args: argparse.Namespace, ask: Callable[[str], str]) -> int:
    x, y = args.xy
    for z in STAGES_M:
        target = np.array([x, y, z])
        d.move_to(target, speed=SLOW)
        pos, _ = d.tcp()
        print(f"stage z={z:.2f}: commanded {_fmt(target)}  measured {_fmt(pos)}")
        if z == STAGES_M[-1]:
            break
        answer = ask("Measure the real fingertip height. Enter = lower, 'stop' = stop here: ")
        if answer.strip() == "stop":
            print("stopped; the arm holds this pose. Run `park` to lower and release.")
            return 0
    return 0


def _cmd_step(d: Any, args: argparse.Namespace) -> int:
    from hfteama.arms import generic_skills  # noqa: F401  (registers the skills)
    from hfteama.arms.worker import SkillWorker
    from hfteama.taskgraph import Task

    poses = Poses.load(args.poses)
    worker = SkillWorker(args.arm, d, poses, "real")
    steps = (
        ("pick", {"object": args.object, "from": "input_bin"}),
        ("place", {"object": args.object, "to": "own_output_bin"}),
    )
    for skill, step_args in steps:
        r = worker.execute(Task(id=f"check_{skill}", skill=skill, args=step_args))
        print(f"{skill}: {r.outcome.value} {r.reason} ({r.duration:.1f} s)")
        if r.outcome.value != "ok":
            return 1
    return 0


def _cmd_park(d: Any, args: argparse.Namespace) -> int:
    d.park()
    print("parked slowly and torque is off: the arm is limp.")
    return 0


def _default_connect(arm_id: str, path: str) -> Any:
    from hfteama.arms import real

    return real.from_config(arm_id, path)


def main(
    argv: list[str] | None = None,
    ask: Callable[[str], str] = input,
    connect: Callable[[str, str], Any] | None = None,
) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cc = sub.add_parser("check-config", help="validate the config without connecting")
    cc.add_argument("--config", default=str(ROOT / "configs" / "real_arms.json"))
    cc.add_argument("--poses", default=str(ROOT / "tasks" / "sort" / "poses_real.json"))
    cc.add_argument("--pairs", default=str(ROOT / "configs" / "teleop_pairs.json"))
    pr = sub.add_parser("probe", help="passive: ping the servos on a port and read their state")
    pr.add_argument("--port", required=True, help="e.g. COM7 or /dev/ttyACM0")
    pr.add_argument("--baud", type=int, default=1_000_000)
    for name in ("read", "joint", "point", "descend", "step", "park"):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True)
        p.add_argument("--config", default=str(ROOT / "configs" / "real_arms.json"))
        if name == "joint":
            p.add_argument("--joint", required=True, choices=JOINTS[:5])
            p.add_argument("--delta", type=float, default=0.1)
        if name == "point":
            p.add_argument("--xyz", type=float, nargs=3, required=True)
            p.add_argument("--yaw", type=float, default=None, help="jaw-closing heading, degrees")
        if name == "descend":
            p.add_argument("--xy", type=float, nargs=2, required=True)
        if name == "step":
            p.add_argument("--object", required=True)
            p.add_argument("--poses", default=str(ROOT / "tasks" / "sort" / "poses_real.json"))
    args = ap.parse_args(argv)

    if args.cmd == "check-config":
        problems = check_config(args.config, args.poses, args.pairs)
        print(
            "config OK"
            if not problems
            else "\n".join(["problems:", *[f"  - {p}" for p in problems]])
        )
        return 1 if problems else 0

    if args.cmd == "probe":
        return _cmd_probe(args)

    if args.cmd != "read":
        print(WARNING)
        if ask("Type 'yes' to connect the arm: ").strip() != "yes":
            print("not confirmed: nothing was connected.")
            return 2
    driver = (connect or _default_connect)(args.arm, args.config)
    commands = {
        "read": _cmd_read, "joint": _cmd_joint, "point": _cmd_point,
        "step": _cmd_step, "park": _cmd_park,
    }  # fmt: skip
    try:
        if args.cmd == "descend":
            rc = _cmd_descend(driver, args, ask)
        else:
            rc = commands[args.cmd](driver, args)
    except (MotionError, ArmDown) as exc:
        print(f"refused / failed: {exc.reason}. Nothing further was commanded.")
        return 1
    if args.cmd not in ("read", "park"):
        print("the arm still holds its pose (torque on). Run `park` to lower it and release.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
