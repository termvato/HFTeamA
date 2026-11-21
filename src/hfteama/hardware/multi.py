"""Run several lerobot commands at once (one per pair) from a single terminal.

Each child's output is prefixed with its pair name. LeRobot's per-loop status lines
(loop time, joint table), which redraw in place with cursor moves, are dropped: from two
processes they would overwrite each other.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import threading

from hfteama.hardware.pairs import ArmConfig

# ANSI escape sequences, e.g. the cursor-up LeRobot uses to redraw its status lines.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Per-loop status lines of lerobot-teleoperate / lerobot-record.
_STATUS = re.compile(r"^(Teleop loop time|-{5,}$|NAME\s+\||\S+\.pos\s+\|)")


def check_calibrations(arms: list[ArmConfig]) -> list[str]:
    """IDs of arms whose calibration file is missing or differs from their servos.

    Read-only: connects to each bus without touching torque. With several processes in
    one terminal LeRobot's "Press ENTER to use provided calibration file" prompt would be
    ambiguous, so callers refuse to start instead. Returns [] when LeRobot is not
    installed (the commands themselves would fail anyway).
    """
    try:
        import lerobot.robots.so_follower  # noqa: F401  (registers the arm types)
        import lerobot.teleoperators.so_leader  # noqa: F401
        from lerobot.robots.config import RobotConfig
        from lerobot.robots.utils import make_robot_from_config
        from lerobot.teleoperators.config import TeleoperatorConfig
        from lerobot.teleoperators.utils import make_teleoperator_from_config
    except ImportError:
        return []

    bad = []
    for arm in arms:
        kwargs = dict(port=arm.port, id=arm.id, calibration_dir=arm.calibration_dir)
        if arm.type.endswith("follower"):
            dev = make_robot_from_config(RobotConfig.get_choice_class(arm.type)(**kwargs))
        else:
            dev = make_teleoperator_from_config(
                TeleoperatorConfig.get_choice_class(arm.type)(**kwargs)
            )
        if not dev.calibration:
            bad.append(arm.id)
            continue
        dev.bus.connect(handshake=False)
        try:
            calibrated = dev.bus.is_calibrated
            if not (calibrated() if callable(calibrated) else calibrated):
                bad.append(arm.id)
        finally:
            dev.bus.disconnect(disable_torque=False)
    return bad


def _pump(name: str, stream, out) -> None:
    """Copy a child's output to ours, line by line, prefixed and without status lines."""
    width = 0
    for raw in iter(stream.readline, b""):
        line = _ANSI.sub("", raw.decode(errors="replace")).rstrip("\r\n")
        if not line.strip() or _STATUS.match(line.strip()):
            continue
        width = max(width, len(name))
        out.write(f"[{name:<{width}}] {line}\n")
        out.flush()


def run_parallel(commands: dict[str, list[str]], dry_run: bool) -> int:
    for name, cmd in commands.items():
        print(f"[{name}] $ " + shlex.join(cmd), flush=True)
    if dry_run:
        return 0

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    procs: dict[str, subprocess.Popen] = {}
    threads = []
    for name, cmd in commands.items():
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        procs[name] = p
        t = threading.Thread(target=_pump, args=(name, p.stdout, sys.stdout), daemon=True)
        t.start()
        threads.append(t)

    print("Running. Ctrl+C stops all of them.", flush=True)
    try:
        for p in procs.values():
            p.wait()
    except KeyboardInterrupt:
        # Ctrl+C already reached the children (same process group); give them time to
        # disconnect cleanly (torque off), then make sure they are gone.
        for p in procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in procs.values():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
    for t in threads:
        t.join(timeout=2)

    codes = {name: p.returncode for name, p in procs.items()}
    failed = {n: c for n, c in codes.items() if c not in (0, -signal.SIGINT, 130)}
    for name, code in failed.items():
        print(f"[{name}] exited with code {code}", flush=True)
    return 1 if failed else 0
