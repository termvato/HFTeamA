"""Find SO-101 arms on USB, give each a fixed name, and notice when one is plugged in or out.

USB serial numbers identify the boards, so a name survives replugging and /dev/ttyACM* reordering.
Names live in vla/arms.json (git-ignored, one per rig; see arms.example.json).

    python vla/detect_arms.py list                 # every serial port, its USB serial, its name if known
    python vla/detect_arms.py add follower_a       # unplug -> replug the arm when asked; saves its serial
    python vla/detect_arms.py probe follower_a     # check motors 1-6 answer (needs LeRobot; arm idle)
    python vla/detect_arms.py udev                 # print udev rules: /dev/so101/<name> for every arm
    python vla/detect_arms.py watch                # JSON lines: arm_joined / arm_left, until Ctrl+C

Listing and watching only read sysfs; they never open a port, so they are safe next to a running arm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REGISTRY = Path(__file__).with_name("arms.json")
ROLES = ("follower", "leader")
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


# ---------------------------------------------------------------- USB serial ports


def usb_info(tty: str) -> dict:
    """USB vendor, product and serial of /dev/<tty>, read from sysfs (empty if not a USB device)."""
    node = Path(f"/sys/class/tty/{tty}/device").resolve()
    for d in (node, *node.parents):
        if (d / "idVendor").exists():
            read = lambda f: (d / f).read_text().strip() if (d / f).exists() else ""  # noqa: E731
            return {
                "vid": read("idVendor"),
                "pid": read("idProduct"),
                "serial": read("serial"),
                "product": read("product"),
            }
    return {}


def ports() -> dict[str, dict]:
    """{"/dev/ttyACM0": {vid, pid, serial, product}} for every USB serial port present now."""
    found = {}
    for pattern in ("ttyACM*", "ttyUSB*"):
        for dev in sorted(Path("/dev").glob(pattern)):
            info = usb_info(dev.name)
            if info:
                found[str(dev)] = info
    return found


# ---------------------------------------------------------------- registry


def load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"arms": []}


def save(reg: dict) -> None:
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"saved {REGISTRY}")


def by_serial(reg: dict) -> dict[str, dict]:
    return {a["serial"]: a for a in reg["arms"]}


def port_of(arm_id: str) -> str | None:
    """Current /dev/tty* of a named arm, or None if it is not plugged in."""
    arm = next((a for a in load()["arms"] if a["id"] == arm_id), None)
    if arm is None:
        sys.exit(f"unknown arm {arm_id!r}; known: {[a['id'] for a in load()['arms']]}")
    return next((p for p, i in ports().items() if i["serial"] == arm["serial"]), None)


# ---------------------------------------------------------------- commands


def cmd_list(_: argparse.Namespace) -> None:
    known = by_serial(load())
    now = ports()
    if not now:
        print("no USB serial ports. Is the arm powered and plugged in? (ls /dev/ttyACM*)")
    for port, info in now.items():
        arm = known.get(info["serial"])
        name = f"{arm['id']} ({arm['role']})" if arm else "UNKNOWN -> python vla/detect_arms.py add <name>"
        print(f"{port:14} serial={info['serial'] or '?':14} {info['vid']}:{info['pid']}  {name}")
    present = {i["serial"] for i in now.values()}
    for serial, arm in known.items():
        if serial not in present:
            print(f"{'-':14} serial={serial:14} {arm['id']} ({arm['role']}) NOT PLUGGED IN")


def cmd_add(a: argparse.Namespace) -> None:
    role = a.role or next((r for r in ROLES if a.id.startswith(r)), None)
    if role not in ROLES:
        sys.exit(f"cannot tell the role from {a.id!r}; pass --role follower|leader")
    input(f"Plug in every arm except {a.id}, or unplug just {a.id}. Press Enter when done.")
    before = ports()
    input(f"Now plug in {a.id} (USB + power). Press Enter when done.")
    time.sleep(1.0)
    new = {p: i for p, i in ports().items() if p not in before}
    if len(new) != 1:
        sys.exit(f"expected exactly one new port, saw {list(new) or 'none'}; try again")
    port, info = next(iter(new.items()))
    if not info["serial"]:
        sys.exit(f"{port} reports no USB serial number, so it cannot be told apart after a replug")
    reg = load()
    reg["arms"] = [x for x in reg["arms"] if x["id"] != a.id and x["serial"] != info["serial"]]
    reg["arms"].append({"id": a.id, "role": role, "serial": info["serial"], "vid": info["vid"],
                        "pid": info["pid"]})
    reg["arms"].sort(key=lambda x: x["id"])
    print(f"{a.id}: {port} serial={info['serial']}")
    save(reg)


def cmd_probe(a: argparse.Namespace) -> None:
    """Read each motor's model number. Opens the port, so nothing else may be using it."""
    port = port_of(a.id)
    if port is None:
        sys.exit(f"{a.id} is not plugged in")
    from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

    motors = {m: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, m in enumerate(MOTORS, 1)}
    bus = SerialMotorsBus(port=port, motors=motors)
    bus.connect(handshake=False)
    try:
        answers = {m: bus.ping(m, num_retry=2) for m in MOTORS}
    finally:
        bus.disconnect(disable_torque=False)  # probing must not change what the arm is doing
    for m, model in answers.items():
        print(f"  {m:14} id={motors[m].id}  {'model ' + str(model) if model is not None else 'NO ANSWER'}")
    missing = [m for m, v in answers.items() if v is None]
    if missing:
        sys.exit(f"{a.id} on {port}: no answer from {missing}. Check power, daisy chain, motor ids.")
    print(f"{a.id} on {port}: all 6 motors answer")


def cmd_udev(_: argparse.Namespace) -> None:
    reg = load()
    if not reg["arms"]:
        sys.exit("no arms registered yet; run: python vla/detect_arms.py add follower_a")
    print("# /etc/udev/rules.d/99-so101.rules  (generated by vla/detect_arms.py udev)")
    for arm in reg["arms"]:
        print(f'SUBSYSTEM=="tty", ATTRS{{idVendor}}=="{arm["vid"]}", ATTRS{{idProduct}}=="{arm["pid"]}", '
              f'ATTRS{{serial}}=="{arm["serial"]}", SYMLINK+="so101/{arm["id"]}", MODE="0666"')


def cmd_watch(a: argparse.Namespace) -> None:
    """Print one JSON line per change; the orchestrator's arm registry can consume the same events."""
    known = by_serial(load())
    seen: dict[str, str] = {}  # serial -> port
    while True:
        now = {i["serial"]: p for p, i in ports().items() if i["serial"]}
        events = [("arm_joined", s, p) for s, p in now.items() if s not in seen]
        events += [("arm_left", s, p) for s, p in seen.items() if s not in now]
        for kind, serial, port in events:
            arm = known.get(serial, {})
            print(json.dumps({"t": round(time.time(), 3), "type": kind, "arm_id": arm.get("id"),
                              "role": arm.get("role"), "port": port, "serial": serial}), flush=True)
        seen = now
        time.sleep(a.period)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    s = sub.add_parser("add")
    s.add_argument("id", help="e.g. follower_a, leader_a, follower_b, leader_b")
    s.add_argument("--role", choices=ROLES)
    s.set_defaults(fn=cmd_add)
    s = sub.add_parser("probe")
    s.add_argument("id")
    s.set_defaults(fn=cmd_probe)
    sub.add_parser("udev").set_defaults(fn=cmd_udev)
    s = sub.add_parser("watch")
    s.add_argument("--period", type=float, default=0.5)
    s.set_defaults(fn=cmd_watch)
    args = p.parse_args()
    try:
        args.fn(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
