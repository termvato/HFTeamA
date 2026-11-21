"""Stage 1 of docs/handoff/action-to-hardware.md on one real SO-101: read only, then the gripper.

No joint is commanded. The only commanded motion is the gripper, in the gripper-threshold test.

    cd /data/diego/Projects/HFTeam-action
    source ../HFTeam/.venv/bin/activate
    export PYTHONPATH=$PWD/src
    python scripts/real_stage1.py --arm rakesh_follower      # or vato_follower

Menu:
  r  read the joints once
  t  torque off + live readout (move the arm by hand)    -> arm goes limp: SUPPORT IT
  p  save the current pose as park_q (torque off, arm folded by hand)
  o  measure an offset: put one joint at a known angle, type it in
  g  gripper threshold: 10 empty closes, 10 on the block  (needs torque on: reconnect after t)
  c  reconnect                                            -> torque goes off then on: SUPPORT IT
  w  write the results into configs/real_arms.json
  q  quit (disconnects: torque off, SUPPORT THE ARM)
"""

import argparse
import builtins
import json
import select
import sys
import time
from pathlib import Path

import numpy as np

from hfteama.arms.kinematics import BODY, DEFAULT_OFFSETS_DEG, JOINTS, LIMITS_RAD
from hfteama.arms.real import from_config

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "configs" / "real_arms.json"
EXAMPLE = ROOT / "configs" / "real_arms.example.json"
PAIRS = ROOT / "configs" / "teleop_pairs.json"
SHORT = dict(zip(JOINTS, ("pan", "lift", "elbow", "w_flex", "w_roll", "grip")))


def ask(msg: str) -> bool:
    return input(f"{msg} [y/N] ").strip().lower() == "y"


def ensure_entry(arm: str) -> None:
    """Create configs/real_arms.json / the arm's entry from the template if missing."""
    cfg = json.loads((REAL if REAL.exists() else EXAMPLE).read_text())
    if arm in cfg["arms"]:
        if not REAL.exists():
            REAL.write_text(json.dumps(cfg, indent=2) + "\n")
        return
    pairs = json.loads(PAIRS.read_text())["pairs"]
    pair = next((k for k, v in pairs.items() if v["follower"]["id"] == arm), None)
    if pair is None:
        raise SystemExit(f"{arm} is no follower in {PAIRS}")
    template = json.loads(EXAMPLE.read_text())["arms"]["rakesh_follower"]
    cfg["arms"][arm] = {**template, "pair": pair, "offsets_deg": {}}
    REAL.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"added {arm} (pair {pair}) to {REAL} from the template")


class Session:
    def __init__(self, arm: str):
        self.arm = arm
        self.d = None
        self.torque_off = False
        self.results: dict = {}

    def connect(self) -> None:
        input("SUPPORT THE ARM (torque comes on and holds the current pose), then Enter ")
        self.d = from_config(self.arm, REAL)
        self.torque_off = False
        print("connected")

    def disconnect(self) -> None:
        if self.d is None:
            return
        input("SUPPORT THE ARM (disconnect turns torque off), then Enter ")
        try:
            self.d._robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"disconnect: {exc}")
        self.d = None

    def obs(self) -> tuple[np.ndarray, dict]:
        raw = self.d._observe()
        return self.d._frame.to_kinematic(raw), raw

    def show(self) -> None:
        q, raw = self.obs()
        model = "  ".join(f"{SHORT[j]} {np.degrees(v):7.1f}" for j, v in zip(JOINTS, q))
        lerobot = "  ".join(f"{raw[f'{j}.pos']:6.1f}" for j in JOINTS)
        print(f"model deg: {model}   | lerobot: {lerobot}")

    def live(self) -> None:
        if not self.torque_off:
            input("SUPPORT THE ARM: torque goes off and a raised arm drops. Enter ")
            with self.d._bus_lock:
                self.d._robot.bus.disable_torque()
            self.torque_off = True
        print(">>> press ENTER (or Ctrl-C) to stop the readout and get the menu back <<<")
        print("check: sign and size of each joint, gripper value goes UP when opening")
        try:
            while True:
                self.show()
                ready, _, _ = select.select([sys.stdin], [], [], 0.4)
                if ready:
                    sys.stdin.readline()
                    break
        except KeyboardInterrupt:
            print()
        print("readout stopped (torque still OFF)")

    def park(self) -> None:
        q = self.obs()[0][:5]
        bad = [j for j, v in zip(BODY, q) if not LIMITS_RAD[j][0] <= v <= LIMITS_RAD[j][1]]
        if bad:
            print(f"outside the joint limits: {bad}; move it and try again")
            return
        self.results["park_q"] = [round(float(v), 3) for v in q]
        print(f"park_q = {self.results['park_q']}")

    def offset(self) -> None:
        print("joints: " + ", ".join(BODY))
        j = input("joint: ").strip()
        if j not in BODY:
            print("unknown joint")
            return
        true = float(input(f"true angle of {j} in the MODEL convention (deg, e.g. 90): "))
        n = 10
        vals = []
        for _ in range(n):
            vals.append(self.obs()[1][f"{j}.pos"])
            time.sleep(0.05)
        raw = float(np.mean(vals))
        sign = self.d._frame.signs[j]
        off = raw - sign * true  # deg_lerobot = sign * deg_kinematic + offset
        cur = self.d._frame.offsets_deg.get(j, 0.0)
        print(f"lerobot {raw:.1f}, sign {sign:+d} -> offset {off:.1f} deg (current {cur:.1f})")
        if ask("keep it?"):
            self.results.setdefault("offsets_deg", {})[j] = round(off, 1)

    def gripper(self) -> None:
        if self.torque_off:
            print("torque is off: reconnect first ('c')")
            return
        if not ask("the gripper will open and close 20 times. Fingers clear?"):
            return

        def close_read() -> float:
            self.d.close_gripper()
            time.sleep(0.3)
            return float(self.d._gripper_pct())

        empty, block = [], []
        for i in range(10):
            self.d.open_gripper()
            empty.append(close_read())
            print(f"empty {i + 1}/10: {empty[-1]:.1f}")
        for i in range(10):
            self.d.open_gripper()
            input(f"block {i + 1}/10: put the block between the fingers, Enter ")
            block.append(close_read())
            print(f"block {i + 1}/10: {block[-1]:.1f}")
        self.d.open_gripper()
        hi_empty, lo_block = max(empty), min(block)
        print(f"empty max {hi_empty:.1f}, block min {lo_block:.1f}")
        if lo_block <= hi_empty:
            print("the groups overlap: the gripper cannot tell holding from empty. Not saved.")
            return
        thr = round((hi_empty + lo_block) / 2, 1)
        print(f"grip_holding_min_pct = {thr}")
        self.results["grip_holding_min_pct"] = thr

    def write(self) -> None:
        if not self.results:
            print("nothing measured yet")
            return
        cfg = json.loads(REAL.read_text())
        entry = cfg["arms"][self.arm]
        print(json.dumps(self.results, indent=1))
        if not ask(f"write into {REAL} [{self.arm}]?"):
            return
        for k, v in self.results.items():
            if k == "offsets_deg":
                entry["offsets_deg"] = {**entry.get("offsets_deg", {}), **v}
            else:
                entry[k] = v
        REAL.write_text(json.dumps(cfg, indent=2) + "\n")
        log = ROOT / "runs" / "stage1" / f"{self.arm}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(self.results, indent=1) + "\n")
        print(f"written; copy in {log}")
        self.results = {}


class Tee:
    """Everything printed (and every answer typed) also goes to the log file."""

    def __init__(self, path: Path):
        self.f = path.open("a", buffering=1)
        self.out = sys.stdout

    def write(self, s: str) -> None:
        self.out.write(s)
        self.f.write(s)

    def flush(self) -> None:
        self.out.flush()
        self.f.flush()


def start_log(arm: str) -> Path:
    log = ROOT / "runs" / "stage1" / f"{arm}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    tee = Tee(log)
    sys.stdout = tee
    raw_input = builtins.input

    def logged_input(prompt: str = "") -> str:
        tee.write(prompt)
        ans = raw_input()
        tee.f.write(ans + "\n")
        return ans

    builtins.input = logged_input
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="rakesh_follower", help="rakesh_follower or vato_follower")
    arm = ap.parse_args().arm
    log = start_log(arm)
    print(f"log: {log}")
    ensure_entry(arm)
    print(f"default offsets (deg): {DEFAULT_OFFSETS_DEG}")
    s = Session(arm)
    s.connect()
    s.show()
    actions = {"r": s.show, "t": s.live, "p": s.park, "o": s.offset, "g": s.gripper, "w": s.write}
    try:
        while True:
            c = input("\n[r]ead [t]orque-off live [p]ark_q [o]ffset [g]ripper re[c]onnect [w]rite [q]uit > ").strip()
            if c == "q":
                break
            if c == "c":
                s.disconnect()
                s.connect()
            elif c in actions:
                try:
                    actions[c]()
                except Exception as exc:  # noqa: BLE001
                    print(f"error: {type(exc).__name__}: {exc}")
    finally:
        if s.results and ask("results not written yet. Write them now?"):
            s.write()
        s.disconnect()


if __name__ == "__main__":
    main()
