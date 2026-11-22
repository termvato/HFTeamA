"""Stage 2 of docs/handoff/action-to-hardware.md: one joint at a time, small and slow.

Each body joint moves +0.1 rad (about 6 deg) and back, at speed 0.3, after you press Enter.
Then the gripper opens and closes once.

    cd /data/diego/Projects/HFTeam-action
    source ../HFTeam/.venv/bin/activate
    export PYTHONPATH=$PWD/src
    python scripts/real_stage2.py --arm rakesh_follower

Hand at the power. Type s + Enter instead of Enter to skip a joint, q + Enter to stop.
Everything printed is logged to runs/stage2/.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_stage1 import ROOT, SHORT, start_log  # noqa: E402  (same folder; logging helper)

from hfteama.arms.kinematics import BODY, JOINTS  # noqa: E402
from hfteama.arms.primitives import MotionError  # noqa: E402
from hfteama.arms.real import from_config  # noqa: E402

STEP_RAD = 0.1
SPEED = 0.3


def fmt(q: np.ndarray) -> str:
    return "  ".join(f"{SHORT[j]} {np.degrees(v):7.1f}" for j, v in zip(JOINTS, q, strict=False))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="rakesh_follower")
    arm = ap.parse_args().arm
    log = start_log(arm)
    (ROOT / "runs" / "stage2").mkdir(parents=True, exist_ok=True)
    final = ROOT / "runs" / "stage2" / log.name
    print(f"log: {final}")

    input("SUPPORT THE ARM (torque comes on), then Enter ")
    d = from_config(arm, ROOT / "configs" / "real_arms.json")
    try:
        q0 = d.read_joints()
        print(f"start: {fmt(q0)}")
        for i, name in enumerate(BODY):
            ans = input(
                f"\n{name}: +{np.degrees(STEP_RAD):.1f} deg and back. "
                "Enter=go  s=skip  q=stop > "
            )
            if ans.strip() == "q":
                break
            if ans.strip() == "s":
                continue
            # move toward the inside of the range: if +0.1 is refused, try -0.1
            for sign in (+1, -1):
                q = d.read_joints()[:5]
                target = q.copy()
                target[i] += sign * STEP_RAD
                try:
                    t = time.time()
                    d.move_joints(target, speed=SPEED)
                    reached = d.read_joints()
                    dt = time.time() - t
                    step_deg = sign * np.degrees(STEP_RAD)
                    print(f"  moved {step_deg:+.1f} deg in {dt:.1f}s: {fmt(reached)}")
                    err = np.degrees(reached[i] - target[i])
                    print(f"  {name} error vs command: {err:+.1f} deg")
                    # back to start, clipped inside limits (rest pose sits beyond them)
                    lo = np.array([d._limits[j][0] for j in BODY])
                    hi = np.array([d._limits[j][1] for j in BODY])
                    d.move_joints(np.clip(q, lo, hi), speed=SPEED)
                    print(f"  back: {fmt(d.read_joints())}")
                    break
                except MotionError as exc:
                    hint = "; trying the other way" if sign > 0 else ""
                    print(f"  refused/failed ({exc}){hint}")
        if input("\ngripper: open and close once. Enter=go  s=skip > ").strip() != "s":
            d.open_gripper()
            print(f"  open: gripper.pos {d._gripper_pct():.1f}")
            g = d.close_gripper()
            print(f"  closed: gripper.pos {d._gripper_pct():.1f}, holding={g.holding}")
        print("\nstage 2 done")
    except KeyboardInterrupt:
        d.stop()
        print("\nstopped: arm holds its pose")
    finally:
        input("SUPPORT THE ARM (disconnect turns torque off), then Enter ")
        try:
            d._robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"disconnect: {exc}")
        sys.stdout.flush()
        log.replace(final)


if __name__ == "__main__":
    main()
