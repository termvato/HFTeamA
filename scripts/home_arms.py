"""Send the real arms back to their home (rest) pose, slowly, then torque off. Nothing else.

    python scripts/home_arms.py                 # both arms (rakesh, hacklab)
    python scripts/home_arms.py rakesh          # one arm

For after a crash or a killed process: the arm is left wherever it stopped. Connecting enables
torque, so the arm holds its pose first: support it. Then it moves to home at the configured
joint speed and torque goes off at home (never with the arm raised). A human runs this.
"""

from __future__ import annotations

import sys

from hfteama.sorter.geometry import load_config
from hfteama.sorter.robot import ArmDriver


def main(argv: list[str]) -> int:
    cfg = load_config()
    sides = argv or list(cfg["arms"])
    print(f"Homing {', '.join(sides)}. Support each arm: torque comes on, then it moves home.")
    if input("Type 'yes' to connect and home: ").strip() != "yes":
        print("not confirmed: nothing was connected.")
        return 2
    for side in sides:
        drv = ArmDriver(side, cfg["arms"][side], cfg["motion"]).connect()
        try:
            print(f"{side}: going home...")
            drv.go_home()
        finally:
            drv.torque_off()
            print(f"{side}: home, torque off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
