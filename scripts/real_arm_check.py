"""Thin wrapper: `python scripts/real_arm_check.py ...` is `python -m hfteama.arm_check ...`."""

from hfteama.arm_check import main

if __name__ == "__main__":
    raise SystemExit(main())
