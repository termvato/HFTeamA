"""Thin wrapper: `python scripts/run_real.py ...` is `python -m hfteama.runner ...`."""

from hfteama.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
