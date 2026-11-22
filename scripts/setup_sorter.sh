#!/usr/bin/env bash
# Set up this machine to run the orchestration on the real arms through the sorter
# (docs/hardware/sorter.md, "With main's orchestration"). Installs into .venv, then checks
# read-only: camera, the followers' USB serials, calibration files. It NEVER connects to or
# moves an arm, and never calibrates.
#
#   bash scripts/setup_sorter.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# LeRobot 0.7 needs Python >= 3.12
newer() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; }
if [ -x .venv/bin/python ] && ! newer .venv/bin/python; then
    echo ".venv has Python < 3.12 (LeRobot 0.7 needs 3.12+): move it away and run this again."
    exit 1
fi
if [ ! -x .venv/bin/python ]; then
    if command -v uv >/dev/null; then
        uv venv --python 3.12 .venv  # uv downloads Python 3.12 if the system has none
    else
        PY=""
        for p in python3.13 python3.12 python3; do
            if command -v "$p" >/dev/null && newer "$p"; then PY=$p; break; fi
        done
        [ -n "$PY" ] || {
            echo "Need Python >= 3.12, or uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        }
        "$PY" -m venv .venv
    fi
fi
if command -v uv >/dev/null; then
    uv pip install --python .venv/bin/python -e ".[sorter,dev]"
else
    .venv/bin/python -m pip install -U pip
    .venv/bin/python -m pip install -e ".[sorter,dev]"
fi

echo
echo "== checks (read-only, no arm is connected)"
fail=0
command -v v4l2-ctl >/dev/null || { echo "MISSING v4l2-ctl: sudo apt install v4l-utils"; fail=1; }
groups | grep -qw dialout || {
    echo "NOT in group dialout (USB serial access): sudo usermod -aG dialout \$USER, then log in again"
    fail=1
}
env -u PYTHONPATH .venv/bin/python - <<'EOF' || fail=1
import json, sys
from pathlib import Path

ok = True
import cv2
if not hasattr(cv2, "aruco"):
    print("MISSING cv2.aruco (OpenCV without the ArUco module)"); ok = False
import lerobot  # noqa: F401
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: F401

from hfteama.discovery import connected_serials
from hfteama.vision.camera import find_device

cfg = json.loads(Path("configs/sorter.json").read_text())
dev = find_device(cfg["camera"]["name"])
print("camera:", cfg["camera"]["name"], "->", dev or "NOT FOUND (plugged in?)")
ok = ok and dev is not None

pairs = json.loads(Path("configs/teleop_pairs.json").read_text())["pairs"]
seen = connected_serials()
for side, arm in cfg["arms"].items():
    f = pairs[arm["pair"]]["follower"]
    cal = Path("configs/calibration/robots/so_follower") / f"{f['id']}.json"
    state = "connected" if f["serial"] in seen else "NOT CONNECTED"
    print(f"arm {side}: follower {f['id']} serial {f['serial']} {state};"
          f" calibration {'ok' if cal.exists() else 'MISSING'}")
    if not cal.exists():
        ok = False  # without the file LeRobot would ask to calibrate: never do that
sys.exit(0 if ok else 1)
EOF
echo
if [ "$fail" = 0 ]; then
    echo "Ready. A human runs the commands that move arms (docs/hardware/sorter.md):"
else
    echo "Fix the lines above first. Then a human runs (docs/hardware/sorter.md):"
fi
echo "  env -u PYTHONPATH .venv/bin/hfteama sort export"
echo "  env -u PYTHONPATH .venv/bin/python -m hfteama.runner --backend sorter \\"
echo "      --arms rakesh_follower,vato_follower \\"
echo "      --task tasks/sorter_boxes/boxes.json --poses tasks/sorter_boxes/poses_real.json"
exit $fail
