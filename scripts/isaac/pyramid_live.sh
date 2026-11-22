#!/usr/bin/env bash
# The pyramid demo live in an Isaac Sim window: MuJoCo + the team's Executor run it, Isaac draws it.
#   scripts/isaac/pyramid_live.sh                    # arm 2 joins after 3 boxes, arm 3 after 7
#   scripts/isaac/pyramid_live.sh --join-after 2,6   # any pyramid_demo.py option
#   VIEW=close scripts/isaac/pyramid_live.sh         # camera: front (default), close, side, top
#   ORBIT=0 scripts/isaac/pyramid_live.sh            # still camera (default: circles at 1.5 deg/s)
set -euo pipefail
cd "$(dirname "$0")/../.."
ISAAC="${ISAAC:-$HOME/isaac-sim}"; PORT="${PORT:-9871}"; PY="${PY:-.venv/bin/python}"
LOG="$(mktemp -t isaac_mirror.XXXX.log)"
env -u PYTHONPATH "$ISAAC/python.sh" scripts/isaac/mirror.py --port "$PORT" --view "${VIEW:-front}" \
    --orbit "${ORBIT:-1.5}" \
    > "$LOG" 2>&1 &
VIEWER=$!
echo "starting Isaac Sim (log: $LOG); about a minute"
until grep -q "listening on" "$LOG"; do
    kill -0 "$VIEWER" 2>/dev/null || { echo "Isaac Sim stopped:"; tail -20 "$LOG"; exit 1; }
    sleep 2
done
sleep 5  # let the window settle
"$PY" scripts/isaac/pyramid_demo.py --isaac "$PORT" "$@"
echo "done; the Isaac Sim window keeps the finished pyramid. Close it to quit."
wait "$VIEWER"
