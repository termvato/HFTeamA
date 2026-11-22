#!/usr/bin/env bash
# Vivek's MuJoCo 3-arm demo, shown live in Isaac Sim. MuJoCo runs it; Isaac Sim draws it.
#   scripts/isaac/run.sh                         # sort12 on 3 arms, Isaac Sim window
#   scripts/isaac/run.sh --task tasks/assemble/assemble3.json   # any sim_demo.py option
#   RECORD=/tmp/frames scripts/isaac/run.sh      # headless, PNG frames instead of a window
# ISAAC (default ~/isaac-sim) is the Isaac Sim folder, PORT (default 9871) the UDP port.
set -euo pipefail
cd "$(dirname "$0")/../.."
ISAAC="${ISAAC:-$HOME/isaac-sim}"; PORT="${PORT:-9871}"
PY="${PY:-.venv/bin/python}"
VIEW_ARGS=(--port "$PORT")
[[ -n "${RECORD:-}" ]] && VIEW_ARGS+=(--headless --record "$RECORD" --exit-idle 5)
LOG="$(mktemp -t isaac_mirror.XXXX.log)"
env -u PYTHONPATH "$ISAAC/python.sh" scripts/isaac/mirror.py "${VIEW_ARGS[@]}" > "$LOG" 2>&1 &
VIEWER=$!
echo "starting Isaac Sim (log: $LOG); the first start takes about a minute"
until grep -q "listening on" "$LOG"; do
    kill -0 "$VIEWER" 2>/dev/null || { echo "Isaac Sim stopped:"; tail -20 "$LOG"; exit 1; }
    sleep 2
done
sleep 3  # let the window settle
"$PY" scripts/sim_demo.py --isaac "$PORT" "$@"
echo "sim done; Isaac Sim keeps the final scene${RECORD:+ (frames in $RECORD)}. Close its window to quit."
wait "$VIEWER"
