#!/usr/bin/env bash
# The pyramid demo as an MP4: simulate (MuJoCo, real time), render (Isaac Sim), caption (ffmpeg).
#   scripts/isaac/pyramid_video.sh                    # -> runs/pyramid/pyramid.mp4
#   OUT=runs/p2 JOIN=2,6 scripts/isaac/pyramid_video.sh   # boxes placed before arm 2, arm 3 join
# About 1.5 min of simulation, then ~20 min of rendering on an RTX 3070 laptop.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-runs/pyramid}"; JOIN="${JOIN:-3,7}"; ISAAC="${ISAAC:-$HOME/isaac-sim}"
PY="${PY:-.venv/bin/python}"
# Pillow for the captions: the venv's, or the first conda env that has it
VIDEO_PY="${VIDEO_PY:-}"
for c in "$PY" "$HOME"/miniconda3/envs/*/bin/python; do
    [[ -z "$VIDEO_PY" ]] && "$c" -c "import PIL, numpy" 2>/dev/null && VIDEO_PY="$c"
done
[[ -n "$VIDEO_PY" ]] || { echo "need Pillow: $PY -m pip install pillow"; exit 1; }

"$PY" scripts/isaac/pyramid_demo.py --join-after "$JOIN" --out "$OUT"
rm -rf "$OUT/frames"
env -u PYTHONPATH "$ISAAC/python.sh" scripts/isaac/mirror.py --replay "$OUT/states.bin" \
    --record "$OUT/frames" --subframes 3 --orbit "${ORBIT:-1.5}" > "$OUT/render.log" 2>&1 \
    || { tail -20 "$OUT/render.log"; exit 1; }
"$VIDEO_PY" scripts/isaac/make_video.py "$OUT"
