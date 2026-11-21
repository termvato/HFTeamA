#!/usr/bin/env bash
# Local web page for data collection: camera feeds, joints, start/save/discard, upload.
#   ./vla/studio.sh single        # open http://localhost:8765
#   ./vla/studio.sh dual
# Settings come from config.sh, like record.sh. The page opens even with nothing plugged in;
# connection errors show on the page.
SKIP_CHECKS=1  # read later by robot_args/teleop_args, so not a one-command prefix
source "$(dirname "$0")/lib.sh"
MODE=$(mode_arg "${1:-}"); shift
REPO=$([[ $MODE == single ]] && echo "$DATASET_SINGLE" || echo "$DATASET_DUAL")
robot_args "$MODE"; teleop_args "$MODE"
exec python "$VLA_DIR/studio.py" --mode "$MODE" --repo-id "$REPO" --task "$TASK" --fps "$FPS" \
    --episode-time-s "$EPISODE_TIME_S" --reset-time-s "$RESET_TIME_S" --private "$PRIVATE" \
    --views "${CAMERA_VIEWS// /,}" --configured "$(IFS=,; echo "${CAMERA_ARMS[*]}")" \
    --port "${STUDIO_PORT:-8765}" "$@" -- "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}"
