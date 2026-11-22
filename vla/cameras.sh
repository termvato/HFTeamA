#!/usr/bin/env bash
# Cameras: list the devices, or check the configured ones together (see cameras.py).
#   ./vla/cameras.sh list
#   ./vla/cameras.sh check single|dual
source "$(dirname "$0")/lib.sh"
case "${1:-}" in
    list)  exec python "$VLA_DIR/cameras.py" list --table "$CAMERA_TABLE" --width "$CAM_W" --height "$CAM_H" ;;
    check) MODE=$(mode_arg "${2:-}"); cameras_for "$MODE"
           exec python "$VLA_DIR/cameras.py" check --cameras "$CAMERAS" ;;
    *)     echo "usage: $0 list | check single|dual" >&2; exit 2 ;;
esac
