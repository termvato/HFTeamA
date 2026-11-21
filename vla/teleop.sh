#!/usr/bin/env bash
# Drive the follower(s) from the leader(s) with the camera views on screen. Nothing is recorded.
# Use it to check wiring, calibration and camera framing before every recording session.
#   ./vla/teleop.sh single      # the SINGLE_ARM pair (config.sh)
#   ./vla/teleop.sh dual        # rakesh pair (left) and hacklab pair (right)
# If LeRobot asks "Press ENTER to use provided calibration file": press ENTER only for rakesh_leader
# (expected once). For any other arm stop with Ctrl+C and ask the Hardware team. Never type c.
source "$(dirname "$0")/lib.sh"
MODE=$(mode_arg "${1:-}")
robot_args "$MODE"; teleop_args "$MODE"
lerobot-teleoperate "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" --display_data=true
