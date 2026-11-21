#!/usr/bin/env bash
# Drive the follower(s) from the leader(s) with the camera views on screen. Nothing is recorded.
# Use it to check wiring, calibration and camera framing before every recording session.
#   ./vla/teleop.sh single      # leader_a -> follower_a
#   ./vla/teleop.sh dual        # leader_a -> follower_a (left), leader_b -> follower_b (right)
source "$(dirname "$0")/lib.sh"
MODE=$(mode_arg "${1:-}")
robot_args "$MODE"; teleop_args "$MODE"
lerobot-teleoperate "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" --display_data=true
