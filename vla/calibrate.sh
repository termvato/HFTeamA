#!/usr/bin/env bash
# Calibrate one arm by name, then link its file under the bimanual names so dual mode reuses it.
#   ./vla/calibrate.sh follower_a | leader_a | follower_b | leader_b
# LeRobot walks you through it: put the arm in the middle of its range, press Enter, then move
# every joint through its full range and press Enter again.
source "$(dirname "$0")/lib.sh"
ID="${1:?usage: $0 follower_a|leader_a|follower_b|leader_b}"
PORT_VAR="$(echo "${ID}" | tr a-z A-Z)_PORT"; PORT="${!PORT_VAR:?unknown arm $ID}"
need_port "$PORT"

case "$ID" in
    follower_*) lerobot-calibrate --robot.type=so101_follower --robot.port="$PORT" --robot.id="$ID"
                KIND=robots/so_follower; BI="$BI_FOLLOWER_ID" ;;
    leader_*)   lerobot-calibrate --teleop.type=so101_leader --teleop.port="$PORT" --teleop.id="$ID"
                KIND=teleoperators/so_leader; BI="$BI_LEADER_ID" ;;
esac

# A is left, B is right in bi_so_follower / bi_so_leader (config.sh)
SIDE=$([[ "$ID" == *_a ]] && echo left || echo right)
ln -sf "$ID.json" "$CALIB_DIR/$KIND/${BI}_${SIDE}.json"
echo "calibration: $CALIB_DIR/$KIND/$ID.json (also used as ${BI}_${SIDE})"
