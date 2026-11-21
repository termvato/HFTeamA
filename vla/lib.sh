# Helpers shared by the vla/ scripts. Settings live in config.sh.
VLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then  # before set -u: conda's activate trips on it
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate lerobot
fi
set -euo pipefail
source "$VLA_DIR/config.sh"

CALIB_DIR="${HF_LEROBOT_CALIBRATION:-$HOME/.cache/huggingface/lerobot/calibration}"

need_port() {  # fail early with a useful hint instead of a LeRobot stack trace
    for p in "$@"; do
        [[ -e "$p" ]] || { echo "missing $p: plugged in? udev rules installed? (python vla/detect_arms.py list)" >&2; exit 1; }
    done
}

need_calib() {  # kind=robots/so_follower or teleoperators/so_leader, then ids
    local kind=$1; shift
    for id in "$@"; do
        [[ -e "$CALIB_DIR/$kind/$id.json" ]] || {
            echo "no calibration $CALIB_DIR/$kind/$id.json: run ./vla/calibrate.sh <arm> for each arm" >&2; exit 1; }
    done
}

# robot_args single|dual -> lerobot CLI flags for the follower side (cameras included)
robot_args() {
    if [[ $1 == single ]]; then
        need_port "$FOLLOWER_A_PORT"; need_calib robots/so_follower follower_a
        ROBOT_ARGS=(--robot.type=so101_follower --robot.id=follower_a --robot.port="$FOLLOWER_A_PORT"
                    --robot.max_relative_target="$MAX_RELATIVE_TARGET" --robot.cameras="$CAMERAS")
    else
        need_port "$FOLLOWER_A_PORT" "$FOLLOWER_B_PORT"
        need_calib robots/so_follower "${BI_FOLLOWER_ID}_left" "${BI_FOLLOWER_ID}_right"
        ROBOT_ARGS=(--robot.type=bi_so_follower --robot.id="$BI_FOLLOWER_ID"
                    --robot.left_arm_config.port="$FOLLOWER_A_PORT"
                    --robot.right_arm_config.port="$FOLLOWER_B_PORT"
                    --robot.left_arm_config.max_relative_target="$MAX_RELATIVE_TARGET"
                    --robot.right_arm_config.max_relative_target="$MAX_RELATIVE_TARGET"
                    --robot.cameras="$CAMERAS")
    fi
}

# teleop_args single|dual -> lerobot CLI flags for the leader side
teleop_args() {
    if [[ $1 == single ]]; then
        need_port "$LEADER_A_PORT"; need_calib teleoperators/so_leader leader_a
        TELEOP_ARGS=(--teleop.type=so101_leader --teleop.id=leader_a --teleop.port="$LEADER_A_PORT")
    else
        need_port "$LEADER_A_PORT" "$LEADER_B_PORT"
        need_calib teleoperators/so_leader "${BI_LEADER_ID}_left" "${BI_LEADER_ID}_right"
        TELEOP_ARGS=(--teleop.type=bi_so_leader --teleop.id="$BI_LEADER_ID"
                     --teleop.left_arm_config.port="$LEADER_A_PORT"
                     --teleop.right_arm_config.port="$LEADER_B_PORT")
    fi
}

mode_arg() {
    case "${1:-}" in single|dual) echo "$1" ;; *) echo "usage: $0 single|dual" >&2; exit 2 ;; esac
}
