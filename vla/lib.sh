# Helpers shared by the vla/ scripts. Settings live in config.sh.
VLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then  # before set -u: conda's activate trips on it
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate lerobot
fi
set -euo pipefail
source "$VLA_DIR/config.sh"

CALIB_DIR="${HF_LEROBOT_CALIBRATION:-$HOME/.cache/huggingface/lerobot/calibration}"

need_port() {  # fail early with a useful hint instead of a LeRobot stack trace
    [[ "${SKIP_CHECKS:-0}" == 1 ]] && return 0
    for p in "$@"; do
        [[ -e "$p" ]] || { echo "missing $p: plugged in? udev rules installed? (python vla/detect_arms.py list)" >&2; exit 1; }
    done
}

need_calib() {  # kind=robots/so_follower or teleoperators/so_leader, then ids
    local kind=$1; shift
    [[ "${SKIP_CHECKS:-0}" == 1 ]] && return 0
    for id in "$@"; do
        [[ -e "$CALIB_DIR/$kind/$id.json" ]] || {
            echo "no calibration $CALIB_DIR/$kind/$id.json: run ./vla/calibrate.sh <arm> for each arm" >&2; exit 1; }
    done
}

# cameras_for single|dual -> CAMERAS, the --robot.cameras value, from CAMERA_TABLE in config.sh.
# Every camera is a top-level robot camera, also in dual mode: per-arm cameras would be renamed
# left_*/right_*, and the same camera must have the same name in single and dual data.
cameras_for() {
    local mode=$1 entries=() names=" " devices=" " name arms rot dev extra
    CAMERA_ARMS=()  # name:arms for every configured camera, both modes (the studio's fixed slots)
    while read -r name arms rot dev extra; do
        [[ -z "$name" || "$name" == \#* ]] && continue
        [[ -n "$dev" && -z "$extra" ]] || { echo "config.sh CAMERA_TABLE: bad line: $name $arms $rot $dev $extra" >&2; exit 1; }
        [[ "$name" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "camera name '$name': use lower case, digits and _" >&2; exit 1; }
        [[ "$arms" =~ ^(both|a|b)$ ]] || { echo "camera $name: arms must be both, a or b (got '$arms')" >&2; exit 1; }
        [[ "$rot" =~ ^(0|90|180|-90)$ ]] || { echo "camera $name: rot must be 0, 90, 180 or -90" >&2; exit 1; }
        [[ "$names" != *" $name "* ]] || { echo "camera name '$name' is used twice" >&2; exit 1; }
        [[ "$devices" != *" $dev "* ]] || { echo "camera device $dev is listed twice" >&2; exit 1; }
        names+="$name "; devices+="$dev "
        CAMERA_ARMS+=("$name:$arms")
        [[ $mode == single && $arms == b ]] && continue
        local w=$CAM_W h=$CAM_H  # LeRobot's width/height are after rotation: a sideways camera is portrait
        [[ $rot == 90 || $rot == -90 ]] && { w=$CAM_H; h=$CAM_W; }
        entries+=("$name: {type: opencv, index_or_path: $dev, width: $w, height: $h, fps: $CAM_FPS, fourcc: $CAM_FOURCC, rotation: $rot}")
    done <<<"$CAMERA_TABLE"
    (( ${#entries[@]} )) || { echo "no cameras for $mode mode in CAMERA_TABLE (config.sh)" >&2; exit 1; }
    local IFS=,
    CAMERAS="{ ${entries[*]} }"
}

# robot_args single|dual -> lerobot CLI flags for the follower side (cameras included)
robot_args() {
    cameras_for "$1"
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
