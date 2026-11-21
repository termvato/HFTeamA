# Helpers shared by the vla/ scripts. Settings live in config.sh.
VLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# In the dialout group (/etc/group) but this terminal started before joining: rerun the calling
# script with the group applied, instead of failing on the serial ports. Logging out and back in
# makes this unnecessary.
if [[ -z "${VLA_SG:-}" ]] && ! id -nG | grep -qw dialout && id -nG "$USER" | grep -qw dialout; then
    export VLA_SG=1
    exec sg dialout -c "$(printf '%q ' "$0" "$@")"
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then  # before set -u: conda's activate trips on it
    source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate lerobot
fi
set -euo pipefail
source "$VLA_DIR/config.sh"

FOLLOWER_CALIB="$CALIB_DIR/robots/so_follower"     # passed as --robot.calibration_dir
LEADER_CALIB="$CALIB_DIR/teleoperators/so_leader"  # passed as --teleop.calibration_dir

# port_of <usb serial> -> /dev/ttyACM* of that board now (empty if it is not plugged in)
port_of() {
    local tty
    for tty in /sys/class/tty/ttyACM* /sys/class/tty/ttyUSB*; do
        [[ -e "$tty/device" ]] || continue
        [[ "$(cat "$(readlink -f "$tty/device")/../serial" 2>/dev/null)" == "$1" ]] && { echo "/dev/${tty##*/}"; return; }
    done
}

# arm_port <ID var prefix, e.g. FOLLOWER_A> -> the port, or exit with what is missing
arm_port() {
    local id_var="${1}_ID" serial_var="${1}_SERIAL" port
    port="$(port_of "${!serial_var}")"
    if [[ -z "$port" ]]; then
        # studio.py looks "serial:<n>" up again on every connect, so arms can be plugged in later
        [[ "${SKIP_CHECKS:-0}" == 1 ]] && { echo "serial:${!serial_var}"; return; }
        echo "${!id_var} (USB serial ${!serial_var}) is not plugged in. Follower: is its 12 V supply on?" \
             "(python vla/detect_arms.py list)" >&2
        exit 1
    fi
    [[ "${SKIP_CHECKS:-0}" == 1 ]] && { echo "serial:${!serial_var}"; return; }
    if [[ ! -w "$port" ]]; then
        if id -nG "$USER" | grep -qw dialout; then  # added to the group, but this session predates it
            echo "${!id_var} is on $port. You are in the dialout group, but this terminal started" \
                 "before that: run 'newgrp dialout' here (or log out and back in), then try again" >&2
        else
            echo "${!id_var} is on $port, but you may not open it: sudo usermod -aG dialout $USER," \
                 "then log out and back in" >&2
        fi
        exit 1
    fi
    echo "$port"
}

need_calib() {  # dir, then ids
    local dir=$1; shift
    [[ "${SKIP_CHECKS:-0}" == 1 ]] && return 0
    for id in "$@"; do
        [[ -e "$dir/$id.json" ]] || {
            echo "no calibration $dir/$id.json: run ./vla/sync_calibration.sh (never recalibrate these arms)" >&2
            exit 1; }
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
        [[ $mode == single && $arms != both && $arms != "${SINGLE_ARM,,}" ]] && continue  # other arm's wrist
        local w=$CAM_W h=$CAM_H  # LeRobot's width/height are after rotation: a sideways camera is portrait
        [[ $rot == 90 || $rot == -90 ]] && { w=$CAM_H; h=$CAM_W; }
        entries+=("$name: {type: opencv, index_or_path: $dev, width: $w, height: $h, fps: $CAM_FPS, fourcc: $CAM_FOURCC, rotation: $rot, warmup_s: $CAM_WARMUP_S}")
    done <<<"$CAMERA_TABLE"
    (( ${#entries[@]} )) || { echo "no cameras for $mode mode in CAMERA_TABLE (config.sh)" >&2; exit 1; }
    local IFS=,
    CAMERAS="{ ${entries[*]} }"
}

# robot_args single|dual -> lerobot CLI flags for the follower side (cameras included)
robot_args() {
    cameras_for "$1"
    local a b s id_var
    if [[ $1 == single ]]; then
        s="FOLLOWER_$(single_arm)"; id_var="${s}_ID"
        a="$(arm_port "$s")"
        need_calib "$FOLLOWER_CALIB" "${!id_var}"
        ROBOT_ARGS=(--robot.type=so101_follower --robot.id="${!id_var}" --robot.port="$a"
                    --robot.calibration_dir="$FOLLOWER_CALIB"
                    --robot.max_relative_target="$MAX_RELATIVE_TARGET" --robot.cameras="$CAMERAS")
    else
        a="$(arm_port FOLLOWER_A)"
        b="$(arm_port FOLLOWER_B)"
        need_calib "$FOLLOWER_CALIB" "${BI_FOLLOWER_ID}_left" "${BI_FOLLOWER_ID}_right"
        ROBOT_ARGS=(--robot.type=bi_so_follower --robot.id="$BI_FOLLOWER_ID"
                    --robot.calibration_dir="$FOLLOWER_CALIB"
                    --robot.left_arm_config.port="$a"
                    --robot.right_arm_config.port="$b"
                    --robot.left_arm_config.max_relative_target="$MAX_RELATIVE_TARGET"
                    --robot.right_arm_config.max_relative_target="$MAX_RELATIVE_TARGET"
                    --robot.cameras="$CAMERAS")
    fi
}

# teleop_args single|dual -> lerobot CLI flags for the leader side
teleop_args() {
    local a b s id_var
    if [[ $1 == single ]]; then
        s="LEADER_$(single_arm)"; id_var="${s}_ID"
        a="$(arm_port "$s")"
        need_calib "$LEADER_CALIB" "${!id_var}"
        TELEOP_ARGS=(--teleop.type=so101_leader --teleop.id="${!id_var}" --teleop.port="$a"
                     --teleop.calibration_dir="$LEADER_CALIB")
    else
        a="$(arm_port LEADER_A)"
        b="$(arm_port LEADER_B)"
        need_calib "$LEADER_CALIB" "${BI_LEADER_ID}_left" "${BI_LEADER_ID}_right"
        TELEOP_ARGS=(--teleop.type=bi_so_leader --teleop.id="$BI_LEADER_ID"
                     --teleop.calibration_dir="$LEADER_CALIB"
                     --teleop.left_arm_config.port="$a"
                     --teleop.right_arm_config.port="$b")
    fi
}

single_arm() {  # SINGLE_ARM from config.sh -> A or B
    case "${SINGLE_ARM,,}" in a) echo A ;; b) echo B ;; *) echo "SINGLE_ARM must be a or b" >&2; exit 1 ;; esac
}

mode_arg() {
    case "${1:-}" in single|dual) echo "$1" ;; *) echo "usage: $0 single|dual" >&2; exit 2 ;; esac
}
