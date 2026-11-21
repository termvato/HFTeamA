# Settings shared by every vla/ script. Edit here, not in the scripts.
# Sourced by bash; values can also be overridden from the shell: TASK="..." ./vla/record_single.sh

# ---- Hugging Face
if [[ -z "${HF_USER:-}" ]]; then
    _whoami="$(hf auth whoami 2>&1 || true)"  # || true: under set -e a failure would exit silently
    HF_USER="$(sed -n 's/^user=//p' <<<"$_whoami")"
fi
if [[ -z "$HF_USER" ]]; then  # offline or not logged in: refuse rather than record to "/so101_..."
    echo "Could not get your Hugging Face username from 'hf auth whoami' ($(command -v hf)):" >&2
    echo "$_whoami" | head -5 >&2
    echo "Fix: hf auth login   or skip the lookup: export HF_USER=<name>" >&2
    exit 1
fi
DATASET_SINGLE="${DATASET_SINGLE:-${HF_USER}/so101_sort_single}"
DATASET_DUAL="${DATASET_DUAL:-${HF_USER}/so101_sort_dual}"
PRIVATE="${PRIVATE:-true}"          # datasets start private on the Hub; flip when ready to share

# ---- Arms: stable names from the udev rules (python vla/detect_arms.py udev)
FOLLOWER_A_PORT="${FOLLOWER_A_PORT:-/dev/so101/follower_a}"
LEADER_A_PORT="${LEADER_A_PORT:-/dev/so101/leader_a}"
FOLLOWER_B_PORT="${FOLLOWER_B_PORT:-/dev/so101/follower_b}"
LEADER_B_PORT="${LEADER_B_PORT:-/dev/so101/leader_b}"

# In dual mode arm A is "left" and arm B is "right"; LeRobot then looks for calibration files named
# <BI_ID>_left / <BI_ID>_right, which calibrate.sh links to the per-arm ones.
BI_FOLLOWER_ID="bi_follower"
BI_LEADER_ID="bi_leader"

# ---- Cameras (lerobot-find-cameras opencv lists them; /dev/v4l/by-id/... paths survive replugging)
# Keep the same camera names in single and dual data so one policy can read both.
CAM_W=640; CAM_H=480; CAM_FPS=30
CAM_FRONT="${CAM_FRONT:-/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_65ECE41F-video-index0}" # front view, sees both arms, the objects and the box
cam() { echo "{type: opencv, index_or_path: $1, width: $CAM_W, height: $CAM_H, fps: $CAM_FPS, fourcc: MJPG}"; }
CAMERAS="{ front: $(cam "$CAM_FRONT") }"
# A second camera later (e.g. overhead or wrist): add it here with its own name, in both modes:
# CAMERAS="{ front: $(cam "$CAM_FRONT"), top: $(cam "$CAM_TOP") }"

# ---- Task: one episode = one object picked and placed in the box (see vla/README.md, "Episodes")
TASK="${TASK:-Pick up the object and put it in the box}"
FPS=30
EPISODE_TIME_S="${EPISODE_TIME_S:-25}"   # recording stops early with the right-arrow key
RESET_TIME_S="${RESET_TIME_S:-10}"
NUM_EPISODES="${NUM_EPISODES:-50}"

# ---- Safety: cap each step's joint change (degrees) on the followers while teleoperating
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
