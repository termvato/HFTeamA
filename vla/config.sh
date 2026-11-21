# Settings shared by every vla/ script. Edit here, not in the scripts.
# Sourced by bash; values can also be overridden from the shell: TASK="..." ./vla/record_single.sh

# ---- Hugging Face
HF_USER="${HF_USER:-$(hf auth whoami 2>/dev/null | sed -n 's/^user=//p')}"
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
CAM_TOP="${CAM_TOP:-/dev/video2}"     # overhead, sees both arms, the objects and the box
CAM_FRONT="${CAM_FRONT:-/dev/video4}" # front/side view
cam() { echo "{type: opencv, index_or_path: $1, width: $CAM_W, height: $CAM_H, fps: $CAM_FPS, fourcc: MJPG}"; }
CAMERAS="{ top: $(cam "$CAM_TOP"), front: $(cam "$CAM_FRONT") }"

# ---- Task: one episode = one object picked and placed in the box (see vla/README.md, "Episodes")
TASK="${TASK:-Pick up the object and put it in the box}"
FPS=30
EPISODE_TIME_S="${EPISODE_TIME_S:-25}"   # recording stops early with the right-arrow key
RESET_TIME_S="${RESET_TIME_S:-10}"
NUM_EPISODES="${NUM_EPISODES:-50}"

# ---- Safety: cap each step's joint change (degrees) on the followers while teleoperating
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
