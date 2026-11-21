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

# ---- Cameras: one line each. ./vla/cameras.sh list shows the devices, ./vla/cameras.sh check tests them.
#   name   a fixed name: it becomes observation.images.<name> in every dataset, so never rename one
#          after recording. Lower case, no spaces: front, top, wrist_a, wrist_b.
#   arms   which recordings use it: both (single and dual), a (arm A's wrist: single and dual),
#          b (arm B's wrist: dual only)
#   device /dev/v4l/by-id/...-video-index0 (survives replugging; index1 is metadata, not video)
#   rot    0, 90, 180 or -90 (a camera mounted sideways or upside down)
# All cameras share one size and frame rate: the policy sees every view at the same resolution,
# and the frame rate must equal the dataset FPS. MJPG keeps several cameras within USB bandwidth.
CAM_W=640; CAM_H=480; CAM_FPS=30; CAM_FOURCC=MJPG
CAMERA_VIEWS="top wrist_a wrist_b"   # the fixed slots on the studio page, in this order
#
# Three cameras: top (sees both arms, the objects and the box) and one wrist camera per arm.
# Uncomment each wrist line with its device once it is plugged in (./vla/cameras.sh list).
# Record only with every camera in place: a dataset keeps the camera set it started with.
CAMERA_TABLE="${CAMERA_TABLE:-
# name     arms   rot   device
top        both   0     /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_65ECE41F-video-index0
# wrist_a  a      0     /dev/v4l/by-id/FILL_ME-video-index0
# wrist_b  b      0     /dev/v4l/by-id/FILL_ME-video-index0
}"

# ---- Task: one episode = one object picked and placed in the box (see vla/README.md, "Episodes")
TASK="${TASK:-Pick up the object and put it in the box}"
FPS=$CAM_FPS
EPISODE_TIME_S="${EPISODE_TIME_S:-25}"   # recording stops early with the right-arrow key
RESET_TIME_S="${RESET_TIME_S:-10}"
NUM_EPISODES="${NUM_EPISODES:-50}"

# ---- Safety: cap each step's joint change (degrees) on the followers while teleoperating
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
