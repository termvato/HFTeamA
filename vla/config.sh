# Settings shared by every vla/ script. Edit here, not in the scripts.
# Sourced by bash; values can also be overridden from the shell: TASK="..." ./vla/record_single.sh

# ---- Hugging Face
if [[ -z "${HF_USER:-}" ]]; then
    _whoami="$(hf auth whoami 2>&1 || true)"  # || true: under set -e a failure would exit silently
    # hf prints "user=<name>" or, in some terminals, "✓ Logged in" then "  user: <name>"
    HF_USER="$(sed -nE 's/^[[:space:]]*user[=:][[:space:]]*([^[:space:]]+).*/\1/p' <<<"$_whoami" | head -1)"
fi
if [[ -z "$HF_USER" ]]; then  # offline or not logged in: refuse rather than record to "/so101_..."
    echo "Could not get your Hugging Face username from 'hf auth whoami' ($(command -v hf)):" >&2
    echo "$_whoami" | head -5 >&2
    echo "Fix: hf auth login   or skip the lookup: export HF_USER=<name>" >&2
    exit 1
fi
PAIR_A=rakesh; PAIR_B=hacklab   # the Hardware team's pair names (arms below)
SINGLE_ARM="${SINGLE_ARM:-a}"   # which pair single-arm mode uses: a (rakesh) or b (hacklab)
# One single-arm dataset per pair: each has its own wrist camera, so the camera sets differ.
_single_pair=$([[ "${SINGLE_ARM,,}" == b ]] && echo "$PAIR_B" || echo "$PAIR_A")
DATASET_SINGLE="${DATASET_SINGLE:-${HF_USER}/so101_stack_single_${_single_pair}}"
DATASET_DUAL="${DATASET_DUAL:-${HF_USER}/so101_stack_dual}"
PRIVATE="${PRIVATE:-true}"          # datasets start private on the Hub; flip when ready to share

# ---- Arms: the Hardware team's two calibrated pairs (hardware branch: configs/teleop_pairs.json,
# configs/calibration/). Never recalibrate them: ./vla/sync_calibration.sh copies their files.
# Arm A = left = pair "rakesh", arm B = right = pair "hacklab" (left/right as the operator sees them).
# Ports are found from each arm's USB serial, so /dev/ttyACM* numbering does not matter.
FOLLOWER_A_ID=rakesh_follower;  FOLLOWER_A_SERIAL=5B79014975
LEADER_A_ID=rakesh_leader;      LEADER_A_SERIAL=5B8E115290
FOLLOWER_B_ID=vato_follower;    FOLLOWER_B_SERIAL=5A7A015861
LEADER_B_ID=hacklab_leader;     LEADER_B_SERIAL=5A7A015820
CALIBRATION_REF="${CALIBRATION_REF:-origin/hardware}"   # where the calibration files come from
CALIB_DIR="${CALIB_DIR:-$HOME/.cache/hfteama/vla_calibration}"

# In dual mode LeRobot looks for <BI_ID>_left / <BI_ID>_right calibration files; sync_calibration.sh
# links those names to the arms' own files.
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
CAM_WARMUP_S=3   # as the Hardware team uses for the wrist cameras: they need a moment to settle
CAMERA_VIEWS="top wrist_a wrist_b"   # the fixed slots on the studio page, in this order
#
# Three cameras: top (sees both arms, the objects and the box) and one wrist camera per arm.
# Record only with every camera in place: a dataset keeps the camera set it started with.
CAMERA_TABLE="${CAMERA_TABLE:-
# name     arms   rot   device
top        both   0     /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_65ECE41F-video-index0
wrist_a    a      0     /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0
wrist_b    b      0     /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._Autodarts_DIY_Cam_SN0001-video-index0
}"
# Wrist cameras: the ones the Hardware team mounted (teleop_pairs.json): wrist_a on the rakesh
# follower, wrist_b on the hacklab follower.

# ---- Task: one episode = one object picked and placed in the box (see vla/README.md, "Episodes")
TASK="${TASK:-Stack the box on top of the other box}"
FPS=$CAM_FPS
# 0 = no limit: an episode runs until you press Stop & save (studio) or the right-arrow key.
# A number = seconds, after which the episode saves itself.
EPISODE_TIME_S="${EPISODE_TIME_S:-0}"
RESET_TIME_S="${RESET_TIME_S:-10}"
NUM_EPISODES="${NUM_EPISODES:-50}"

# ---- Safety: cap each step's joint change (degrees) on the followers while teleoperating
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
