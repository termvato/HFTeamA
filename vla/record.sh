#!/usr/bin/env bash
# Record teleoperated episodes into a LeRobot dataset and push it to the Hugging Face Hub.
#   ./vla/record.sh single            # -> $DATASET_SINGLE
#   ./vla/record.sh dual              # -> $DATASET_DUAL
#   ./vla/record.sh single --resume   # add episodes to a dataset recorded earlier
# Extra flags pass through to lerobot-record, e.g. --dataset.num_episodes=10
#
# Keys while recording:  ->  end this episode early   <-  discard and redo it   Esc  stop and save
source "$(dirname "$0")/lib.sh"
MODE=$(mode_arg "${1:-}"); shift
REPO=$([[ $MODE == single ]] && echo "$DATASET_SINGLE" || echo "$DATASET_DUAL")
[[ -n "$HF_USER" ]] || { echo "not logged in to Hugging Face: hf auth login" >&2; exit 1; }

EXTRA=()
for a in "$@"; do [[ $a == --resume ]] && EXTRA+=(--resume=true) || EXTRA+=("$a"); done

robot_args "$MODE"; teleop_args "$MODE"
echo "recording $MODE -> $REPO  task: \"$TASK\""
lerobot-record "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" \
    --display_data=true \
    --dataset.repo_id="$REPO" \
    --dataset.no_stamp=true \
    --dataset.single_task="$TASK" \
    --dataset.fps="$FPS" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s="$( (( EPISODE_TIME_S > 0 )) && echo "$EPISODE_TIME_S" || echo 3600)" \
    --dataset.reset_time_s="$RESET_TIME_S" \
    --dataset.push_to_hub=true \
    --dataset.private="$PRIVATE" \
    --dataset.tags='["so101","stacking","hfteama"]' \
    "${EXTRA[@]}"
