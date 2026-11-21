#!/usr/bin/env bash
# Train a policy on a recorded dataset, with early stopping on the validation loss.
#   ./vla/train.sh smolvla                       # dataset: DATASET (default: the single-arm one)
#   ./vla/train.sh act
#   DATASET=<user>/<name> MAX_STEPS=30000 ./vla/train.sh smolvla
#
# The last EVAL_SPLIT of the episodes are held out (never trained on). Every EVAL_STEPS it computes
# the loss on them and saves a checkpoint. train_watch.py keeps the best checkpoint and stops once
# the loss stops improving (PATIENCE validations without a MIN_DELTA gain, after MIN_STEPS).
# MAX_STEPS is only a cap. Both run under nohup: closing the terminal does not stop them.
#
# Follow:  tail -f ~/vla_runs/<run>.watch.log     (one line per validation)
# Stop:    kill <pid printed below>                (the best checkpoint so far stays on disk)
source "$(dirname "$0")/lib.sh"
POLICY="${1:?usage: $0 smolvla|act}"
DATASET="${DATASET:-$DATASET_SINGLE}"
MAX_STEPS="${MAX_STEPS:-20000}"; EVAL_STEPS="${EVAL_STEPS:-1000}"; EVAL_SPLIT="${EVAL_SPLIT:-0.1}"
BATCH="${BATCH:-8}"; PATIENCE="${PATIENCE:-3}"; MIN_DELTA="${MIN_DELTA:-0.02}"; MIN_STEPS="${MIN_STEPS:-3000}"
RUNS="${RUNS:-$HOME/vla_runs}"
NAME="${POLICY}_${DATASET#*/}_$(date +%Y%m%d_%H%M)"
RUN="$RUNS/$NAME"; LOG="$RUNS/$NAME.log"
PUSH_REPO="${PUSH_REPO-$HF_USER/$NAME}"   # empty = do not upload

case "$POLICY" in
    smolvla)
        # smolvla_base was trained with cameras named camera1, camera2, ...: map the dataset's
        # own cameras onto them, in the dataset's order (top first, then the wrist cameras)
        RENAME="$(python -I - "$DATASET" <<'EOF' 2>/dev/null
import json, sys
from lerobot.datasets import LeRobotDatasetMetadata
keys = LeRobotDatasetMetadata(sys.argv[1]).camera_keys
print(json.dumps({k: f"observation.images.camera{i}" for i, k in enumerate(keys, 1)}))
EOF
)"
        [[ -n "$RENAME" ]] || { echo "could not read the cameras of $DATASET" >&2; exit 1; }
        POLICY_ARGS=(--policy.path=lerobot/smolvla_base --rename_map="$RENAME") ;;
    act) POLICY_ARGS=(--policy.type=act) ;;
    *) echo "unknown policy $POLICY: smolvla or act" >&2; exit 2 ;;
esac

mkdir -p "$RUNS"
cd "$RUNS"
nohup lerobot-train "${POLICY_ARGS[@]}" \
    --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id="$DATASET" --dataset.eval_split="$EVAL_SPLIT" \
    --batch_size="$BATCH" --steps="$MAX_STEPS" \
    --eval_steps="$EVAL_STEPS" --save_freq="$EVAL_STEPS" --log_freq=100 \
    --output_dir="$RUN" --job_name="$NAME" > "$LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
nohup python "$VLA_DIR/train_watch.py" --log "$LOG" --run "$RUN" --pid "$TRAIN_PID" \
    --patience "$PATIENCE" --min-delta "$MIN_DELTA" --min-steps "$MIN_STEPS" --push-repo "$PUSH_REPO" \
    > "$RUNS/$NAME.watch.log" 2>&1 < /dev/null &

echo "training $POLICY on $DATASET (pid $TRAIN_PID), at most $MAX_STEPS steps, early stopping on"
echo "  validation every $EVAL_STEPS steps on the last ${EVAL_SPLIT} of the episodes; patience $PATIENCE"
echo "  progress: tail -f $RUNS/$NAME.watch.log"
echo "  full log: $LOG"
echo "  best model: $RUN/checkpoints/<best step>/pretrained_model${PUSH_REPO:+, then uploaded to $PUSH_REPO (private)}"
