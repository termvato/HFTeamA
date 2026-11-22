#!/usr/bin/env bash
# Fine-tune X-VLA (lerobot/xvla-base, 0.9B) on a dual-arm SO-101 dataset, in the background.
#   ./vla/train_xvla.sh
#   DATASET=<user>/<name> STEPS=7000 BATCH=16 ./vla/train_xvla.sh
#
# Needs the conda env "lerobot" with LeRobot 0.7 and its xvla, training and dataset extras.
# gpu_guard.sh pauses the training whenever the GPU reaches MAX_C. Both run under nohup.
#
# Follow:  tail -f ~/vla_runs/<run>.log        GPU: tail -f ~/vla_runs/<run>.gpu.log
# Stop:    kill <pid printed below>             (the checkpoints saved so far stay on disk)
VLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then  # before set -u: conda's activate trips on it
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate lerobot
fi
set -euo pipefail
# the env's C++ runtime first: the system one can be too old for the env's sqlite (trackio)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DATASET="${DATASET:-diegoloayza789/hfteam_box_middle}"
STEPS="${STEPS:-7000}"; BATCH="${BATCH:-16}"; SAVE_FREQ="${SAVE_FREQ:-2000}"; WORKERS="${WORKERS:-12}"
# xvla-base names its cameras image, image2, image3. The first one is the main view: it goes
# through the language model with the instruction. The others are extra views.
DEFAULT_RENAME='{"observation.images.front":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}'
RENAME="${RENAME:-$DEFAULT_RENAME}"
RUNS="${RUNS:-$HOME/vla_runs}"
NAME="xvla_${DATASET#*/}_$(date +%Y%m%d_%H%M)"
RUN="$RUNS/$NAME"; LOG="$RUNS/$NAME.log"

mkdir -p "$RUNS"
cd "$RUNS"
# action_mode=auto: the 12 joint targets are padded to the base model's 20, loss on the 12 only.
# State is normalised too (the base model leaves it raw, but ours is in degrees, up to +-130).
# Weights stay float32 with bf16 autocast: with policy.dtype=bfloat16 the weights themselves are
# bf16 and most small updates round away.
nohup lerobot-train --policy.path=lerobot/xvla-base --policy.action_mode=auto \
    --policy.device=cuda --policy.push_to_hub=false \
    --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
    --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
    --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
    --dataset.repo_id="$DATASET" --rename_map="$RENAME" \
    --accelerator.mixed_precision=bf16 --batch_size="$BATCH" --num_workers="$WORKERS" \
    --steps="$STEPS" --save_freq="$SAVE_FREQ" --log_freq=100 \
    --output_dir="$RUN" --job_name="$NAME" > "$LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
nohup "$VLA_DIR/gpu_guard.sh" "$TRAIN_PID" > "$RUNS/$NAME.gpu.log" 2>&1 < /dev/null &

echo "training xvla on $DATASET (pid $TRAIN_PID): $STEPS steps, batch $BATCH"
echo "  log:         $LOG"
echo "  GPU guard:   $RUNS/$NAME.gpu.log (pauses at ${MAX_C:-87} C, resumes at ${RESUME_C:-85} C)"
echo "  checkpoints: $RUN/checkpoints/<step>/pretrained_model (every $SAVE_FREQ steps and the last)"
