#!/usr/bin/env bash
# Fine-tune X-VLA (lerobot/xvla-base, 0.9B) on a dual-arm SO-101 dataset, in a detached
# container (docker_train.sh): it keeps running after you disconnect.
#   ./vla/train_xvla.sh                                  a new run: STEPS=7000 BATCH=16
#   ./vla/train_xvla.sh resume <run dir>                 continue a run that was stopped
#   ./vla/train_xvla.sh resume <run dir> 10000           train a finished run on, to 10000 steps
#
# Follow:  docker logs -f xvla_box    or    tail -f ~/vla_runs/<run>.log   (GPU: <run>.gpu.log)
# Stop:    docker stop xvla_box       (the last checkpoint stays on disk and can be resumed)
set -euo pipefail
VLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="${RUNS:-$HOME/vla_runs}"
NAME="${NAME:-xvla_box}"   # the container's name

if [[ "${1:-}" == resume ]]; then
    RUN="$(readlink -f "${2:?usage: $0 resume <run dir> [total steps]}")"
    ARGS=(--config_path="$RUN/checkpoints/last/pretrained_model/train_config.json" --resume=true)
    # More steps than the run was planned for also raises the learning rate again: the cosine
    # schedule is stretched to the new total.
    [[ -n "${3:-}" ]] && ARGS+=(--steps="$3")
    [[ -n "${SAVE_FREQ:-}" ]] && ARGS+=(--save_freq="$SAVE_FREQ")
else
    DATASET="${DATASET:-diegoloayza789/hfteam_box_middle}"
    # xvla-base names its cameras image, image2, image3. The first one is the main view: it goes
    # through the language model with the instruction. The others are extra views.
    DEFAULT_RENAME='{"observation.images.front":"observation.images.image","observation.images.left_wrist":"observation.images.image2","observation.images.right_wrist":"observation.images.image3"}'
    RUN="$RUNS/xvla_${DATASET#*/}_$(date +%Y%m%d_%H%M)"
    # action_mode=auto: the 12 joint targets are padded to the base model's 20, loss on the 12 only.
    # State is normalised too (the base model leaves it raw, but ours is in degrees, up to +-130).
    # Weights stay float32 with bf16 autocast: with policy.dtype=bfloat16 the weights themselves
    # are bf16 and most small updates round away.
    ARGS=(--policy.path=lerobot/xvla-base --policy.action_mode=auto
          --policy.device=cuda --policy.push_to_hub=false
          --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false
          --policy.train_policy_transformer=true --policy.train_soft_prompts=true
          --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}'
          --dataset.repo_id="$DATASET" --rename_map="${RENAME:-$DEFAULT_RENAME}"
          --accelerator.mixed_precision=bf16 --batch_size="${BATCH:-16}" --num_workers="${WORKERS:-12}"
          --steps="${STEPS:-7000}" --save_freq="${SAVE_FREQ:-2000}" --log_freq=100
          --output_dir="$RUN" --job_name="$(basename "$RUN")")
fi

OUT="$RUN" LOG="$RUN.log" "$VLA_DIR/docker_train.sh" "$NAME" "${ARGS[@]}"
echo "  checkpoints: $RUN/checkpoints/last/pretrained_model (the newest; older ones are pruned)"
