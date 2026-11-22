#!/usr/bin/env bash
# Run lerobot-train in a detached Docker container, so it keeps running after you disconnect.
#   OUT=<output_dir> ./vla/docker_train.sh <container name> <lerobot-train args...>
# The container uses the host's conda env "lerobot", ~/lerobot, and the Hugging Face and torch
# caches, mounted at the same paths. A checkpoint therefore resumes in a container or on the
# host alike. container_entry.sh adds the GPU guard and checkpoint pruning (KEEP_OLDER).
#   docker logs -f <name>     follow          docker stop <name>     stop (checkpoints stay)
set -euo pipefail
NAME="${1:?usage: OUT=<output_dir> $0 <container name> <lerobot-train args...>}"; shift
: "${OUT:?set OUT to the output_dir of the run}"
VLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/miniforge3/envs/lerobot}"
LEROBOT_SRC="${LEROBOT_SRC:-$HOME/lerobot}"
RUNS="${RUNS:-$HOME/vla_runs}"
IMAGE="${IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}"  # only for the driver mounts
LOG="${LOG:-$RUNS/$NAME.log}"

if [[ -n "$(docker ps -q --filter "name=^${NAME}$")" ]]; then
    echo "container $NAME is still running: docker stop $NAME, or pick another name" >&2; exit 1
fi
docker rm "$NAME" > /dev/null 2>&1 || true   # an earlier, finished container of this name

# a copy per container: bash reads a script while running it, so never edit one in use
BIN="$RUNS/.bin/$NAME"
mkdir -p "$BIN" "$HOME/.cache/torch"
cp "$VLA_DIR/container_entry.sh" "$VLA_DIR/gpu_guard.sh" "$BIN/"

docker run -d --name "$NAME" --init --gpus all --ipc host --network host \
    --log-opt max-size=20m --log-opt max-file=1 \
    --user "$(id -u):$(id -g)" -w "$RUNS" \
    -e HOME="$HOME" -e USER="$USER" -e ENV_DIR="$ENV_DIR" -e OUT="$OUT" -e LOG="$LOG" \
    -e KEEP_OLDER="${KEEP_OLDER:-1}" -e MAX_C="${MAX_C:-87}" -e RESUME_C="${RESUME_C:-85}" \
    -e HF_HOME="$HOME/.cache/huggingface" -e TORCH_HOME="$HOME/.cache/torch" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" -e XDG_CACHE_HOME=/tmp/cache -e MPLCONFIGDIR=/tmp/mpl \
    -e http_proxy -e https_proxy -e no_proxy -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
    -v "$ENV_DIR:$ENV_DIR:ro" -v "$LEROBOT_SRC:$LEROBOT_SRC:ro" -v "$RUNS:$RUNS" \
    -v "$HOME/.cache/huggingface:$HOME/.cache/huggingface" \
    -v "$HOME/.cache/torch:$HOME/.cache/torch" \
    "$IMAGE" bash "$BIN/container_entry.sh" "$@"

echo "container $NAME started"
echo "  log:  $LOG   (or: docker logs -f $NAME)"
echo "  GPU:  ${LOG%.log}.gpu.log"
echo "  stop: docker stop $NAME"
