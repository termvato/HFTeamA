#!/usr/bin/env bash
# Runs inside the training container started by docker_train.sh: lerobot-train, gpu_guard.sh,
# and checkpoint pruning. Exits with lerobot-train's exit code.
#   OUT        the run's output_dir (checkpoints are pruned in OUT/checkpoints)
#   LOG        lerobot-train's log file (also shown by docker logs)
#   KEEP_OLDER how many checkpoints before the newest keep their weights (default 1). Only the
#              newest keeps its optimizer state (needed to resume); older ones lose it.
set -u
export PATH="$ENV_DIR/bin:$PATH" LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BIN="$(cd "$(dirname "$0")" && pwd)"
KEEP_OLDER="${KEEP_OLDER:-1}"

prune() {
    local ckpts="$OUT/checkpoints" last d n=0
    [[ -L "$ckpts/last" ]] || return 0
    last="$(basename "$(readlink "$ckpts/last")")"  # lerobot moves "last" once a save is complete
    for d in $(ls -d "$ckpts"/[0-9]* 2>/dev/null | sort -r); do
        [[ "$(basename "$d")" < "$last" ]] || continue
        n=$((n + 1))
        if (( n > KEEP_OLDER )); then
            rm -rf "$d" && echo "$(date +%T) pruned checkpoint $(basename "$d")"
        elif [[ -d "$d/training_state" ]]; then
            rm -rf "$d/training_state" && echo "$(date +%T) pruned optimizer state of $(basename "$d")"
        fi
    done
}

lerobot-train "$@" > >(tee -a "$LOG") 2>&1 &
PID=$!
"$BIN/gpu_guard.sh" "$PID" >> "${LOG%.log}.gpu.log" 2>&1 &
while kill -0 "$PID" 2>/dev/null; do
    prune
    sleep 30
done
wait "$PID"; rc=$?
prune
echo "$(date +%T) lerobot-train exited with $rc"
exit "$rc"
