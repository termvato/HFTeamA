#!/usr/bin/env bash
# Keep the GPU below a temperature while a training process runs.
#   ./vla/gpu_guard.sh <pid>            MAX_C=85 RESUME_C=83 ./vla/gpu_guard.sh <pid>
# At MAX_C the process is paused (SIGSTOP), at RESUME_C it continues (SIGCONT). Nothing is lost:
# the process only waits. Needs no root (a power limit with nvidia-smi -pl would).
# Default 87 C: the RTX 6000 Ada slows itself down at 88 C, and below 85 C its fan stays slow.
# The temperature is read every PERIOD_MS: short pauses keep the card just under MAX_C, which
# costs less training time than long ones. Prints a line every 30 s. Ends when the process ends.
set -u
PID="${1:?usage: $0 <pid of the training process>}"
MAX_C="${MAX_C:-87}"; RESUME_C="${RESUME_C:-85}"; PERIOD_MS="${PERIOD_MS:-250}"; GPU="${GPU:-0}"
paused=0; n=0; pauses=0; paused_ticks=0
trap 'kill -CONT "$PID" 2>/dev/null' EXIT   # never leave the training paused

while kill -0 "$PID" 2>/dev/null; do   # restarts the reader if nvidia-smi stops
    while IFS=', ' read -r temp power; do
        kill -0 "$PID" 2>/dev/null || break
        [[ "$temp" =~ ^[0-9]+$ ]] || continue
        if (( temp >= MAX_C )); then
            kill -STOP "$PID"   # every time, in case something else resumed the process
            (( paused )) || pauses=$((pauses + 1))
            paused=1
        elif (( paused && temp <= RESUME_C )); then
            kill -CONT "$PID"; paused=0
        fi
        (( paused )) && paused_ticks=$((paused_ticks + 1))
        if (( n % (30000 / PERIOD_MS) == 0 )); then
            echo "$(date +%T) ${temp} C ${power} W; ${pauses} pauses," \
                 "$((paused_ticks * PERIOD_MS / 1000)) s paused in total"
        fi
        n=$((n + 1))
    done < <(nvidia-smi --id="$GPU" --query-gpu=temperature.gpu,power.draw \
                 --format=csv,noheader,nounits -lms "$PERIOD_MS")
    kill -CONT "$PID" 2>/dev/null; paused=0
done
echo "$(date +%T) process $PID ended; ${pauses} pauses, $((paused_ticks * PERIOD_MS / 1000)) s paused in total"
