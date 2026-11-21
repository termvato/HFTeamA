#!/usr/bin/env bash
# One-time machine setup: LeRobot from source in a conda env "lerobot", serial access, ffmpeg.
# Safe to re-run. Steps that need sudo are printed, not run.
set -euo pipefail
LEROBOT_SRC="${LEROBOT_SRC:-$HOME/lerobot}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env list | grep -q '^lerobot ' || conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg

[[ -d "$LEROBOT_SRC" ]] || git clone https://github.com/huggingface/lerobot.git "$LEROBOT_SRC"
pip install -e "$LEROBOT_SRC[feetech,smolvla]"
python -c "import lerobot, torch; print('lerobot', lerobot.__version__, '| cuda', torch.cuda.is_available())"

hf auth whoami >/dev/null 2>&1 || echo "NEXT: hf auth login   (token with write access, for pushing datasets)"
id -nG | grep -qw dialout || echo "NEXT: sudo usermod -aG dialout $USER   then log out and back in"
echo "Then plug in the arms and follow vla/README.md, Part 1."
