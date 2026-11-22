#!/usr/bin/env bash
# Make a folder to copy to another machine (SFTP) and publish to the Hugging Face Hub from
# there: this workstation cannot upload through its proxy.
#   ./vla/make_bundle.sh <run dir> <hub repo> [checkpoint]      checkpoint: default "last"
#   CARD=card.md ./vla/make_bundle.sh ~/vla_runs/<run> VivekSai07/xvla_hfteam_box_middle
# Result: ~/vla_bundles/<repo name>/model (what gets published) and PUBLISH.md (the steps).
# The weights are hard links to the checkpoint, so the bundle takes no extra disk space.
set -euo pipefail
RUN="${1:?usage: $0 <run dir> <hub repo, e.g. user/name> [checkpoint]}"
REPO="${2:?usage: $0 <run dir> <hub repo, e.g. user/name> [checkpoint]}"
CKPT="${3:-last}"
SRC="$(readlink -f "$RUN/checkpoints/$CKPT")/pretrained_model"
STEP="$(basename "$(dirname "$SRC")")"
NAME="${REPO#*/}"
OUT="${BUNDLES:-$HOME/vla_bundles}/$NAME"
[[ -s "$SRC/model.safetensors" ]] || { echo "no model in $SRC" >&2; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/model"
cp -al "$SRC/." "$OUT/model/"
rm -f "$OUT/model/train_config.json"   # resume settings with local paths: not for the Hub
cp --remove-destination "$SRC/config.json" "$OUT/model/config.json"   # a real copy: edited below
[[ -n "${CARD:-}" ]] && cp "$CARD" "$OUT/model/README.md"

# A fine-tuned checkpoint keeps the base model's state size in config.json: write the real one.
ENV_PY="${ENV_DIR:-$HOME/miniforge3/envs/lerobot}/bin/python"
"$ENV_PY" - "$OUT/model" <<'EOF'
import glob, json, sys
from safetensors import safe_open

model = sys.argv[1]
config = json.load(open(f"{model}/config.json"))
state = config.get("input_features", {}).get("observation.state")
for path in glob.glob(f"{model}/policy_preprocessor_step_*_normalizer_processor.safetensors"):
    with safe_open(path, "pt") as f:
        if state and "observation.state.mean" in f.keys():
            size = list(f.get_slice("observation.state.mean").get_shape())
            if state["shape"] != size:
                print(f"config.json: observation.state shape {state['shape']} -> {size}")
                state["shape"] = size
json.dump(config, open(f"{model}/config.json", "w"), indent=4)
EOF

HOST="$(hostname -I | cut -d' ' -f1)"
SIZE="$(du -shL "$OUT/model" | cut -f1)"
cat > "$OUT/PUBLISH.md" <<EOF
# Publish $REPO

Checkpoint: step $STEP of \`$RUN\` ($SIZE).

## 1. Copy the folder to Windows (PowerShell)

\`\`\`powershell
sftp -r $USER@$HOST:vla_bundles/$NAME .
\`\`\`
Or with WinSCP: host \`$HOST\`, user \`$USER\`, folder \`$OUT\`.

## 2. Publish from Windows

\`\`\`powershell
pip install -U huggingface_hub
hf auth login                      # a token with write access
hf upload $REPO $NAME\\model . --repo-type model
\`\`\`
The repo is created if it does not exist, as a public repo. Add \`--private\` for a private one.
Running step 2 again with a newer bundle replaces the files.

## 3. Use it at the rig

\`--policy.path=$REPO\` (see the model card, \`model/README.md\`).
EOF
echo "bundle: $OUT ($SIZE, step $STEP)"
echo "steps:  $OUT/PUBLISH.md"
