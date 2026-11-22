"""Train a LeRobot policy on a Hub dataset on a Modal GPU, and push the model to the Hub.

    pip install modal && modal setup                     # once: log in (browser)
    modal secret create hfteam-hf HF_TOKEN=hf_...        # once: YOUR write token, to push the model
    modal run --detach scripts/train_modal.py            # ACT on diegoloayza789/hfteam_box_middle
    modal run --detach scripts/train_modal.py --policy smolvla --steps 20000 --batch-size 64

The command returns once the training is started (spawned) on Modal: closing the terminal or
turning the laptop off does not stop it (`--detach` is required for that). Follow it in the Modal
dashboard (Apps) or
with `modal app logs hfteam-train`. Checkpoints go to the Modal volume `hfteam-train`
(`modal volume ls hfteam-train`); the final model is pushed to the Hub as
<hub-user>/<policy>_<dataset name> (private unless --public).

The same LeRobot version the rig runs (0.7.0), so the model loads on the laptop as it is.
"""

from __future__ import annotations

import modal

LEROBOT = "lerobot[training,smolvla]==0.7.0"  # smolvla: transformers, for SmolVLA
# Tried in this order: the first type with free capacity runs it (all have >= 24 GB, enough for
# ACT and SmolVLA). A single type can wait a long time when Modal is short of it.
GPU = ["L4", "A10G", "L40S", "A100-40GB"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(LEROBOT)
)
app = modal.App("hfteam-train", image=image)
runs = modal.Volume.from_name("hfteam-train", create_if_missing=True)


@app.function(
    gpu=GPU,
    timeout=8 * 3600,
    secrets=[modal.Secret.from_name("hfteam-hf")],
    volumes={"/runs": runs},
)
def train(
    dataset: str,
    policy: str,
    model_repo: str,
    steps: int,
    batch_size: int,
    save_freq: int,
    private: bool,
) -> str:
    import subprocess
    import time

    name = model_repo.split("/")[-1]
    out = f"/runs/{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={dataset}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        "--log_freq=100",
        "--num_workers=8",
        f"--output_dir={out}",
        f"--job_name={name}",
        "--policy.device=cuda",
        "--policy.push_to_hub=true",
        f"--policy.repo_id={model_repo}",
        f"--policy.private={'true' if private else 'false'}",
    ]
    if policy == "smolvla":
        # Fine-tune the pretrained SmolVLA; the base model expects camera1..3.
        cmd += [
            "--policy.path=lerobot/smolvla_base",
            '--rename_map={"observation.images.front": "observation.images.camera1", '
            '"observation.images.left_wrist": "observation.images.camera2", '
            '"observation.images.right_wrist": "observation.images.camera3"}',
        ]
    else:
        cmd.append(f"--policy.type={policy}")
    print("$", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    finally:
        runs.commit()  # keep the checkpoints even if training stops early
    return f"https://huggingface.co/{model_repo}"


@app.local_entrypoint()
def main(
    dataset: str = "diegoloayza789/hfteam_box_middle",
    policy: str = "act",
    hub_user: str = "diegoloayza789",
    steps: int = 40000,
    batch_size: int = 8,
    save_freq: int = 5000,
    public: bool = False,
):
    if policy not in ("act", "smolvla"):
        raise SystemExit("--policy act or smolvla")
    model_repo = f"{hub_user}/{policy}_{dataset.split('/')[-1]}"
    print(f"Training {policy} on {dataset} ({steps} steps, batch {batch_size}, GPU: {GPU})")
    print(f"Model -> https://huggingface.co/{model_repo}")
    # spawn, not remote: this command returns at once and the training does not depend on it. With
    # .remote() the local process had to stay alive, and killing it (closed terminal, reboot,
    # timeout) sent a cancellation that stopped the training even with --detach.
    call = train.spawn(dataset, policy, model_repo, steps, batch_size, save_freq, not public)
    print(f"Started ({call.object_id}). Safe to close this terminal or turn off the laptop.")
    print("Follow it: modal app list / modal app logs <app id>, or the Modal dashboard.")
