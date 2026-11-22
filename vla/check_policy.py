"""Open-loop check of a trained policy: predicted action chunks against the recorded ones.

    python vla/check_policy.py ~/vla_runs/<run>/checkpoints/last/pretrained_model
    python vla/check_policy.py <checkpoint> --dataset <user>/<name> --frames 64

It runs the checkpoint's own pre/post-processors on recorded frames and prints the mean absolute
error per joint, in the dataset's units (degrees; gripper 0-100). No arm is involved. A low error
shows the checkpoint loads and reproduces the demonstrations; it is not a success rate.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--dataset", default="diegoloayza789/hfteam_box_middle")
parser.add_argument("--policy", default="xvla")
parser.add_argument("--frames", type=int, default=64)
parser.add_argument("--batch", type=int, default=8)
args = parser.parse_args()

policy = get_policy_class(args.policy).from_pretrained(args.checkpoint)
policy.eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

meta = LeRobotDatasetMetadata(args.dataset)
chunk = policy.config.chunk_size
steps = [i / meta.fps for i in range(chunk)]
dataset = LeRobotDataset(args.dataset, delta_timestamps={"action": steps})
# spread over all episodes
picks = np.linspace(0, len(dataset) - 1, args.frames).astype(int).tolist()

errors = []
for start in range(0, len(picks), args.batch):
    items = [dataset[i] for i in picks[start : start + args.batch]]
    batch = {
        k: torch.stack([it[k] for it in items])
        if isinstance(items[0][k], torch.Tensor)
        else [it[k] for it in items]
        for k in items[0]
    }
    target, pad = batch["action"].clone(), batch["action_is_pad"]
    with torch.no_grad():
        pred = post(policy.predict_action_chunk(pre(batch))).float().cpu()
    err = (pred - target).abs()
    err[pad] = float("nan")  # steps past the end of the episode have no recorded action
    errors.append(err)

err = torch.cat(errors)  # frames x chunk x joints
per_joint = np.nanmean(err.numpy(), axis=(0, 1))
names = meta.features["action"]["names"]
value_range = np.asarray(meta.stats["action"]["max"]) - np.asarray(meta.stats["action"]["min"])
print(f"{len(picks)} frames, chunks of {chunk} steps ({chunk / meta.fps:.1f} s)")
for name, e, r in zip(names, per_joint, value_range, strict=True):
    print(f"  {name:28s} {e:6.2f}  ({100 * e / r:4.1f}% of the recorded range)")
print(
    f"  mean {per_joint.mean():.2f}; first step of the chunk {np.nanmean(err[:, 0].numpy()):.2f}, "
    f"last step {np.nanmean(err[:, -1].numpy()):.2f}"
)

one = pre({k: v[:1] for k, v in batch.items()})
t0 = time.perf_counter()
with torch.no_grad():
    policy.predict_action_chunk(one).float().cpu()  # .cpu() waits for the GPU to finish
print(f"  one chunk for one observation: {1000 * (time.perf_counter() - t0):.0f} ms")
