"""Early stopping for lerobot-train, which has none: stop when the validation loss stops improving.

train.sh starts lerobot-train with held-out episodes (--dataset.eval_split), a validation loss every
--eval_steps, and a checkpoint at each of those steps. This watcher follows the training log:

- After each validation it keeps the best one so far. A new loss counts as better only when it is
  at least MIN_DELTA (relative) below the best.
- It deletes every checkpoint except the best and the newest, to save disk space.
- It stops training after PATIENCE validations in a row without improvement, once MIN_STEPS are
  done. The best checkpoint is then the model to use.
- Optionally it uploads the best checkpoint's pretrained_model to the Hub (--push-repo).

Progress: <run>/watch.json, and a one-line summary per validation in the watcher log.

A low validation loss means the model predicts the held-out demonstrations' actions well. It is
not the success rate: test the best checkpoint on the arm.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import time
from pathlib import Path

EVAL_RE = re.compile(r"step (\d+): eval_loss=([0-9.]+)")
TRAIN_RE = re.compile(r"step:(\d+(?:\.\d+)?)([KM]?) .*?\bloss:([0-9.]+)")


def parse_step(num: str, suffix: str) -> int:
    return int(float(num) * {"": 1, "K": 1_000, "M": 1_000_000}[suffix])


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def find_trainer(run: Path, pid: int) -> int:
    """The training process: `pid` if it is the one writing to `run`, else the process whose
    command line names `run` (a launcher such as setsid can fork, so $! is not always it)."""
    def cmdline(p: str) -> str:
        try:
            return Path(f"/proc/{p}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return ""

    if str(run) in cmdline(str(pid)):
        return pid
    for proc in Path("/proc").glob("[0-9]*"):
        c = cmdline(proc.name)
        if "train_watch" not in c and f"--output_dir={run}" in c:
            return int(proc.name)
    return pid


def checkpoints(run: Path) -> dict[int, Path]:
    d = run / "checkpoints"
    return {int(p.name): p for p in d.iterdir() if p.name.isdigit()} if d.exists() else {}


def prune(run: Path, keep: set[int], keep_last: bool = True) -> None:
    """Delete checkpoints not in keep. While training, also keep the one 'last' points to (a
    resume needs it); once training is over, keep only `keep` and drop the 'last' link if dangling."""
    last = run / "checkpoints" / "last"
    last_target = last.resolve().name if last.is_symlink() else None
    for step, path in checkpoints(run).items():
        if step not in keep and not (keep_last and path.name == last_target):
            shutil.rmtree(path, ignore_errors=True)
    if not keep_last and last.is_symlink() and not last.exists():
        last.unlink()


def stop(pid: int) -> None:
    """Stop training and its data-loader workers. SIGTERM, not SIGINT: a process started in the
    background of a script ignores SIGINT. The best checkpoint is already on disk."""
    try:
        target, kill = (os.getpgid(pid), os.killpg) if os.getpgid(pid) == pid else (pid, os.kill)
        kill(target, signal.SIGTERM)
        for _ in range(30):
            if not alive(pid):
                return
            time.sleep(1)
        kill(target, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, required=True, help="lerobot-train's log file")
    p.add_argument("--run", type=Path, required=True, help="lerobot-train's --output_dir")
    p.add_argument("--pid", type=int, required=True, help="lerobot-train's process id")
    p.add_argument("--patience", type=int, default=3, help="validations without improvement before stopping")
    p.add_argument("--min-delta", type=float, default=0.02, help="relative improvement that counts (0.02 = 2%%)")
    p.add_argument("--min-steps", type=int, default=3000, help="never stop before this step")
    p.add_argument("--push-repo", default="", help="upload the best checkpoint here when done (private)")
    a = p.parse_args()

    state = {"evals": [], "train_loss": [], "best_step": None, "best_loss": None, "bad": 0,
             "status": "training", "stop_reason": ""}
    status_file = a.run / "watch.json"

    def save_state() -> None:
        a.run.mkdir(parents=True, exist_ok=True)
        status_file.write_text(json.dumps(state, indent=1))

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    time.sleep(3)  # let the trainer start
    a.pid = find_trainer(a.run, a.pid)
    log(f"watching training process {a.pid}: {a.run}")
    pos = 0
    stop_at: int | None = None  # stop once the checkpoint for this step is written
    while True:
        text = a.log.read_text(errors="replace") if a.log.exists() else ""
        new, pos = text[pos:], len(text)
        for line in new.replace("\r", "\n").splitlines():
            if m := TRAIN_RE.search(line):
                state["train_loss"].append([parse_step(m[1], m[2]), float(m[3])])
            if m := EVAL_RE.search(line):
                step, loss = int(m[1]), float(m[2])
                state["evals"].append([step, loss])
                best = state["best_loss"]
                if best is None or loss < best * (1 - a.min_delta):
                    state.update(best_step=step, best_loss=loss, bad=0)
                    verdict = "new best"
                else:
                    state["bad"] += 1
                    verdict = f"no improvement ({state['bad']}/{a.patience})"
                train = state["train_loss"][-1][1] if state["train_loss"] else float("nan")
                log(f"step {step}: eval_loss {loss:.4f} (train {train:.4f}) {verdict}; "
                    f"best {state['best_loss']:.4f} at step {state['best_step']}")
                if state["bad"] >= a.patience and step >= a.min_steps and stop_at is None:
                    stop_at = step
                    state["stop_reason"] = (f"eval_loss did not improve by {a.min_delta:.0%} for "
                                            f"{a.patience} validations; best at step {state['best_step']}")
                save_state()

        keep = {state["best_step"], max(checkpoints(a.run), default=None)} - {None}
        prune(a.run, keep)

        finished = not alive(a.pid)
        if stop_at is not None and not finished and stop_at in checkpoints(a.run):
            time.sleep(20)  # let the checkpoint finish writing
            log(f"early stop at step {stop_at}: {state['stop_reason']}")
            stop(a.pid)
            state["status"], finished = "stopped early", True
        if finished:
            if state["status"] == "training":
                state["status"] = "finished (all steps)" if "Training: 100%" in text or "End of training" in text \
                    else "ended (check the log)"
            prune(a.run, {state["best_step"]} - {None}, keep_last=False)
            best_dir = (a.run / "checkpoints" / f"{state['best_step']:06d}" / "pretrained_model"
                        if state["best_step"] is not None else None)
            state["best_checkpoint"] = str(best_dir) if best_dir and best_dir.exists() else ""
            log(f"{state['status']}; best eval_loss {state['best_loss']} at step {state['best_step']}: "
                f"{state['best_checkpoint'] or 'no checkpoint'}")
            if a.push_repo and state["best_checkpoint"]:
                from huggingface_hub import HfApi

                api = HfApi()
                api.create_repo(a.push_repo, private=True, exist_ok=True)
                api.upload_folder(folder_path=state["best_checkpoint"], repo_id=a.push_repo,
                                  commit_message=f"best checkpoint: step {state['best_step']}, "
                                                 f"eval_loss {state['best_loss']:.4f}")
                state["pushed_to"] = f"https://huggingface.co/{a.push_repo}"
                log(f"uploaded to {state['pushed_to']} (private)")
            save_state()
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
