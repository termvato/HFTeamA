"""Teleoperated VLA datasets: record, check, discard and save, one fixed dataset per task.

`hfteama vla <task> record|status|discard|save`. The tasks (instruction text, dataset name,
episodes per session) come from configs/vla_tasks.json, so nothing here knows about a task.

- **record**: both pairs of the 'bimanual' config as one LeRobot robot (12 joints, the pairs'
  wrist cameras as left_* / right_*) plus the config's top-level cameras (e.g. `front`, names
  kept as they are). The dataset name is fixed (`no_stamp`): the first session creates it,
  every later one (night, day) appends to it with `--resume`. Never pushes.
- **status**: episodes, frames and per-episode length, from the metadata only.
- **discard**: delete episodes (default: the last one) after recording. The dataset before the
  edit is kept as `<name>_old` (one level of undo). Discarding every episode moves the whole
  dataset aside to `<name>_discarded_<time>` instead.
- **save**: check the data (features, camera set, task text, every episode's first and last
  frame decodes, no NaN), then push to the Hub.

lerobot is imported lazily: it is not a dependency of the package.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hfteama.hardware.pairs import PROJECT_ROOT, _flag, cameras_arg

DEFAULT_TASKS = PROJECT_ROOT / "configs" / "vla_tasks.json"
# Keys of a camera entry that are ours (how to find / set it up), not lerobot camera fields.
_CAMERA_KEYS = {"find", "controls", "controls_note", "note"}
MIN_EPISODE_S = 1.0


@dataclass(frozen=True)
class VlaTask:
    name: str
    task: str
    repo_id: str
    episodes: int
    episode_time_s: float
    reset_time_s: float
    fps: int
    private: bool = True
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)


def lerobot_home() -> Path:
    """Where LeRobot keeps local datasets (same rule as lerobot.utils.constants)."""
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return Path(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot")).expanduser()


def dataset_root(task: VlaTask) -> Path:
    return lerobot_home() / task.repo_id


def load_tasks(path: Path | str | None = None) -> dict[str, VlaTask]:
    path = Path(path or os.environ.get("HFTEAM_VLA_TASKS", DEFAULT_TASKS))
    data = json.loads(path.read_text())
    shared = data.get("record", {})
    tasks = {}
    for name, t in data["tasks"].items():
        rec = {**shared, **t}
        tasks[name] = VlaTask(
            name=name,
            task=rec["task"],
            repo_id=rec["repo_id"],
            episodes=int(rec.get("episodes", 25)),
            episode_time_s=float(rec.get("episode_time_s", 60)),
            reset_time_s=float(rec.get("reset_time_s", 20)),
            fps=int(rec.get("fps", 30)),
            private=bool(rec.get("private", True)),
            cameras=data.get("cameras", {}),
        )
    return tasks


def get_task(name: str, path: Path | str | None = None) -> VlaTask:
    tasks = load_tasks(path)
    if name not in tasks:
        raise SystemExit(f"Unknown VLA task {name!r}. Available: {', '.join(tasks)}")
    return tasks[name]


# ---------------------------------------------------------------- record


V4L_BY_ID = Path("/dev/v4l/by-id")


def stable_path(device: str, by_id: Path = V4L_BY_ID) -> str:
    """The /dev/v4l/by-id link of /dev/videoN, or the device itself if it has none.

    /dev/videoN changes when a camera reconnects (a loose cable); the by-id link does not, so a
    long-running recorder can reopen the camera after a replug.
    """
    try:
        links = sorted(by_id.glob("*-video-index0"))
    except OSError:
        return device
    target = Path(device).resolve()
    for link in links:
        if link.resolve() == target:
            return str(link)
    return device


def resolve_cameras(cameras: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """lerobot camera configs: a `find` name becomes that camera's stable /dev/v4l/by-id path."""
    from hfteama.vision.camera import find_device

    out = {}
    for name, cam in cameras.items():
        cam = dict(cam)
        if "find" in cam and "index_or_path" not in cam:
            device = find_device(cam["find"])
            if device is None:
                raise SystemExit(
                    f"Camera {name!r} ({cam['find']!r}) not found: is it plugged in? "
                    "(v4l2-ctl --list-devices; needs v4l-utils)"
                )
            cam["index_or_path"] = stable_path(device)
        out[name] = {k: v for k, v in cam.items() if k not in _CAMERA_KEYS}
    return out


def set_camera_controls(cameras: dict[str, dict[str, Any]], resolved: dict[str, dict]) -> None:
    from hfteama.vision.camera import apply_controls

    for name, cam in cameras.items():
        if cam.get("controls"):
            failed = apply_controls(str(resolved[name]["index_or_path"]), cam["controls"])
            if failed:
                print(f"warning: camera {name}: could not set {', '.join(failed)}")


def hardware_args(
    task: VlaTask,
    cameras: dict[str, dict[str, Any]] | None = None,
    pairs_config: Path | str | None = None,
    wrists_top_level: bool = False,
) -> list[str]:
    """--robot.* / --teleop.* flags: both pairs as one robot plus the task's top-level cameras.

    `cameras` are the resolved top-level cameras (default: resolve the task's now).
    `wrists_top_level`: declare the pairs' cameras as top-level ones named `left_<cam>` /
    `right_<cam>` instead of per arm. The dataset keys are the same either way; the studio needs
    it because it only shows the robot's top-level cameras.
    """
    from hfteama.hardware.bimanual import bimanual_args, load_bimanual

    bi = load_bimanual(pairs_config)
    if cameras is None:
        cameras = resolve_cameras(task.cameras)
    wrist = {f"left_{c}": cam for c, cam in bi.left.cameras.items()}
    wrist |= {f"right_{c}": cam for c, cam in bi.right.cameras.items()}
    clash = set(cameras) & (set(wrist) | set(bi.left.cameras) | set(bi.right.cameras))
    if clash:
        raise SystemExit(f"Camera names clash with the wrist cameras: {sorted(clash)}")
    args = bimanual_args(bi, with_cameras=not wrists_top_level)
    if wrists_top_level:
        cameras = {**wrist, **cameras}
    if cameras:
        args.append(f"--robot.cameras={cameras_arg(cameras)}")
    return args


def record_command(
    task: VlaTask,
    episodes: int | None = None,
    cameras: dict[str, dict[str, Any]] | None = None,
    pairs_config: Path | str | None = None,
) -> list[str]:
    """The lerobot-record command: bimanual robot, the task's fixed dataset, resume if it exists.

    `cameras` are the resolved top-level cameras (default: resolve the task's now).
    """
    from hfteama.hardware.cli import lerobot_cmd

    root = dataset_root(task)
    dataset = {
        "repo_id": task.repo_id,
        "root": str(root),
        "single_task": task.task,
        "num_episodes": episodes or task.episodes,
        "episode_time_s": task.episode_time_s,
        "reset_time_s": task.reset_time_s,
        "fps": task.fps,
        "push_to_hub": False,
        "no_stamp": True,
    }
    cmd = [lerobot_cmd("lerobot-record"), *hardware_args(task, cameras, pairs_config)]
    cmd += [f"--dataset.{k}={_flag(v)}" for k, v in dataset.items()]
    cmd.append("--display_data=true")
    if (root / "meta" / "info.json").is_file():
        cmd.append("--resume=true")
    return cmd


def rollout_command(
    task: VlaTask,
    model: str,
    cameras: dict[str, dict[str, Any]] | None = None,
    pairs_config: Path | str | None = None,
    duration: float = 60,
    device: str = "cuda",
    max_step_deg: float = 10,
    rename_map: dict[str, str] | None = None,
) -> list[str]:
    """lerobot-rollout: a trained policy drives both followers (no leaders), interactive.

    Cameras are top-level `front`, `left_wrist`, `right_wrist`: the dataset keys the policy was
    trained on. `rename_map` maps them to the policy's own camera names when they differ
    (SmolVLA camera1..3, X-VLA image..image3; see policy_rename_map). `max_step_deg` caps each
    joint's move per control step on both arms.
    """
    from hfteama.hardware.cli import lerobot_cmd

    robot = [
        a
        for a in hardware_args(task, cameras, pairs_config, wrists_top_level=True)
        if not a.startswith("--teleop.")
    ]
    if max_step_deg > 0:
        robot += [
            f"--robot.{side}_arm_config.max_relative_target={max_step_deg:g}"
            for side in ("left", "right")
        ]
    return [
        lerobot_cmd("lerobot-rollout"),
        "--strategy.type=base",
        f"--policy.path={model}",
        f"--device={device}",
        *robot,
        f"--task={task.task}",
        f"--fps={task.fps}",
        f"--duration={duration:g}",
        "--interactive=true",
        "--display_data=true",
        *([f"--rename_map={json.dumps(rename_map)}"] if rename_map else []),
    ]


def policy_rename_map(model: str) -> dict[str, str]:
    """The camera renaming a trained policy was trained with, read from the policy itself.

    LeRobot saves it in the preprocessor (`rename_observations_processor`). Empty when the policy
    uses the dataset's camera names (ACT) or the files can't be read.
    """
    path = Path(model)
    try:
        if path.is_dir():
            pre = path / "policy_preprocessor.json"
        else:
            from huggingface_hub import hf_hub_download

            pre = Path(hf_hub_download(model, "policy_preprocessor.json"))
        steps = json.loads(pre.read_text()).get("steps", [])
    except Exception:  # noqa: BLE001 - no preprocessor: no renaming
        return {}
    for step in steps:
        if step.get("registry_name") == "rename_observations_processor":
            return dict(step.get("config", {}).get("rename_map", {}))
    return {}


def expected_cameras(task: VlaTask, pairs_config: Path | str | None = None) -> set[str]:
    """Image keys a dataset of this task must have."""
    from hfteama.hardware.bimanual import load_bimanual

    bi = load_bimanual(pairs_config)
    names = {f"left_{c}" for c in bi.left.cameras} | {f"right_{c}" for c in bi.right.cameras}
    return {f"observation.images.{n}" for n in names | set(task.cameras)}


# ---------------------------------------------------------------- inspect


def _meta(task: VlaTask, root: Path | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = root or dataset_root(task)
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"No dataset for {task.name} yet ({root}). Record first.")
    return LeRobotDatasetMetadata(task.repo_id, root=root)


def status(task: VlaTask, root: Path | None = None) -> dict[str, Any]:
    meta = _meta(task, root)
    eps = meta.episodes
    lengths = [int(n) for n in eps["length"]] if meta.total_episodes else []
    return {
        "repo_id": task.repo_id,
        "root": str(root or dataset_root(task)),
        "episodes": meta.total_episodes,
        "frames": meta.total_frames,
        "fps": meta.fps,
        "minutes": round(meta.total_frames / meta.fps / 60, 1),
        "lengths_s": [round(n / meta.fps, 1) for n in lengths],
        "tasks": list(meta.tasks.index) if meta.tasks is not None else [],
        "cameras": sorted(meta.camera_keys),
        "action_dim": meta.features["action"]["shape"][0],
    }


def verify(task: VlaTask, root: Path | None = None, pairs_config=None) -> list[str]:
    """Problems that would make the dataset useless for training; [] if it is fine."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = root or dataset_root(task)
    meta = _meta(task, root)
    problems = []
    if meta.total_episodes == 0:
        return ["no episodes"]
    if meta.features["action"]["shape"][0] != 12:
        problems.append(f"action has {meta.features['action']['shape'][0]} values, expected 12")
    want, got = expected_cameras(task, pairs_config), set(meta.camera_keys)
    if want != got:
        problems.append(f"cameras {sorted(got)}, expected {sorted(want)}")
    tasks = list(meta.tasks.index)
    if tasks != [task.task]:
        problems.append(f"task text {tasks}, expected [{task.task!r}]")
    if meta.fps != task.fps:
        problems.append(f"fps {meta.fps}, expected {task.fps}")

    ds = LeRobotDataset(task.repo_id, root=root)
    for ep in range(meta.total_episodes):
        start = int(meta.episodes["dataset_from_index"][ep])
        end = int(meta.episodes["dataset_to_index"][ep])
        if end - start < MIN_EPISODE_S * meta.fps:
            problems.append(f"episode {ep}: only {end - start} frames")
        for idx in {start, end - 1}:
            try:
                frame = ds[idx]
            except Exception as e:  # noqa: BLE001 - report any decode failure
                problems.append(f"episode {ep} frame {idx - start}: {type(e).__name__}: {e}")
                continue
            for key in ("action", "observation.state"):
                if not all(math.isfinite(float(v)) for v in frame[key]):
                    problems.append(f"episode {ep} frame {idx - start}: {key} not finite")
            for cam in meta.camera_keys:
                img = frame[cam]
                if float(img.float().std()) < 1e-3:
                    problems.append(f"episode {ep} frame {idx - start}: {cam} is a flat image")
    return problems


# ---------------------------------------------------------------- discard / save


def discard(task: VlaTask, episodes: list[int] | None = None, root: Path | None = None) -> Path:
    """Delete episodes (default: the last). Returns where the previous version was kept."""
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = root or dataset_root(task)
    total = _meta(task, root).total_episodes
    if total == 0:
        raise SystemExit("The dataset has no episodes.")
    episodes = sorted(set(episodes if episodes is not None else [total - 1]))
    bad = [e for e in episodes if not 0 <= e < total]
    if bad:
        raise SystemExit(f"No episode {bad} (the dataset has episodes 0-{total - 1}).")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(episodes) == total:
        aside = root.with_name(f"{root.name}_discarded_{stamp}")
        shutil.move(root, aside)
        return aside

    tmp = root.with_name(f"{root.name}_editing_{stamp}")
    delete_episodes(
        LeRobotDataset(task.repo_id, root=root), episodes, output_dir=tmp, repo_id=task.repo_id
    )
    old = root.with_name(f"{root.name}_old")
    if old.exists():
        shutil.rmtree(old)
    shutil.move(root, old)
    shutil.move(tmp, root)
    return old


def save(
    task: VlaTask,
    root: Path | None = None,
    private: bool | None = None,
    force: bool = False,
    pairs_config=None,
) -> list[str]:
    """Check, then push to the Hub. Returns the problems found (pushes only if none, or force)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = root or dataset_root(task)
    problems = verify(task, root, pairs_config)
    if problems and not force:
        return problems
    ds = LeRobotDataset(task.repo_id, root=root)
    ds.push_to_hub(
        private=task.private if private is None else private, tags=["so101", "bimanual", "hfteam"]
    )
    return problems


# ---------------------------------------------------------------- CLI


def add_parser(sub) -> None:
    v = sub.add_parser("vla", help="VLA datasets per task: record, status, discard, save")
    v.add_argument("task", help="Task name from configs/vla_tasks.json, or 'list'")
    v.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["record", "studio", "run", "status", "discard", "save"],
    )
    v.add_argument(
        "--episodes",
        help="record: how many (default: the task's); "
        "discard: indices, e.g. 3,7 (default: the last one)",
    )
    v.add_argument("--all", action="store_true", help="discard: every episode")
    v.add_argument("--yes", action="store_true", help="discard/save: don't ask to confirm")
    v.add_argument("--public", action="store_true", help="save: public dataset on the Hub")
    v.add_argument("--force", action="store_true", help="save: push even if the check fails")
    v.add_argument("--tasks-config", help="default: configs/vla_tasks.json")
    v.add_argument("--port", type=int, default=8765, help="studio: web page port")
    v.add_argument("--model", help="run: trained policy, Hub repo id or local folder")
    v.add_argument("--duration", type=float, default=60, help="run: seconds per /start")
    v.add_argument("--device", default="cuda", help="run: cuda or cpu")
    v.add_argument(
        "--max-step-deg",
        type=float,
        default=10,
        help="run: largest joint move per control step (safety; 0 = no limit)",
    )


def _confirm(prompt: str, yes: bool) -> bool:
    return yes or input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes", "s", "si", "sí")


def dispatch(args) -> int:
    from hfteama.hardware.cli import run

    if args.task == "list":
        for t in load_tasks(args.tasks_config).values():
            have = (dataset_root(t) / "meta" / "info.json").is_file()
            n = _meta(t).total_episodes if have else 0
            print(f'{t.name:12} {n:3d} episodes  {t.repo_id}  "{t.task}"')
        return 0
    task = get_task(args.task, args.tasks_config)

    if args.action == "record":
        n = int(args.episodes) if args.episodes else None
        if args.dry_run:
            fake = {
                k: {
                    **{c: v for c, v in cam.items() if c not in _CAMERA_KEYS},
                    "index_or_path": cam.get("index_or_path", f"<{cam.get('find')}>"),
                }
                for k, cam in task.cameras.items()
            }
            return run(record_command(task, n, fake, args.config), True)
        cams = resolve_cameras(task.cameras)
        set_camera_controls(task.cameras, cams)
        from hfteama.hardware.bimanual import arms, load_bimanual
        from hfteama.hardware.cli import refuse_if_uncalibrated

        bi = load_bimanual(args.config)
        refuse_if_uncalibrated(
            arms(bi), f"`hfteama --pair {bi.left.pair} teleop` / `--pair {bi.right.pair}`"
        )
        code = run(record_command(task, n, cams, args.config), False)
        if (dataset_root(task) / "meta" / "info.json").is_file():
            _print_status(task)
        return code

    if args.action == "run":
        if not args.model:
            raise SystemExit("run needs --model <hub repo id or folder> (a trained policy)")
        if args.dry_run:
            fake = {
                k: {
                    **{c: v for c, v in cam.items() if c not in _CAMERA_KEYS},
                    "index_or_path": cam.get("index_or_path", f"<{cam.get('find')}>"),
                }
                for k, cam in task.cameras.items()
            }
            cmd = rollout_command(
                task,
                args.model,
                fake,
                args.config,
                args.duration,
                args.device,
                args.max_step_deg,
                policy_rename_map(args.model),
            )
            return run(cmd, True)
        cams = resolve_cameras(task.cameras)
        set_camera_controls(task.cameras, cams)
        from hfteama.hardware.bimanual import load_bimanual
        from hfteama.hardware.cli import refuse_if_uncalibrated

        bi = load_bimanual(args.config)
        refuse_if_uncalibrated(
            [bi.left.follower, bi.right.follower],
            f"`hfteama --pair {bi.left.pair} teleop` / `--pair {bi.right.pair}`",
        )
        print(
            "Interactive: the arms stay still until you type /start; /stop ends a run.\n"
            "Keep a hand near the arms. At the end they return to their starting pose."
        )
        rename = policy_rename_map(args.model)
        if rename:
            print("Cameras -> policy: " + ", ".join(f"{k} -> {v}" for k, v in rename.items()))
        return run(
            rollout_command(
                task,
                args.model,
                cams,
                args.config,
                args.duration,
                args.device,
                args.max_step_deg,
                rename,
            ),
            False,
        )

    if args.action == "studio":
        from hfteama.hardware.vla_studio import serve

        return serve(task, port=args.port, pairs_config=args.config)

    if args.action == "status":
        _print_status(task)
        problems = verify(task, pairs_config=args.config)
        print("check: OK" if not problems else "check: PROBLEMS\n  " + "\n  ".join(problems))
        return 1 if problems else 0

    if args.action == "discard":
        total = _meta(task).total_episodes
        eps = (
            list(range(total))
            if args.all
            else [int(e) for e in args.episodes.split(",")]
            if args.episodes
            else None
        )
        what = (
            "ALL episodes"
            if args.all
            else f"episodes {eps}"
            if eps
            else (f"the last episode ({total - 1})")
        )
        if not _confirm(f"Discard {what} of {task.repo_id}?", args.yes):
            print("Nothing discarded.")
            return 1
        kept = discard(task, eps)
        print(f"Discarded. Previous version kept in {kept}")
        if (dataset_root(task) / "meta" / "info.json").is_file():
            _print_status(task)
        return 0

    if args.action == "save":
        private = not args.public
        where = "PRIVATE" if private else "PUBLIC"
        if not _confirm(f"Push {task.repo_id} to the Hugging Face Hub ({where})?", args.yes):
            print("Not pushed.")
            return 1
        problems = save(task, private=private, force=args.force, pairs_config=args.config)
        if problems and not args.force:
            print("Not pushed, the check failed:\n  " + "\n  ".join(problems))
            return 1
        print(f"Pushed: https://huggingface.co/datasets/{task.repo_id}")
        return 0
    raise SystemExit(f"Unknown action {args.action}")


def _print_status(task: VlaTask) -> None:
    s = status(task)
    print(f"{s['repo_id']}  ({s['root']})")
    print(f"  {s['episodes']} episodes, {s['frames']} frames, {s['minutes']} min at {s['fps']} fps")
    print(f"  task: {s['tasks']}   action: {s['action_dim']}   cameras: {s['cameras']}")
    if s["lengths_s"]:
        print(
            "  episode lengths (s): " + ", ".join(f"{i}:{n}" for i, n in enumerate(s["lengths_s"]))
        )
