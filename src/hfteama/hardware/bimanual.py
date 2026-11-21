"""Two pairs as one LeRobot bimanual robot (bi_so_follower / bi_so_leader).

One process, one dataset with both arms: 12 joints, cameras prefixed `left_` / `right_`.
Use it for tasks where the two arms work together; for independent arms run the pairs
in parallel instead (`hfteama --pair a,b teleop`).

LeRobot names the arms' calibration `<id>_left.json` / `<id>_right.json` inside one
calibration_dir. Rather than duplicating calibration files in the repo, we build that
directory under ~/.cache/hfteama with symlinks to each pair's committed file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hfteama.hardware.pairs import (
    ArmConfig,
    PairConfig,
    _flag,
    cameras_arg,
    load_pair,
    read_config,
)

LINK_ROOT = Path(os.environ.get("HFTEAM_BIMANUAL_CACHE", "~/.cache/hfteama/bimanual")).expanduser()


@dataclass
class BimanualConfig:
    id: str
    left: PairConfig
    right: PairConfig
    record: dict[str, Any]


def load_bimanual(path: Path | str | None = None) -> BimanualConfig:
    data = read_config(path)
    bi = data.get("bimanual")
    if not bi:
        raise SystemExit('No "bimanual" section in the pairs config.')
    return BimanualConfig(
        id=bi["id"],
        left=load_pair(path, bi["left"]),
        right=load_pair(path, bi["right"]),
        record={**data.get("record", {}), **bi.get("record", {})},
    )


def link_calibrations(cfg: BimanualConfig, root: Path = LINK_ROOT) -> tuple[Path, Path]:
    """Create <root>/<id>/{followers,leaders}/<id>_{left,right}.json -> committed files."""
    base = root / cfg.id
    dirs = {"followers": base / "followers", "leaders": base / "leaders"}
    for role, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        for side, pair in (("left", cfg.left), ("right", cfg.right)):
            arm = pair.follower if role == "followers" else pair.leader
            target = (arm.calibration_dir / f"{arm.id}.json").resolve()
            if not target.is_file():
                raise SystemExit(f"Missing calibration {target} for {pair.pair} {arm.id}")
            link = d / f"{cfg.id}_{side}.json"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
    return dirs["followers"], dirs["leaders"]


def _side_args(prefix: str, side: str, arm: ArmConfig, cameras: dict | None) -> list[str]:
    p = f"--{prefix}.{side}_arm_config"
    args = [f"{p}.port={arm.port}"]
    args += [f"{p}.{k}={_flag(v)}" for k, v in arm.extra.items()]
    if cameras:
        args.append(f"{p}.cameras={cameras_arg(cameras)}")
    return args


def bimanual_args(cfg: BimanualConfig, with_cameras: bool, root: Path = LINK_ROOT) -> list[str]:
    followers, leaders = link_calibrations(cfg, root)
    args = [
        "--robot.type=bi_so_follower",
        f"--robot.id={cfg.id}",
        f"--robot.calibration_dir={followers}",
        "--teleop.type=bi_so_leader",
        f"--teleop.id={cfg.id}",
        f"--teleop.calibration_dir={leaders}",
    ]
    for side, pair in (("left", cfg.left), ("right", cfg.right)):
        args += _side_args("robot", side, pair.follower, pair.cameras if with_cameras else None)
        args += _side_args("teleop", side, pair.leader, None)
    return args


def arms(cfg: BimanualConfig) -> list[ArmConfig]:
    """The four arms, for the calibration pre-check (with their own committed IDs)."""
    return [a for p in (cfg.left, cfg.right) for a in (p.follower, p.leader)]
