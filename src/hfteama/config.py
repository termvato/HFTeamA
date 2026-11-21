"""Load the arm configuration and turn it into lerobot CLI arguments."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "arms.toml"
CALIBRATION_ROOT = PROJECT_ROOT / "calibration"

# lerobot stores calibration under <root>/robots/<robot.name>/ and
# <root>/teleoperators/<teleop.name>/; SO-101 classes are named so_follower / so_leader.
FOLLOWER_CALIBRATION_DIR = CALIBRATION_ROOT / "robots" / "so_follower"
LEADER_CALIBRATION_DIR = CALIBRATION_ROOT / "teleoperators" / "so_leader"

# Keys of an arm section that are ours, not lerobot config fields.
_ARM_KEYS = {"type", "port", "id", "serial"}


@dataclass
class ArmConfig:
    type: str
    port: str
    id: str
    calibration_dir: Path
    serial: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    pair: str
    follower: ArmConfig
    leader: ArmConfig
    cameras: dict[str, dict[str, Any]]
    record: dict[str, Any]


def find_port_by_serial(serial: str) -> str | None:
    """Return the /dev path of the connected USB serial device with this serial number."""
    from serial.tools import list_ports

    for p in list_ports.comports():
        if p.serial_number == serial:
            return p.device
    return None


def _arm(section: dict[str, Any], calibration_dir: Path, env_prefix: str) -> ArmConfig:
    serial = section.get("serial")
    port = os.environ.get(f"{env_prefix}_PORT")
    if port is None and serial:
        port = find_port_by_serial(serial)
    if port is None:
        port = section["port"]
    return ArmConfig(
        type=section["type"],
        port=port,
        id=section["id"],
        calibration_dir=calibration_dir,
        serial=serial,
        extra={k: v for k, v in section.items() if k not in _ARM_KEYS},
    )


def load_config(path: Path | str | None = None, pair: str | None = None) -> ProjectConfig:
    path = Path(path or os.environ.get("HFTEAM_CONFIG", DEFAULT_CONFIG))
    with path.open("rb") as f:
        data = tomllib.load(f)
    pairs = data["pairs"]
    name = pair or os.environ.get("HFTEAM_PAIR") or data.get("default_pair")
    if name not in pairs:
        raise SystemExit(f"Unknown pair {name!r}. Available: {', '.join(pairs)}")
    return ProjectConfig(
        pair=name,
        follower=_arm(pairs[name]["follower"], FOLLOWER_CALIBRATION_DIR, "HFTEAM_FOLLOWER"),
        leader=_arm(pairs[name]["leader"], LEADER_CALIBRATION_DIR, "HFTEAM_LEADER"),
        cameras=pairs[name].get("cameras", {}),
        record=data.get("record", {}),
    )


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def cameras_arg(cameras: dict[str, dict[str, Any]]) -> str:
    """Render cameras in the draccus dict syntax lerobot expects."""
    parts = []
    for name, cam in cameras.items():
        fields = ", ".join(f'"{k}": {_format_value(v)}' for k, v in cam.items())
        parts.append(f"{name}: {{{fields}}}")
    return "{" + ", ".join(parts) + "}"


def _flag(value: Any) -> str:
    return _format_value(value) if isinstance(value, bool) else str(value)


def arm_args(prefix: str, arm: ArmConfig, cameras: dict[str, dict[str, Any]] | None = None):
    args = [
        f"--{prefix}.type={arm.type}",
        f"--{prefix}.port={arm.port}",
        f"--{prefix}.id={arm.id}",
        f"--{prefix}.calibration_dir={arm.calibration_dir}",
    ]
    args += [f"--{prefix}.{k}={_flag(v)}" for k, v in arm.extra.items()]
    if cameras:
        args.append(f"--{prefix}.cameras={cameras_arg(cameras)}")
    return args


def follower_args(cfg: ProjectConfig, with_cameras: bool = False) -> list[str]:
    return arm_args("robot", cfg.follower, cfg.cameras if with_cameras else None)


def leader_args(cfg: ProjectConfig) -> list[str]:
    return arm_args("teleop", cfg.leader)


def record_args(cfg: ProjectConfig, overrides: dict[str, Any] | None = None) -> list[str]:
    rec = {**cfg.record, **(overrides or {})}
    return [f"--dataset.{k}={_flag(v)}" for k, v in rec.items()]
