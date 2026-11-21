"""Check that the CLI builds the lerobot commands we expect."""

from pathlib import Path

import pytest

from hfteama.__main__ import build_command, parser
from hfteama.config import DEFAULT_CONFIG, cameras_arg, load_config


def cmd(*argv: str) -> list[str]:
    return build_command(parser().parse_args(list(argv)))


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.pair == "rakesh"
    assert cfg.follower.id == "rakesh_follower" and cfg.leader.id == "rakesh_leader"
    assert (cfg.follower.calibration_dir / f"{cfg.follower.id}.json").is_file()
    assert (cfg.leader.calibration_dir / f"{cfg.leader.id}.json").is_file()
    assert cfg.follower.type == "so101_follower"
    assert cfg.leader.type == "so101_leader"
    assert cfg.follower.calibration_dir.parts[-2:] == ("robots", "so_follower")
    assert cfg.leader.calibration_dir.parts[-2:] == ("teleoperators", "so_leader")


def test_calibrate_follower() -> None:
    c = cmd("calibrate", "follower")
    assert Path(c[0]).name == "lerobot-calibrate"
    assert "--robot.type=so101_follower" in c
    assert any(a.startswith("--robot.calibration_dir=") for a in c)
    assert not any(a.startswith("--teleop.") for a in c)


def test_calibrate_leader() -> None:
    c = cmd("calibrate", "leader")
    assert "--teleop.type=so101_leader" in c
    assert not any(a.startswith("--robot.") for a in c)


def test_teleop_uses_both_arms() -> None:
    c = cmd("teleop")
    assert "--robot.type=so101_follower" in c and "--teleop.type=so101_leader" in c
    assert not any(a.startswith("--robot.cameras=") for a in c)


def test_record_overrides() -> None:
    c = cmd("record", "--episodes", "3", "--task", "Stack the cubes")
    assert "--dataset.num_episodes=3" in c
    assert "--dataset.single_task=Stack the cubes" in c
    assert "--dataset.push_to_hub=false" in c
    cams = next(a for a in c if a.startswith("--robot.cameras="))
    assert "wrist: {" in cams and "/dev/v4l/by-id/" in cams


def test_port_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HFTEAM_LEADER_PORT", "/dev/ttyACM9")
    assert "--teleop.port=/dev/ttyACM9" in cmd("calibrate", "leader")


def test_cameras_arg_format() -> None:
    out = cameras_arg({"front": {"type": "opencv", "index_or_path": 0, "fps": 30}})
    assert out == '{front: {"type": "opencv", "index_or_path": 0, "fps": 30}}'


def test_port_resolved_by_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    import hfteama.config as config

    monkeypatch.setattr(config, "find_port_by_serial", lambda s: f"/dev/by-serial-{s}")
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.follower.port == f"/dev/by-serial-{cfg.follower.serial}"
    assert not any(k == "serial" for k in cfg.follower.extra)


def test_port_fallback_when_not_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    import hfteama.config as config

    monkeypatch.setattr(config, "find_port_by_serial", lambda s: None)
    assert load_config(DEFAULT_CONFIG).leader.port == "/dev/ttyACM0"


def test_unknown_pair() -> None:
    with pytest.raises(SystemExit):
        load_config(DEFAULT_CONFIG, pair="nope")
