"""Hardware CLI: pairs config loading and the lerobot commands it builds."""

from pathlib import Path

import pytest

import hfteama.hardware.pairs as pairs
from hfteama.__main__ import parser
from hfteama.hardware.cli import build_command
from hfteama.hardware.pairs import DEFAULT_CONFIG, cameras_arg, load_pair, read_config


@pytest.fixture(autouse=True)
def no_usb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on which arms are plugged in (or on pyserial)."""
    monkeypatch.setattr(pairs, "find_port_by_serial", lambda s: None)


def cmd(*argv: str) -> list[str]:
    return build_command(parser().parse_args(list(argv)))


def test_every_pair_has_calibration_files() -> None:
    """A committed pair must point at committed calibration (leaders may still be pending)."""
    for name in read_config()["pairs"]:
        cfg = load_pair(DEFAULT_CONFIG, name)
        assert (cfg.follower.calibration_dir / f"{cfg.follower.id}.json").is_file(), name


def test_default_pair() -> None:
    cfg = load_pair(DEFAULT_CONFIG)
    assert cfg.pair == "rakesh"
    assert cfg.follower.calibration_dir.parts[-3:] == ("calibration", "robots", "so_follower")
    assert cfg.leader.calibration_dir.parts[-3:] == ("calibration", "teleoperators", "so_leader")
    assert (cfg.leader.calibration_dir / f"{cfg.leader.id}.json").is_file()


def test_calibrate_follower() -> None:
    c = cmd("calibrate", "follower")
    assert Path(c[0]).name == "lerobot-calibrate"
    assert "--robot.type=so101_follower" in c
    assert any(a.startswith("--robot.calibration_dir=") for a in c)
    assert not any(a.startswith("--teleop.") for a in c)


def test_calibrate_leader_of_other_pair() -> None:
    c = cmd("--pair", "hacklab", "calibrate", "leader")
    assert "--teleop.id=hacklab_leader" in c
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


def test_port_resolved_by_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pairs, "find_port_by_serial", lambda s: f"/dev/by-serial-{s}")
    cfg = load_pair(DEFAULT_CONFIG)
    assert cfg.follower.port == f"/dev/by-serial-{cfg.follower.serial}"
    assert "serial" not in cfg.follower.extra


def test_port_fallback_when_not_connected() -> None:
    assert load_pair(DEFAULT_CONFIG).leader.port == "/dev/ttyACM0"


def test_unknown_pair() -> None:
    with pytest.raises(SystemExit):
        load_pair(DEFAULT_CONFIG, pair="nope")


def test_cameras_arg_format() -> None:
    out = cameras_arg({"front": {"type": "opencv", "index_or_path": 0, "fps": 30}})
    assert out == '{front: {"type": "opencv", "index_or_path": 0, "fps": 30}}'
