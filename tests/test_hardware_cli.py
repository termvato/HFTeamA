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


def test_each_pair_records_to_its_own_dataset() -> None:
    """Two pairs recording at once must not write into the same dataset."""
    repos = [load_pair(DEFAULT_CONFIG, name).record["repo_id"] for name in read_config()["pairs"]]
    assert len(repos) == len(set(repos))
    hacklab = cmd("--pair", "hacklab", "record")
    assert "--dataset.repo_id=diegoloayza789/hfteam_hacklab_so101" in hacklab


def test_several_pairs_run_in_parallel(capsys: pytest.CaptureFixture[str]) -> None:
    from hfteama.hardware.cli import dispatch

    assert dispatch(parser().parse_args(["-n", "--pair", "rakesh,hacklab", "teleop"])) == 0
    out = capsys.readouterr().out
    assert "[rakesh] $ " in out and "--robot.id=rakesh_follower" in out
    assert "[hacklab] $ " in out and "--robot.id=vato_follower" in out


def test_several_pairs_only_for_teleop_and_record() -> None:
    from hfteama.hardware.cli import dispatch

    with pytest.raises(SystemExit):
        dispatch(parser().parse_args(["--pair", "rakesh,hacklab", "calibrate", "leader"]))


def test_parallel_output_drops_status_lines() -> None:
    import io

    from hfteama.hardware.multi import _pump

    src = io.BytesIO(b"hello\nTeleop loop time: 16.7ms (60 Hz)\x1b[1A\nshoulder_pan.pos |  1.0\n")
    out = io.StringIO()
    _pump("rakesh", src, out)
    assert out.getvalue() == "[rakesh] hello\n"


def test_bimanual_command(tmp_path: Path) -> None:
    from hfteama.hardware.bimanual import bimanual_args, load_bimanual

    bi = load_bimanual()
    args = bimanual_args(bi, with_cameras=True, root=tmp_path)
    assert "--robot.type=bi_so_follower" in args and "--teleop.type=bi_so_leader" in args
    assert any(a.startswith("--robot.left_arm_config.cameras=") for a in args)
    assert any(a.startswith("--teleop.right_arm_config.port=") for a in args)
    left = tmp_path / bi.id / "followers" / f"{bi.id}_left.json"
    assert left.resolve().name == f"{bi.left.follower.id}.json"
    right = tmp_path / bi.id / "leaders" / f"{bi.id}_right.json"
    assert right.resolve().name == f"{bi.right.leader.id}.json"


def test_bimanual_record_has_its_own_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import hfteama.hardware.bimanual as bimanual
    from hfteama.hardware.cli import build_bimanual

    monkeypatch.setattr(bimanual, "LINK_ROOT", tmp_path)
    monkeypatch.setattr(bimanual.bimanual_args, "__defaults__", (tmp_path,))
    c = build_bimanual(parser().parse_args(["bimanual", "record", "--episodes", "2"]))
    assert "--dataset.repo_id=diegoloayza789/hfteam_bimanual_so101" in c
    assert "--dataset.num_episodes=2" in c
