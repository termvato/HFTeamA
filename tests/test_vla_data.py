"""VLA datasets: the tasks config and the lerobot-record command (no lerobot needed).

End to end with LeRobot's own recorder: tests/test_vla_recording.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hfteama.hardware.bimanual as bimanual
import hfteama.hardware.pairs as pairs
from hfteama.hardware import vla_data
from hfteama.hardware.vla_data import VlaTask, load_tasks, record_command

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FRONT = {"type": "opencv", "index_or_path": "/dev/video9", "width": 640, "height": 480, "fps": 30}


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No USB, calibration links and datasets under tmp_path."""
    monkeypatch.setattr(pairs, "find_port_by_serial", lambda s: None)
    monkeypatch.setattr(bimanual.bimanual_args, "__defaults__", (tmp_path / "links",))
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lerobot"))


def task(name: str = "box_middle") -> VlaTask:
    return load_tasks()[name]


# ---------------------------------------------------------------- config and command


def test_three_tasks_each_with_its_own_dataset() -> None:
    tasks = load_tasks()
    assert set(tasks) == {"box_right", "box_left", "box_middle"}
    assert len({t.repo_id for t in tasks.values()}) == 3
    assert len({t.task for t in tasks.values()}) == 3
    assert tasks["box_middle"].episodes == 40 and tasks["box_right"].episodes == 25
    assert all(t.private for t in tasks.values())
    assert all("front" in t.cameras for t in tasks.values())


def test_expected_cameras() -> None:
    assert vla_data.expected_cameras(task()) == {
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.front",
    }


def test_record_command_creates_then_resumes(tmp_path: Path) -> None:
    t = task()
    cmd = record_command(t, episodes=3, cameras={"front": FRONT})
    assert "--robot.type=bi_so_follower" in cmd
    assert any(a.startswith("--robot.cameras={front: ") and "/dev/video9" in a for a in cmd)
    assert f"--dataset.repo_id={t.repo_id}" in cmd
    assert f"--dataset.root={tmp_path / 'lerobot' / t.repo_id}" in cmd
    assert f"--dataset.single_task={t.task}" in cmd
    assert "--dataset.num_episodes=3" in cmd
    assert "--dataset.no_stamp=true" in cmd and "--dataset.push_to_hub=false" in cmd
    assert "--resume=true" not in cmd

    meta = tmp_path / "lerobot" / t.repo_id / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")
    assert "--resume=true" in record_command(t, cameras={"front": FRONT})
    assert f"--dataset.num_episodes={t.episodes}" in record_command(t, cameras={"front": FRONT})


def test_front_camera_found_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    import hfteama.vision.camera as camera

    found = {"C922": "/dev/video4"}
    monkeypatch.setattr(camera, "find_device", found.get)
    monkeypatch.setattr(vla_data, "stable_path", lambda d: d)
    cams = vla_data.resolve_cameras(task().cameras)
    assert cams["front"]["index_or_path"] == "/dev/video4"
    assert not {"find", "controls", "controls_note"} & set(cams["front"])

    monkeypatch.setattr(camera, "find_device", lambda name: None)
    with pytest.raises(SystemExit, match="not found"):
        vla_data.resolve_cameras(task().cameras)


def test_dry_run_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from hfteama.__main__ import main

    assert main(["-n", "vla", "box_left", "record", "--episodes", "2"]) == 0
    out = capsys.readouterr().out
    assert "lerobot-record" in out and "Move the box to the left" in out and "<C922>" in out


def test_stable_path_survives_a_replug(tmp_path: Path) -> None:
    """/dev/videoN -> its by-id link, which still points at the camera after it reconnects."""
    dev = tmp_path / "video7"
    dev.touch()
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    (by_id / "usb-046d_C922-video-index0").symlink_to(dev)
    (by_id / "usb-046d_C922-video-index1").symlink_to(tmp_path / "video8")
    assert vla_data.stable_path(str(dev), by_id) == str(by_id / "usb-046d_C922-video-index0")
    assert vla_data.stable_path(str(tmp_path / "video9"), by_id) == str(tmp_path / "video9")
