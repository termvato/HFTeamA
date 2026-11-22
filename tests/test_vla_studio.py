"""VLA studio: Sahil's Studio engine with our setup, the extra actions and the HTTP page.

No arms or cameras: the Studio is built but not connected; the dataset comes from LeRobot's own
recorder with the fake two-arm robot of test_vla_recording.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("lerobot", reason="the studio needs lerobot installed")

from hfteama.hardware import vla_data, vla_studio  # noqa: E402
from tests.test_vla_data import FRONT, isolated, task  # noqa: E402, F401
from tests.test_vla_recording import record_session  # noqa: E402


@pytest.fixture
def studio():
    return vla_studio.make_studio(task("box_right"), cameras={"front": FRONT})


def test_studio_uses_our_robot_with_all_cameras_top_level(studio) -> None:
    hw = studio.hw
    assert hw.robot.type == "bi_so_follower" and hw.teleop.type == "bi_so_leader"
    assert set(hw.robot.cameras) == {"front", "left_wrist", "right_wrist"}
    assert not hw.robot.left_arm_config.cameras and not hw.robot.right_arm_config.cameras
    assert studio.views == ["front", "left_wrist", "right_wrist"]
    t = task("box_right")
    assert studio.repo_id == t.repo_id and studio.task == t.task
    assert studio.root == vla_data.dataset_root(t)
    assert studio.state == "disconnected"


def test_ops_delete_check_upload(studio, monkeypatch: pytest.MonkeyPatch) -> None:
    t = task("box_right")
    ops = vla_studio.Ops(studio, t)
    assert ops.dataset()["target"] == t.episodes
    assert not ops.check()["ok"] and not ops.delete_last()["ok"]  # nothing recorded yet

    record_session(monkeypatch, t, episodes=3)
    info = ops.dataset()
    assert info["episodes"] == 3 and len(info["lengths_s"]) == 3
    assert ops.check() == {"ok": True, "message": "Check OK.", "problems": []}

    r = ops.delete_last()
    assert r["ok"] and "Deleted episode 2" in r["message"]
    assert ops.dataset()["episodes"] == 2

    studio.state = "recording"  # never edit while an episode is being written
    assert not ops.delete_last()["ok"] and ops.dataset()["episodes"] == 2
    studio.state = "disconnected"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    pushed = []
    monkeypatch.setattr(LeRobotDataset, "push_to_hub", lambda self, **kw: pushed.append(kw))
    studio.state = "idle"
    assert ops.upload()["ok"]
    for _ in range(100):
        if studio.state == "idle" and pushed:
            break
        time.sleep(0.05)
    assert pushed and pushed[0]["private"] is True
    assert studio.url.endswith(t.repo_id)


def test_http_page_and_actions(studio, monkeypatch: pytest.MonkeyPatch) -> None:
    t = task("box_right")
    record_session(monkeypatch, t, episodes=2)
    ops = vla_studio.Ops(studio, t)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), vla_studio.make_handler(studio, ops, vla_studio.load_engine())
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path: str) -> bytes:
        return urllib.request.urlopen(base + path, timeout=10).read()

    def post(path: str) -> dict:
        req = urllib.request.Request(base + path, data=b"{}", method="POST")
        return json.loads(urllib.request.urlopen(req, timeout=60).read())

    try:
        assert b"VLA Studio" in get("/")
        status = json.loads(get("/api/status"))
        assert [s["name"] for s in status["slots"]] == ["front", "left_wrist", "right_wrist"]
        assert json.loads(get("/api/dataset"))["episodes"] == 2
        assert post("/api/check")["ok"]
        assert post("/api/delete_last")["ok"]
        assert json.loads(get("/api/dataset"))["episodes"] == 1
        assert post("/api/save")["state"] == "disconnected"  # engine action: nothing to save
    finally:
        server.shutdown()


def test_page_exists() -> None:
    assert Path(vla_studio.PAGE).is_file()
