"""The camera pipeline behind the demo's hooks, on a rendered table with tagged boxes (no GPU)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

pytest.importorskip("cv2")

from hfteama.events import EventLog  # noqa: E402
from hfteama.perception.live import LivePipeline  # noqa: E402
from hfteama.taskgraph import Task  # noqa: E402
from tests.test_tags import BOX_H, render  # noqa: E402

BOWL = (0.317, 0.126)
TABLE = {
    "markers": {
        "dictionary": "DICT_4X4_50",
        "size_m": 0.07,
        "reference_id": 0,
        "table_ids": [0, 2, 3],
    },
    "arms": [
        {"id": "left", "base_pose_m": [0.089, 0.42, 0.0], "base_yaw_deg": -90.0, "reach": "so101"},
        {"id": "right", "base_pose_m": [0.488, 0.44, 0.0], "base_yaw_deg": -90.0, "reach": "so101"},
    ],
    "bins": [{"id": "bowl", "label": "the bowl", "xyz": [*BOWL, 0.10], "radius_m": 0.08}],
    "tags": {
        "dictionary": "DICT_APRILTAG_36h11",
        "height_m": BOX_H,
        "only_listed": True,
        "boxes": {
            "5": {"label": "red box", "bin": "bowl"},
            "9": {"label": "blue box", "bin": "bowl"},
        },
    },
}


class NoVLM:
    def ask(self, *a, **k):  # every box is a listed tag with a bin: Qwen must never be asked
        raise AssertionError("the VLM was called")


class Snapshot:
    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.status: dict[str, str] = {}

    def snapshot(self) -> dict:
        return {"status": dict(self.status)}


@pytest.fixture
def rig(tmp_path: Path):
    table = tmp_path / "table.json"
    table.write_text(json.dumps(TABLE))
    frame = {"img": render({5: (0.20, 0.22, 30.0), 9: (0.45, 0.30, -60.0)})}
    p = LivePipeline(table, lambda: frame["img"], vlm=NoVLM(), clearance_m=0.15)
    log = EventLog(tmp_path / "events.jsonl")
    ex = Snapshot(log)
    p.attach(ex)
    return p, frame, ex, log


def test_plan_gives_the_resolved_graph_the_screen_needs(rig) -> None:
    p, _, _, _ = rig
    assert p.tags is not None and p.no_vlm
    g = p.plan()
    assert sorted(g.tasks) == ["pick_tag_05", "pick_tag_09", "place_tag_05", "place_tag_09"]
    pick = g.tasks["pick_tag_05"]
    assert pick.args["object"] == "tag_05" and pick.args["label"] == "red box"
    assert math.dist(pick.args["pick_xyz"][:2], (0.20, 0.22)) < 0.005
    assert pick.args["pick_yaw_deg"] == pytest.approx(30.0, abs=2.0)
    assert g.tasks["pick_tag_09"].args["pick_yaw_deg"] == pytest.approx(-60.0, abs=2.0)
    assert set(pick.allowed_arms) <= {"left", "right"} and pick.allowed_arms
    assert pick.durations, "travel time per arm (resolved once, not twice)"
    assert p.scene_data(g)["objects"][0]["id"] == "tag_05"


def test_rescan_follows_a_moved_box_and_writes_the_scene(rig) -> None:
    p, frame, ex, log = rig
    first = p.plan()
    assert p.rescan() is None  # nothing changed
    ex.status = {"pick_tag_05": "done", "place_tag_05": "running"}  # 5 is in an arm's gripper
    frame["img"] = render({5: (0.30, 0.20, 0.0), 9: (0.40, 0.28, 0.0)})
    g = p.rescan()
    assert g is not None
    assert g.tasks["pick_tag_05"].args == first.tasks["pick_tag_05"].args  # running work kept
    pick9 = g.tasks["pick_tag_09"].args
    assert math.dist(pick9["pick_xyz"][:2], (0.40, 0.28)) < 0.005
    assert pick9["pick_yaw_deg"] == pytest.approx(0.0, abs=2.0)
    scenes = [e for e in log.read() if e.kind == "scene"]
    assert len(scenes) == 3  # the plan's (log attached early here), then one per re-scan
    xy = {o["id"]: o["xy"] for o in scenes[-1].data["objects"]}
    assert math.dist(xy["tag_09"], (0.40, 0.28)) < 0.005


def test_tag_check_confirms_a_place_only_when_the_box_is_in_the_bin(rig) -> None:
    p, frame, _, _ = rig
    p.plan()
    verify = p.verifier()
    place = Task(id="place_tag_05", skill="place", args={"object": "tag_05", "target": "bowl"})
    ok, why = verify(place, None)
    assert not ok and why == "tag_05 seen outside bowl"
    frame["img"] = render({5: (*BOWL, 0.0), 9: (0.45, 0.30, -60.0)})
    assert verify(place, None) == (True, "tag_05 in bowl")
    frame["img"] = render({9: (0.45, 0.30, -60.0)})
    assert verify(place, None) == (False, "tag of tag_05 not seen")
    assert verify(Task(id="pick_tag_09", skill="pick"), None) == (True, "not checked")


def test_rescan_before_plan_is_refused(rig) -> None:
    with pytest.raises(RuntimeError, match="before the first plan"):
        rig[0].rescan()


def test_periodic_rescan_hands_changes_to_the_executor(tmp_path: Path) -> None:
    from hfteama.demo import periodic_rescan

    class Ex:
        finished = False
        log = EventLog(tmp_path / "events.jsonl")
        graphs: list = []

        def update_graph(self, g) -> None:
            self.graphs.append(g)

    answers = iter(["graph", None, RuntimeError("camera unplugged"), "graph2"])

    def rescan():
        a = next(answers, None)
        if isinstance(a, Exception):
            raise a
        return a

    ex = Ex()
    periodic_rescan(ex, rescan, 0.01)  # type: ignore[arg-type]
    end = time.monotonic() + 5
    while len(ex.graphs) < 2 and time.monotonic() < end:
        time.sleep(0.01)
    ex.finished = True
    assert ex.graphs[:2] == ["graph", "graph2"]
    assert any(e.data.get("reason") == "rescan_failed" for e in ex.log.read())
