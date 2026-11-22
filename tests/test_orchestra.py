"""Orchestra API on fake arms: USB plug-ins (a fake scan) show up as nodes, commands are checked."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from hfteama.clock import Clock  # noqa: E402
from hfteama.demo import build, fake_factory  # noqa: E402
from hfteama.events import EventLog  # noqa: E402
from hfteama.orchestra.api import create_app  # noqa: E402
from hfteama.orchestra.hub import StateHub  # noqa: E402
from hfteama.taskgraph import load  # noqa: E402

ROOT = Path(__file__).parent.parent
TASK = load(ROOT / "tasks" / "sort" / "hacklab6.json")
PAIRS = ROOT / "configs" / "teleop_pairs.json"
TABLE = ROOT / "configs" / "hacklab_table.json"
KNOWN = {"5B79014975": "rakesh_follower", "5A7A015861": "vato_follower"}


def until(pred, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.01)


@pytest.fixture
def rig(tmp_path: Path):
    usb: set[str] = set()  # serials "plugged in"
    clock = Clock(50.0)
    log = EventLog(tmp_path / "events.jsonl")
    ex, watcher, rig = build(
        TASK, log, KNOWN, fake_factory(clock, noise=0.0), clock=clock, scan=lambda: set(usb)
    )
    hub = StateHub(
        log,
        ex,
        watcher,
        rig,
        clock,
        "fake",
        PAIRS,
        table_path=TABLE,
        watching=True,
        task_name="hacklab6",
    )
    client = TestClient(create_app(hub, rig))
    yield usb, watcher, ex, client, rig
    ex.stop()


def plug_usb(usb: set[str], watcher, serial: str) -> None:
    usb.add(serial)
    watcher.poll_once()
    watcher.poll_once()  # debounce: two scans in a row


def unplug_usb(usb: set[str], watcher, serial: str) -> None:
    usb.discard(serial)
    watcher.poll_once()
    watcher.poll_once()


def node(state: dict, arm: str) -> dict:
    return next(n for n in state["nodes"]["items"] if n["arm_id"] == arm)


def test_config_describes_the_batch_and_the_catalog(rig) -> None:
    _, _, _, client, _ = rig
    cfg = client.get("/api/config").json()
    assert cfg["backend"] == "fake" and cfg["watching"] is True
    assert cfg["catalog"] == ["rakesh_follower", "vato_follower"]
    assert cfg["task"]["name"] == "hacklab6"
    steps = {s["id"]: s for s in cfg["task"]["steps"]}
    assert steps["place_lip_balm"]["label"] == "lip balm"  # label carried over from the pick
    assert steps["place_lip_balm"]["target"] == "bowl"
    assert cfg["commands"]["inject_failure"] is True


def test_usb_plug_ins_show_up_as_nodes_live(rig) -> None:
    usb, watcher, _, client, _ = rig
    st = client.get("/api/state").json()
    assert st["nodes"] == {**st["nodes"], "known": 2, "connected": 0, "joined": 0}
    assert node(st, "rakesh_follower")["node_state"] == "known"
    assert st["run"]["phase"] == "empty"  # nothing to start without an arm
    assert node(st, "rakesh_follower")["label"] == "Black arm, left"

    plug_usb(usb, watcher, "5B79014975")
    st = client.get("/api/state").json()
    assert st["nodes"]["joined"] == 1 and st["nodes"]["connected"] == 1
    r = node(st, "rakesh_follower")
    assert r["node_state"] == "ready" and r["joined"] and r["heartbeat_age_s"] is not None
    assert st["run"]["phase"] == "estimate" and st["run"]["estimate"] is not None

    plug_usb(usb, watcher, "5A7A015861")
    st = client.get("/api/state").json()
    assert st["nodes"]["joined"] == 2

    unplug_usb(usb, watcher, "5A7A015861")
    st = client.get("/api/state").json()
    v = node(st, "vato_follower")
    assert v["node_state"] == "lost" and v["last_reason"] == "unplugged" and not v["connected"]
    assert st["nodes"]["joined"] == 1


def test_go_runs_and_cards_show_the_running_step(rig) -> None:
    usb, watcher, ex, client, _ = rig
    assert client.post("/api/run/go").status_code == 409  # no arm yet
    plug_usb(usb, watcher, "5B79014975")
    plug_usb(usb, watcher, "5A7A015861")
    assert client.post("/api/run/go").json()["accepted"] is True
    assert client.post("/api/run/go").status_code == 409  # only once

    def busy() -> bool:
        return any(
            n["node_state"] == "busy" for n in client.get("/api/state").json()["nodes"]["items"]
        )

    until(busy)
    n = next(
        n for n in client.get("/api/state").json()["nodes"]["items"] if n["node_state"] == "busy"
    )
    assert n["task"] and n["task_estimate"] and n["task_elapsed"] is not None
    tasks = {t["id"]: t for t in client.get("/api/tasks").json()}
    assert tasks[n["task"]]["state"] == "running" and tasks[n["task"]]["arm"] == n["arm_id"]
    assert ex.wait(timeout=10)
    st = client.get("/api/state").json()
    assert st["run"]["phase"] == "finished" and st["run"]["done"] == st["run"]["total"] == 12


def test_pause_stops_new_steps_and_resume_continues(rig) -> None:
    usb, watcher, ex, client, _ = rig
    assert client.post("/api/run/pause").status_code == 409  # not running
    plug_usb(usb, watcher, "5B79014975")
    client.post("/api/run/go")
    assert client.post("/api/run/pause").json()["accepted"] is True
    assert client.get("/api/state").json()["run"]["paused"] is True
    until(lambda: not ex.snapshot()["busy"] and not ex.snapshot()["held"])
    done = ex.snapshot()["status"]
    time.sleep(0.3)  # at 50x: 15 s of run time, three steps' worth
    assert ex.snapshot()["status"] == done  # nothing new started
    assert client.post("/api/run/resume").json()["accepted"] is True
    assert ex.wait(timeout=10)


def test_arm_commands_are_checked_and_audited(rig, tmp_path: Path) -> None:
    usb, watcher, ex, client, _ = rig
    assert client.post("/api/arms/nobody/remove").status_code == 404
    assert client.post("/api/arms/rakesh_follower/remove").status_code == 409  # not in the run
    # with USB watching, the screen cannot plug in an arm whose cable is out
    r = client.post("/api/arms/rakesh_follower/plug")
    assert r.status_code == 409 and "USB serial not seen" in r.json()["message"]
    plug_usb(usb, watcher, "5B79014975")
    assert client.post("/api/arms/rakesh_follower/plug").status_code == 409  # already in
    assert client.post("/api/arms/rakesh_follower/inject-failure").json()["accepted"] is True
    assert client.post("/api/arms/rakesh_follower/remove", json={"abort": False}).json()["accepted"]
    until(lambda: "rakesh_follower" not in ex.snapshot()["arms"])
    st = client.get("/api/state").json()
    assert node(st, "rakesh_follower")["node_state"] == "connected"  # parked, cable still in
    lines = (tmp_path / "commands.jsonl").read_text().splitlines()
    assert len(lines) == 6 and '"accepted": false' in lines[0]
    assert '"accepted": true' in lines[-1]


def test_real_backend_never_offers_a_simulated_unplug(rig) -> None:
    _, watcher, ex, _, controls = rig
    hub = StateHub(ex.log, ex, watcher, controls, Clock(), "real", PAIRS)
    real = TestClient(create_app(hub, controls))
    assert real.get("/api/config").json()["commands"]["simulate_unplug"] is False
    assert real.post("/api/arms/rakesh_follower/simulate-unplug").status_code == 409


def test_token_guards_commands(rig) -> None:
    _, watcher, ex, _, controls = rig
    hub = StateHub(ex.log, ex, watcher, controls, Clock(), "fake", PAIRS)
    guarded = TestClient(create_app(hub, controls, token="s3cret"))
    assert guarded.post("/api/run/go").status_code == 403
    assert guarded.post("/api/run/go", headers={"X-Orchestra-Token": "s3cret"}).status_code == 409
    assert guarded.get("/api/state").status_code == 200  # reading needs no token


def test_stream_says_hello_then_state(rig) -> None:
    usb, watcher, _, client, _ = rig
    plug_usb(usb, watcher, "5B79014975")
    with client.stream("GET", "/api/stream") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        seen: list[str] = []
        for line in r.iter_lines():
            if line.startswith("event: "):
                seen.append(line[7:])
            if "state" in seen:
                break
    assert seen[0] == "hello" and "event" in seen and "state" in seen


def test_screen_is_served(rig) -> None:
    _, _, _, client, _ = rig
    page = client.get("/")
    assert page.status_code == 200 and "Orchestra" in page.text
    assert client.get("/js/live.js").status_code == 200


def test_state_carries_plan_against_actual_and_what_if(rig) -> None:
    usb, watcher, ex, client, _ = rig
    plug_usb(usb, watcher, "5B79014975")
    st = client.get("/api/state").json()
    sched = st["schedule"]
    assert sched["plan"] and not sched["actual"] and sched["finish"] is None  # estimate only
    assert {b["arm"] for b in sched["plan"]} == {"rakesh_follower"}
    assert all(b["end"] >= b["start"] for b in sched["plan"])
    assert st["run"]["whatif"]  # the whole batch with 1..N arms
    plug_usb(usb, watcher, "5A7A015861")
    client.post("/api/run/go")
    assert ex.wait(timeout=10)
    sched = client.get("/api/state").json()["schedule"]
    assert len(sched["actual"]) == 12 and {b["state"] for b in sched["actual"]} == {"ok"}
    assert sched["finish"] == pytest.approx(max(b["end"] for b in sched["actual"]), abs=0.5)


def test_state_carries_the_table_map(rig) -> None:
    from hfteama.demo import table_scene
    from hfteama.events import Event

    _, _, ex, client, _ = rig
    assert client.get("/api/state").json()["scene"] is None  # nothing logged: no map
    ex.log.write(Event("scene", data=table_scene(TASK, str(TABLE), list(KNOWN.values()))))
    scene = client.get("/api/state").json()["scene"]
    assert {a["id"] for a in scene["arms"]} == set(KNOWN.values())
    assert scene["bins"] and len(scene["objects"]) == 6


def test_page_has_the_one_dashboard(rig) -> None:
    _, _, _, client, _ = rig
    page = client.get("/").text
    for part in ('id="gantt"', 'id="map"', 'id="whatif"', 'id="cards"', 'id="robot-list"'):
        assert part in page, part
    for js in ("demo", "live", "schedule", "app"):
        assert client.get(f"/js/{js}.js").status_code == 200, js
