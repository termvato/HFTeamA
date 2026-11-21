"""Tests for the frozen interfaces and the lower bound on the demo batch."""

import json
from pathlib import Path

import pytest

from hfteama.events import Event, EventLog
from hfteama.scheduler import lower_bound
from hfteama.taskgraph import TaskGraphError, load

TASKS = Path(__file__).parent.parent / "tasks"
SORT12 = TASKS / "sort" / "sort12.json"
ASSEMBLE3 = TASKS / "assemble" / "assemble3.json"


def test_sort12_loads() -> None:
    g = load(SORT12)
    assert len(g.tasks) == 24
    assert g.tasks["place_01"].same_arm_as == "pick_01"
    order = g.topo_order()
    assert order.index("pick_01") < order.index("place_01")


@pytest.mark.parametrize(
    "n, value, binding",
    [
        (1, 120.0, "arms"),
        (2, 60.0, "arms"),
        (3, 54.0, "resource:input_bin"),
        (8, 54.0, "resource:input_bin"),
    ],
)
def test_bound_names_the_limit(n: int, value: float, binding: str) -> None:
    lb = lower_bound(load(SORT12), [f"arm_{i}" for i in range(n)])
    assert lb.value == pytest.approx(value)
    assert lb.binding == binding


@pytest.mark.parametrize(
    "n, value, binding", [(1, 93.0, "arms"), (2, 46.5, "arms"), (4, 41.0, "resource:output_tray")]
)
def test_bound_on_ordered_task(n: int, value: float, binding: str) -> None:
    lb = lower_bound(load(ASSEMBLE3), [f"arm_{i}" for i in range(n)])
    assert lb.value == pytest.approx(value)
    assert lb.binding == binding


def test_every_task_graph_loads() -> None:
    graphs = sorted(p for p in TASKS.glob("*/*.json") if not p.name.startswith("poses"))
    assert len(graphs) >= 2
    for p in graphs:
        load(p)


def test_skill_registry() -> None:
    from hfteama.arms.skills import REGISTRY, missing_skills, skill

    @skill("_test_skill")
    def _f(arm, task, poses):  # noqa: ANN001, ANN202
        return "ok"

    try:
        assert missing_skills({"_test_skill", "nope"}) == {"nope"}
        with pytest.raises(ValueError):
            skill("_test_skill")(_f)
    finally:
        REGISTRY.pop("_test_skill")


def test_cycle_rejected(tmp_path: Path) -> None:
    bad = {
        "tasks": [
            {"id": "a", "skill": "s", "deps": ["b"]},
            {"id": "b", "skill": "s", "deps": ["a"]},
        ]
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(TaskGraphError, match="cycle"):
        load(p)


def test_unknown_dep_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"tasks": [{"id": "a", "skill": "s", "deps": ["nope"]}]}))
    with pytest.raises(TaskGraphError, match="unknown dep"):
        load(p)


def test_event_log_roundtrip(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run" / "events.jsonl")
    log.write(Event(kind="task_end", arm="arm_a", task="pick_01", outcome="ok"))
    (e,) = log.read()
    assert e.kind == "task_end" and e.outcome == "ok"


def test_event_log_skips_a_line_still_being_written(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.write(Event(kind="task_start", task="a"))
    with log.path.open("a") as f:
        f.write('{"kind": "task_end", "task": "a", "outc')  # writer mid-line
    assert [e.kind for e in log.read()] == ["task_start"]
