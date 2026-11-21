"""Live screen app (#17): builds, and the Gantt has plan and actual lanes."""

from pathlib import Path

import pytest

pytest.importorskip("dash")

from hfteama.events import EventLog  # noqa: E402
from hfteama.view.app import create_app, gantt  # noqa: E402
from hfteama.view.state import Bar, Screen  # noqa: E402


def test_app_builds(tmp_path: Path) -> None:
    app = create_app(EventLog(tmp_path / "e.jsonl"))
    assert app.layout is not None


def test_gantt_lanes_and_lines() -> None:
    sc = Screen(
        phase="running",
        now=12.0,
        arms={"a": "real", "b": "fake"},
        plan=[Bar("p1", "a", 0, 4, "plan"), Bar("p2", "b", 0, 6, "plan")],
        actual=[Bar("p1", "a", 0, 4.2, "actual", "ok"), Bar("p2", "b", 0, 12, "running")],
        finish=30.0,
    )
    fig = gantt(sc)
    names = {t.name for t in fig.data}
    assert {"plan", "done", "running"} <= names
    assert {"a · REAL", "b · FAKE"} <= {y for t in fig.data for y in t.y}
    assert len(fig.layout.shapes) == 2  # now and predicted finish
