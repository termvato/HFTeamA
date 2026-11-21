"""Duration estimates (#13)."""

from pathlib import Path

from hfteama.events import Event, EventLog
from hfteama.scheduler.durations import DurationModel
from hfteama.taskgraph import Task

T = Task("pick_01", "pick", duration=4.0, durations={"b": 5.0})


def test_prior_until_enough_samples() -> None:
    m = DurationModel()
    assert m.estimate(T, "a") == 4.0 and m.estimate(T, "b") == 5.0
    m.record("a", "pick", 6.0)
    m.record("a", "pick", 7.0)
    assert m.estimate(T, "a") == 4.0
    m.record("a", "pick", 8.0)
    assert m.estimate(T, "a") == 8.0  # 80th percentile of 6, 7, 8 (nearest rank)


def test_p80_with_more_samples() -> None:
    m = DurationModel()
    for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        m.record("a", "pick", float(s))
    assert m.estimate(T, "a") == 8.0
    assert m.summary() == {"a": {"pick": 8.0}}


def test_load_from_event_log(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    for d in (3.0, 3.5, 4.0):
        log.write(Event("task_end", arm="a", outcome="ok", data={"skill": "pick", "duration": d}))
    log.write(Event("task_end", arm="a", outcome="failed", data={"skill": "pick", "duration": 99}))
    m = DurationModel()
    assert m.load(log) == 3
    assert m.estimate(T, "a") == 4.0
