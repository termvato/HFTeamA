"""Tests for TabPFN duration and failure-risk estimator with empirical and timeout fallbacks."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from hfteama.events import Event, EventLog
from hfteama.scheduler.tabpfn_costs import TabPFNCostEstimator, load_records


def _sample_events(path: Path, n: int, fail_every: int = 4) -> EventLog:
    log = EventLog(path)
    for i in range(n):
        arm = "rakesh_follower" if i % 2 == 0 else "vato_follower"
        outcome = "failed" if (i % fail_every == 0) else "ok"
        log.write(
            Event(
                kind="task_end",
                arm=arm,
                task=f"pick_{i}",
                outcome=outcome,
                data={"duration_s": 3.0 + 0.1 * i, "dist": 0.25, "reach_r": 0.22},
            )
        )
    return log


def test_empirical_fallback_below_min_samples(tmp_path: Path) -> None:
    log = _sample_events(tmp_path / "events.jsonl", n=6)
    records = load_records(log)
    est = TabPFNCostEstimator(min_samples=15)
    est.fit(records)
    preds = est.predict(
        [{"arm_id": "rakesh_follower", "dist": 0.25}, {"arm_id": "unseen", "dist": 0.3}]
    )
    assert len(preds) == 2
    assert all(p.source == "empirical" for p in preds)
    assert preds[0].q80 >= preds[0].q50
    assert preds[0].expected_cost == pytest.approx(preds[0].q80 * (1.0 + preds[0].fail_prob))


def test_default_prediction_when_no_records() -> None:
    est = TabPFNCostEstimator(min_samples=15)
    est.fit([])
    [pred] = est.predict([{"arm_id": "rakesh_follower"}], default_duration=4.5)
    assert pred.source == "default"
    assert pred.q50 == 4.5 and pred.timeout_s == 13.5


def test_mocked_tabpfn_fit_and_predict(tmp_path: Path) -> None:
    class FakeReg:
        def fit(self, _x: np.ndarray, _y: np.ndarray) -> FakeReg:
            return self

        def predict(self, x: np.ndarray, **_kwargs: object) -> np.ndarray:
            return np.column_stack([np.full(len(x), 3.2), np.full(len(x), 4.0)])

    class FakeClf:
        def fit(self, _x: np.ndarray, _y: np.ndarray) -> FakeClf:
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.column_stack([np.full(len(x), 0.8), np.full(len(x), 0.2)])

    records = load_records(_sample_events(tmp_path / "events.jsonl", n=16))
    est = TabPFNCostEstimator(min_samples=15, client_factory=lambda: (FakeReg(), FakeClf()))
    est.fit(records)
    [pred] = est.predict([{"arm_id": "rakesh_follower", "dist": 0.25}])
    assert pred.source == "tabpfn"
    assert pred.q50 == pytest.approx(3.2)
    assert pred.q80 == pytest.approx(4.0)
    assert pred.fail_prob == pytest.approx(0.2)
    assert pred.expected_cost == pytest.approx(4.8)
    assert pred.timeout_s == pytest.approx(6.0)


def test_timeout_falls_back_to_empirical(tmp_path: Path) -> None:
    class SlowReg:
        def fit(self, _x: np.ndarray, _y: np.ndarray) -> SlowReg:
            time.sleep(0.2)
            return self

    class SlowClf:
        def fit(self, _x: np.ndarray, _y: np.ndarray) -> SlowClf:
            return self

    records = load_records(_sample_events(tmp_path / "events.jsonl", n=16))
    est = TabPFNCostEstimator(
        min_samples=15, timeout_s=0.02, client_factory=lambda: (SlowReg(), SlowClf())
    )
    est.fit(records)
    [pred] = est.predict([{"arm_id": "rakesh_follower"}])
    assert pred.source == "empirical"
