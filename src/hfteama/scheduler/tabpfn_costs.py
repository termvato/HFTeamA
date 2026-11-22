"""TabPFN-3.5-Plus per-arm task duration and failure-risk estimator with empirical fallback."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hfteama.events import Event, EventLog

FEATURE_COLS = ("arm_idx", "is_sim", "dist", "reach_r", "joint_delta", "servo_temp", "object_width")


@dataclass(frozen=True)
class CostPrediction:
    q50: float
    q80: float
    fail_prob: float
    expected_cost: float
    timeout_s: float
    source: str  # "tabpfn" | "empirical" | "default"


def _row_from_event(ev: Event, arm_map: dict[str, int]) -> dict[str, float] | None:
    if ev.kind != "task_end" or not ev.arm:
        return None
    d = ev.data or {}
    if "duration_s" in d:
        duration = float(d["duration_s"])
    elif "duration" in d:
        duration = float(d["duration"])
    elif "start" in d and "end" in d:
        duration = float(d["end"]) - float(d["start"])
    else:
        return None
    if duration <= 0.0:
        return None
    arm_idx = float(arm_map.setdefault(ev.arm, len(arm_map)))
    return {
        "arm_id": ev.arm,
        "arm_idx": arm_idx,
        "is_sim": float(bool(d.get("is_sim", False))),
        "dist": float(d.get("dist", 0.25)),
        "reach_r": float(d.get("reach_r", 0.22)),
        "joint_delta": float(d.get("joint_delta", 0.5)),
        "servo_temp": float(d.get("servo_temp", 35.0)),
        "object_width": float(d.get("object_width", 0.03)),
        "duration_s": duration,
        "failed": 0.0 if ev.outcome == "ok" else 1.0,
    }


def load_records(log_or_path: EventLog | str | Path) -> list[dict[str, Any]]:
    """Extract completed-task feature/target rows from an EventLog or JSONL path."""
    log = log_or_path if isinstance(log_or_path, EventLog) else EventLog(log_or_path)
    arm_map: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for ev in log.read():
        row = _row_from_event(ev, arm_map)
        if row is not None:
            rows.append(row)
    return rows


class TabPFNCostEstimator:
    """Predict q50/q80 durations and failure probability with a hard API timeout fallback."""

    def __init__(
        self,
        min_samples: int = 15,
        timeout_s: float = 1.5,
        client_factory: Callable[[], tuple[Any, Any]] | None = None,
    ) -> None:
        self.min_samples = min_samples
        self.timeout_s = timeout_s
        self._client_factory = client_factory
        self._arm_map: dict[str, int] = {}
        self._emp_by_arm: dict[str, tuple[float, float, float]] = {}
        self._emp_global: tuple[float, float, float] | None = None
        self._reg: Any = None
        self._clf: Any = None
        self._has_both_classes = False
        self._const_fail_prob = 0.0

    def _build_clients(self) -> tuple[Any, Any] | None:
        if self._client_factory is not None:
            return self._client_factory()
        if not os.environ.get("TABPFN_TOKEN"):
            return None
        try:
            from tabpfn_client import TabPFNClassifier, TabPFNRegressor
        except ImportError:
            return None
        return TabPFNRegressor(fit_mode="fit_with_cache"), TabPFNClassifier()

    def _matrix(self, rows: list[dict[str, Any]]) -> np.ndarray:
        mat = np.empty((len(rows), len(FEATURE_COLS)), dtype=float)
        for i, row in enumerate(rows):
            arm = str(row.get("arm_id", ""))
            arm_idx = float(self._arm_map.get(arm, len(self._arm_map)))
            mat[i] = [
                arm_idx,
                float(bool(row.get("is_sim", False))),
                float(row.get("dist", 0.25)),
                float(row.get("reach_r", 0.22)),
                float(row.get("joint_delta", 0.5)),
                float(row.get("servo_temp", 35.0)),
                float(row.get("object_width", 0.03)),
            ]
        return mat

    def fit(self, records: list[dict[str, Any]]) -> None:
        """Fit empirical summaries and, when enough samples exist, TabPFN models."""
        self._arm_map.clear()
        self._emp_by_arm.clear()
        self._emp_global = None
        self._reg = None
        self._clf = None
        if not records:
            return

        by_arm: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            arm = str(rec.get("arm_id", "unknown"))
            self._arm_map.setdefault(arm, len(self._arm_map))
            by_arm.setdefault(arm, []).append(rec)

        for arm, group in by_arm.items():
            durs = np.array([float(r["duration_s"]) for r in group], dtype=float)
            fails = np.array([float(r.get("failed", 0.0)) for r in group], dtype=float)
            self._emp_by_arm[arm] = (
                float(np.quantile(durs, 0.5)),
                float(np.quantile(durs, 0.8)),
                float(np.mean(fails)),
            )

        all_durs = np.array([float(r["duration_s"]) for r in records], dtype=float)
        all_fails = np.array([float(r.get("failed", 0.0)) for r in records], dtype=float)
        self._emp_global = (
            float(np.quantile(all_durs, 0.5)),
            float(np.quantile(all_durs, 0.8)),
            float(np.mean(all_fails)),
        )
        self._const_fail_prob = float(np.mean(all_fails))
        if len(records) < self.min_samples:
            return

        clients = self._build_clients()
        if clients is None:
            return
        reg, clf = clients
        x_train = self._matrix(records)
        y_dur = all_durs
        y_fail = all_fails.astype(int)
        has_both = len(np.unique(y_fail)) >= 2

        def _do_fit() -> tuple[Any, Any | None]:
            reg.fit(x_train, y_dur)
            fitted_clf = clf.fit(x_train, y_fail) if has_both else None
            return reg, fitted_clf

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_do_fit)
            try:
                self._reg, self._clf = fut.result(timeout=self.timeout_s)
                self._has_both_classes = has_both
            except (FuturesTimeoutError, Exception):  # noqa: BLE001
                self._reg = None
                self._clf = None

    def _empirical_predict(
        self, features: list[dict[str, Any]], default_duration: float
    ) -> list[CostPrediction]:
        out: list[CostPrediction] = []
        for row in features:
            arm = str(row.get("arm_id", ""))
            stats = self._emp_by_arm.get(arm) or self._emp_global
            if stats is None:
                q50 = q80 = float(default_duration)
                out.append(
                    CostPrediction(
                        q50=q50,
                        q80=q80,
                        fail_prob=0.0,
                        expected_cost=q80,
                        timeout_s=3.0 * q50,
                        source="default",
                    )
                )
            else:
                q50, q80, fp = stats
                out.append(
                    CostPrediction(
                        q50=q50,
                        q80=q80,
                        fail_prob=fp,
                        expected_cost=q80 * (1.0 + fp),
                        timeout_s=1.5 * q80,
                        source="empirical",
                    )
                )
        return out

    def predict(
        self, features: list[dict[str, Any]], default_duration: float = 5.0
    ) -> list[CostPrediction]:
        """Predict batch costs via TabPFN, falling back cleanly on timeout or error."""
        if not features:
            return []
        if self._reg is None:
            return self._empirical_predict(features, default_duration)

        x_pred = self._matrix(features)

        def _do_predict() -> tuple[np.ndarray, np.ndarray]:
            qs = np.asarray(
                self._reg.predict(x_pred, output_type="quantiles", quantiles=[0.5, 0.8]),
                dtype=float,
            )
            if self._has_both_classes and self._clf is not None:
                probs = np.asarray(self._clf.predict_proba(x_pred), dtype=float)[:, 1]
            else:
                probs = np.full(len(features), self._const_fail_prob, dtype=float)
            return qs, probs

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_do_predict)
            try:
                qs, probs = fut.result(timeout=self.timeout_s)
            except (FuturesTimeoutError, Exception):  # noqa: BLE001
                return self._empirical_predict(features, default_duration)

        q50_arr = qs[:, 0] if qs.ndim == 2 else qs[0]
        q80_arr = qs[:, 1] if qs.ndim == 2 else qs[1]
        out: list[CostPrediction] = []
        for q50, q80, fp in zip(q50_arr, q80_arr, probs, strict=True):
            q50_f = max(0.05, float(q50))
            q80_f = max(q50_f, float(q80))
            fp_f = float(np.clip(fp, 0.0, 1.0))
            out.append(
                CostPrediction(
                    q50=q50_f,
                    q80=q80_f,
                    fail_prob=fp_f,
                    expected_cost=q80_f * (1.0 + fp_f),
                    timeout_s=1.5 * q80_f,
                    source="tabpfn",
                )
            )
        return out
