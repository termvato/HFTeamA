"""Scaling benchmark (#15), planning part only: no executor runs, so it is fast."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hfteama.taskgraph import load

SCRIPT = Path(__file__).parent.parent / "scripts" / "scaling_benchmark.py"


def _bench():
    spec = importlib.util.spec_from_file_location("scaling_benchmark", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


bench = _bench()


@pytest.mark.parametrize("name", sorted(bench.TASKS))
def test_planned_finish_falls_and_respects_the_bound(name: str) -> None:
    rows = bench.plan_rows(name, load(bench.TASKS[name]), range(1, 5))
    assert [r.n for r in rows] == [1, 2, 3, 4]
    finishes = [r.planned for r in rows]
    assert all(b <= a + 1e-9 for a, b in zip(finishes, finishes[1:], strict=False))
    assert all(r.planned >= r.bound - 1e-9 for r in rows)
    assert rows[0].planned_speedup == pytest.approx(1.0)
    assert rows[1].planned < rows[0].planned  # a second arm helps on both tasks
    assert all(r.ideal == pytest.approx(rows[0].planned / r.n) for r in rows)


def test_outputs_without_measurement(tmp_path: Path) -> None:
    rows = bench.plan_rows("sort12", load(bench.TASKS["sort12"]), range(1, 5))
    table = bench.markdown(rows)
    assert table.count("\n") == 1 + len(rows)
    bench.write_csv(rows, tmp_path / "s.csv")
    assert len((tmp_path / "s.csv").read_text().splitlines()) == 1 + len(rows)
    assert bench.binding_text("resource:input_bin") == "input bin is the bottleneck"
