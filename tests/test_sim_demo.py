"""scripts/sim_demo.py runs sort12 on 3 MuJoCo arms through SkillWorker."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

pytestmark = pytest.mark.sim

SCRIPT = Path(__file__).parent.parent / "scripts" / "sim_demo.py"


def _main():
    spec = importlib.util.spec_from_file_location("sim_demo", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def _ends(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r["kind"] == "task_end"]


def test_sim_demo_runs_sort12(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    assert _main()(["--speed", "8", "--events", str(events)]) == 0
    ends = _ends(events)
    assert len(ends) == 24 and all(e["outcome"] == "ok" for e in ends)
    assert len({e["arm"] for e in ends}) >= 2


def test_sim_demo_recovers_from_a_grasp_failure(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    assert _main()(["--speed", "8", "--fail", "arm_b:grasp", "--events", str(events)]) == 0
    ends = _ends(events)
    failed = [e for e in ends if e["outcome"] == "failed"]
    assert any(e["data"]["reason"] == "grasp" for e in failed)  # the injected failure was seen
    for e in failed:  # and every failed step was retried until it succeeded
        assert any(x["task"] == e["task"] and x["outcome"] == "ok" for x in ends), e["task"]


def test_measure_durations_writes_percentiles_per_skill_and_arm(tmp_path: Path) -> None:
    script = SCRIPT.with_name("measure_durations.py")
    spec = importlib.util.spec_from_file_location("measure_durations", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "durations.json"
    assert mod.main(["--cycles", "1", "--speed", "8", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["backend"] == "sim"
    for skill in ("pick", "place"):
        assert set(data["skills"][skill]) == {"arm_a", "arm_b", "arm_c"}
        for stats in data["skills"][skill].values():
            assert stats["n"] == 1 and 0 < stats["p50"] <= stats["p80"]


def test_measure_durations_real_backend_needs_arms() -> None:
    script = SCRIPT.with_name("measure_durations.py")
    spec = importlib.util.spec_from_file_location("measure_durations", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):
        mod.main(["--backend", "real"])
