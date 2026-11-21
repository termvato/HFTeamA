"""CP-SAT planner (#7): valid plans, never worse than HEFT, handles running and held steps."""

from pathlib import Path

import pytest

pytest.importorskip("ortools")

from hfteama.scheduler.check import check  # noqa: E402
from hfteama.scheduler.cpsat import cpsat  # noqa: E402
from hfteama.scheduler.heft import heft  # noqa: E402
from hfteama.scheduler.plan import Assignment  # noqa: E402
from hfteama.scheduler.problem import Problem  # noqa: E402
from hfteama.scheduler.solve import solve  # noqa: E402
from hfteama.taskgraph import load  # noqa: E402

TASKS = Path(__file__).parent.parent / "tasks"
GRAPHS = [load(TASKS / "sort" / "sort12.json"), load(TASKS / "assemble" / "assemble3.json")]


@pytest.mark.parametrize("graph", GRAPHS, ids=["sort12", "assemble3"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_valid_and_not_worse_than_heft(graph, n: int) -> None:
    p = Problem(graph, [f"arm_{i}" for i in range(n)])
    h = heft(p)
    c = cpsat(p, time_limit=3.0, hint=h)
    assert check(c, p) == []
    assert c.makespan <= h.makespan + 1e-6


def test_running_and_held() -> None:
    g = GRAPHS[0]
    p = Problem(
        g,
        ["arm_0", "arm_1"],
        now=10.0,
        done=frozenset({"pick_01", "pick_02"}),
        running={"place_02": Assignment("place_02", "arm_1", 8.0, 14.0)},
        held={"arm_0": "place_01"},
    )
    c = cpsat(p, time_limit=3.0, hint=heft(p))
    assert check(c, p) == []
    first = min((a for a in c.assignments if a.arm_id == "arm_0"), key=lambda a: a.start)
    assert first.task_id == "place_01"


def test_solve_with_cpsat_names_limit() -> None:
    plan = solve(Problem(GRAPHS[0], ["a", "b", "c"]), method="cpsat", time_limit=3.0)
    assert plan.solver == "cpsat" and plan.binding == "resource:input_bin"
    assert plan.makespan == pytest.approx(54.0)
