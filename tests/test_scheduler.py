"""Planners (#6, #8), validator (#16), blocked/unblocked steps and what-if (#12)."""

from pathlib import Path

import pytest

from hfteama.scheduler.check import check
from hfteama.scheduler.greedy import greedy
from hfteama.scheduler.heft import heft
from hfteama.scheduler.plan import Assignment
from hfteama.scheduler.problem import Problem, blocked_tasks, unblocked_by
from hfteama.scheduler.solve import retime, solve, whatif
from hfteama.taskgraph import Task, TaskGraph, load

TASKS = Path(__file__).parent.parent / "tasks"
SORT12 = load(TASKS / "sort" / "sort12.json")
ASSEMBLE3 = load(TASKS / "assemble" / "assemble3.json")


def arms(n: int) -> list[str]:
    return [f"arm_{i}" for i in range(n)]


@pytest.mark.parametrize("graph", [SORT12, ASSEMBLE3], ids=["sort12", "assemble3"])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("planner", [heft, greedy], ids=["heft", "greedy"])
def test_valid_full_plans(graph: TaskGraph, n: int, planner) -> None:
    p = Problem(graph, arms(n))
    plan = planner(p)
    assert check(plan, p) == []
    assert len(plan.assignments) == len(graph.tasks)


@pytest.mark.parametrize("graph", [SORT12, ASSEMBLE3], ids=["sort12", "assemble3"])
def test_solve_respects_lower_bound_and_names_limit(graph: TaskGraph) -> None:
    for n in (1, 2, 3, 4):
        plan = solve(Problem(graph, arms(n)))
        assert plan.makespan >= plan.lower_bound - 1e-6
        assert plan.binding
    assert solve(Problem(SORT12, arms(3))).binding == "resource:input_bin"
    assert solve(Problem(ASSEMBLE3, arms(4))).binding == "order"


def test_one_arm_is_sequential_and_more_arms_help() -> None:
    one = solve(Problem(SORT12, arms(1))).makespan
    two = solve(Problem(SORT12, arms(2))).makespan
    assert one == pytest.approx(120.0)
    assert two < one


def test_arm_holding_a_part_places_it_before_anything_else() -> None:
    plan = heft(Problem(SORT12, arms(2)))
    by = {a.task_id: a for a in plan.assignments}
    for i in range(1, 13):
        pick, place = by[f"pick_{i:02d}"], by[f"place_{i:02d}"]
        assert pick.arm_id == place.arm_id
        between = [
            a
            for a in plan.assignments
            if a.arm_id == pick.arm_id and pick.end - 1e-9 < a.start < place.start - 1e-9
        ]
        assert between == []


def test_running_steps_stay_and_held_part_goes_first() -> None:
    p = Problem(
        SORT12,
        arms(2),
        now=10.0,
        done=frozenset({"pick_01", "pick_02"}),
        running={"place_02": Assignment("place_02", "arm_1", 8.0, 14.0)},
        held={"arm_0": "place_01"},
    )
    plan = heft(p)
    assert check(plan, p) == []
    first_on_0 = min((a for a in plan.assignments if a.arm_id == "arm_0"), key=lambda a: a.start)
    assert first_on_0.task_id == "place_01"
    assert Assignment("place_02", "arm_1", 8.0, 14.0) in plan.assignments


def test_blocked_and_unblocked_by_new_arm() -> None:
    g = TaskGraph(
        "reach",
        {
            "a": Task("a", "pick_place", duration=2.0),
            "far": Task("far", "pick_place", allowed_arms=("arm_9",), duration=3.0),
            "after": Task("after", "inspect", deps=("far",), duration=1.0),
        },
    )
    p = Problem(g, arms(1))
    assert blocked_tasks(p) == {"far", "after"}
    plan = solve(p)
    assert plan.blocked == ["after", "far"] and check(plan, p) == []
    assert unblocked_by(p, "arm_9") == {"far", "after"}


def test_retime_keeps_decisions_and_detects_change() -> None:
    p0 = Problem(SORT12, arms(2))
    plan = heft(p0)
    later = Problem(SORT12, arms(2), now=5.0)
    rt = retime(plan, later)
    assert rt is not None and check(rt, later) == []
    assert {(a.task_id, a.arm_id) for a in rt.assignments} == {
        (a.task_id, a.arm_id) for a in plan.assignments
    }
    assert retime(plan, Problem(SORT12, arms(1))) is None


def test_whatif_is_non_increasing() -> None:
    est = whatif(Problem(SORT12, arms(1)), (1, 2, 3, 4))
    assert set(est) == {1, 2, 3, 4}
    assert est[1] >= est[2] >= est[3] >= est[4]
    assert est[1] == pytest.approx(120.0)
