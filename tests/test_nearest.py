"""Each box to its nearest arm, real-arm step estimates, park outcome, camera API (no hardware)."""

from __future__ import annotations

import time
from pathlib import Path

from hfteama.events import EventLog
from hfteama.perception.live import prefer_nearest
from hfteama.registry import resolve
from hfteama.registry.arms import SO101_REACH, ArmNode
from hfteama.scheduler.problem import Problem
from hfteama.scheduler.solve import solve
from hfteama.taskgraph import Task, TaskGraph
from hfteama.view.state import build

ARMS = [
    ArmNode("left", base_pose_m=(0.163, 0.35, 0.0), base_yaw_deg=-90.0, reach=SO101_REACH),
    ArmNode("right", base_pose_m=(0.571, 0.35, 0.0), base_yaw_deg=-90.0, reach=SO101_REACH),
]
BOWL = [0.367, 0.12, 0.08]


def boxes(xs: list[float]) -> TaskGraph:
    tasks = {}
    for i, x in enumerate(xs):
        tasks[f"pick_{i}"] = Task(
            f"pick_{i}", "pick", {"object": f"b{i}", "pick_xyz": [x, 0.2, 0.015]}, duration=3.0
        )
        tasks[f"place_{i}"] = Task(
            f"place_{i}",
            "place",
            {"object": f"b{i}", "place_xyz": BOWL},
            deps=(f"pick_{i}",),
            resources=("bowl",),
            same_arm_as=f"pick_{i}",
            duration=3.0,
        )
    return TaskGraph(name="boxes", tasks=tasks)


def nearest_graph(xs: list[float], factor: float = 6.0, min_step: float = 15.0) -> TaskGraph:
    g = resolve(boxes(xs), ARMS, travel_s_per_m=4.0, clearance_m=0.15).graph
    return prefer_nearest(g, ARMS, min_step, factor)


def test_every_box_goes_to_its_nearest_arm() -> None:
    # 0.30 and 0.44 sit between the arms: both reach them, the plan used to balance them out
    xs = [0.05, 0.20, 0.30, 0.44, 0.58, 0.70]
    g = nearest_graph(xs)
    plan = solve(Problem(g, ["left", "right"]), method="cpsat")
    by_task = {a.task_id: a.arm_id for a in plan.assignments}
    shared = [i for i in range(len(xs)) if len(g.tasks[f"pick_{i}"].allowed_arms) == 2]
    assert shared, "some boxes must be reachable by both arms"
    for i in shared:  # where only one arm reaches, reach alone decides
        x = xs[i]
        want = "left" if abs(x - 0.163) < abs(x - 0.571) else "right"
        assert by_task[f"pick_{i}"] == want, (x, by_task)
        assert by_task[f"place_{i}"] == want  # the place stays with its pick


def test_the_far_arm_takes_over_when_the_nearest_is_gone() -> None:
    g = nearest_graph([0.30])
    assert set(g.tasks["pick_0"].allowed_arms) == {"left", "right"}  # still allowed: re-split works
    plan = solve(Problem(g, ["right"]), method="cpsat")
    assert {a.arm_id for a in plan.assignments} == {"right"}


def test_real_steps_are_never_estimated_below_the_floor() -> None:
    g = nearest_graph([0.20], factor=1.0, min_step=15.0)
    for t in g.tasks.values():
        assert t.duration >= 15.0
        assert all(d >= 15.0 for d in t.durations.values())
    # off by default: the estimates are what resolve gave
    plain = resolve(boxes([0.20]), ARMS, travel_s_per_m=4.0).graph
    same = prefer_nearest(plain, ARMS)
    assert same.tasks["pick_0"].durations == plain.tasks["pick_0"].durations


def test_park_outcome_is_on_the_screen(tmp_path: Path) -> None:
    from hfteama.events import Event

    log = EventLog(tmp_path / "events.jsonl")
    log.write(Event("arm_added", t=1.0, arm="a", data={"backend": "real"}))
    log.write(Event("arm_removed", t=2.0, arm="a"))
    log.write(Event("arm_parked", t=3.0, arm="a", data={"ok": False, "error": "bus: timeout"}))
    feed = build(log.read()).feed
    assert feed[0] == "Arm a did NOT park: bus: timeout"
    assert feed[1].startswith("Arm a removed: going home")


def test_executor_logs_the_park(tmp_path: Path) -> None:
    from hfteama.executor import Executor
    from tests.test_executor import arm, make

    ex, log, clock = make(tmp_path)
    ex.add_arm(arm("a", clock))
    ex.add_arm(arm("b", clock))
    ex.start()
    ex.remove_arm("b", abort=True)
    assert ex.wait(10)
    deadline = time.time() + 5
    while time.time() < deadline:
        parked = [e for e in log.read() if e.kind == "arm_parked"]
        if parked:
            break
        time.sleep(0.05)
    assert parked and parked[0].arm == "b" and parked[0].data["ok"] is True
    assert isinstance(ex, Executor)
