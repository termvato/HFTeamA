"""Arm registry and reachability (backlog #3)."""

import json
from pathlib import Path

import pytest

from hfteama.arms.base import ArmState
from hfteama.events import EventLog
from hfteama.registry import ArmNode, ArmRegistry, Reach, load_catalog, newly_unblocked, resolve
from hfteama.registry.arms import SO101_REACH
from hfteama.scheduler import lower_bound
from hfteama.taskgraph import Task, TaskGraph, load

CONFIGS = Path(__file__).parent.parent / "configs"
SORT12 = Path(__file__).parent.parent / "tasks" / "sort" / "sort12.json"

R = Reach(r_min=0.05, r_max=0.30)
ARM_A = ArmNode("arm_a", base_pose_m=(0.0, 0.0, 0.0), reach=R, state=ArmState.READY)
ARM_B = ArmNode("arm_b", base_pose_m=(0.45, 0.0, 0.0), reach=R, state=ArmState.READY)


def pair(obj: str, pick: tuple, place: tuple) -> list[Task]:
    return [
        Task(id=f"pick_{obj}", skill="pick", args={"object": obj, "pick_xyz": pick}, duration=4),
        Task(
            id=f"place_{obj}",
            skill="place",
            args={"object": obj, "place_xyz": place},
            deps=(f"pick_{obj}",),
            same_arm_as=f"pick_{obj}",
            duration=6,
        ),
    ]


def graph(*tasks: Task) -> TaskGraph:
    g = TaskGraph(name="t", tasks={t.id: t for t in tasks})
    g.validate()
    return g


def test_example_catalog_loads() -> None:
    cat = load_catalog(CONFIGS / "arms.example.json")
    assert set(cat) == {"arm_a", "arm_b", "arm_c"}
    assert cat["arm_b"].base_pose_m == (0.45, 0.0, 0.0)
    assert cat["arm_a"].reach == SO101_REACH and cat["arm_a"].base_yaw_deg == 90.0


def test_example_bins_reachable_at_full_extension() -> None:
    from hfteama.perception import load_bins

    cat = load_catalog(CONFIGS / "arms.example.json")
    for b in load_bins(CONFIGS / "bins.example.json"):
        assert any(a.reaches(b.xyz) for a in cat.values()), b.id


def test_so101_reach_sector() -> None:
    # facing +y (yaw 90): pan axis at (0, 0.0388); reach 0.44 m; hip +-110 deg
    arm = ArmNode("a", base_yaw_deg=90.0, reach=SO101_REACH)
    y0 = 0.0388
    assert arm.reaches((0.0, y0 + 0.43))
    assert not arm.reaches((0.0, y0 + 0.45)), "beyond full extension"
    assert arm.reaches((0.40, y0)), "90 deg to the side is inside +-110"
    assert not arm.reaches((0.0, y0 - 0.30)), "straight behind is outside +-110"
    assert not arm.reaches((0.20, y0 - 0.20)), "135 deg off forward"


def test_reach_config(tmp_path: Path) -> None:
    arms = [
        {"id": "x", "reach": {"r_min": 0.1, "r_max": 0.3}},
        {"id": "y", "reach": {"preset": "so101", "r_max": 0.35}},
        {"id": "z", "reach": "*"},
    ]
    p = tmp_path / "arms.json"
    p.write_text(json.dumps({"arms": arms}))
    cat = load_catalog(p)
    x = cat["x"]
    assert x.reaches((0.2, 0.0)) and not x.reaches((0.05, 0.0)) and not x.reaches((0.4, 0.0))
    assert cat["y"].reach.pan_limit_deg == SO101_REACH.pan_limit_deg
    assert cat["y"].reach.r_max == 0.35
    assert cat["z"].reaches((5.0, -5.0)), '"*" reaches everything'


def test_allowed_arms_follow_reach_and_same_arm() -> None:
    # obj1: both ends near arm_a. obj2: pick near a, place only near b -> no single arm: blocked.
    g = graph(*pair("o1", (0.1, 0.1, 0), (0.0, 0.2, 0)), *pair("o2", (-0.2, 0.0, 0), (0.6, 0, 0)))
    r = resolve(g, [ARM_A, ARM_B])
    assert r.graph.tasks["pick_o1"].allowed_arms == ("arm_a",)
    assert r.graph.tasks["place_o1"].allowed_arms == ("arm_a",)
    assert r.blocked == {"pick_o2", "place_o2"}
    assert r.unreachable == {"pick_o2", "place_o2"}
    assert set(r.runnable().tasks) == {"pick_o1", "place_o1"}


def test_dependents_of_blocked_are_blocked() -> None:
    far = Task(id="far", skill="pick", args={"pick_xyz": (2.0, 0, 0)})
    after = Task(id="after", skill="inspect", deps=("far",))
    r = resolve(graph(far, after), [ARM_A])
    assert r.blocked == {"far", "after"} and r.unreachable == {"far"}


def test_new_arm_unblocks_steps() -> None:
    g = graph(*pair("o1", (0.1, 0.0, 0), (0.2, 0, 0)), *pair("o2", (0.5, 0.1, 0), (0.6, 0, 0)))
    before = resolve(g, [ARM_A])
    after = resolve(g, [ARM_A, ARM_B])
    assert before.blocked == {"pick_o2", "place_o2"}
    assert after.blocked == frozenset()
    assert newly_unblocked(before, after) == {"pick_o2", "place_o2"}
    # the scheduler's bound runs on the result
    lb = lower_bound(after.runnable(), ["arm_a", "arm_b"])
    assert lb.value == pytest.approx(10.0)


def test_tasks_without_points_run_anywhere() -> None:
    r = resolve(load(SORT12), [ARM_A, ARM_B])
    assert not r.blocked
    assert r.graph.tasks["pick_01"].allowed_arms == ("arm_a", "arm_b")


def test_registry_events_and_heartbeat(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    seen: list[tuple[str, str]] = []
    reg = ArmRegistry(log=log, heartbeat_timeout=1.5)
    reg.subscribe(lambda kind, arm: seen.append((kind, arm)))
    reg.join(ARM_A, now=0.0)
    reg.join(ARM_B, now=0.0)
    assert reg.ready() == ["arm_a", "arm_b"]

    reg.heartbeat("arm_a", 1.0)
    assert reg.check_heartbeats(now=2.0) == ["arm_b"]
    assert reg.get("arm_b").state is ArmState.DOWN
    assert reg.ready() == ["arm_a"]
    assert reg.check_heartbeats(now=2.1) == [], "a down arm is reported once"

    reg.leave("arm_b")
    assert "arm_b" not in reg
    assert seen == [
        ("arm_added", "arm_a"),
        ("arm_added", "arm_b"),
        ("arm_failed", "arm_b"),
        ("arm_removed", "arm_b"),
    ]
    kinds = [e.kind for e in log.read()]
    assert kinds == ["arm_added", "arm_added", "arm_failed", "arm_removed"]
