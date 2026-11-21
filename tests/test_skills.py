"""Generic skills against the in-memory driver (no MuJoCo)."""

import json
from pathlib import Path

import numpy as np
import pytest

from hfteama.arms import generic_skills as gs
from hfteama.arms.poses import Poses
from hfteama.arms.primitives import MotionError, SkillFailed
from hfteama.arms.skills import REGISTRY
from hfteama.taskgraph import Task
from tests.fake_driver import FakeDriver

POSES = Poses(
    locations={"input_bin": [0.20, 0.00, 0.02]},
    per_arm={
        "arm_a": {"own_output_bin": [0.10, 0.20, 0.02]},
        "arm_b": {"own_output_bin": [0.10, -0.20, 0.02]},
    },
    objects={"block_01": [0.21, 0.01, 0.015]},
)


def _task(skill: str, **args: object) -> Task:
    return Task(id="t1", skill=skill, args=dict(args))


def _names(d: FakeDriver) -> list[str]:
    return [n for n, _ in d.calls]


def _targets(d: FakeDriver) -> list[tuple]:
    return [a[0] for n, a in d.calls if n == "move_to"]


def test_pick_call_order_and_heights() -> None:
    d = FakeDriver()
    REGISTRY["pick"](d, _task("pick", object="block_01", **{"from": "input_bin"}), POSES)
    assert _names(d) == [
        "open_gripper",
        "move_to",
        "move_to",
        "close_gripper",
        "move_to",
        "holding",
    ]
    x, y, z = POSES.point("block_01", "arm_a")
    assert _targets(d) == [
        (x, y, z + gs.APPROACH_DZ),
        (x, y, z + gs.GRASP_DZ),
        (x, y, z + gs.LIFT_DZ),
    ]


def test_pick_uses_pick_xyz_over_object() -> None:
    d = FakeDriver()
    REGISTRY["pick"](d, _task("pick", object="block_01", pick_xyz=[0.3, 0.1, 0.015]), POSES)
    assert _targets(d)[1] == (0.3, 0.1, 0.015 + gs.GRASP_DZ)


def test_place_resolves_own_bin_per_arm() -> None:
    for arm, y in (("arm_a", 0.20), ("arm_b", -0.20)):
        d = FakeDriver(arm_id=arm)
        REGISTRY["place"](d, _task("place", object="block_01", to="own_output_bin"), POSES)
        assert _targets(d)[1][:2] == (0.10, y)
        assert _names(d) == ["move_to", "move_to", "open_gripper", "move_to"]


def test_place_accepts_target_or_to_or_place_xyz() -> None:
    for args in ({"target": "input_bin"}, {"to": "input_bin"}, {"place_xyz": [0.2, 0.0, 0.02]}):
        d = FakeDriver()
        REGISTRY["place"](d, _task("place", **args), POSES)
        assert _targets(d)[1][:2] == (0.2, 0.0)


def test_pick_place_picks_then_places() -> None:
    d = FakeDriver()
    task = _task("pick_place", object="block_01", target="input_bin")
    REGISTRY["pick_place"](d, task, POSES)
    names = _names(d)
    assert names.count("open_gripper") == 2  # one before the grasp, one to release
    assert names.index("close_gripper") < len(names) - 1 - names[::-1].index("open_gripper")


def test_grasp_failure_raises_grasp() -> None:
    with pytest.raises(SkillFailed, match="grasp"):
        REGISTRY["pick"](FakeDriver(fail_grasp=True), _task("pick", object="block_01"), POSES)


def test_failed_grasp_retreats_upward_before_raising() -> None:
    # A missed grasp must not leave the arm sitting in the shared area for the next arm to hit.
    d = FakeDriver(fail_grasp=True)
    with pytest.raises(SkillFailed, match="grasp"):
        REGISTRY["pick"](d, _task("pick", object="block_01"), POSES)
    assert _names(d)[-2:] == ["open_gripper", "move_to"]
    x, y, z = POSES.point("block_01", "arm_a")
    assert _targets(d)[-1] == (x, y, z + gs.APPROACH_DZ)


def test_drop_after_lift_raises_dropped() -> None:
    with pytest.raises(SkillFailed, match="dropped"):
        REGISTRY["pick"](FakeDriver(drop_after_lift=True), _task("pick", object="block_01"), POSES)


@pytest.mark.parametrize("skill", ["pick", "place"])
def test_unknown_location_fails_cleanly(skill: str) -> None:
    d = FakeDriver()
    args = {"object": "mystery"} if skill == "pick" else {"to": "mystery"}
    with pytest.raises(SkillFailed, match="unknown location: mystery"):
        REGISTRY[skill](d, _task(skill, **args), POSES)
    assert "move_to" not in _names(d)


def test_no_location_in_args_fails_cleanly() -> None:
    with pytest.raises(SkillFailed, match="location"):
        REGISTRY["pick"](FakeDriver(), _task("pick", product="p1", step="base"), POSES)


def test_park_skill_parks() -> None:
    d = FakeDriver()
    REGISTRY["park"](d, _task("park"), POSES)
    assert _names(d) == ["park"]


def test_inspect_moves_to_named_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gs, "INSPECT_HOLD_S", 0.0)
    poses = Poses(locations={"inspect": [0.2, 0.0, 0.1]})
    d = FakeDriver()
    REGISTRY["inspect"](d, _task("inspect"), poses)
    assert _targets(d) == [(0.2, 0.0, 0.1)]


def test_registered() -> None:
    assert {"pick", "place", "pick_place", "inspect", "park"} <= set(REGISTRY)


def test_poses_load_roundtrip(tmp_path: Path) -> None:
    data = {
        "table_z": 0.0,
        "locations": {"input_bin": [0.2, 0.0, 0.02]},
        "per_arm": {"arm_a": {"own_output_bin": [0.1, 0.2, 0.02]}},
        "objects": {"block_01": [0.21, 0.01, 0.015]},
        "arms": {"arm_a": {"base": [0.0, 0.26, 0.0], "yaw": 1.5}},
    }
    p = tmp_path / "poses.json"
    p.write_text(json.dumps(data))
    poses = Poses.load(p)
    assert poses.point("own_output_bin", "arm_a") == pytest.approx([0.1, 0.2, 0.02])
    assert poses.has("block_01", "arm_b") and not poses.has("own_output_bin", "arm_b")
    assert poses.arms["arm_a"]["yaw"] == 1.5
    assert Poses.empty().arms == {} and not Poses.empty().has("x", "a")
    assert isinstance(poses.point("input_bin", "arm_a"), np.ndarray)


def test_below_table_target_is_refused_before_any_motion() -> None:
    d = FakeDriver()
    with pytest.raises(MotionError, match="below table"):
        REGISTRY["pick"](d, _task("pick", pick_xyz=[0.2, 0.0, -0.05]), POSES)
    assert d.calls == []  # the approach point is above the table, but nothing may move first


def test_unreachable_waypoint_is_refused_before_any_motion() -> None:
    d = FakeDriver(unreachable_above=0.09)  # the lift point of a block on the table is 0.115
    with pytest.raises(MotionError, match="unreachable"):
        REGISTRY["pick"](d, _task("pick", object="block_01"), POSES)
    assert d.calls == []


def test_pick_place_validates_the_place_before_picking_anything_up() -> None:
    d = FakeDriver()
    task = _task("pick_place", object="block_01", place_xyz=[0.2, 0.0, -0.05])
    with pytest.raises(MotionError, match="below table"):
        REGISTRY["pick_place"](d, task, POSES)
    assert d.calls == []


def test_motion_failure_retreats_upward_before_raising() -> None:
    d = FakeDriver(stall_at_move=2)  # stalls on the descent to the part
    with pytest.raises(MotionError, match="stall"):
        REGISTRY["pick"](d, _task("pick", object="block_01"), POSES)
    x, y, z = POSES.point("block_01", "arm_a")
    assert _targets(d)[-1] == (x, y, z + gs.APPROACH_DZ)


def test_a_stopped_motion_does_not_retreat() -> None:
    d = FakeDriver(stall_at_move=2, stall_reason="stopped")
    with pytest.raises(MotionError, match="stopped"):
        REGISTRY["pick"](d, _task("pick", object="block_01"), POSES)
    assert len(_targets(d)) == 2


def test_pick_carries_the_part_out_of_the_shared_area_before_returning() -> None:
    # The executor releases the area's lock when pick returns: the arm must already be clear of it.
    poses = Poses(
        locations={"input_bin": [0.20, 0.00, 0.02]},
        per_arm={"arm_a": {"carry": [0.05, 0.20, 0.12]}},
        objects={"block_01": [0.21, 0.01, 0.015]},
    )
    d = FakeDriver()
    REGISTRY["pick"](d, _task("pick", object="block_01"), poses)
    assert _targets(d)[-1] == (0.05, 0.20, 0.12)
    assert _names(d)[-2:] == ["holding", "move_to"]  # the grasp is re-checked before carrying


def test_unreachable_carry_pose_is_refused_before_any_motion() -> None:
    poses = Poses(
        objects={"block_01": [0.21, 0.01, 0.015]}, per_arm={"arm_a": {"carry": [0, 0, 0.5]}}
    )
    d = FakeDriver(unreachable_above=0.2)
    with pytest.raises(MotionError, match="unreachable"):
        REGISTRY["pick"](d, _task("pick", object="block_01"), poses)
    assert d.calls == []
