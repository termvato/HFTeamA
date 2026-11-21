"""SimTable + MujocoDriver: real MuJoCo arms moved through IK, block carried by option B."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hfteama.arms import generic_skills  # noqa: E402, F401  (registers the skills)
from hfteama.arms.ik import ArmBase, Kinematics  # noqa: E402
from hfteama.arms.poses import Poses  # noqa: E402
from hfteama.arms.primitives import ArmDown, MotionError  # noqa: E402
from hfteama.arms.sim import SimTable  # noqa: E402
from hfteama.arms.skills import REGISTRY  # noqa: E402
from hfteama.taskgraph import Task  # noqa: E402

pytestmark = pytest.mark.sim

POSES = Poses(
    locations={"input_bin": [0.20, 0.00, 0.02], "out": [0.15, 0.15, 0.02]},
    objects={"block_01": [0.20, 0.00, 0.015]},
    arms={
        "arm_a": {"base": [0.0, 0.0, 0.0], "yaw": 0.0},
        "arm_b": {"base": [0.0, 0.6, 0.0], "yaw": 0.0},
        "arm_c": {"base": [0.0, -0.6, 0.0], "yaw": 0.0},
    },
)
SPEED = 4.0


@pytest.fixture
def table() -> Iterator[SimTable]:
    with SimTable(POSES, speed=SPEED) as t:
        yield t


def _tcp(table: SimTable, arm_id: str) -> np.ndarray:
    cfg = POSES.arms[arm_id]
    kin = Kinematics(ArmBase(tuple(cfg["base"]), cfg["yaw"]))
    return kin.fk(table.driver(arm_id).read_joints()[:5])[0]


def test_move_to_reaches_the_point(table: SimTable) -> None:
    target = np.array([0.2, 0.05, 0.05])
    table.driver("arm_a").move_to(target)
    assert np.linalg.norm(_tcp(table, "arm_a") - target) < 0.005


def test_move_to_rejects_unreachable_and_below_table(table: SimTable) -> None:
    d = table.driver("arm_a")
    before = d.read_joints()
    with pytest.raises(MotionError, match="unreachable"):
        d.move_to(np.array([2.0, 0.0, 0.1]))
    with pytest.raises(MotionError, match="below table"):
        d.move_to(np.array([0.2, 0.0, 0.005]))
    assert d.read_joints() == pytest.approx(before, abs=1e-3)


def test_pick_and_place_moves_the_block(table: SimTable) -> None:
    d = table.driver("arm_a")
    pick = Task(id="p", skill="pick", args={"object": "block_01", "from": "input_bin"})
    place = Task(id="q", skill="place", args={"to": "out"})
    REGISTRY["pick"](d, pick, POSES)
    assert d.holding()
    REGISTRY["place"](d, place, POSES)
    assert not d.holding()
    time.sleep(0.3)  # let the released block settle
    x, y, z = table.object_pos("block_01")
    assert (x, y) == pytest.approx((0.15, 0.15), abs=0.03)
    assert z < 0.05


def test_injected_grasp_failure(table: SimTable) -> None:
    d = table.driver("arm_a")
    table.inject("arm_a", "grasp")
    d.open_gripper()
    d.move_to(np.array([0.2, 0.0, 0.095]))
    d.move_to(np.array([0.2, 0.0, 0.015]))
    assert not d.close_gripper().holding
    assert not d.holding()


def test_injected_down_raises_arm_down_and_clears(table: SimTable) -> None:
    d = table.driver("arm_a")
    table.inject("arm_a", "down")
    with pytest.raises(ArmDown):
        d.move_to(np.array([0.2, 0.0, 0.05]))
    table.inject("arm_a", "clear")
    d.move_to(np.array([0.2, 0.0, 0.05]))


def test_three_arms_move_in_parallel(table: SimTable) -> None:
    def go(arm: str, y: float) -> None:
        table.driver(arm).move_to(np.array([0.2, y, 0.05]))

    t0 = table.sim_time()
    go("arm_a", 0.0)
    single = table.sim_time() - t0
    for arm in ("arm_a", "arm_b", "arm_c"):  # back to a known start for a fair comparison
        table.driver(arm).move_joints(np.zeros(5))
    t0 = table.sim_time()
    threads = [
        threading.Thread(target=go, args=(a, y))
        for a, y in (("arm_a", 0.0), ("arm_b", 0.6), ("arm_c", -0.6))
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert table.sim_time() - t0 < 1.6 * single


def test_set_joint_target_is_nonblocking_and_converges(table: SimTable) -> None:
    d = table.driver("arm_a")
    goal = np.array([0.4, -0.3, 0.5, 0.2, 0.0])
    start = time.time()
    d.set_joint_target(goal)
    assert time.time() - start < 0.05
    end = time.time() + 5.0
    while np.abs(d.read_joints()[:5] - goal).max() > 0.03:
        assert time.time() < end, "did not converge"
        d.set_joint_target(goal)
        time.sleep(0.02)


def test_reset_restores_objects(table: SimTable) -> None:
    d = table.driver("arm_a")
    REGISTRY["pick"](d, Task(id="p", skill="pick", args={"object": "block_01"}), POSES)
    assert np.linalg.norm(table.object_pos("block_01") - [0.2, 0.0, 0.015]) > 0.05
    table.reset()
    assert table.object_pos("block_01") == pytest.approx([0.2, 0.0, 0.015], abs=0.002)
    assert not d.holding()
