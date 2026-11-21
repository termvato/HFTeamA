"""LeRobotDriver against a fake robot: limits, table checks, conversion, grasp threshold, park."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hfteama.arms.ik import ArmBase, Kinematics  # noqa: E402
from hfteama.arms.kinematics import BODY, DEFAULT_OFFSETS_DEG, ArmFrame  # noqa: E402
from hfteama.arms.primitives import ArmDown, ArmDriver, MotionError  # noqa: E402
from hfteama.arms.real import LeRobotDriver  # noqa: E402

pytestmark = pytest.mark.sim

MAX_STEP = 0.05


class FakeBus:
    def __init__(self, robot: FakeRobot) -> None:
        self._robot = robot

    def disable_torque(self) -> None:
        self._robot.calls.append("disable_torque")


class FakeRobot:
    """Echoes commands back as observations. `fail` simulates a dead bus."""

    def __init__(self) -> None:
        self.state = {f"{j}.pos": 0.0 for j in BODY} | {"gripper.pos": 50.0}
        self.actions: list[dict] = []
        self.calls: list[str] = []
        self.fail = False
        self.grip_floor: float | None = None  # gripper stalls here when "holding" something
        self.on_send = None
        self.bus = FakeBus(self)

    def get_observation(self) -> dict:
        if self.fail:
            raise ConnectionError("no status packet")
        return dict(self.state)

    def send_action(self, action: dict) -> dict:
        if self.fail:
            raise ConnectionError("no status packet")
        self.calls.append("send_action")
        self.actions.append(dict(action))
        self.state.update(action)
        if self.grip_floor is not None:
            self.state["gripper.pos"] = max(self.state["gripper.pos"], self.grip_floor)
        if self.on_send is not None:
            self.on_send(len(self.actions))
        return action

    def disconnect(self) -> None:
        self.calls.append("disconnect")


PARK_Q = np.array([1.0, -1.2, 1.2, 0.8, 0.0])


@pytest.fixture
def robot() -> FakeRobot:
    return FakeRobot()


@pytest.fixture
def driver(robot: FakeRobot) -> LeRobotDriver:
    return LeRobotDriver(
        arm_id="arm_a",
        robot=robot,
        frame=ArmFrame(),
        kin=Kinematics(ArmBase((0.0, 0.0, 0.0), 0.0)),
        table_z=0.0,
        park_q=PARK_Q,
        grip_holding_min_pct=15.0,
        max_step_rad=MAX_STEP,
        rate_hz=10_000.0,
        gripper_wait_s=0.0,
    )


def _as_q(action: dict) -> np.ndarray:
    return ArmFrame().to_kinematic(action)


def test_satisfies_the_driver_protocol(driver: LeRobotDriver) -> None:
    assert isinstance(driver, ArmDriver)


def test_move_joints_steps_never_exceed_max_step(driver: LeRobotDriver, robot: FakeRobot) -> None:
    start = driver.read_joints()
    goal = np.array([0.5, -0.6, 0.9, 0.3, 0.2])
    driver.move_joints(goal)
    path = [start[:5]] + [_as_q(a)[:5] for a in robot.actions]
    assert np.abs(np.diff(path, axis=0)).max() <= MAX_STEP + 1e-6
    assert path[-1] == pytest.approx(goal, abs=1e-6)


def test_rejects_limit_and_below_table_before_sending(
    driver: LeRobotDriver, robot: FakeRobot
) -> None:
    with pytest.raises(MotionError, match="joint limit: shoulder_pan"):
        driver.move_joints(np.array([3.0, 0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(MotionError, match="below table"):
        driver.move_to(np.array([0.2, 0.0, 0.005]))
    with pytest.raises(MotionError, match="unreachable"):
        driver.move_to(np.array([2.0, 0.0, 0.1]))
    assert robot.actions == []


def test_commands_are_converted_with_armframe(driver: LeRobotDriver, robot: FakeRobot) -> None:
    goal = np.array([0.3, -0.4, 0.8, 0.2, 0.1])
    driver.move_joints(goal)
    last = robot.actions[-1]
    elbow_deg = np.degrees(0.8) + DEFAULT_OFFSETS_DEG["elbow_flex"]
    assert last["elbow_flex.pos"] == pytest.approx(elbow_deg)
    assert last["shoulder_pan.pos"] == pytest.approx(np.degrees(0.3))


def test_move_to_reaches_a_reachable_point(driver: LeRobotDriver) -> None:
    target = np.array([0.2, 0.05, 0.05])
    driver.move_to(target)
    p, _ = Kinematics().fk(driver.read_joints()[:5])
    assert np.linalg.norm(p - target) < 0.005


def test_grasp_threshold(driver: LeRobotDriver, robot: FakeRobot) -> None:
    robot.grip_floor = 30.0  # fingers stall on a part
    grasp = driver.close_gripper()
    assert grasp.holding and driver.holding()
    robot.grip_floor = None
    driver.open_gripper()
    assert not driver.close_gripper().holding  # closed to ~0 %: nothing between the fingers


def test_bus_error_is_arm_down(driver: LeRobotDriver, robot: FakeRobot) -> None:
    robot.fail = True
    with pytest.raises(ArmDown, match="bus"):
        driver.read_joints()
    with pytest.raises(ArmDown):
        driver.move_joints(np.zeros(5))


def test_stop_during_a_move_raises_stopped(driver: LeRobotDriver, robot: FakeRobot) -> None:
    robot.on_send = lambda n: driver.stop() if n == 3 else None
    with pytest.raises(MotionError, match="stopped"):
        driver.move_joints(np.array([0.8, -0.8, 1.0, 0.5, 0.0]))
    assert len(robot.actions) < 20  # froze early instead of finishing the path


def test_park_moves_slowly_then_torque_off(driver: LeRobotDriver, robot: FakeRobot) -> None:
    driver.park()
    assert robot.calls[-1] == "disable_torque"
    assert _as_q(robot.actions[-1])[:5] == pytest.approx(PARK_Q, abs=1e-6)
    assert "disable_torque" not in robot.calls[:-1]


def test_set_joint_target_is_rate_limited(driver: LeRobotDriver, robot: FakeRobot) -> None:
    before = driver.read_joints()[:5]
    driver.set_joint_target(np.array([1.5, 0.0, 0.0, 0.0, 0.0]))
    after = _as_q(robot.actions[-1])[:5]
    assert np.abs(after - before).max() <= MAX_STEP + 1e-6


def test_alive_follows_successful_bus_traffic(driver: LeRobotDriver, robot: FakeRobot) -> None:
    driver.read_joints()
    t = driver.alive()
    assert t > 0
    robot.fail = True
    with pytest.raises(ArmDown):
        driver.read_joints()
    assert driver.alive() == t  # a dead bus does not look alive
