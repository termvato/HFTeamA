"""LeRobotDriver against a fake robot: limits, table checks, conversion, grasp threshold, park."""

from __future__ import annotations

import threading
import time

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
        self.sag_deg = 0.0  # observed shoulder/elbow sit this far below the commanded pose
        self.delay_s = 0.0  # time each bus call takes
        self.overlap = False  # True if two bus calls were ever in flight at once
        self._in_flight = 0
        self.bus = FakeBus(self)

    def _begin_call(self) -> None:
        if self._in_flight:
            self.overlap = True
        self._in_flight += 1
        if self.delay_s:
            time.sleep(self.delay_s)

    def _end_call(self) -> None:
        self._in_flight -= 1

    def get_observation(self) -> dict:
        if self.fail:
            raise ConnectionError("no status packet")
        self._begin_call()
        obs = dict(self.state)
        obs["shoulder_lift.pos"] -= self.sag_deg
        obs["elbow_flex.pos"] -= self.sag_deg
        self._end_call()
        return obs

    def send_action(self, action: dict) -> dict:
        if self.fail:
            raise ConnectionError("no status packet")
        self._begin_call()
        self.calls.append("send_action")
        self.actions.append(dict(action))
        self.state.update(action)
        if self.grip_floor is not None:
            self.state["gripper.pos"] = max(self.state["gripper.pos"], self.grip_floor)
        self._end_call()
        if self.on_send is not None:
            self.on_send(len(self.actions))
        return action

    def disconnect(self) -> None:
        self.calls.append("disconnect")


PARK_Q = np.array([1.0, -1.2, 1.2, 0.8, 0.0])


@pytest.fixture
def robot() -> FakeRobot:
    return FakeRobot()


def make_driver(robot: FakeRobot, **overrides: object) -> LeRobotDriver:
    kwargs: dict = dict(
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
    kwargs.update(overrides)
    return LeRobotDriver(**kwargs)


@pytest.fixture
def driver(robot: FakeRobot) -> LeRobotDriver:
    return make_driver(robot)


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


GOAL = np.array([0.3, -0.4, 0.8, 0.2, 0.1])


def test_a_stop_during_ik_is_not_forgotten(driver: LeRobotDriver, robot: FakeRobot) -> None:
    real_solve = driver._kin.solve

    def slow_solve(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        time.sleep(0.3)
        return real_solve(*args, **kwargs)

    driver._kin.solve = slow_solve
    box: dict = {}

    def run() -> None:
        try:
            driver.move_to(np.array([0.2, 0.05, 0.05]))
            box["r"] = "finished"
        except MotionError as exc:
            box["r"] = exc.reason

    th = threading.Thread(target=run)
    th.start()
    time.sleep(0.1)
    driver.stop()
    th.join(5.0)
    assert box["r"] == "stopped" and len(robot.actions) <= 1
    driver.resume()
    driver.move_to(np.array([0.2, 0.05, 0.05]))  # the latch clears on resume


def test_commands_build_on_the_last_command_not_the_sagging_measurement(
    robot: FakeRobot,
) -> None:
    robot.sag_deg = 3.0  # the arm hangs 3 deg below where it was told to be
    d = make_driver(robot)
    d.move_joints(GOAL)
    d.open_gripper()
    assert _as_q(robot.actions[-1])[:5] == pytest.approx(GOAL, abs=1e-6)


def test_the_squeeze_survives_the_lift(driver: LeRobotDriver, robot: FakeRobot) -> None:
    robot.grip_floor = 30.0  # fingers stall on the part
    assert driver.close_gripper().holding
    squeeze = robot.actions[-1]["gripper.pos"]
    n = len(robot.actions)
    driver.move_to(np.array([0.2, 0.0, 0.12]))
    assert len(robot.actions) > n
    assert all(a["gripper.pos"] == pytest.approx(squeeze) for a in robot.actions[n:])


def test_stop_while_holding_keeps_the_grip(driver: LeRobotDriver, robot: FakeRobot) -> None:
    robot.grip_floor = 30.0
    driver.close_gripper()
    squeeze = robot.actions[-1]["gripper.pos"]
    driver.stop()
    assert robot.actions[-1]["gripper.pos"] == pytest.approx(squeeze)


def test_stop_before_any_command_sends_nothing(driver: LeRobotDriver, robot: FakeRobot) -> None:
    driver.stop()
    assert robot.actions == []


def test_bus_calls_never_overlap(driver: LeRobotDriver, robot: FakeRobot) -> None:
    robot.delay_s = 0.002
    th = threading.Thread(target=lambda: driver.move_joints(GOAL))
    th.start()
    for _ in range(30):  # another thread (heartbeat, cancel) reading while the move streams
        driver.read_joints()
    th.join(5.0)
    assert not robot.overlap


def test_alive_polls_an_idle_bus(driver: LeRobotDriver) -> None:
    driver.read_joints()
    driver._last_ok -= 10.0  # the arm has been idle for 10 s
    assert time.time() - driver.alive() < 1.0


def test_a_dead_bus_is_not_made_to_look_alive(driver: LeRobotDriver, robot: FakeRobot) -> None:
    driver.read_joints()
    stale = driver._last_ok - 10.0
    driver._last_ok = stale
    robot.fail = True
    assert driver.alive() == stale


def test_after_park_motion_is_refused_not_sent(driver: LeRobotDriver, robot: FakeRobot) -> None:
    driver.park()
    n = len(robot.actions)
    with pytest.raises(MotionError, match="torque off"):
        driver.move_joints(np.zeros(5))
    with pytest.raises(MotionError, match="torque off"):
        driver.open_gripper()
    assert len(robot.actions) == n  # nothing is commanded to a limp arm


def test_speed_scale_slows_every_step(robot: FakeRobot) -> None:
    d = make_driver(robot, speed_scale=0.5)
    start = d.read_joints()
    d.move_joints(GOAL)
    path = [start[:5]] + [_as_q(a)[:5] for a in robot.actions]
    assert np.abs(np.diff(path, axis=0)).max() <= MAX_STEP * 0.5 + 1e-6


def test_settle_tolerance_is_configurable(robot: FakeRobot) -> None:
    robot.sag_deg = 3.0
    tight = make_driver(robot, settle_tol_rad=0.01, settle_timeout_s=0.05)
    with pytest.raises(MotionError, match="stall"):
        tight.move_joints(GOAL)
    make_driver(robot, settle_tol_rad=0.1).move_joints(GOAL)


def _q_with_tcp_below(z: float) -> np.ndarray:
    kin = Kinematics()
    rng = np.random.default_rng(3)
    for _ in range(2000):
        q = rng.uniform(-1.2, 1.2, 5)
        if kin.fk(q)[0][2] < z:
            return q
    raise AssertionError("no joint pose below the table found")


def test_a_joint_move_whose_path_dips_below_the_table_is_refused(
    driver: LeRobotDriver, robot: FakeRobot
) -> None:
    goal = _q_with_tcp_below(-0.03)  # the end pose itself is under the table
    with pytest.raises(MotionError, match="below table"):
        driver.move_joints(goal)
    assert robot.actions == []


def test_move_to_with_yaw_turns_the_gripper(driver: LeRobotDriver) -> None:
    driver.move_to(np.array([0.2, 0.05, 0.05]), yaw=1.0)
    err = (
        Kinematics().tool_heading(driver.read_joints()[:5]) - 1.0 + np.pi / 2
    ) % np.pi - np.pi / 2
    assert abs(err) < 0.08
