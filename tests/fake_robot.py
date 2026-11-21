"""A fake LeRobot follower for tests: echoes commands back, can sag, stall the grip, or die."""

from __future__ import annotations

import time

import numpy as np

from hfteama.arms.ik import ArmBase, Kinematics
from hfteama.arms.kinematics import BODY, ArmFrame
from hfteama.arms.real import LeRobotDriver


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
MAX_STEP = 0.05


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
