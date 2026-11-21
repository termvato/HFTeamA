"""Real SO-101 backend: `ArmDriver` over a LeRobot `SOFollower`.

Same primitives and IK as the simulator; the differences are the ones that matter on hardware:
every command is limit- and table-checked before it is sent, moves are interpolated so no joint
steps more than `max_step_rad` per control tick, a bus error becomes `ArmDown`, and `park()` is a
slow move followed by torque off (never a torque cut with the arm raised).

LeRobot is imported only in `from_config`, so tests (and anything that merely imports this module)
run without it. IK needs MuJoCo (`pip install -e ".[sim]"`). Arm ports, calibration files and the
shared table frame come from Hardware (`hfteama.hardware.pairs`, `configs/real_arms.json`).

Not yet run on hardware: follow `docs/handoff/action-to-hardware.md` before the first motion.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from hfteama.arms.ik import ArmBase, Kinematics
from hfteama.arms.kinematics import DEFAULT_OFFSETS_DEG, LIMITS_RAD, ArmFrame
from hfteama.arms.primitives import ArmDown, Grasp, MotionError, check_joints, check_point

OPEN_PCT = 70.0  # gripper opening for approach and release, LeRobot 0..100
SETTLE_TOL_RAD = 0.05
SETTLE_TIMEOUT_S = 1.5
PARK_SPEED = 0.3
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "real_arms.json"


class LeRobotDriver:
    def __init__(
        self,
        arm_id: str,
        robot: Any,
        frame: ArmFrame,
        kin: Kinematics,
        table_z: float,
        park_q: np.ndarray,
        grip_holding_min_pct: float,
        max_step_rad: float = 0.05,
        rate_hz: float = 30.0,
        gripper_wait_s: float = 0.3,
    ) -> None:
        self.arm_id = arm_id
        self.table_z = table_z
        self._robot = robot
        self._frame = frame
        self._kin = kin
        self._park_q = np.asarray(park_q, float)
        self._min_pct = grip_holding_min_pct
        self._max_step = max_step_rad
        self._dt = 1.0 / rate_hz
        self._gripper_wait = gripper_wait_s
        self._stop = threading.Event()
        self._target: np.ndarray | None = None
        self._last_ok = 0.0
        self._closed = False  # last gripper command was close

    # ---- bus

    def _observe(self) -> dict:
        try:
            obs = self._robot.get_observation()
        except Exception as exc:  # noqa: BLE001  (any bus failure means the arm is gone)
            raise ArmDown(f"bus: {exc}") from exc
        self._last_ok = time.time()
        return obs

    def _send(self, q6: np.ndarray) -> None:
        q6 = np.asarray(q6, float)
        check_joints(q6)
        try:
            self._robot.send_action(self._frame.to_lerobot(q6, clip=False))
        except Exception as exc:  # noqa: BLE001
            raise ArmDown(f"bus: {exc}") from exc
        self._last_ok = time.time()
        self._target = q6.copy()

    def _begin(self) -> None:
        self._stop.clear()

    def _tick(self) -> None:
        if self._stop.is_set():
            raise MotionError("stopped")
        time.sleep(self._dt)

    # ---- ArmDriver

    def read_joints(self) -> np.ndarray:
        return self._frame.to_kinematic(self._observe())

    def set_joint_target(self, q: np.ndarray) -> None:
        self._begin()
        q = np.asarray(q, float)
        base = self._target if self._target is not None else self.read_joints()
        goal = base.copy()
        goal[: len(q)] = q
        check_joints(goal)
        step = np.clip(goal - base, -self._max_step, self._max_step)
        self._send(base + step)

    def move_joints(self, q: np.ndarray, speed: float = 1.0) -> None:
        self._begin()
        q = np.asarray(q, float)
        check_joints(q)
        start = self.read_joints()
        goal = start.copy()
        goal[: len(q)] = q
        step = self._max_step * min(max(speed, 0.05), 1.0)
        n = max(1, int(np.ceil(np.abs(goal - start)[: len(q)].max() / step)))
        for k in range(1, n + 1):
            self._tick()
            self._send(start + (goal - start) * (k / n))
        waited = 0.0
        while np.abs(self.read_joints()[:5] - goal[:5]).max() > SETTLE_TOL_RAD:
            if waited >= SETTLE_TIMEOUT_S:
                raise MotionError("stall")
            self._tick()
            waited += self._dt

    def move_to(self, xyz: np.ndarray, speed: float = 1.0) -> None:
        xyz = np.asarray(xyz, float)
        check_point(xyz, self.table_z)
        result = self._kin.solve(xyz, seed=self.read_joints()[:5], down=True)
        if not result.ok:
            raise MotionError("unreachable")
        self.move_joints(result.q, speed)

    def _set_gripper(self, pct: float) -> None:
        q = self.read_joints()
        lo, hi = LIMITS_RAD["gripper"]
        q[5] = lo + (hi - lo) * pct / 100.0
        self._send(q)
        time.sleep(self._gripper_wait)

    def _gripper_pct(self) -> float:
        return float(self._observe()["gripper.pos"])

    def open_gripper(self) -> None:
        self._begin()
        self._closed = False
        self._set_gripper(OPEN_PCT)

    def close_gripper(self) -> Grasp:
        self._begin()
        self._set_gripper(0.0)
        self._closed = True
        pct = self._gripper_pct()
        return Grasp(holding=pct >= self._min_pct, position=float(self.read_joints()[5]))

    def holding(self) -> bool:
        """Closed and stalled short of fully closed: something is between the fingers."""
        return self._closed and self._gripper_pct() >= self._min_pct

    def stop(self) -> None:
        self._stop.set()
        try:
            self._send(self.read_joints())  # freeze where it is
        except (ArmDown, MotionError):
            pass  # a lost arm cannot be frozen; the caller sees ArmDown on its next call

    def park(self) -> None:
        self._begin()
        self.move_joints(self._park_q, speed=PARK_SPEED)
        try:
            self._robot.bus.disable_torque()
        except Exception as exc:  # noqa: BLE001
            raise ArmDown(f"bus: {exc}") from exc
        self._target = None

    def alive(self) -> float:
        return self._last_ok


def from_config(arm_id: str, path: str | Path = DEFAULT_CONFIG) -> LeRobotDriver:
    """Build a driver from `configs/real_arms.json` and Hardware's pair config; connects the arm.

    Not unit-tested (needs LeRobot and a powered arm). The arm's port, id and calibration come
    from the named pair in `configs/teleop_pairs.json`.
    """
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    from hfteama.hardware.pairs import load_pair

    cfg = json.loads(Path(path).read_text())["arms"][arm_id]
    pair = load_pair(pair=cfg["pair"])
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=pair.follower.port,
            id=pair.follower.id,
            calibration_dir=pair.follower.calibration_dir,
            use_degrees=True,
            max_relative_target=float(cfg.get("max_relative_target_deg", 10.0)),
        )
    )
    robot.connect(calibrate=False)
    return LeRobotDriver(
        arm_id=arm_id,
        robot=robot,
        frame=ArmFrame(offsets_deg={**DEFAULT_OFFSETS_DEG, **cfg.get("offsets_deg", {})}),
        kin=Kinematics(ArmBase(tuple(cfg["base"]), float(cfg.get("yaw", 0.0)))),
        table_z=float(cfg.get("table_z", 0.0)),
        park_q=np.asarray(cfg["park_q"], float),
        grip_holding_min_pct=float(cfg["grip_holding_min_pct"]),
    )
