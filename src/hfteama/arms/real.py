"""Real SO-101 backend: `ArmDriver` over a LeRobot `SOFollower`.

Same primitives and IK as the simulator; the differences are the ones that matter on hardware:
every command is limit- and table-checked before it is sent, moves are interpolated so no joint
steps more than `max_step_rad` per control tick, a bus error becomes `ArmDown`, and `park()` is a
slow move followed by torque off (never a torque cut with the arm raised). After `park()` the arm is
limp and the driver refuses motion until the arm is reconnected.

Commands are always built on the last *commanded* pose, never on the measured one: under gravity the
measured pose sags below the command, and re-commanding it would lower the arm and release a grip.
All bus access goes through one lock (the Feetech bus is not thread-safe, and `stop`, `alive` and
the worker's cancel run on other threads).

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
from hfteama.arms.kinematics import DEFAULT_OFFSETS_DEG, JOINTS, LIMITS_RAD, ArmFrame
from hfteama.arms.primitives import ArmDown, Grasp, MotionError, check_point

OPEN_PCT = 70.0  # gripper opening for approach and release, LeRobot 0..100
SETTLE_TOL_RAD = 0.1  # STS3215 under arm load sits a few degrees off its goal
SETTLE_TIMEOUT_S = 1.5
PARK_SPEED = 0.3
GRIP_STEP_PCT = 5.0  # gripper ramp step per tick; below LeRobot's max_relative_target (10)
GRIP_SETTLE_TIMEOUT_S = 1.5
GRIP_DONE_PCT = 2.0  # within this of the goal: reached
GRIP_STILL_PCT = 0.3  # reading changed less than this in one tick: stalled (on an object)
HEARTBEAT_POLL_S = 0.5  # an idle arm does no bus traffic, so alive() reads the bus after this long
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
        speed_scale: float = 1.0,
        settle_tol_rad: float = SETTLE_TOL_RAD,
        settle_timeout_s: float = SETTLE_TIMEOUT_S,
        limits_rad: dict | None = None,
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
        self._speed_scale = speed_scale  # global slow-down for first runs (0.3 = 30 % speed)
        self._settle_tol = settle_tol_rad
        # per-arm limits (measured on the real arm), never wider than the model's
        self._limits = dict(LIMITS_RAD)
        for name, (lo, hi) in (limits_rad or {}).items():
            mlo, mhi = LIMITS_RAD[name]
            self._limits[name] = (max(lo, mlo), min(hi, mhi))
        self._settle_timeout = settle_timeout_s
        self._bus_lock = threading.RLock()  # re-entrant: a callback inside a bus call may stop()
        self._torque_off = False
        self._stop = threading.Event()
        self._target: np.ndarray | None = None
        self._last_ok = 0.0
        self._closed = False  # last gripper command was close

    # ---- bus

    def _observe(self) -> dict:
        with self._bus_lock:
            try:
                obs = self._robot.get_observation()
            except Exception as exc:  # noqa: BLE001  (any bus failure means the arm is gone)
                raise ArmDown(f"bus: {exc}") from exc
            self._last_ok = time.time()
        return obs

    def _check(self, q: np.ndarray, ref: np.ndarray | None = None) -> None:
        """Raise MotionError("joint limit: <name>") for a joint outside this arm's limits.

        With `ref` (the pose the command starts from), a joint that is already outside may still
        move back toward its range: the arm rests on its mechanical stop slightly beyond the
        model's limit (shoulder_lift reads about -102 deg folded), and must be able to leave it.
        """
        for i, (name, value) in enumerate(zip(JOINTS, np.asarray(q, float), strict=False)):
            if name == "gripper":
                continue  # the gripper stalls on objects; its command range is fixed (0..100 %)
            lo, hi = self._limits[name]
            if lo <= value <= hi:
                continue
            if ref is not None and i < len(ref):
                r = float(ref[i])
                if (value < lo and r <= value) or (value > hi and r >= value):
                    continue  # outside, but not further out than where it starts
            raise MotionError(f"joint limit: {name}")

    def _send(self, q6: np.ndarray) -> None:
        q6 = np.asarray(q6, float)
        ref = self._target if self._target is not None else self.read_joints()
        self._check(q6, ref)
        with self._bus_lock:
            try:
                self._robot.send_action(self._frame.to_lerobot(q6, clip=False))
            except Exception as exc:  # noqa: BLE001
                raise ArmDown(f"bus: {exc}") from exc
            self._last_ok = time.time()
            self._target = q6.copy()

    def _base(self) -> np.ndarray:
        """The pose commands are built on: the last one sent, else the measured one (first move)."""
        return self._target.copy() if self._target is not None else self.read_joints()

    def _begin(self) -> None:
        """Start of every motion call. A latched stop stays latched until `resume()`."""
        if self._torque_off:
            raise MotionError("torque off (parked): reconnect the arm before moving it")
        if self._stop.is_set():
            raise MotionError("stopped")

    def resume(self) -> None:
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
        base = self._base()
        goal = base.copy()
        goal[: len(q)] = q
        self._check(goal, base)
        step = np.clip(goal - base, -self._max_step, self._max_step)
        self._send(base + step)

    def move_joints(self, q: np.ndarray, speed: float = 1.0) -> None:
        self._begin()
        q = np.asarray(q, float)
        start = self._base()
        goal = start.copy()
        goal[: len(q)] = q
        self._check(goal, start)  # a joint resting beyond its limit may stay there or move back
        step = self._max_step * min(max(speed, 0.05), 1.0) * self._speed_scale
        n = max(1, int(np.ceil(np.abs(goal - start)[: len(q)].max() / step)))
        for k in range(1, n + 1):
            self._tick()
            self._send(start + (goal - start) * (k / n))
        waited = 0.0
        while np.abs(self.read_joints()[:5] - goal[:5]).max() > self._settle_tol:
            if waited >= self._settle_timeout:
                raise MotionError("stall")
            self._tick()
            waited += self._dt

    def reachable(self, xyz: np.ndarray) -> bool:
        xyz = np.asarray(xyz, float)
        try:
            check_point(xyz, self.table_z)
        except MotionError:
            return False
        result = self._kin.solve(xyz, seed=self.read_joints()[:5], down=True)
        if not result.ok:
            return False
        try:
            self._check(result.q)
        except MotionError:
            return False
        return True

    def move_to(self, xyz: np.ndarray, speed: float = 1.0) -> None:
        self._begin()
        xyz = np.asarray(xyz, float)
        check_point(xyz, self.table_z)
        result = self._kin.solve(xyz, seed=self.read_joints()[:5], down=True)
        if not result.ok:
            raise MotionError("unreachable")
        self.move_joints(result.q, speed)  # re-checks the stop latch: a stop during IK is kept

    def _set_gripper(self, pct: float) -> None:
        """Ramp the gripper to `pct` (LeRobot 0..100), then wait until it stops moving.

        One big jump does not work on the real arm: LeRobot's `max_relative_target` (10) clips every
        command to 10 units from the present position, so a single "open to 70" opened it by 10.
        The ramp sends steps of GRIP_STEP_PCT, one per tick, like `move_joints` does for the body.
        """
        lo, hi = LIMITS_RAD["gripper"]
        q = self._base()
        start = (q[5] - lo) / (hi - lo) * 100.0
        n = max(1, int(np.ceil(abs(pct - start) / GRIP_STEP_PCT)))
        for k in range(1, n + 1):
            self._tick()
            q[5] = lo + (hi - lo) * (start + (pct - start) * k / n) / 100.0
            self._send(q)
        # settle: done when it reaches the goal, or stalls on an object (reading stops changing).
        # The goal is re-sent every tick: the ramp is faster than the gripper, so the last ramp
        # command was clipped to 10 units from where the gripper was then, short of the goal.
        waited, prev = 0.0, None
        while waited < GRIP_SETTLE_TIMEOUT_S:
            self._tick()
            self._send(q)
            waited += self._dt
            cur = self._gripper_pct()
            if abs(cur - pct) < GRIP_DONE_PCT:
                break
            if prev is not None and waited >= self._gripper_wait and abs(cur - prev) < GRIP_STILL_PCT:
                break
            prev = cur

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
        """Latch the stop and hold the last commanded pose (the grip included)."""
        self._stop.set()
        if self._target is None:
            return  # nothing was ever commanded: the arm is where its own torque holds it
        try:
            self._send(self._target)
        except (ArmDown, MotionError):
            pass  # a lost arm cannot be frozen; the caller sees ArmDown on its next call

    def park(self) -> None:
        if self._torque_off:
            return
        self._begin()
        self.move_joints(self._park_q, speed=PARK_SPEED)
        with self._bus_lock:
            try:
                self._robot.bus.disable_torque()
            except Exception as exc:  # noqa: BLE001
                raise ArmDown(f"bus: {exc}") from exc
        self._torque_off = True
        self._target = None

    def alive(self) -> float:
        """Last good bus traffic; an idle arm is polled so it is not mistaken for a lost one."""
        if time.time() - self._last_ok > HEARTBEAT_POLL_S:
            try:
                self._observe()
            except ArmDown:
                pass  # stays stale: the executor sees the heartbeat go quiet
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
        speed_scale=float(cfg.get("speed_scale", 0.3)),
        settle_tol_rad=float(np.radians(cfg.get("settle_tol_deg", 5.7))),
        limits_rad={k: tuple(np.radians(v)) for k, v in cfg.get("limits_deg", {}).items()},
    )
