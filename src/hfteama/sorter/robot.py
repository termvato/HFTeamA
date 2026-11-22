"""One SO-101 follower driven by joint targets from the IK (LeRobot Python API).

Safety:
- Never calibrates. The calibration file must already match the servos (checked read-only
  before connecting); if not, it refuses to start instead of letting LeRobot prompt
  "Press ENTER to use provided calibration file".
- Every move is interpolated in joint space at max_joint_speed_deg_s, and LeRobot's
  max_relative_target caps each command as a second guard.
- The pose at connect is "home" (the arm folded at rest, clear of the camera's view). The arm
  returns there before torque is released.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np

from hfteama.arms.kinematics import BODY, ArmFrame
from hfteama.arms.so101_ik import calibrated_limits, fk, roll_window, solve_ik


class Stopped(RuntimeError):
    """A stop was requested (ArmDriver.stop_event) while the arm was moving."""


class ArmDriver:
    # Timing of every move (measurement only, see _note): set `on_move` to a function taking
    # one dict per move, e.g. the bridge's logger. Nothing here changes a command or a wait.
    on_move = None
    _io: dict | None = None  # bus time and call counts of the move in progress (_begin)

    def __init__(self, side: str, arm_cfg: dict, motion: dict, pairs_config=None):
        from hfteama.hardware.pairs import load_pair

        self.side = side
        self.arm_cfg = arm_cfg
        self.motion = motion
        self.pair = load_pair(pairs_config, arm_cfg["pair"])
        self.arm = self.pair.follower
        self.frame = ArmFrame(
            offsets_deg={**ArmFrame().offsets_deg, **arm_cfg.get("offsets_deg", {})}
        )
        cal_file = self.arm.calibration_dir / f"{self.arm.id}.json"
        self.limits = roll_window(
            calibrated_limits(json.loads(cal_file.read_text()), self.frame),
            self.frame,
            arm_cfg.get("wrist_roll_center_deg", 0.0),
            motion.get("wrist_roll_half_range_deg", 90.0),
        )
        self.robot = None
        self.home: dict | None = None
        # set from another thread to freeze the arm: every motion loop checks it each tick and
        # raises Stopped, leaving the arm holding its last commanded pose
        self.stop_event = threading.Event()
        self.last_io = 0.0  # wall time of the last bus read/write that worked
        self.gripper_target = motion.get("gripper_open", 60.0)

    # --- connection -------------------------------------------------------------------------

    def connect(self) -> ArmDriver:
        from hfteama.hardware.multi import check_calibrations
        from hfteama.hardware.pairs import find_port_by_serial

        if self.arm.serial:
            # never fall back to the config's port: it may now be another arm
            port = find_port_by_serial(self.arm.serial)
            if port is None:
                raise SystemExit(
                    f"{self.side} arm ({self.arm.id}, serial {self.arm.serial}) is not connected."
                    " Plug it in (check with `hfteama pairs`)."
                )
            self.arm.port = port

        if check_calibrations([self.arm]):
            raise SystemExit(
                f"{self.side} arm ({self.arm.id}): calibration file and servos differ. "
                "STOP: do not recalibrate; check the arm and the calibration file."
            )
        import lerobot.robots.so_follower  # noqa: F401  (registers the arm types)
        from lerobot.robots.config import RobotConfig
        from lerobot.robots.utils import make_robot_from_config

        cfg = RobotConfig.get_choice_class(self.arm.type)(
            port=self.arm.port,
            id=self.arm.id,
            calibration_dir=self.arm.calibration_dir,
            max_relative_target=self.motion.get("max_relative_target", 10.0),
        )
        # LeRobot logs every command max_relative_target trims: a joint lagging its ramp
        # under load trips it by a fraction of a degree, harmlessly. Keep errors only.
        import logging

        logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)
        self.robot = make_robot_from_config(cfg)
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:  # belt and braces: never write a calibration
            self.robot.disconnect()
            raise SystemExit(f"{self.arm.id}: not calibrated, refusing to move it.")
        obs = self.read()
        self.home = (
            self.arm_cfg.get("home")
            or self.motion.get("home")
            or {f"{j}.pos": obs[f"{j}.pos"] for j in BODY}
        )
        self.gripper_target = obs["gripper.pos"]
        return self

    def disconnect(self, go_home: bool = True) -> None:
        if self.robot is None:
            return
        try:
            if go_home and self.home is not None:
                self.go_home()
        finally:
            self.robot.disconnect()  # releases torque: only after the arm is back at rest
            self.robot = None

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise Stopped("stopped")

    def torque_off(self) -> None:
        self.robot.bus.disable_torque()

    # --- state ------------------------------------------------------------------------------

    def read(self) -> dict:
        t = time.perf_counter()
        obs = self.robot.get_observation()
        self._count("read", time.perf_counter() - t)
        self.last_io = time.time()
        return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}

    def _send(self, action: dict) -> None:
        t = time.perf_counter()
        self.robot.send_action(action)
        self._count("send", time.perf_counter() - t)
        self.last_io = time.time()  # the bus answered: the arm is alive (heartbeat mid-motion)

    def _count(self, what: str, seconds: float) -> None:
        if self._io is None:
            self._begin()
        self._io[f"{what}_s"] += seconds
        self._io[f"{what}s"] += 1

    def _begin(self) -> float:
        """Start timing a move: zero the bus counters, return the start time."""
        self._io = {"send_s": 0.0, "sends": 0, "read_s": 0.0, "reads": 0}
        return time.perf_counter()

    def _note(self, kind: str, began: float, **data) -> None:
        """Report one finished move to `on_move` (if set): where its time went.

        ticks / planned_s: the ramp as planned (ticks at rate_hz). ramp_s: how long the ramp
        really took. late_ticks / late_max_ms: ticks that ended after their slot, and the worst
        overrun. send_ms / read_ms: mean time inside one send_action / get_observation (servo
        bus). err_deg: how far the arm was behind the goal when the ramp ended. settle_s: the
        wait for it afterwards. Ticks late with slow send/read = the bus; ticks late with fast
        send/read = Python was busy elsewhere; ticks on time but a large err_deg and settle_s =
        the arm itself lags its commands (load, heat, supply)."""
        if self.on_move is None:
            return
        io = self._io or {"send_s": 0.0, "sends": 0, "read_s": 0.0, "reads": 0}
        rec = {
            "kind": kind,
            "total_s": round(time.perf_counter() - began, 3),
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in data.items()},
            "send_ms": round(1000 * io["send_s"] / io["sends"], 2) if io["sends"] else None,
            "read_ms": round(1000 * io["read_s"] / io["reads"], 2) if io["reads"] else None,
            "sends": io["sends"],
            "reads": io["reads"],
        }
        try:
            self.on_move(rec)
        except Exception:  # noqa: BLE001  (a logging problem must never stop a move)
            pass

    def q(self) -> np.ndarray:
        """Body joint angles (rad, model convention)."""
        return self.frame.to_kinematic(self.read())[:5]

    def tcp(self) -> np.ndarray:
        """Tool point in the arm frame (m)."""
        return fk(self.q())[:3, 3]

    # --- motion -----------------------------------------------------------------------------

    def move_deg(self, goal: dict, speed: float | None = None, settle: float = 1.5) -> None:
        """Move body joints to `goal` ("<joint>.pos": LeRobot deg), gripper held, interpolated
        at `speed` deg/s; then wait until within 3 deg (or `settle` s)."""
        speed = speed or self.motion.get("max_joint_speed_deg_s", 45.0)
        rate = self.motion.get("rate_hz", 50)
        began = self._begin()
        start = self.read()
        keys = [f"{j}.pos" for j in BODY if f"{j}.pos" in goal]
        d = np.array([goal[k] - start[k] for k in keys])
        n = max(1, int(np.ceil(np.abs(d).max() / (speed / rate))))
        late_ticks, late_max = 0, 0.0
        t0 = time.perf_counter()
        for i in range(1, n + 1):
            self._check_stop()
            s = i / n
            s = s * s * (3 - 2 * s)  # smoothstep: no jerk at the ends
            act = {k: start[k] + d[m] * s for m, k in enumerate(keys)}
            act["gripper.pos"] = self.gripper_target
            self._send(act)
            late = time.perf_counter() - (t0 + i / rate)  # > 0: this tick overran its slot
            if late > 0:
                late_ticks, late_max = late_ticks + 1, max(late_max, late)
            time.sleep(max(0.0, -late))
        ramp_s = time.perf_counter() - t0
        err0 = None
        deadline = time.perf_counter() + settle
        while time.perf_counter() < deadline:
            now = self.read()
            err = max(abs(now[k] - goal[k]) for k in keys)
            err0 = err if err0 is None else err0
            if err < 3.0:
                break
            self._send({**{k: goal[k] for k in keys}, "gripper.pos": self.gripper_target})
            time.sleep(1 / rate)
        self._note(
            "move_deg", began, ticks=n, planned_s=n / rate, ramp_s=ramp_s,
            settle_s=time.perf_counter() - t0 - ramp_s, late_ticks=late_ticks,
            late_max_ms=1000 * late_max, err_deg=err0,
        )  # fmt: skip

    def move_q(self, q, speed: float | None = None) -> None:
        q = np.asarray(q, float)
        lo, hi = self.limits
        if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
            raise ValueError(f"{self.side}: joint target outside the calibrated range")
        self.move_deg(self.frame.to_lerobot(q, clip=False), speed)

    def move_line(
        self, q_from, target_to, yaw: float | None, steps: int = 8, speed=None
    ) -> np.ndarray:
        """Straight TCP line (arm frame) from the pose q_from to target_to, through IK
        waypoints seeded one from the other, driven as ONE smooth motion (no stop at each
        waypoint). Returns the final joint angles."""
        path = self.line_path(q_from, target_to, yaw, steps)
        self.move_path(path, speed)
        return path[-1]

    def line_path(self, q_from, target_to, yaw: float | None, steps: int = 8) -> list:
        """The joint waypoints of move_line (IK only, no bus: it takes ~1 s, so callers that
        hold a bus lock can solve it first)."""
        p0 = fk(q_from)[:3, 3]
        q = np.asarray(q_from, float)
        path = []
        for s in np.linspace(0, 1, steps + 1)[1:]:
            sol = solve_ik(
                p0 + (np.asarray(target_to) - p0) * s, yaw=yaw, seed=q, limits=self.limits
            )
            if not sol.ok:
                raise RuntimeError(f"{self.side}: IK failed on the way to {np.round(target_to, 3)}")
            q = sol.q
            path.append(q)
        return path

    def move_path(self, qs, speed: float | None = None, settle: float = 1.5) -> None:
        """Drive through joint waypoints qs (rad, model) in one motion: the poses are spaced
        by joint distance and timed with a single smooth start and stop, at most `speed`
        deg/s on any joint; it only waits for the arm at the end."""
        lo, hi = self.limits
        for q in qs:
            if np.any(np.asarray(q) < lo - 1e-6) or np.any(np.asarray(q) > hi + 1e-6):
                raise ValueError(f"{self.side}: joint target outside the calibrated range")
        speed = speed or self.motion.get("max_joint_speed_deg_s", 45.0)
        rate = self.motion.get("rate_hz", 50)
        keys = [f"{j}.pos" for j in BODY]
        began = self._begin()
        now = self.read()
        pts = [np.array([now[k] for k in keys])]
        pts += [np.array([self.frame.to_lerobot(q, clip=False)[k] for k in keys]) for q in qs]
        pts = np.array(pts)
        seg = np.abs(np.diff(pts, axis=0)).max(axis=1)  # joint distance of each segment
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-6:
            return
        # smoothstep peaks at 1.5x the mean speed: size the steps so that stays <= speed
        n = max(1, int(np.ceil(1.5 * total / (speed / rate))))
        late_ticks, late_max = 0, 0.0
        t0 = time.perf_counter()
        for i in range(1, n + 1):
            self._check_stop()
            u = i / n
            d = total * u * u * (3 - 2 * u)
            pos = np.array([np.interp(d, cum, pts[:, m]) for m in range(len(keys))])
            act = {k: float(pos[m]) for m, k in enumerate(keys)}
            act["gripper.pos"] = self.gripper_target
            self._send(act)
            late = time.perf_counter() - (t0 + i / rate)  # > 0: this tick overran its slot
            if late > 0:
                late_ticks, late_max = late_ticks + 1, max(late_max, late)
            time.sleep(max(0.0, -late))
        ramp_s = time.perf_counter() - t0
        goal = {k: float(pts[-1][m]) for m, k in enumerate(keys)}
        err0 = None
        deadline = time.perf_counter() + settle
        while time.perf_counter() < deadline:
            cur = self.read()
            err = max(abs(cur[k] - goal[k]) for k in keys)
            err0 = err if err0 is None else err0
            if err < 3.0:
                break
            self._send({**goal, "gripper.pos": self.gripper_target})
            time.sleep(1 / rate)
        self._note(
            "move_path", began, ticks=n, planned_s=n / rate, ramp_s=ramp_s,
            settle_s=time.perf_counter() - t0 - ramp_s, late_ticks=late_ticks,
            late_max_ms=1000 * late_max, err_deg=err0,
        )  # fmt: skip

    def gripper(self, value: float, wait: float | None = None) -> float:
        """Move the gripper (LeRobot 0 closed .. 100 open) and wait until it gets there or
        stops on an object; the arm holds still meanwhile. Returns where the gripper ended up.

        Each command is at most 0.8 x max_relative_target from the gripper's PRESENT position
        (closed loop): LeRobot clips any bigger step, and the motor is slower than a fixed
        ramp, so an open-loop ramp would run ahead of it."""
        rate = self.motion.get("rate_hz", 50)
        cap = self.motion.get("max_relative_target", 10.0) * 0.8
        timeout = wait if wait is not None else self.motion.get("gripper_timeout_s", 3.0)
        began = self._begin()
        now = self.read()
        hold = {f"{j}.pos": now[f"{j}.pos"] for j in BODY}
        g = now["gripper.pos"]
        best, since = g, time.perf_counter()  # stall detection: no progress for 0.3 s
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline and abs(value - g) > 1.0:
            self._check_stop()
            self._send({**hold, "gripper.pos": g + float(np.clip(value - g, -cap, cap))})
            time.sleep(1 / rate)
            g = self.read()["gripper.pos"]
            if abs(value - g) < abs(value - best) - 0.3:
                best, since = g, time.perf_counter()
            elif time.perf_counter() - since > 0.3:
                break  # blocked by the piece
        # hold where it stopped, squeezing a little (within the cap): the arm's next moves
        # resend this, and a far target would be clipped (and warned) on every step
        self.gripper_target = float(g + np.clip(value - g, -cap, cap))
        self._note("gripper", began, asked=float(value), reached=float(g))
        return g

    def go_upright(self) -> None:
        """Stand the arm straight up, keeping its pan and wrist roll."""
        from hfteama.arms.so101_ik import upright

        self.move_q(upright(self.q(), self.limits))

    def lift(self, dz: float | None = None) -> None:
        """Raise the tool dz (m, default motion.lift_after_drop) straight up from where it
        is, any gripper orientation; if that is out of reach, go halfway to upright."""
        from hfteama.arms.so101_ik import upright

        dz = self.motion.get("lift_after_drop", 0.05) if dz is None else dz
        q = self.q()
        sol = solve_ik(fk(q)[:3, 3] + [0.0, 0.0, dz], down=False, seed=q, limits=self.limits)
        if sol.ok:
            self.move_q(sol.q)
        else:
            self.move_q(q + 0.5 * (upright(q, self.limits) - q))

    def go_home(self) -> None:
        self.move_deg(self.home)
