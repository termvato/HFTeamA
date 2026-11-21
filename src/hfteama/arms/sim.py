"""Simulated arms: one MuJoCo scene with N SO-101s on a table, driven through `ArmDriver`.

`SimTable` builds the scene (floor, the arms at their base poses, the objects from `Poses`) and runs
physics in a background thread, paced at `speed` x real time. `MujocoDriver` implements the
primitives for one arm: moves are IK + interpolated joint targets for the model's position
actuators, so motion and timing are physical.

Grasping is option B (design doc): closing the gripper with an object within `GRASP_RADIUS_M` of
the tool centre attaches the object, which is then carried kinematically until the gripper opens.
Objects collide with the floor and each other but not with the arms. Contact grasping (option A)
is a follow-up on the same scene.

All driver waits are in simulated time, so tests can run at `speed` > 1 and durations measured on
`SimTable.sim_time()` are comparable to a real arm's wall-clock durations.
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Any, Literal

import mujoco
import numpy as np

from hfteama.arms.ik import ArmBase, Kinematics
from hfteama.arms.kinematics import JOINTS
from hfteama.arms.model import ARM_JOINTS, MODEL_XML, TCP_SITE
from hfteama.arms.poses import Poses
from hfteama.arms.primitives import (
    ArmDown,
    Grasp,
    MotionError,
    check_joints,
    check_point,
)

GRIPPER_OPEN = 1.2  # rad
GRIPPER_CLOSED = -0.1  # rad
GRASP_RADIUS_M = 0.03
JOINT_SPEED = 1.5  # rad/s at speed=1.0
TICK_S = 0.02  # command period, simulated seconds (50 Hz)
GRIPPER_SETTLE_S = 0.4
SETTLE_TIMEOUT_S = 3.0
SETTLE_TOL_RAD = 0.02
MAX_TARGET_STEP_RAD = 0.05  # per set_joint_target call
OBJECT_HALF_M = 0.015
# Folded sideways rest/park pose (pan, lift, elbow, wrist flex, wrist roll). The zero pose stretches
# every arm toward the bin, so arms that face each other interpenetrate; this one keeps all
# arm geometry >= 0.2 m from the bin centre with no self or arm-arm contact (checked for the sort
# layout in tasks/sort/poses.json).
REST_Q = np.array([1.85, -1.6, 1.6, 1.0, 0.0])
PARK_Q = REST_Q

# MuJoCo collision bits: the model's arm geoms use bit 1. Objects use bit 2 and the floor both, so
# objects rest on the floor and bump each other but never touch an arm (option B carries them).
_OBJECT_BITS = 2
_FLOOR_BITS = 3

Injection = Literal["grasp", "down", "slow", "clear"]


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


class SimTable:
    def __init__(
        self,
        poses: Poses,
        arm_ids: list[str] | None = None,
        speed: float = 1.0,
        viewer: bool = False,
    ) -> None:
        self.poses = poses
        self.speed = speed
        self.arm_ids = list(arm_ids or poses.arms or ["arm_a", "arm_b", "arm_c"])
        self._want_viewer = viewer
        self._viewer: Any = None
        self._lock = threading.Lock()
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_step_wall = time.time()
        self._attached: dict[str, str] = {}  # arm_id -> object name
        self._build()
        self._drivers = {
            a: MujocoDriver(self, a, self._base(a, i)) for i, a in enumerate(self.arm_ids)
        }
        self.reset()

    # ---- scene

    def _base(self, arm_id: str, index: int) -> ArmBase:
        cfg = self.poses.arms.get(arm_id)
        if cfg is None:
            return ArmBase((0.45 * index, 0.0, self.poses.table_z), yaw=np.pi / 2)
        return ArmBase(tuple(cfg["base"]), float(cfg.get("yaw", 0.0)))

    def _build(self) -> None:
        spec = mujoco.MjSpec()
        spec.option.timestep = 0.0025  # same numerics as the vendored model
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.iterations = 20
        spec.option.ls_iterations = 100
        spec.option.impratio = 100
        spec.option.tolerance = 1e-6
        world = spec.worldbody
        world.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
        world.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[3, 3, 0.1],
            pos=[0, 0, self.poses.table_z],
            rgba=[0.55, 0.55, 0.6, 1],
            contype=_FLOOR_BITS,
            conaffinity=_FLOOR_BITS,
        )
        for i, arm_id in enumerate(self.arm_ids):
            base = self._base(arm_id, i)
            child = mujoco.MjSpec.from_file(str(MODEL_XML))
            frame = world.add_frame(pos=list(base.xyz), quat=_yaw_quat(base.yaw))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                frame.attach_body(child.worldbody.first_body(), f"{arm_id}/", "")
        for name, xyz in self.poses.objects.items():
            body = world.add_body(name=name, pos=list(xyz))
            body.add_freejoint()
            body.add_geom(
                name=f"{name}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[OBJECT_HALF_M] * 3,
                rgba=[0.9, 0.3, 0.2, 1],
                friction=[1.0, 0.03, 0.003],
                contype=_OBJECT_BITS,
                conaffinity=_OBJECT_BITS,
            )
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)

    def _jnt(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def reset(self) -> None:
        """Objects back at their poses, arms at the rest pose, no grasps."""
        with self._lock:
            m, d = self._model, self._data
            for arm_id in self.arm_ids:
                for j, v in zip(JOINTS, (*REST_Q, GRIPPER_CLOSED), strict=True):
                    d.qpos[m.jnt_qposadr[self._jnt(f"{arm_id}/{j}")]] = v
            for name, xyz in self.poses.objects.items():
                jid = m.body(name).jntadr[0]
                qadr, dadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
                d.qpos[qadr : qadr + 7] = [*xyz, 1.0, 0.0, 0.0, 0.0]
                d.qvel[dadr : dadr + 6] = 0.0
            d.qvel[:] = 0.0
            self._attached.clear()
            mujoco.mj_forward(m, d)
            for drv in self._drivers.values():
                drv._sync_target()

    # ---- physics thread

    def __enter__(self) -> SimTable:
        if self._want_viewer:
            from mujoco import viewer

            self._viewer = viewer.launch_passive(self._model, self._data)
        self._quit.clear()
        self._thread = threading.Thread(target=self._run, name="sim-physics", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._quit.set()
        if self._thread is not None:
            self._thread.join(2.0)
        if self._viewer is not None:
            self._viewer.close()

    def _run(self) -> None:
        wall0, sim0 = time.perf_counter(), self._data.time
        while not self._quit.is_set():
            target = sim0 + (time.perf_counter() - wall0) * self.speed
            with self._lock:
                if self._data.time < target - 0.5:  # fell behind: re-anchor, don't spiral
                    wall0, sim0 = time.perf_counter(), self._data.time
                    target = sim0
                steps = 0
                while self._data.time < target and steps < 200:
                    mujoco.mj_step(self._model, self._data)
                    self._carry()
                    steps += 1
                self._last_step_wall = time.time()
                if self._viewer is not None and self._viewer.is_running():
                    self._viewer.sync()
            time.sleep(0.002)

    def sim_time(self) -> float:
        """Simulated seconds since the scene was built."""
        return float(self._data.time)

    def wait(self, seconds: float, check: Any = None) -> None:
        """Block for `seconds` of simulated time, calling `check()` every millisecond."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("SimTable is not running; use `with SimTable(...) as table:`")
        end = self.sim_time() + seconds
        while self.sim_time() < end:
            if check is not None:
                check()
            time.sleep(0.001)

    def last_step_wall(self) -> float:
        return self._last_step_wall

    # ---- access

    def driver(self, arm_id: str) -> MujocoDriver:
        return self._drivers[arm_id]

    def object_pos(self, name: str) -> np.ndarray:
        with self._lock:
            return self._data.xpos[self._model.body(name).id].copy()

    def inject(self, arm_id: str, kind: Injection) -> None:
        """Failure injection: grasp (next close misses), down (arm lost), slow, or clear all."""
        self._drivers[arm_id]._inject(kind)

    # ---- grasp (option B)

    def _tcp(self, arm_id: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, f"{arm_id}/{TCP_SITE}")
        return self._data.site_xpos[sid].copy()

    def _carry(self) -> None:  # called with the lock held, after every physics step
        for arm_id, obj in self._attached.items():
            jid = self._model.body(obj).jntadr[0]
            qadr, dadr = self._model.jnt_qposadr[jid], self._model.jnt_dofadr[jid]
            self._data.qpos[qadr : qadr + 3] = self._tcp(arm_id)
            self._data.qvel[dadr : dadr + 6] = 0.0

    def _attach_nearest(self, arm_id: str) -> str | None:
        with self._lock:
            tcp = self._tcp(arm_id)
            free = [o for o in self.poses.objects if o not in self._attached.values()]
            dist = {
                o: float(np.linalg.norm(self._data.xpos[self._model.body(o).id] - tcp))
                for o in free
            }
            near = [o for o in free if dist[o] <= GRASP_RADIUS_M]
            if not near:
                return None
            obj = min(near, key=lambda o: dist[o])
            self._attached[arm_id] = obj
            return obj

    def _release(self, arm_id: str) -> None:
        with self._lock:
            self._attached.pop(arm_id, None)

    def attached(self, arm_id: str) -> str | None:
        with self._lock:
            return self._attached.get(arm_id)


class MujocoDriver:
    """`ArmDriver` for one arm of a `SimTable`."""

    def __init__(self, table: SimTable, arm_id: str, base: ArmBase) -> None:
        self.arm_id = arm_id
        self.table_z = table.poses.table_z
        self._table = table
        m = table._model
        names = [f"{arm_id}/{j}" for j in JOINTS]
        self._qadr = np.array([m.jnt_qposadr[table._jnt(n)] for n in names])
        self._act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names])
        self._kin = Kinematics(base)
        self._target = np.zeros(6)
        self._stop = threading.Event()
        self._down = False
        self._down_at = 0.0
        self._fail_grasp = False
        self._slow = 1.0

    # ---- plumbing

    def _sync_target(self) -> None:  # called with the table lock held
        self._target = self._table._data.qpos[self._qadr].copy()
        self._table._data.ctrl[self._act] = self._target
        self._down, self._fail_grasp, self._slow = False, False, 1.0
        self._stop.clear()

    def _command(self, q6: np.ndarray) -> None:
        self._target = np.asarray(q6, float).copy()
        with self._table._lock:
            self._table._data.ctrl[self._act] = self._target

    def _check_alive(self) -> None:
        if self._down:
            raise ArmDown("injected")

    def _tick(self) -> None:
        self._check_alive()
        if self._stop.is_set():
            raise MotionError("stopped")

    def _begin(self) -> None:
        """Start of every motion call. A latched stop stays latched until `resume()`."""
        self._check_alive()
        if self._stop.is_set():
            raise MotionError("stopped")

    def resume(self) -> None:
        self._stop.clear()

    def reachable(self, xyz: np.ndarray) -> bool:
        xyz = np.asarray(xyz, float)
        try:
            check_point(xyz, self.table_z)
        except MotionError:
            return False
        return self._kin.solve(xyz, seed=self.read_joints()[:5], down=True).ok

    def _inject(self, kind: str) -> None:
        if kind == "grasp":
            self._fail_grasp = True
        elif kind == "down":
            self._down, self._down_at = True, time.time()
        elif kind == "slow":
            self._slow = 0.3
        elif kind == "clear":
            self._down, self._fail_grasp, self._slow = False, False, 1.0
        else:
            raise ValueError(f"unknown injection: {kind}")

    # ---- ArmDriver

    def read_joints(self) -> np.ndarray:
        with self._table._lock:
            return self._table._data.qpos[self._qadr].copy()

    def set_joint_target(self, q: np.ndarray) -> None:
        self._begin()
        q = np.asarray(q, float)
        check_joints(q)
        goal = self._target.copy()
        goal[: len(q)] = q
        step = np.clip(goal - self._target, -MAX_TARGET_STEP_RAD, MAX_TARGET_STEP_RAD)
        self._command(self._target + step)

    def move_joints(self, q: np.ndarray, speed: float = 1.0) -> None:
        self._begin()
        q = np.asarray(q, float)
        check_joints(q)
        origin = self._target.copy()
        goal = origin.copy()
        goal[: len(q)] = q
        dist = float(np.abs(goal[:5] - self.read_joints()[:5]).max())
        duration = max(dist / (JOINT_SPEED * max(speed, 1e-3) * self._slow), 0.1)
        n = int(duration / TICK_S) + 1
        for k in range(1, n + 1):
            self._tick()
            s = k / n
            self._command(origin + (goal - origin) * (s * s * (3.0 - 2.0 * s)))
            self._table.wait(TICK_S, self._tick)
        waited = 0.0
        while np.abs(self.read_joints()[:5] - goal[:5]).max() > SETTLE_TOL_RAD:
            if waited >= SETTLE_TIMEOUT_S:
                raise MotionError("stall")
            self._tick()
            self._table.wait(TICK_S, self._tick)
            waited += TICK_S

    def move_to(self, xyz: np.ndarray, speed: float = 1.0) -> None:
        self._begin()
        xyz = np.asarray(xyz, float)
        check_point(xyz, self.table_z)
        result = self._kin.solve(xyz, seed=self.read_joints()[:5], down=True)
        if not result.ok:
            raise MotionError("unreachable")
        self.move_joints(result.q, speed)

    def open_gripper(self) -> None:
        self._begin()
        self._command(np.r_[self._target[:5], GRIPPER_OPEN])
        self._table._release(self.arm_id)
        self._table.wait(GRIPPER_SETTLE_S, self._tick)

    def close_gripper(self) -> Grasp:
        self._begin()
        self._command(np.r_[self._target[:5], GRIPPER_CLOSED])
        self._table.wait(GRIPPER_SETTLE_S, self._tick)
        if self._fail_grasp:
            self._fail_grasp = False
            attached = None
        else:
            attached = self._table._attach_nearest(self.arm_id)
        return Grasp(holding=attached is not None, position=float(self.read_joints()[5]))

    def holding(self) -> bool:
        self._check_alive()
        return self._table.attached(self.arm_id) is not None

    def stop(self) -> None:
        self._stop.set()
        self._command(self.read_joints())  # freeze where it is

    def park(self) -> None:
        self._begin()
        self.move_joints(PARK_Q, speed=0.3)

    def alive(self) -> float:
        return self._down_at if self._down else self._table.last_step_wall()


__all__ = ["ARM_JOINTS", "MujocoDriver", "SimTable"]
