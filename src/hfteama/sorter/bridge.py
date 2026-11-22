"""The sorter's calibrated arms behind Action's `ArmDriver` primitives (main's orchestration).

Operations' executor and Action's skills (`pick`, `place`, `pick_place`, ...) drive arms only
through `hfteama.arms.primitives.ArmDriver`. `SorterArmDriver` implements it with the sorter's
stack, measured on the real arms: the numpy IK with each arm's calibrated joint limits, the
per-arm aim corrections fitted from logged grasps (`configs/sorter_placements.json`), the
closed-loop gripper, smooth straight-line moves, and the same safety (never calibrates, refuses a
USB serial that is not connected, home pose before torque off).

Frames: the skills speak main's table frame (`configs/hacklab_table.json`: origin marker 0, x
towards marker 3, y towards marker 2). The sorter's frame is the same four tags with another
origin (`configs/sorter.json`), so the conversion is exact: `MainFrame`.

A point given to `move_to` is where the gripped piece's centre should be (a pick point, a place
point): the arm's aim correction turns it into the tool command, as in `hfteama sort`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from hfteama.arms.kinematics import JOINTS, LIMITS_RAD
from hfteama.arms.primitives import ArmDown, Grasp, MotionError, check_point
from hfteama.arms.so101_ik import fk, solve_ik
from hfteama.sorter.geometry import PROJECT_ROOT, load_config
from hfteama.sorter.plan import ArmModel, drop_solve
from hfteama.vision.detect import Piece

MAIN_TABLE = PROJECT_ROOT / "configs" / "hacklab_table.json"
HEARTBEAT_POLL_S = 0.5


class MainFrame:
    """main's table frame <-> the sorter's, both defined by the reference tags.

    main: origin at tag `origin`, x towards tag `x_towards`, y towards tag `y_towards`, z up
    (`hacklab_table.json`: 0, 3, 2). The tag centres come from the sorter config."""

    def __init__(self, tags: dict[int, tuple[float, float]], origin=0, x_towards=3, y_towards=2):
        o = np.asarray(tags[origin], float)
        ex = np.asarray(tags[x_towards], float) - o
        ex /= np.linalg.norm(ex)
        ey = np.array([-ex[1], ex[0]])  # right-handed with z up
        if np.dot(ey, np.asarray(tags[y_towards], float) - o) < 0:
            raise ValueError("main frame is left-handed with these tags")
        self.o, self.R = o, np.stack([ex, ey], axis=1)  # columns: main's axes in sorter coords

    @classmethod
    def from_config(cls, cfg: dict) -> MainFrame:
        tags = {int(k): tuple(v) for k, v in cfg["table"]["tags"].items()}
        mf = cfg.get("main_frame", {})
        return cls(tags, mf.get("origin", 0), mf.get("x_towards", 3), mf.get("y_towards", 2))

    def to_sorter(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        return np.concatenate([self.o + self.R @ p[:2], p[2:]])

    def to_main(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        return np.concatenate([self.R.T @ (p[:2] - self.o), p[2:]])

    def yaw_to_sorter(self, yaw: float) -> float:
        return float(yaw + np.arctan2(self.R[1, 0], self.R[0, 0]))


class SorterArmDriver:
    """`ArmDriver` (hfteama.arms.primitives) over a sorter arm. Positions in main's frame."""

    def __init__(
        self, arm_id: str, side: str, driver, model: ArmModel, frame: MainFrame, cfg: dict
    ):
        self.arm_id = arm_id
        self.side = side
        self.table_z = 0.0
        self._drv = driver  # hfteama.sorter.robot.ArmDriver, connected
        self._arm = model
        self._frame = frame
        self._cfg = cfg
        self._mc, self._pc = cfg["motion"], cfg["pieces"]
        self._lock = threading.RLock()  # the Feetech bus is not thread-safe
        self._last_ok = time.time()
        self._parked = False
        self._last_xy: np.ndarray | None = None  # sorter-frame xy of the last move_to
        self._last_yaw: float | None = None  # its wrist yaw, reused while only z changes
        self._last_down = False  # its pose had the gripper down (within max_tilt)
        self._last_mode = "grasp"  # how its pose was solved: "grasp" or "drop" (see _command)
        self._carrying = False  # the last close_gripper held a piece (until open_gripper)
        self._aims: dict = {}  # grasp_aim cache (see _grasp_aim)

    # ---- helpers

    def _guard(self):
        if self._parked:
            raise MotionError("torque off")

    def _run(self, fn, *a, **k):
        from hfteama.sorter.robot import Stopped

        self._guard()
        with self._lock:
            try:
                out = fn(*a, **k)
            except Stopped:
                raise MotionError("stopped") from None
            except (RuntimeError, ValueError) as exc:  # IK on the way, joint range
                raise MotionError(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001  (any bus failure means the arm is gone)
                raise ArmDown(f"bus: {exc}") from exc
            self._last_ok = time.time()
            return out

    def _command(self, xyz_main, yaw_main: float | None, mode: str = "grasp"):
        """IK for a piece centre at xyz_main (main's frame): (sol, xy, yaw_arm, down), down =
        gripper pointing down within max_tilt (a straight vertical line works from there).

        mode "grasp": the grasp aim (`grasp_aim`), gripper down within max_tilt; above the
        pieces it may tilt. mode "drop" (a held piece taken somewhere, e.g. over the bowl): the
        sorter's drop (`plan.drop_solve`), gripper down within max_tilt, else within
        drop_max_tilt_deg, else any orientation (pointing forward reaches further out)."""
        p = self._frame.to_sorter(xyz_main)
        z = float(p[2])
        max_tilt = np.radians(self._mc.get("max_tilt_deg", 30.0))
        if mode == "drop":
            drop_tilt = np.radians(self._mc.get("drop_max_tilt_deg", 60.0))
            for down, tilt in ((True, max_tilt), (True, drop_tilt), (False, drop_tilt)):
                sol = drop_solve(self._arm, float(p[0]), float(p[1]), z, down, tilt)
                if sol.ok:
                    break
            return sol, p[:2], None, sol.ok and down and tilt == max_tilt
        same_spot = self._last_xy is not None and np.hypot(*(p[:2] - self._last_xy)) < 0.01
        if yaw_main is not None:
            piece_yaw = self._frame.yaw_to_sorter(yaw_main) - np.pi / 2  # jaws close along yaw
        else:
            piece_yaw = 0.0
        w = self._pc.get("object_width", 0.03)
        L = self._pc.get("object_length", w)
        ax, ay, yaw_arm, _, dz = self._grasp_aim(
            round(float(p[0]), 4), round(float(p[1]), 4), round(z, 4), float(piece_yaw), L, w
        )
        if same_spot and yaw_main is None and self._last_yaw is not None:
            yaw_arm = self._last_yaw  # same wrist all the way down and up again
        target = self._arm.placement.to_arm([ax, ay, z + dz])
        sol = solve_ik(target, yaw=yaw_arm, limits=self._arm.limits, max_tilt=max_tilt)
        down = sol.ok
        if not sol.ok and self._high(z):  # above the pieces: any gripper orientation
            sol = solve_ik(target, down=False, limits=self._arm.limits)
        return sol, p[:2], yaw_arm, down

    def _grasp_aim(self, x, y, z, yaw, length, width):
        """`ArmModel.grasp_aim`, cached: a step checks every point before moving (reachable),
        then moves through the same points, and each costs ~1 s of IK."""
        key = (x, y, z, yaw, length, width)
        if key not in self._aims:
            piece = Piece(x=x, y=y, length=length, width=width, yaw=yaw, area_cm2=0)
            self._aims[key] = self._arm.grasp_aim(piece, z)
        return self._aims[key]

    def _high(self, z: float) -> bool:
        """Above the pieces on the table, where the gripper may tilt."""
        return z > self._pc.get("object_height", 0.04) + 0.02

    def _mode_for(self, xy_sorter) -> str:
        """The same spot keeps its mode (down, grasp, lift: one aim); a new spot reached while
        carrying a piece is a drop."""
        if self._last_xy is not None and np.hypot(*(xy_sorter - self._last_xy)) < 0.01:
            return self._last_mode
        return "drop" if self._carrying else "grasp"

    # ---- ArmDriver

    def read_joints(self) -> np.ndarray:
        obs = self._run(self._drv.read)
        q = self._drv.frame.to_kinematic(obs)
        return np.asarray(q, float)

    def set_joint_target(self, q: np.ndarray) -> None:
        q = np.asarray(q, float)
        self._run(self._drv.move_q, q[:5], self._mc.get("max_joint_speed_deg_s", 45.0))

    def move_joints(self, q: np.ndarray, speed: float = 1.0) -> None:
        q = np.asarray(q, float)
        deg_s = self._mc.get("max_joint_speed_deg_s", 45.0) * min(max(speed, 0.05), 1.0)
        self._run(self._drv.move_q, q[:5], deg_s)
        if len(q) > 5:
            lo, hi = LIMITS_RAD[JOINTS[5]]
            self._run(self._drv.gripper, float(np.clip((q[5] - lo) / (hi - lo) * 100, 0, 100)))

    def move_to(self, xyz: np.ndarray, speed: float = 1.0, yaw: float | None = None) -> None:
        check_point(xyz, self.table_z, clearance=0.0)
        self._guard()
        mode = self._mode_for(self._frame.to_sorter(xyz)[:2])
        sol, xy, yaw_arm, down = self._command(xyz, yaw, mode)
        if not sol.ok and mode == "drop":  # the drop aim only matters over the bowl: anywhere
            mode = "grasp"  # else (e.g. the carry pose) the grasp aim is as good
            sol, xy, yaw_arm, down = self._command(xyz, yaw, mode)
        if not sol.ok:
            raise MotionError("unreachable")
        deg_s = self._mc.get("max_joint_speed_deg_s", 45.0) * min(max(speed, 0.05), 1.0)
        same_xy = self._last_xy is not None and np.hypot(*(xy - self._last_xy)) < 0.01
        if same_xy and down and self._last_down:
            # only the height changes, gripper down at both ends: a straight vertical line, one
            # smooth motion (its IK keeps the gripper down, so not from or to a tilted pose)
            line_yaw = yaw_arm if yaw_arm is not None else self._last_yaw
            q0 = self._run(self._drv.q)
            try:  # the line's IK outside the bus lock: ~1 s with no bus traffic (heartbeat)
                path = self._drv.line_path(q0, fk(sol.q)[:3, 3], line_yaw, 8)
            except RuntimeError as exc:
                raise MotionError(str(exc)) from exc
            self._run(self._drv.move_path, path, deg_s)
        else:
            self._run(self._drv.move_q, sol.q, deg_s)
        self._last_xy, self._last_down, self._last_mode = xy, down, mode
        if yaw_arm is not None:
            self._last_yaw = yaw_arm

    def open_gripper(self) -> None:
        self._run(self._drv.gripper, self._mc.get("gripper_open", 100.0))
        self._carrying = False

    def close_gripper(self) -> Grasp:
        pct = self._run(self._drv.gripper, self._mc.get("gripper_closed", 2.0))
        lo, hi = LIMITS_RAD[JOINTS[5]]
        self._carrying = self._holding_pct(pct)
        return Grasp(holding=self._carrying, position=lo + (hi - lo) * pct / 100.0)

    def _holding_pct(self, pct: float) -> bool:
        return pct > self._mc.get("gripper_closed", 2.0) + self._mc.get("grasp_min_open", 4.0)

    def holding(self) -> bool:
        return self._holding_pct(self._run(self._drv.read)["gripper.pos"])

    def stop(self) -> None:
        self._drv.stop_event.set()  # checked every tick by the motion loops, no lock needed

    def resume(self) -> None:
        self._drv.stop_event.clear()

    def reachable(self, xyz: np.ndarray, yaw: float | None = None) -> bool:
        z = float(np.asarray(xyz)[2])
        if z < self.table_z:
            return False
        if self._command(xyz, yaw)[0].ok:
            return True
        # checked before the grasp: a place point is reached carrying the piece, as a drop
        return self._high(z) and bool(self._command(xyz, yaw, "drop")[0].ok)

    def park(self) -> None:
        """Back to the arm's home (rest) pose slowly, then torque off. Refuses motion after."""
        self._drv.stop_event.clear()
        with self._lock:
            try:
                self._drv.go_home()
            finally:
                self._drv.torque_off()
                self._parked = True

    def alive(self) -> float:
        """Wall time the arm last answered. A move holds the bus for seconds; its loop talks to
        the servos every tick, so that counts (the executor drops an arm silent for 1.5 s)."""
        self._last_ok = max(self._last_ok, getattr(self._drv, "last_io", 0.0))
        if time.time() - self._last_ok > HEARTBEAT_POLL_S and not self._parked:
            if self._lock.acquire(blocking=False):  # never wait behind a running move
                try:
                    self._drv.read()
                    self._last_ok = time.time()
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    self._lock.release()
        return self._last_ok


def side_of(arm_id: str, cfg: dict, pairs_config=None) -> str:
    """The sorter arm (config key) whose pair's follower id is `arm_id` (catalog id)."""
    from hfteama.hardware.pairs import read_config

    pairs = read_config(pairs_config)["pairs"]
    for side, a in cfg["arms"].items():
        if pairs[a["pair"]]["follower"]["id"] == arm_id or side == arm_id:
            return side
    raise KeyError(f"no sorter arm for {arm_id!r} (configs/sorter.json arms)")


def from_sorter(arm_id: str, sorter_config: str | Path | None = None, pairs_config=None):
    """Connect the sorter arm behind catalog id `arm_id` and wrap it as an `ArmDriver`."""
    from hfteama.sorter.robot import ArmDriver
    from hfteama.sorter.run import arm_models

    cfg = load_config(sorter_config)
    side = side_of(arm_id, cfg, pairs_config)
    model = arm_models(cfg, [side])[side]
    drv = ArmDriver(side, cfg["arms"][side], cfg["motion"], pairs_config).connect()
    return SorterArmDriver(arm_id, side, drv, model, MainFrame.from_config(cfg), cfg)


def sorter_worker_factory(poses, sorter_config=None, pairs_config=None):
    """Workers for the executor (runner `--backend sorter`): each arm through the sorter."""
    from hfteama.arms.worker import SkillWorker

    def make(arm_id: str) -> SkillWorker:
        return SkillWorker(arm_id, from_sorter(arm_id, sorter_config, pairs_config), poses, "real")

    return make
