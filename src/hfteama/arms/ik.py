"""Inverse kinematics for one SO-101 on the vendored MuJoCo model.

Sim and real arms share this solver, so a sim run predicts the joint angles of the real arm (the
real backend adds only the LeRobot calibration offsets in `kinematics.ArmFrame`).

The arm has 5 body joints, so a 3-D position is solved exactly and the tool direction is best
effort: with `down=True` the tool z axis (wrist toward fingertips) is pulled toward straight down,
weighted well below the position (0.01: a 0.5 rad tilt costs as much as 5 mm). Damped least
squares on the MuJoCo Jacobian, adapted from so101-nexus `kinematics.py` (Apache-2.0).

Measured reach with the tool pointing down (base at the origin, arm facing +x): x about 0.10 to
0.30 m for table-height targets (z 0 to 0.03), |y| up to 0.2 m, and up to z about 0.12 m at
x = 0.2 (tool alignment stays above 0.9). Nothing behind the base. Plan poses inside that band.

Frames: the table frame is the shared world frame; an `ArmBase` places the arm in it. Only
`mj_kinematics` and `mj_jacSite` are used, never `mj_step`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hfteama.arms.kinematics import LIMITS_RAD
from hfteama.arms.model import ARM_JOINTS, MODEL_XML, TCP_SITE

_DAMPING = 0.05
_ORIENT_WEIGHT = 0.01
_MAX_ITERS = 200
_N_RANDOM_SEEDS = 10
_LIMIT_MARGIN = 0.05  # rad: solutions stay this far inside the joint limits (a joint on its
# limit cannot settle on the target, so the move would stall)
_DOWN = np.array([0.0, 0.0, -1.0])
ELBOW_MAX_RAD = 1.45  # Menagerie issue #325: elbow self-collision starts near 1.474 rad


@dataclass(frozen=True)
class ArmBase:
    """Where an arm's base sits in the table frame: position (m) and yaw about z (rad)."""

    xyz: tuple[float, float, float]
    yaw: float = 0.0

    def _rot(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_table(self, p_base: np.ndarray) -> np.ndarray:
        return self._rot() @ np.asarray(p_base, float) + np.asarray(self.xyz, float)

    def to_base(self, p_table: np.ndarray) -> np.ndarray:
        return self._rot().T @ (np.asarray(p_table, float) - np.asarray(self.xyz, float))


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray  # 5 body joints, rad
    error_m: float
    ok: bool
    heading_error_rad: float | None = None  # jaw-closing heading minus the requested yaw (mod pi)


class Kinematics:
    """FK and IK for one arm placed at `base` in the table frame."""

    def __init__(self, base: ArmBase = ArmBase((0.0, 0.0, 0.0))) -> None:
        self.base = base
        self._model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        self._data = mujoco.MjData(self._model)
        self._precheck_data = mujoco.MjData(self._model)
        self._site = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        jids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self._qadr = np.array([self._model.jnt_qposadr[j] for j in jids])
        self._dadr = np.array([self._model.jnt_dofadr[j] for j in jids])
        self._lo = np.array([LIMITS_RAD[j][0] for j in ARM_JOINTS]) + _LIMIT_MARGIN
        self._hi = np.array([LIMITS_RAD[j][1] for j in ARM_JOINTS]) - _LIMIT_MARGIN
        self._elbow_idx = ARM_JOINTS.index("elbow_flex")
        self._hi[self._elbow_idx] = min(self._hi[self._elbow_idx], ELBOW_MAX_RAD)
        self._rng = np.random.default_rng(0)
        self._rot_inv = self.base._rot().T

    def _set(self, q: np.ndarray) -> None:
        self._data.qpos[self._qadr] = q
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)

    def _tcp_base(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self._data.site_xpos[self._site].copy()
        z_axis = self._data.site_xmat[self._site].reshape(3, 3)[:, 2].copy()
        return pos, z_axis

    def fk(self, q5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position and TCP z axis (wrist toward fingertips), both in the table frame."""
        self._set(np.asarray(q5, float))
        pos, z_axis = self._tcp_base()
        return self.base.to_table(pos), self.base._rot() @ z_axis

    def tool_heading(self, q5: np.ndarray) -> float:
        """Table-frame heading (rad) of the jaw-closing direction: the tool x axis, seen from above.

        With the tool pointing down, wrist roll turns this heading one-for-one. The jaws are
        symmetric, so only the heading modulo pi matters.
        """
        self._set(np.asarray(q5, float))
        x_axis = self._data.site_xmat[self._site].reshape(3, 3)[:, 0]
        v = self.base._rot() @ x_axis
        return float(np.arctan2(v[1], v[0]))

    def min_tcp_z(self, q_from: np.ndarray, q_to: np.ndarray, samples: int = 20) -> float:
        """Lowest tool-centre height (table frame) along the straight joint-space path."""
        a, b = np.asarray(q_from, float)[:5], np.asarray(q_to, float)[:5]
        return min(float(self.fk(a + (b - a) * k / samples)[0][2]) for k in range(samples + 1))

    def _iterate(
        self, q0: np.ndarray, target: np.ndarray, down: bool, fix_roll: bool = False
    ) -> tuple[np.ndarray, float]:
        q = np.clip(q0, self._lo, self._hi)
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        for _ in range(_MAX_ITERS):
            self._set(q)
            pos, z_axis = self._tcp_base()
            err = target - pos
            if np.linalg.norm(err) < 1e-4 and not down:
                break
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._site)
            jp = jacp[:, self._dadr]
            if down:
                rot_err = _ORIENT_WEIGHT * np.cross(z_axis, _DOWN)
                jac = np.vstack([jp, _ORIENT_WEIGHT * jacr[:, self._dadr]])
                rhs = np.concatenate([err, rot_err])
            else:
                jac, rhs = jp, err
            if fix_roll:
                jac = jac.copy()
                jac[:, 4] = 0.0  # wrist roll stays where it was set
            step = jac.T @ np.linalg.solve(jac @ jac.T + _DAMPING**2 * np.eye(len(rhs)), rhs)
            if np.linalg.norm(step) < 1e-7:
                break
            q = np.clip(q + step, self._lo, self._hi)
        self._set(q)
        pos, _ = self._tcp_base()
        return q, float(np.linalg.norm(target - pos))

    def solve(
        self,
        xyz_table: np.ndarray,
        seed: np.ndarray | None = None,
        down: bool = True,
        tol: float = 0.002,
        yaw: float | None = None,
    ) -> IKResult:
        """Joint angles that put the TCP at `xyz_table`; `ok` when within `tol` metres.

        Tries `seed`, the zero pose and a few fixed random starts, and keeps the best. With
        `down` the tool is also pulled toward vertical, and a start is preferred while it points
        down (alignment > 0.95).
        """
        target = self.base.to_base(np.asarray(xyz_table, float))
        starts = [np.zeros(len(ARM_JOINTS))]
        if seed is not None:
            starts.insert(0, np.asarray(seed, float))
        rng = np.random.default_rng(0)  # per call: the answer must not depend on earlier calls
        starts += [rng.uniform(self._lo, self._hi) for _ in range(_N_RANDOM_SEEDS)]
        best: tuple[float, np.ndarray, float] | None = None
        for q0 in starts:
            q, err = self._iterate(q0, target, down)
            align = float(self._tcp_base()[1] @ _DOWN) if down else 1.0
            score = err + (1.0 - align) * 0.01
            if best is None or score < best[0]:
                best = (score, q, err)
            if err <= tol and align > 0.95:
                break
        assert best is not None
        q, err = best[1], best[2]
        heading_err: float | None = None
        if yaw is not None:
            q, err, heading_err = self._turn_to_yaw(q, target, down, yaw)
        return IKResult(q=q, error_m=err, ok=err <= tol, heading_error_rad=heading_err)

    def _heading_error(self, q: np.ndarray, yaw: float) -> float:
        return float((self.tool_heading(q) - yaw + np.pi / 2) % np.pi - np.pi / 2)

    def _turn_to_yaw(
        self, q: np.ndarray, target: np.ndarray, down: bool, yaw: float
    ) -> tuple[np.ndarray, float, float]:
        """Set wrist roll so the jaws close along `yaw`, re-solving position with roll fixed.

        Roll sits 12 mm off the tool centre and pan also changes the heading, so this repeats a
        few times. The jaws are symmetric: of the two roll angles that work, take the nearer one
        inside the joint limits.
        """
        q = q.copy()
        err = float("inf")
        for _ in range(6):
            delta = -self._heading_error(q, yaw)
            if abs(delta) < 0.01 and err < float("inf"):
                break
            options = [q[4] + delta + k * np.pi for k in (-1, 0, 1)]
            options = [o for o in options if self._lo[4] <= o <= self._hi[4]]
            if not options:
                break
            q[4] = min(options, key=lambda o: abs(o - q[4]))
            q, err = self._iterate(q, target, down, fix_roll=True)
        return q, err, self._heading_error(q, yaw)

    def precheck(
        self,
        pick_xyz: np.ndarray,
        place_xyz: np.ndarray,
        table_z: float = 0.0,
        hover_dz: float = 0.07,
        tol: float = 0.010,
        min_sv: float = 1e-3,
    ) -> tuple[bool, str]:
        """Verify pick/place and hover waypoints on a scratch MjData before real motion."""
        pick = np.asarray(pick_xyz, float)
        place = np.asarray(place_xyz, float)
        waypoints = (
            pick + np.array([0.0, 0.0, hover_dz]),
            pick,
            place + np.array([0.0, 0.0, hover_dz]),
            place,
        )
        seed: np.ndarray | None = None
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        for tgt in waypoints:
            if float(tgt[2]) < table_z + 0.01:
                return False, "below table"
            res = self.solve(tgt, seed=seed, down=True, tol=tol)
            if not res.ok:
                return False, f"unreachable ({res.error_m:.3f} m)"
            if float(res.q[self._elbow_idx]) > ELBOW_MAX_RAD + 1e-6:
                return False, "elbow limit"
            self._precheck_data.qpos[self._qadr] = res.q
            mujoco.mj_kinematics(self._model, self._precheck_data)
            mujoco.mj_comPos(self._model, self._precheck_data)
            mujoco.mj_jacSite(self._model, self._precheck_data, jacp, jacr, self._site)
            sv = float(np.linalg.svd(jacp[:, self._dadr], compute_uv=False)[-1])
            if sv < min_sv:
                return False, f"singular ({sv:.2e})"
            seed = res.q
        return True, "ok"
