"""SO-101 forward and inverse kinematics in pure numpy (no PyBullet, MuJoCo or LeRobot).

The chain is copied from TheRobotStudio's so101_new_calib.urdf (the model the MuJoCo Menagerie
SO-101 is built from), so joint angles are the same as `hfteama.arms.kinematics.ArmFrame`
produces: radians, order shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.

Arm frame (base_link): x forward (away from the base, the way the arm points at pan = 0),
y to the arm's left, z up, origin on the underside of the base, i.e. on the table.

The tool point (TCP) is the URDF's gripper_frame_link: between the finger tips. Its z axis is
the approach direction (from the wrist towards the tips); the jaws close along its x axis.

Usage:
    from hfteama.arms.so101_ik import fk, solve_ik
    T = fk(q)                                   # 4x4 pose of the TCP in the arm frame
    sol = solve_ik([0.25, 0.0, 0.02], yaw=0.3)  # gripper pointing down, jaws turned 0.3 rad
    if sol.ok: q = sol.q
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hfteama.arms.kinematics import BODY, LIMITS_RAD


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _tf(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T


# Joint origins (parent link -> joint frame) from so101_new_calib.urdf; every joint rotates
# about its local z axis.
_JOINT_ORIGINS = [
    _tf((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),  # shoulder_pan
    _tf((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0)),  # shoulder_lift
    _tf((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),  # elbow_flex
    _tf((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),  # wrist_flex
    _tf((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),  # wrist_roll
]
_TCP = _tf((-0.0079, -0.000218121, -0.0981274), (0, 3.14159, 0))  # gripper_frame_link

LOWER = np.array([LIMITS_RAD[j][0] for j in BODY])
UPPER = np.array([LIMITS_RAD[j][1] for j in BODY])


def calibrated_limits(calibration: dict, frame=None, margin_deg: float = 3.0):
    """Joint limits (lower, upper; rad, model convention) from a LeRobot calibration dict
    (the JSON in configs/calibration/robots/so_follower/<id>.json), `margin_deg` inside the
    calibrated range. LeRobot degrees are centred on the middle of each joint's range.
    frame: the arm's ArmFrame (offsets/signs); default ArmFrame()."""
    from hfteama.arms.kinematics import ArmFrame

    frame = frame or ArmFrame()
    lo, hi = LOWER.copy(), UPPER.copy()
    for k, j in enumerate(BODY):
        c = calibration.get(j)
        if not c:
            continue
        half = (c["range_max"] - c["range_min"]) / 2 * 360 / 4095 - margin_deg
        ends = [np.radians((d - frame.offsets_deg[j]) / frame.signs[j]) for d in (-half, half)]
        lo[k], hi[k] = min(ends), max(ends)
    return lo, hi


def _rotz(a: float) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(a), np.sin(a)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def fk(q) -> np.ndarray:
    """4x4 pose of the TCP in the arm frame for the 5 body joint angles (rad)."""
    T = np.eye(4)
    for origin, a in zip(_JOINT_ORIGINS, np.asarray(q, float)[:5], strict=True):
        T = T @ origin @ _rotz(a)
    return T @ _TCP


def tcp_position(q) -> np.ndarray:
    return fk(q)[:3, 3]


def chain_points(q) -> np.ndarray:
    """Points along the arm (arm frame, (7, 3)): base, every joint, the TCP. A polyline to
    mask the arm out of camera images."""
    T = np.eye(4)
    pts = [np.zeros(3)]
    for origin, a in zip(_JOINT_ORIGINS, np.asarray(q, float)[:5], strict=True):
        T = T @ origin @ _rotz(a)
        pts.append(T[:3, 3].copy())
    pts.append((T @ _TCP)[:3, 3])
    return np.array(pts)


def jaw_yaw(T: np.ndarray) -> float:
    """Heading (rad, in the arm's xy plane) of the TCP x axis, which is the direction the
    jaws close along (the moving jaw's tip moves along TCP x). Meaningful when the gripper
    points down."""
    return float(np.arctan2(T[1, 0], T[0, 0]))


@dataclass
class IKSolution:
    q: np.ndarray  # 5 body joint angles, rad
    position_error: float  # m, distance from the TCP to the target
    tilt: float  # rad, angle between the approach axis and straight down (if requested)
    ok: bool


# A pose with the gripper hanging down in front of the base: a good seed for table work.
SEED = np.array([0.0, -0.3, 0.6, 1.3, 0.0])


def solve_ik(
    target,
    yaw: float | None = None,
    down: bool = True,
    seed=None,
    tol: float = 0.002,
    max_tilt: float = np.radians(30),
    iters: int = 100,
    limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> IKSolution:
    """Joint angles that put the TCP at `target` (m, arm frame).

    Position comes first; with `down`, the gripper is then turned as close to straight down as
    the remaining freedom allows (task-priority IK). Straight down reaches about 0.05-0.29 m
    from the base at table height; farther out the wrist leans towards the target.
    yaw: turn wrist_roll so the TCP x axis heads along `yaw` (rad, arm frame); jaw yaw is
         periodic in pi, so the closest of yaw and yaw + pi is used.
    ok is False when the target is out of reach (error > tol) or, with `down`, the gripper
    is tilted more than max_tilt.
    limits: (lower, upper) joint limits, rad; default the model's (see calibrated_limits).
    """
    lo, hi = limits if limits is not None else (LOWER, UPPER)
    target = np.asarray(target, float)
    seeds = [np.asarray(seed, float)] if seed is not None else []
    pan = -np.arctan2(target[1], target[0] - _JOINT_ORIGINS[0][0, 3])  # +pan turns to -y
    seeds.append(np.array([pan, *SEED[1:]]))
    # a coarse grid of arm shapes: the wrist_flex limit (95 deg) traps a single seed
    seeds += [np.array([pan, a, b, 1.2, 0.0]) for a in (-0.9, -0.2, 0.5, 1.0) for b in (-1.0, 0.9)]
    best = None
    for s in seeds:
        q = np.clip(s.copy(), lo, hi)
        for _ in range(3 if yaw is not None else 1):
            # wrist_roll moves the TCP a little (it is 8 mm off the roll axis): alternate
            if yaw is not None:
                q = _set_yaw(q, yaw, lo, hi)
            q = _task_priority(q, target, down, iters, lo, hi)
        err = float(np.linalg.norm(tcp_position(q) - target))
        tilt = _tilt(q)
        ok = err <= tol and (not down or tilt <= max_tilt)
        sol = IKSolution(q=q, position_error=err, tilt=tilt, ok=ok)
        if best is None or _score(sol) < _score(best):
            best = sol
        if ok and (not down or tilt < np.radians(1)):
            break
    return best


def _score(sol: IKSolution) -> tuple:
    # solutions within tolerance first, then the least tilted; otherwise the closest
    return (0, sol.tilt) if sol.position_error <= 0.002 else (1, sol.position_error)


def _tilt(q) -> float:
    """Angle (rad) between the approach axis and straight down."""
    return float(np.arccos(np.clip(-fk(q)[2, 2], -1.0, 1.0)))


def _jacobians(q, eps=1e-6):
    """Position (3x4) and approach-xy (2x4) Jacobians over the first 4 joints."""
    T = fk(q)
    f0 = np.concatenate([T[:3, 3], T[:2, 2]])
    J = np.empty((5, 4))
    for k in range(4):
        dq = np.zeros(5)
        dq[k] = eps
        T1 = fk(q + dq)
        J[:, k] = (np.concatenate([T1[:3, 3], T1[:2, 2]]) - f0) / eps
    return f0, J[:3], J[3:]


def _dpinv(J, damping=1e-4):
    return J.T @ np.linalg.inv(J @ J.T + damping * np.eye(J.shape[0]))


def _task_priority(q, target, down, iters, lo, hi):
    """Damped least squares on the first 4 joints (wrist_roll only spins the tool): reach the
    target, and use what freedom is left to point the approach axis down. A joint at a limit
    whose step would push it further is taken out of the solve (else the clipped step stalls
    every other joint)."""
    for _ in range(iters):
        f, Jp, Jo = _jacobians(q)
        rp = target - f[:3]
        free = np.ones(4, bool)
        for _ in range(4):
            step = _step(Jp * free, Jo * free, rp, f[3:], down) * free
            blocked = ((q[:4] <= lo[:4] + 1e-9) & (step < 0)) | (
                (q[:4] >= hi[:4] - 1e-9) & (step > 0)
            )
            if not blocked.any():
                break
            free &= ~blocked
        step = np.clip(step, -0.15, 0.15)
        q = q.copy()
        q[:4] = np.clip(q[:4] + step, lo[:4], hi[:4])
        if np.abs(step).max() < 1e-6:
            break
    return q


def _step(Jp, Jo, rp, approach_xy, down):
    Jp_pinv = _dpinv(Jp)
    step = Jp_pinv @ rp
    if down:
        N = np.eye(4) - Jp_pinv @ Jp
        ro = -approach_xy - Jo @ step  # approach x, y should be 0
        step = step + N @ _dpinv(Jo @ N, 1e-3) @ ro
    return step


def _set_yaw(q, yaw, lo=LOWER, hi=UPPER):
    """Choose wrist_roll so the TCP x axis heads along yaw (or yaw + pi), within limits."""
    q = q.copy()
    q[4] = 0.0
    base = jaw_yaw(fk(q))  # heading at roll 0 depends only on the first 4 joints
    # turning wrist_roll by +a turns the jaw heading by +/- a depending on which way the
    # gripper points; measure the sign numerically
    q2 = q.copy()
    q2[4] = 0.1
    sign = np.sign(_wrap(jaw_yaw(fk(q2)) - base)) or 1.0
    best = None
    for y in (yaw, yaw + np.pi, yaw - np.pi):
        a = _wrap(sign * (y - base))
        for cand in (a, a + 2 * np.pi, a - 2 * np.pi):
            if lo[4] <= cand <= hi[4] and (best is None or abs(cand) < abs(best)):
                best = cand
    q[4] = best if best is not None else 0.0
    return q


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def interpolate(q0, q1, max_step: float) -> list[np.ndarray]:
    """Joint-space waypoints from q0 to q1 (excluded) to q1, no joint moving more than
    max_step (rad) between consecutive waypoints."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    n = max(1, int(np.ceil(np.max(np.abs(q1 - q0)) / max_step)))
    return [q0 + (q1 - q0) * (k / n) for k in range(1, n + 1)]
