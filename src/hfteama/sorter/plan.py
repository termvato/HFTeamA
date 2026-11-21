"""Who picks what: IK targets for each piece and for the drop over the box, per arm.

Pure geometry (no hardware): a Scene from the camera in, PickPlans out.

Rules:
- A piece goes to the arm on its side of the table (the arm whose base is nearest along the
  table's x axis); if that arm can't reach it but another can, that one.
- Grasp: gripper straight down (within max_tilt) at approach_z above the piece, then down to
  grasp_z, jaws closing across the piece's short side.
- Drop: over the box at rim height + drop_above. The point is moved from the box centre
  towards the arm, up to (radius - 3 cm), until the arm can reach it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hfteama.arms.so101_ik import IKSolution, solve_ik
from hfteama.sorter.geometry import Placement
from hfteama.vision.detect import Box, Piece


@dataclass
class ArmModel:
    side: str
    placement: Placement
    limits: tuple[np.ndarray, np.ndarray] | None = None  # rad; None = model limits
    # grasp aim shift = a fixed part in the arm frame (forward, left as seen from behind the
    # arm) + a part that turns with the gripper, in the tool frame (along the jaws' closing
    # axis, across it): the open jaws are not centred on the tool point. See fit_aim.
    aim_offset: tuple[float, float] = (0.0, 0.0)
    jaw_offset: tuple[float, float] = (0.0, 0.0)
    # optional: the fixed part as a function of where the piece is, seen from the arm:
    # [(bearing_deg, forward, left, dz), ...] sorted by bearing (0 = straight ahead, + = to
    # the arm's left), linearly interpolated, held at the ends. Overrides aim_offset; dz (m)
    # is added to every height over that piece.
    aim_table: tuple = ()
    # part proportional to where the piece is: G @ (forward, left) of the piece from the base
    # (arm frame, m); [[0, -a], [a, 0]] is a small turn by a rad about the base
    aim_gain: tuple = ((0.0, 0.0), (0.0, 0.0))
    # the jaw offset was fitted on pieces about jaw_ref_width wide (m); a piece of width w
    # needs jaw_width_gain * (w - jaw_ref_width) more along the closing axis (0.5: half the
    # extra width, the fixed jaw stays on the piece's edge)
    jaw_ref_width: float = 0.025
    jaw_width_gain: float = 0.0
    object_width: float | None = None  # known piece width (m); None = the camera's estimate
    square: bool = False  # the pieces are square: grasp from the side nearest the wrist's middle
    # extra height (m) by distance from the base: [(r_m, dz_m), ...] sorted by r, linearly
    # interpolated, held at the ends (e.g. lower close to the base)
    z_by_distance: tuple = ()
    # pieces this arm takes: bearing from the arm (deg, 0 = ahead, + = its left), min/max
    bearing_range: tuple[float, float] = (-180.0, 180.0)

    def bearing(self, x: float, y: float) -> float:
        a = self.placement.to_arm([x, y])
        return float(np.degrees(np.arctan2(a[1], a[0])))

    # pieces this arm takes: table x range (m), e.g. its half of the table
    x_range: tuple[float, float] = (-np.inf, np.inf)

    def takes(self, x: float, y: float) -> bool:
        lo, hi = self.bearing_range
        return lo <= self.bearing(x, y) <= hi and self.x_range[0] <= x <= self.x_range[1]

    def correction(self, x: float, y: float) -> tuple[float, float, float]:
        """(forward, left, dz) to apply for a piece seen at table (x, y)."""
        a = self.placement.to_arm([x, y])
        dz = 0.0
        if self.z_by_distance:
            k = np.array(self.z_by_distance, float)
            dz = float(np.interp(np.hypot(a[0], a[1]), k[:, 0], k[:, 1]))
        if not self.aim_table:
            return (*self.aim_offset, dz)
        b = np.degrees(np.arctan2(a[1], a[0]))
        k = np.array(self.aim_table, float)
        fwd, left, dzt = (float(np.interp(b, k[:, 0], k[:, i])) for i in (1, 2, 3))
        return fwd, left, dzt + dz

    def aim(
        self, x: float, y: float, heading: float = 0.0, width: float | None = None
    ) -> tuple[float, float]:
        """Table xy to aim the tool at to grasp a piece seen at (x, y), with the jaws'
        closing axis heading along `heading` (rad, table frame). `width`: the piece's width
        across the jaws (m); a wider piece sits further from the fixed jaw, so the aim moves
        along the closing axis by jaw_width_gain x (width - jaw_ref_width)."""
        c, s = np.cos(heading), np.sin(heading)
        j = np.asarray(self.jaw_offset, float).copy()
        if width is not None:
            j[0] += self.jaw_width_gain * (width - self.jaw_ref_width)
        fwd, left, _ = self.correction(x, y)
        p = self.placement.to_arm([x, y])
        a = p + np.array([fwd, left]) + np.asarray(self.aim_gain, float) @ p
        t = self.placement.to_table(a) + [c * j[0] - s * j[1], s * j[0] + c * j[1]]
        return float(t[0]), float(t[1])

    def grasp_aim(self, piece: Piece, z: float):
        """(x, y, yaw_arm, heading, dz): where to put the tool to grasp `piece` at height z,
        the wrist yaw for the IK, the jaws' table heading the IK picks (yaw or yaw + pi), and
        the height correction for this piece (add it to every z over it)."""
        from hfteama.arms.so101_ik import LOWER, UPPER, fk, jaw_yaw

        square = self.square or abs(piece.length - piece.width) < 0.1 * piece.length
        lo, hi = self.limits if self.limits is not None else (LOWER, UPPER)
        mid = (lo[4] + hi[4]) / 2
        width = self.object_width or piece.width

        def choose(x, y, yaw):
            """IK at (x, y, z) for jaw yaw `yaw`; for a square piece also yaw + 90 deg, keeping
            the one with the wrist nearest the middle of its range (at most 45 deg off)."""
            target = self.placement.to_arm([x, y, z])
            sol = solve_ik(target, yaw=yaw, limits=self.limits)
            if square:
                other = solve_ik(target, yaw=yaw + np.pi / 2, limits=self.limits)
                if other.ok and (not sol.ok or abs(other.q[4] - mid) < abs(sol.q[4] - mid)):
                    sol, yaw = other, yaw + np.pi / 2
            return sol, yaw

        yaw = self.placement.yaw_to_arm(piece.yaw)
        ax, ay = piece.x, piece.y
        for _ in range(2):  # the choice at the piece, then again at the aim point it gives
            sol, yaw_used = choose(ax, ay, yaw)
            heading = jaw_yaw(fk(sol.q)) + self.placement.yaw
            ax, ay = self.aim(piece.x, piece.y, heading, width)
        yaw = yaw_used
        return ax, ay, yaw, heading, self.correction(piece.x, piece.y)[2]


@dataclass
class PickPlan:
    side: str
    piece: Piece
    above: IKSolution  # TCP at approach height over the piece
    grasp: IKSolution  # TCP at grasp height
    target_arm: np.ndarray  # grasp point, arm frame


@dataclass
class Plan:
    picks: list[PickPlan] = field(default_factory=list)
    drops: dict[str, IKSolution] = field(default_factory=dict)  # side -> drop pose
    unreachable: list[Piece] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def for_arm(self, side: str) -> list[PickPlan]:
        return [p for p in self.picks if p.side == side]


def plan_pick(arm: ArmModel, piece: Piece, mc: dict, pc: dict) -> PickPlan | None:
    if not arm.takes(piece.x, piece.y):
        return None
    max_tilt = np.radians(mc.get("max_tilt_deg", 30.0))
    ax, ay, yaw, _, dz = arm.grasp_aim(piece, pc.get("approach_z", 0.05))
    g = arm.placement.to_arm([ax, ay, pc.get("grasp_z", 0.005) + dz])
    a = arm.placement.to_arm([ax, ay, pc.get("approach_z", 0.05) + dz])
    above = solve_ik(a, yaw=yaw, limits=arm.limits)
    if not above.ok or above.tilt > max_tilt:
        return None
    grasp = solve_ik(g, yaw=yaw, seed=above.q, limits=arm.limits)
    if not grasp.ok or grasp.tilt > max_tilt:
        return None
    return PickPlan(side=arm.side, piece=piece, above=above, grasp=grasp, target_arm=g)


def plan_drop(arm: ArmModel, box: Box, height: float, drop_above: float, max_tilt: float):
    """Drop pose over the box for this arm, or None: drop_above over the rim if reachable,
    else lower in 1 cm steps down to 3 cm; at each height the point may move from the box
    centre towards the arm, up to 3 cm inside the rim. Gripper pointing down (within
    max_tilt) first; if the box is too far for that, any orientation (pointing forward)."""
    toward = np.array([arm.placement.x - box.x, arm.placement.y - box.y])
    toward /= max(np.linalg.norm(toward), 1e-9)
    # gripper pointing down if possible; else any way (pointing forward reaches further out)
    for down in (True, False):
        for above in np.arange(drop_above, 0.03 - 1e-9, -0.01):
            z = height + above
            for shift in np.arange(0.0, max(box.radius - 0.03, 0.0) + 1e-9, 0.01):
                xy = [box.x + shift * toward[0], box.y + shift * toward[1], z]
                sol = solve_ik(
                    arm.placement.to_arm(xy), down=down, limits=arm.limits, max_tilt=max_tilt
                )
                if sol.ok:
                    return sol
    return None


def make_plan(scene, arms: dict[str, ArmModel], cfg: dict) -> Plan:
    plan = Plan()
    mc, pc, bc = cfg["motion"], cfg["pieces"], cfg["box"]
    if scene.box is None:
        plan.notes.append("box not found")
    else:
        for side, arm in arms.items():
            d = plan_drop(
                arm,
                scene.box,
                bc["height"],
                bc.get("drop_above", 0.04),
                np.radians(mc.get("drop_max_tilt_deg", 60.0)),
            )
            if d is None:
                plan.notes.append(f"{side}: can't reach over the box")
            else:
                plan.drops[side] = d
    # picks are planned even for an arm that can't reach the box, so the preview shows them;
    # the runner skips arms without a drop
    usable = arms
    for piece in scene.pieces:
        order = sorted(usable, key=lambda s: abs(usable[s].placement.x - piece.x))
        for side in order:
            pick = plan_pick(usable[side], piece, mc, pc)
            if pick is not None:
                plan.picks.append(pick)
                break
        else:
            plan.unreachable.append(piece)
    # each arm takes its pieces nearest first (shortest moves, and the far ones stay visible)
    plan.picks.sort(key=lambda p: (p.side, float(np.hypot(*p.target_arm[:2]))))
    return plan
