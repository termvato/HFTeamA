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
    # grasp aim shift in the arm frame (m): (forward, left) as seen from behind the arm
    aim_offset: tuple[float, float] = (0.0, 0.0)

    def aim(self, x: float, y: float) -> tuple[float, float]:
        """Table xy to aim the tool at to grasp a piece seen at (x, y)."""
        a = self.placement.to_arm([x, y]) + np.asarray(self.aim_offset)
        t = self.placement.to_table(a)
        return float(t[0]), float(t[1])


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
    max_tilt = np.radians(mc.get("max_tilt_deg", 30.0))
    ax, ay = arm.aim(piece.x, piece.y)
    g = arm.placement.to_arm([ax, ay, pc.get("grasp_z", 0.005)])
    a = arm.placement.to_arm([ax, ay, pc.get("approach_z", 0.05)])
    yaw = arm.placement.yaw_to_arm(piece.yaw)
    above = solve_ik(a, yaw=yaw, limits=arm.limits)
    if not above.ok or above.tilt > max_tilt:
        return None
    grasp = solve_ik(g, yaw=yaw, seed=above.q, limits=arm.limits)
    if not grasp.ok or grasp.tilt > max_tilt:
        return None
    return PickPlan(side=arm.side, piece=piece, above=above, grasp=grasp, target_arm=g)


def plan_drop(arm: ArmModel, box: Box, height: float, drop_above: float, max_tilt: float):
    """Drop pose over the box for this arm, or None."""
    toward = np.array([arm.placement.x - box.x, arm.placement.y - box.y])
    toward /= max(np.linalg.norm(toward), 1e-9)
    z = height + drop_above
    for shift in np.arange(0.0, max(box.radius - 0.03, 0.0) + 1e-9, 0.01):
        p = arm.placement.to_arm([box.x + shift * toward[0], box.y + shift * toward[1], z])
        sol = solve_ik(p, limits=arm.limits, max_tilt=max_tilt)
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
