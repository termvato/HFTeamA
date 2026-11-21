"""Sorter geometry (arm placement, fit) and planning, without hardware or camera."""

import json

import pytest

np = pytest.importorskip("numpy")

from hfteama.arms.so101_ik import fk  # noqa: E402
from hfteama.sorter.geometry import (  # noqa: E402
    Placement,
    fit_placement,
    load_config,
    save_placement,
)
from hfteama.sorter.plan import ArmModel, make_plan  # noqa: E402
from hfteama.vision.detect import Box, Piece, Scene  # noqa: E402

CFG = load_config(placements="/nonexistent")
LEFT = Placement(0.163, 0.37, np.radians(-90))
RIGHT = Placement(0.567, 0.37, np.radians(-90))


def test_placement_round_trip_and_facing() -> None:
    # facing the camera: the arm's forward (+x) is table -y, its left (+y) is table +x
    assert LEFT.to_arm([0.163, 0.17, 0.02]) == pytest.approx([0.2, 0.0, 0.02])
    assert LEFT.to_arm([0.263, 0.37]) == pytest.approx([0.0, 0.1])
    p = np.array([0.31, 0.12, 0.05])
    assert LEFT.to_table(LEFT.to_arm(p)) == pytest.approx(p)


def test_fit_placement_recovers_pose() -> None:
    truth = Placement(0.17, 0.38, np.radians(-87))
    arm_pts = np.array([[0.15, 0.05], [0.25, -0.1], [0.2, 0.12], [0.12, -0.04]])
    table_pts = np.array([truth.to_table(p) for p in arm_pts])
    fit, rms = fit_placement(arm_pts, table_pts)
    assert (fit.x, fit.y, fit.yaw) == pytest.approx((truth.x, truth.y, truth.yaw), abs=1e-9)
    assert rms < 1e-9
    one, _ = fit_placement(arm_pts[:1], table_pts[:1], yaw_guess=truth.yaw)
    assert (one.x, one.y) == pytest.approx((truth.x, truth.y))


def test_saved_placement_overrides_config(tmp_path) -> None:
    f = tmp_path / "placements.json"
    save_placement("left", Placement(0.1, 0.4, np.radians(-80)), 0.002, 3, f)
    cfg = load_config(placements=f)
    assert cfg["arms"]["left"]["base"] == [0.1, 0.4]
    assert cfg["arms"]["left"]["yaw_deg"] == pytest.approx(-80)
    assert json.loads(f.read_text())["left"]["fit_rms_mm"] == 2.0


def piece(x, y, yaw=0.0):
    return Piece(x=x, y=y, length=0.06, width=0.02, yaw=yaw, area_cm2=12.0)


def test_plan_assigns_sides_and_drops() -> None:
    arms = {"left": ArmModel("left", LEFT), "right": ArmModel("right", RIGHT)}
    scene = Scene(
        box=Box(0.365, 0.15, 0.09),
        pieces=[piece(0.20, 0.20), piece(0.52, 0.22, yaw=0.5), piece(0.60, -0.30)],
    )
    plan = make_plan(scene, arms, CFG)
    assert set(plan.drops) == {"left", "right"}
    assert [p.side for p in plan.picks] == ["left", "right"]
    assert len(plan.unreachable) == 1  # 67 cm away from both arms
    for pick in plan.picks:
        arm = arms[pick.side].placement
        tcp = arm.to_table(fk(pick.grasp.q)[:3, 3])
        assert tcp[:2] == pytest.approx([pick.piece.x, pick.piece.y], abs=3e-3)
        assert tcp[2] == pytest.approx(CFG["pieces"]["grasp_z"], abs=3e-3)
        # jaws close along the piece's short side (TCP x, periodic in pi)
        T = fk(pick.grasp.q)
        heading = np.arctan2(T[1, 0], T[0, 0]) + arm.yaw
        d = (heading - pick.piece.yaw) % np.pi
        assert min(d, np.pi - d) < np.radians(5)


def test_piece_on_the_far_side_goes_to_the_arm_that_reaches_it() -> None:
    # only the left arm is there: a piece right of the middle still goes to it if reachable
    arms = {"left": ArmModel("left", LEFT), "right": ArmModel("right", RIGHT)}
    scene = Scene(box=Box(0.365, 0.15, 0.09), pieces=[piece(0.40, 0.25)])
    plan = make_plan(scene, {"left": arms["left"]}, CFG)
    assert [p.side for p in plan.picks] == ["left"]


def test_box_out_of_reach_is_reported() -> None:
    arms = {"left": ArmModel("left", LEFT)}
    plan = make_plan(Scene(box=Box(0.365, -0.2, 0.09), pieces=[]), arms, CFG)
    assert not plan.drops
    assert any("box" in n for n in plan.notes)
