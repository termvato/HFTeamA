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
RAKESH = Placement(0.163, 0.37, np.radians(-90))
HACKLAB = Placement(0.567, 0.37, np.radians(-90))


def test_placement_round_trip_and_facing() -> None:
    # facing the camera: the arm's forward (+x) is table -y, its own left (+y) is table +x
    assert RAKESH.to_arm([0.163, 0.17, 0.02]) == pytest.approx([0.2, 0.0, 0.02])
    assert RAKESH.to_arm([0.263, 0.37]) == pytest.approx([0.0, 0.1])
    p = np.array([0.31, 0.12, 0.05])
    assert RAKESH.to_table(RAKESH.to_arm(p)) == pytest.approx(p)


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
    save_placement("rakesh", Placement(0.1, 0.4, np.radians(-80)), 0.002, 3, f)
    cfg = load_config(placements=f)
    assert cfg["arms"]["rakesh"]["base"] == [0.1, 0.4]
    assert cfg["arms"]["rakesh"]["yaw_deg"] == pytest.approx(-80)
    assert json.loads(f.read_text())["rakesh"]["fit_rms_mm"] == 2.0


def piece(x, y, yaw=0.0):
    return Piece(x=x, y=y, length=0.06, width=0.02, yaw=yaw, area_cm2=12.0)


def test_plan_assigns_sides_and_drops() -> None:
    arms = {"rakesh": ArmModel("rakesh", RAKESH), "hacklab": ArmModel("hacklab", HACKLAB)}
    scene = Scene(
        box=Box(0.365, 0.15, 0.09),
        pieces=[piece(0.20, 0.20), piece(0.52, 0.22, yaw=0.5), piece(0.60, -0.30)],
    )
    plan = make_plan(scene, arms, CFG)
    assert set(plan.drops) == {"rakesh", "hacklab"}
    assert {round(p.piece.x, 2): p.side for p in plan.picks} == {0.2: "rakesh", 0.52: "hacklab"}
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
    # only rakesh is there: a piece on hacklab's side still goes to it if reachable
    arms = {"rakesh": ArmModel("rakesh", RAKESH), "hacklab": ArmModel("hacklab", HACKLAB)}
    scene = Scene(box=Box(0.365, 0.15, 0.09), pieces=[piece(0.40, 0.25)])
    plan = make_plan(scene, {"rakesh": arms["rakesh"]}, CFG)
    assert [p.side for p in plan.picks] == ["rakesh"]


def test_box_out_of_reach_is_reported() -> None:
    arms = {"rakesh": ArmModel("rakesh", RAKESH)}
    plan = make_plan(Scene(box=Box(0.365, -0.2, 0.09), pieces=[]), arms, CFG)
    assert not plan.drops
    assert any("box" in n for n in plan.notes)


def test_z_offset_shifts_heights_only() -> None:
    # the real tip is 1.5 cm lower than the model: aim the model 1.5 cm higher
    p = Placement(0.138, 0.37, np.radians(-90), z=-0.015)
    assert p.to_arm([0.114, 0.163, 0.045]) == pytest.approx([0.207, -0.024, 0.06])
    assert p.to_table(p.to_arm([0.2, 0.1, 0.03])) == pytest.approx([0.2, 0.1, 0.03])
    assert p.to_arm([0.2, 0.1]) == pytest.approx(Placement(0.138, 0.37, p.yaw).to_arm([0.2, 0.1]))


def test_fit_aim_separates_fixed_and_turning_offsets() -> None:
    from hfteama.sorter.geometry import fit_aim

    yaw = np.radians(-90)
    arm = ArmModel("rakesh", Placement(0.14, 0.35, yaw), None, (0.02, 0.01), (0.015, -0.005))
    entries = []
    for i, h in enumerate(np.radians([0, 70, 140, 230])):
        cam = (0.1 + 0.03 * i, 0.15)
        entries.append({"camera": cam, "aim": arm.aim(*cam, h), "heading_deg": np.degrees(h)})
    aim, jaw, rms = fit_aim(entries, yaw)
    assert aim == pytest.approx([0.02, 0.01], abs=1e-9)
    assert jaw == pytest.approx([0.015, -0.005], abs=1e-9)
    assert rms < 1e-9
    # all jaws heading the same way: only the fixed part, from the mean
    same = [dict(e, heading_deg=10.0) for e in entries[:2]]
    aim1, jaw1, _ = fit_aim(same, yaw)
    assert not jaw1.any()


def test_grasp_aim_turns_with_the_jaws() -> None:
    arm = ArmModel("rakesh", RAKESH, None, (0.0, 0.0), (0.02, 0.0))
    a0 = np.subtract(arm.aim(0.2, 0.2, 0.0), (0.2, 0.2))
    a90 = np.subtract(arm.aim(0.2, 0.2, np.pi / 2), (0.2, 0.2))
    assert a0 == pytest.approx([0.02, 0.0])
    assert a90 == pytest.approx([0.0, 0.02])
