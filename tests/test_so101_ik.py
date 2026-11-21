"""Pure-numpy SO-101 FK/IK."""

import pytest

np = pytest.importorskip("numpy")

from hfteama.arms.so101_ik import (  # noqa: E402
    LOWER,
    UPPER,
    fk,
    interpolate,
    jaw_yaw,
    solve_ik,
)


def test_zero_pose_matches_urdf() -> None:
    # all joints at 0: the arm stretched forward and up, tool pointing forward (+x)
    T = fk(np.zeros(5))
    assert T[:3, 3] == pytest.approx([0.391, 0.0, 0.226], abs=2e-3)
    assert T[:3, 2] == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)


@pytest.mark.parametrize(
    "target", [(0.20, 0.0, 0.02), (0.25, -0.10, 0.06), (0.12, 0.05, 0.02), (0.18, 0.12, 0.10)]
)
def test_ik_reaches_table_targets_pointing_down(target) -> None:
    sol = solve_ik(target, yaw=0.4)
    assert sol.ok
    assert fk(sol.q)[:3, 3] == pytest.approx(target, abs=2e-3)
    assert np.degrees(sol.tilt) < 5
    assert np.all(sol.q >= LOWER - 1e-9) and np.all(sol.q <= UPPER + 1e-9)


def test_ik_jaw_yaw_is_periodic_in_pi() -> None:
    for yaw in (-1.2, 0.0, 0.7, 2.5):
        sol = solve_ik((0.2, 0.05, 0.03), yaw=yaw)
        d = (jaw_yaw(fk(sol.q)) - yaw) % np.pi
        assert min(d, np.pi - d) < np.radians(3)


def test_ik_reports_out_of_reach() -> None:
    sol = solve_ik((0.6, 0.0, 0.0))
    assert not sol.ok
    assert sol.position_error > 0.1


def test_ik_covers_the_table_work_area() -> None:
    # every point 12-30 cm from the base, +-60 deg around the front, 2-6 cm above the table
    for r in (0.12, 0.17, 0.24, 0.30):
        for a in np.radians([-60, -20, 20, 60]):
            for z in (0.02, 0.06):
                target = (r * np.cos(a), r * np.sin(a), z)
                sol = solve_ik(target, yaw=a)
                assert sol.ok and np.degrees(sol.tilt) < 15, (target, sol)


def test_interpolate_limits_step() -> None:
    q0, q1 = np.zeros(5), np.array([0.5, -0.2, 0.1, 0.0, 1.0])
    path = interpolate(q0, q1, 0.1)
    assert path[-1] == pytest.approx(q1)
    steps = np.diff(np.vstack([q0, *path]), axis=0)
    assert np.abs(steps).max() <= 0.1 + 1e-12


def test_calibrated_limits_from_lerobot_range() -> None:
    from hfteama.arms.so101_ik import calibrated_limits

    # 2300 ticks of range = +-101.1 deg; minus a 3 deg margin; elbow shifted by its 6.9 deg
    cal = {j: {"range_min": 900, "range_max": 3200} for j in ("wrist_flex", "elbow_flex")}
    lo, hi = calibrated_limits(cal)
    half = 2300 / 2 * 360 / 4095 - 3
    assert np.degrees(hi[3]) == pytest.approx(half)
    assert np.degrees(lo[2]) == pytest.approx(-half - 6.9)
    assert lo[0] == LOWER[0]  # joints missing from the calibration keep the model's limits


def test_chain_points_run_from_base_to_tcp() -> None:
    from hfteama.arms.so101_ik import chain_points

    q = np.array([0.3, -0.2, 0.5, 1.0, 0.4])
    pts = chain_points(q)
    assert pts.shape == (7, 3)
    assert pts[0] == pytest.approx([0, 0, 0])
    assert pts[-1] == pytest.approx(fk(q)[:3, 3])
