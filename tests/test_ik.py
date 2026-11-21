"""IK on the vendored SO-101 model: round trips, pointing down, unreachable, base transform."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hfteama.arms.ik import ArmBase, Kinematics  # noqa: E402
from hfteama.arms.kinematics import BODY, LIMITS_RAD  # noqa: E402

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def kin() -> Kinematics:
    return Kinematics()


def _random_q(rng: np.random.Generator) -> np.ndarray:
    lo = np.array([LIMITS_RAD[j][0] for j in BODY]) * 0.8
    hi = np.array([LIMITS_RAD[j][1] for j in BODY]) * 0.8
    return rng.uniform(lo, hi)


def test_fk_ik_roundtrip(kin: Kinematics) -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        p, _z = kin.fk(_random_q(rng))
        r = kin.solve(p, down=False)
        assert r.ok and r.error_m < 0.002
        assert kin.fk(r.q)[0] == pytest.approx(p, abs=0.002)


@pytest.mark.parametrize("x", [0.15, 0.2, 0.25, 0.28])
@pytest.mark.parametrize("y", [-0.1, 0.0, 0.1])
def test_down_pose_points_down(kin: Kinematics, x: float, y: float) -> None:
    r = kin.solve(np.array([x, y, 0.03]), down=True)
    _p, z_axis = kin.fk(r.q)
    assert r.ok
    assert z_axis @ np.array([0.0, 0.0, -1.0]) > 0.9


def test_solution_respects_joint_limits(kin: Kinematics) -> None:
    r = kin.solve(np.array([0.2, 0.05, 0.03]), down=True)
    for q, j in zip(r.q, BODY, strict=True):
        assert LIMITS_RAD[j][0] - 1e-9 <= q <= LIMITS_RAD[j][1] + 1e-9


def test_unreachable_is_not_ok(kin: Kinematics) -> None:
    assert not kin.solve(np.array([2.0, 0.0, 0.1])).ok


def test_base_transform_roundtrip() -> None:
    base = ArmBase((0.3, -0.2, 0.0), yaw=np.pi / 2)
    p = np.array([0.1, 0.2, 0.3])
    assert base.to_table(base.to_base(p)) == pytest.approx(p)
    assert base.to_table(np.zeros(3)) == pytest.approx([0.3, -0.2, 0.0])


def test_fk_is_expressed_in_table_frame(kin: Kinematics) -> None:
    base = ArmBase((0.3, -0.2, 0.0), yaw=np.pi / 2)
    moved = Kinematics(base)
    q = np.array([0.2, -0.3, 0.5, 0.4, 0.0])
    p0, _ = kin.fk(q)
    p1, _ = moved.fk(q)
    assert p1 == pytest.approx(base.to_table(p0))


@pytest.mark.parametrize("z", [0.095, 0.115, 0.13])
def test_down_pose_reaches_approach_and_lift_heights(kin: Kinematics, z: float) -> None:
    # Skills approach 0.08 m and lift 0.10 m above a block resting on the table; position must win
    # over tool direction, or these poses come back unreachable although position-only IK is fine.
    r = kin.solve(np.array([0.2, 0.0, z]), down=True)
    _p, z_axis = kin.fk(r.q)
    assert r.ok
    assert z_axis @ np.array([0.0, 0.0, -1.0]) > 0.9


@pytest.mark.parametrize("yaw", [0.0, 0.5, 1.2, 2.0, -1.0])
@pytest.mark.parametrize("xy", [(0.2, 0.0), (0.15, 0.08), (0.25, -0.1)])
def test_solve_aligns_the_jaw_closing_direction_with_the_requested_yaw(
    kin: Kinematics, yaw: float, xy: tuple[float, float]
) -> None:
    target = np.array([xy[0], xy[1], 0.03])
    r = kin.solve(target, down=True, yaw=yaw)
    assert r.ok and r.error_m < 0.002
    err = (kin.tool_heading(r.q) - yaw + np.pi / 2) % np.pi - np.pi / 2  # the jaws are symmetric
    assert abs(err) < 0.05
    assert kin.fk(r.q)[1] @ np.array([0.0, 0.0, -1.0]) > 0.9  # still pointing down


def test_yaw_is_reported_when_it_cannot_be_met(kin: Kinematics) -> None:
    r = kin.solve(np.array([0.2, 0.0, 0.03]), down=True, yaw=0.7)
    assert r.heading_error_rad is not None and abs(r.heading_error_rad) < 0.05
    assert kin.solve(np.array([0.2, 0.0, 0.03]), down=True).heading_error_rad is None
