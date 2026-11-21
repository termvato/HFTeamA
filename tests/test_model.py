"""The vendored SO-101 MuJoCo model loads and agrees with arms.kinematics."""

import pytest

mujoco = pytest.importorskip("mujoco")

from hfteama.arms.kinematics import LIMITS_RAD  # noqa: E402
from hfteama.arms.model import ARM_JOINTS, GRIPPER, MODEL_XML, TCP_SITE  # noqa: E402

pytestmark = pytest.mark.sim


def test_vendored_model_loads() -> None:
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    assert m.nu == 6
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE) >= 0
    for j in (*ARM_JOINTS, GRIPPER):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0


def test_model_limits_match_kinematics() -> None:
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    for j in (*ARM_JOINTS, GRIPPER):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = m.jnt_range[jid]
        assert (lo, hi) == pytest.approx(LIMITS_RAD[j], abs=1e-3)
