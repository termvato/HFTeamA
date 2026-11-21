"""tasks/sort/poses.json: every point the sort12 steps use is reachable by every arm."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hfteama.arms.generic_skills import APPROACH_DZ, LIFT_DZ  # noqa: E402
from hfteama.arms.ik import ArmBase, Kinematics  # noqa: E402
from hfteama.arms.poses import Poses  # noqa: E402
from hfteama.taskgraph import load  # noqa: E402

pytestmark = pytest.mark.sim

SORT = Path(__file__).parent.parent / "tasks" / "sort"


def test_sort_poses_reachable_by_every_arm() -> None:
    poses = Poses.load(SORT / "poses.json")
    assert len(poses.arms) == 3
    for arm, cfg in poses.arms.items():
        kin = Kinematics(ArmBase(tuple(cfg["base"]), cfg["yaw"]))
        names = ["input_bin", "own_output_bin", "carry", *poses.objects]
        for name in names:
            p = poses.point(name, arm)
            for dz in (0.0, APPROACH_DZ, LIFT_DZ):
                r = kin.solve(p + np.array([0.0, 0.0, dz]), down=True)
                assert r.ok, f"{arm} cannot reach {name} at +{dz} (error {r.error_m:.3f} m)"


def test_every_sort12_object_has_a_pose() -> None:
    poses = Poses.load(SORT / "poses.json")
    graph = load(SORT / "sort12.json")
    for t in graph.tasks.values():
        if t.skill == "pick":
            assert t.args["object"] in poses.objects
            assert poses.has(t.args["from"], "arm_a")
        else:
            assert poses.has(t.args["to"], "arm_a")


def test_output_bins_are_apart_from_each_other_and_the_input_bin() -> None:
    poses = Poses.load(SORT / "poses.json")
    pts = [np.array(poses.per_arm[a]["own_output_bin"]) for a in poses.arms]
    pts.append(np.array(poses.locations["input_bin"]))
    for i, p in enumerate(pts):
        for q in pts[i + 1 :]:
            assert np.linalg.norm(p[:2] - q[:2]) > 0.12
