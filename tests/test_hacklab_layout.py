"""tasks/hacklab: the real table (two SO-101 side by side, 30 cm apart, one shared bowl)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hfteama.arms.generic_skills import APPROACH_DZ, GRASP_DZ, LIFT_DZ  # noqa: E402
from hfteama.arms.ik import ArmBase, Kinematics  # noqa: E402
from hfteama.arms.poses import Poses  # noqa: E402
from hfteama.taskgraph import load  # noqa: E402

pytestmark = pytest.mark.sim

ROOT = Path(__file__).parent.parent
HACK = ROOT / "tasks" / "hacklab"
TABLE = json.loads((ROOT / "configs" / "hacklab_table.json").read_text())


def _poses() -> Poses:
    return Poses.load(HACK / "poses.json")


def test_arm_bases_are_the_ones_in_the_table_config() -> None:
    poses = _poses()
    for a in TABLE["arms"]:
        cfg = poses.arms[a["id"]]
        assert cfg["base"][:2] == pytest.approx(a["base_pose_m"][:2], abs=1e-3)
        assert cfg["yaw"] == pytest.approx(math.radians(a["base_yaw_deg"]), abs=1e-3)


def test_the_two_arms_are_about_40_cm_apart_centre_to_centre() -> None:
    xs = sorted(c["base"][0] for c in _poses().arms.values())
    assert xs[1] - xs[0] == pytest.approx(0.40, abs=0.02)  # 30 cm gap + ~10 cm base


def test_the_bowl_is_where_the_table_config_says() -> None:
    bowl = TABLE["bins"][0]
    poses = _poses()
    assert poses.locations["bowl"][:2] == pytest.approx(bowl["xyz"][:2], abs=1e-3)
    assert poses.radii["bowl"] == pytest.approx(bowl["radius_m"], abs=1e-3)


def _nearest_arm(poses: Poses, xy: np.ndarray) -> str:
    return min(
        poses.arms, key=lambda a: float(np.linalg.norm(xy - np.array(poses.arms[a]["base"][:2])))
    )


def test_every_arm_can_run_its_own_picks_and_place_into_the_shared_bowl() -> None:
    poses = _poses()
    for arm, cfg in poses.arms.items():
        kin = Kinematics(ArmBase(tuple(cfg["base"]), cfg["yaw"]))
        bowl, carry = poses.point("bowl", arm), poses.point("carry", arm)
        for dz in (APPROACH_DZ, GRASP_DZ):
            assert kin.solve(bowl + [0, 0, dz], down=True).ok, f"{arm} cannot reach the bowl +{dz}"
        assert kin.solve(carry, down=True).ok, f"{arm} cannot reach its carry pose"
        mine = [n for n, p in poses.objects.items() if _nearest_arm(poses, np.array(p[:2])) == arm]
        assert len(mine) == 6, f"{arm} should be nearest to 6 objects, is nearest to {mine}"
        for n in mine:
            for dz in (APPROACH_DZ, GRASP_DZ, LIFT_DZ):
                target = poses.objects[n] + np.array([0.0, 0.0, dz])
                assert kin.solve(target, down=True).ok, (arm, n, dz)


def test_objects_stay_clear_of_the_bowl_and_of_each_other() -> None:
    poses = _poses()
    bx, by, _ = poses.locations["bowl"]
    pts = {n: np.array(p[:2]) for n, p in poses.objects.items()}
    assert len(pts) == 12
    for n, p in pts.items():
        assert math.hypot(p[0] - bx, p[1] - by) > poses.radii["bowl"] + 0.03, n
    names = list(pts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert np.linalg.norm(pts[a] - pts[b]) > 0.035, (a, b)


def test_the_task_matches_the_poses_and_serialises_the_bowl() -> None:
    poses, graph = _poses(), load(HACK / "hacklab12.json")
    assert len(graph.tasks) == 24
    for t in graph.tasks.values():
        if t.skill == "pick":
            assert t.args["object"] in poses.objects and t.resources == ()
        else:
            assert t.args["to"] == "bowl" and t.resources == ("bowl",)  # one arm over the bowl
            assert t.same_arm_as is not None


def test_each_block_goes_to_its_own_spot_in_the_bowl_that_its_arm_can_reach() -> None:
    poses, graph = _poses(), load(HACK / "hacklab12.json")
    bx, by, _ = poses.locations["bowl"]
    spots = []
    for t in graph.tasks.values():
        if t.skill != "place":
            continue
        xyz = np.array(t.args["place_xyz"], float)  # one spot per block, not one pile
        spots.append(xyz[:2])
        assert math.hypot(xyz[0] - bx, xyz[1] - by) < poses.radii["bowl"] - 0.03  # inside the bowl
        (arm,) = t.allowed_arms
        cfg = poses.arms[arm]
        kin = Kinematics(ArmBase(tuple(cfg["base"]), cfg["yaw"]))
        for dz in (APPROACH_DZ, GRASP_DZ):
            assert kin.solve(xyz + [0, 0, dz], down=True).ok, (t.id, arm, dz)
    for i, a in enumerate(spots):
        for b in spots[i + 1 :]:
            assert np.linalg.norm(a - b) > 0.035  # blocks do not land on each other


def test_operations_reach_model_gives_every_step_an_arm() -> None:
    # The orchestrator (registry.resolve) drops a step no arm reaches; the executor never runs it.
    from hfteama.registry import load_catalog, resolve

    catalog = load_catalog(ROOT / "configs" / "hacklab_table.json")
    resolved = resolve(load(HACK / "hacklab12.json"), catalog.values(), clearance_m=0.15)
    assert not resolved.blocked
    assert all(t.allowed_arms for t in resolved.graph.tasks.values())


def test_each_arm_places_on_its_own_side_of_the_bowl() -> None:
    graph, bx = load(HACK / "hacklab12.json"), _poses().locations["bowl"][0]
    for t in graph.tasks.values():
        if t.skill == "place":
            side = t.args["place_xyz"][0] < bx
            assert side == (t.allowed_arms == ("rakesh_follower",)), t.id
