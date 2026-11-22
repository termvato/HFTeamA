"""The Isaac Sim mirror's sender: the scene file and the pose packets (no Isaac Sim needed)."""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from hfteama.arms.poses import Poses  # noqa: E402
from hfteama.arms.sim import SimTable  # noqa: E402
from hfteama.view.isaac_mirror import (  # noqa: E402
    IsaacMirror,
    parse_scene,
    parse_state,
    read_recording,
)

ROOT = Path(__file__).resolve().parent.parent


def test_scene_file_and_packets(tmp_path: Path) -> None:
    poses = Poses.load(ROOT / "tasks" / "sort" / "poses.json")
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    port = rx.getsockname()[1]
    with (
        SimTable(poses, speed=4.0) as table,
        IsaacMirror(table, port=port, cache=tmp_path) as mirror,
    ):
        scene = state = None
        while scene is None or state is None:
            data, _ = rx.recvfrom(65535)
            scene = scene or parse_scene(data)
            state = state or parse_state(data)
        block = table.object_pos("block_01")
    rx.close()

    sid, path = scene
    assert sid == mirror.scene_id and Path(path) == mirror.scene_path
    z = np.load(path)
    names = [str(n) for n in z["body_names"]]
    for arm in table.arm_ids:  # every arm link with a visual mesh, every block
        assert f"{arm}/gripper" in names
    assert {f"block_{i:02d}" for i in range(1, 13)} <= set(names)
    assert (z["g_type"] == mujoco.mjtGeom.mjGEOM_MESH).any()
    assert len(z["mesh_vert"]) and z["mesh_face"].max() < z["mesh_vnum"].max()

    sid2, _t, body_poses = state
    assert sid2 == sid and body_poses.shape == (len(names), 7)
    k = names.index("block_01")  # blocks stay put without an arm: the packet has its position
    assert np.allclose(body_poses[k, :3], block, atol=1e-3)
    assert np.allclose(np.linalg.norm(body_poses[:, 3:], axis=1), 1.0, atol=1e-3)


def test_parsers_reject_other_packets() -> None:
    assert parse_state(b"hello") is None
    assert parse_scene(b"MJM1....") is None


def test_recording_starts_with_the_scene(tmp_path: Path) -> None:
    poses = Poses.load(ROOT / "tasks" / "sort" / "poses.json")
    rec = tmp_path / "states.bin"
    with SimTable(poses, speed=4.0) as table, IsaacMirror(table, cache=tmp_path, record=rec, hz=50):
        table.wait(0.5)
    packets = list(read_recording(rec))
    assert parse_scene(packets[0]) is not None
    times = [parse_state(p)[1] for p in packets[1:]]
    assert len(times) >= 5 and times == sorted(times)


def test_pyramid_task_is_reachable_and_locked() -> None:
    """Each box is its own arm's, gripped square at every height; places hold the pyramid lock
    and stay within 0.40 m of the arm (it stalls near 0.43 m)."""
    import json
    import math

    from hfteama.arms.ik import ArmBase, Kinematics

    folder = ROOT / "tasks" / "pyramid"
    p = json.loads((folder / "poses.json").read_text())
    g = json.loads((folder / "pyramid14.json").read_text())
    kin = {a: Kinematics(ArmBase(tuple(c["base"]), c["yaw"])) for a, c in p["arms"].items()}
    for t in g["tasks"]:
        assert len(t["allowed_arms"]) == 1, t["id"]
        arm = t["allowed_arms"][0]
        if t["skill"] == "pick_square":
            assert t["resources"] == []  # inside the arm's own wedge
            box = np.array(p["objects"][t["args"]["object"]])
            square = [
                all(
                    (r := kin[arm].solve(box + [0, 0, dz], yaw=h)).ok
                    and abs(r.heading_error_rad or 0.0) < 0.05
                    for dz in (-0.007, 0.0, 0.08, 0.10)
                )
                for h in (0.0, np.pi / 2)
            ]
            assert any(square), (arm, t["id"])
        else:
            assert t["skill"] == "place_square" and t["resources"] == ["pyramid"]
            slot = t["args"]["place_xyz"]
            assert math.dist(slot[:2], p["arms"][arm]["base"][:2]) <= 0.40, t["id"]
            assert kin[arm].solve(np.array(slot) + [0, 0, 0.08]).ok, t["id"]
