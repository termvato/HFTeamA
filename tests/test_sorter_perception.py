"""The sorter's calibrated vision behind Operations' live pipeline (no camera, no arm)."""

import pytest

np = pytest.importorskip("numpy")

from hfteama.sorter.bridge import MainFrame  # noqa: E402
from hfteama.sorter.geometry import load_config  # noqa: E402


class Calib:
    """A stand-in for the pipeline's marker calibration, deliberately off: a camera pose that
    is a few cm and a few degrees wrong must not move the boxes."""

    def __init__(self):
        a = np.radians(4.0)
        self.T = np.eye(4)
        self.T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        self.T[:3, 3] = (0.03, -0.02, 1.1)

    def to_camera(self, T_table):
        return self.T @ T_table

    def to_table(self, T_cam):
        return np.linalg.inv(self.T) @ T_cam


def fake_scene(monkeypatch, pieces):
    from hfteama.sorter.run import Eye
    from hfteama.vision.detect import Box, Scene
    from hfteama.vision.table import CameraPose

    pose = CameraPose(R=np.diag([1.0, -1.0, -1.0]), t=np.array([-0.37, 0.17, 1.2]), rms_px=0.5)

    def look_frame(self, frame, arm_chains=(), keep_in_box=False):
        assert keep_in_box  # the place check must still see boxes in the bowl
        assert frame.shape == (1080, 1920, 3)
        return frame, pose, Scene(box=Box(0.37, 0.25, 0.09), pieces=pieces)

    monkeypatch.setattr(Eye, "look_frame", look_frame)


def test_pipeline_gets_the_sorters_positions_and_the_jaws_close_on_the_box_sides(
    monkeypatch,
) -> None:
    """Detector -> perception's build_scene -> sort_steps (the pick args) -> the bridge."""
    pytest.importorskip("cv2")
    from hfteama.arms.generic_skills import pick_point, pick_yaw
    from hfteama.arms.poses import Poses
    from hfteama.arms.so101_ik import fk, jaw_yaw
    from hfteama.perception.decomposer import Bin, sort_steps
    from hfteama.perception.scene import build_scene
    from hfteama.sorter.perception import SorterTagDetector
    from hfteama.vision.detect import Piece

    closing = np.radians(35.0)  # Piece.yaw: the jaws' closing direction, sorter frame
    fake_scene(
        monkeypatch,
        [
            Piece(0.62, 0.18, 0.058, 0.058, closing, 33.6, tag_id=3),
            Piece(0.12, 0.20, 0.058, 0.058, 0.0, 33.6),  # no tag: not a box
        ],
    )
    calib = Calib()
    det = SorterTagDetector(calib)
    scene = build_scene(det.detect(np.zeros((1080, 1920, 3), np.uint8)), calib)
    assert list(scene.objects) == ["tag_03"]

    cfg = load_config()
    mf = MainFrame.from_config(cfg)
    bowl = Bin("bowl", "the bowl", (0.355, 0.089, 0.082), 0.09)
    pick, place = sort_steps(scene.objects["tag_03"], "to_bowl", bowl)
    want = mf.to_main([0.62, 0.18, cfg["pieces"]["grasp_z"]])
    assert pick_point(pick, Poses.empty(), "x") == pytest.approx(want, abs=1e-6)

    from tests.test_sorter_bridge import make

    d, low, _ = make()  # rakesh, a fake bus
    d.move_to(pick_point(pick, Poses.empty(), "x") + [0, 0, 0.08], yaw=pick_yaw(pick))
    heading = jaw_yaw(fk(low.q())) + d._arm.placement.yaw
    off = np.degrees(heading - closing) % 90.0  # a square box: any of its four sides
    assert min(off, 90.0 - off) < 3.0


def test_carried_to_the_bowl_centre_the_drop_moves_towards_the_arm_gripper_down() -> None:
    """The pipeline places every box at the bowl's centre; from the arms' bases that point
    needs a tilted gripper, so the bridge moves the drop towards the arm (as plan_drop)."""
    from tests.test_sorter_bridge import make

    d, low, cfg = make()
    centre = np.array([0.355, 0.089, 0.082])  # main frame
    sol, xy, _, down = d._command(centre, None, "drop")
    assert sol.ok and down
    assert xy == pytest.approx(MainFrame.from_config(cfg).to_sorter(centre)[:2])  # same spot id
    cfg_off = {**cfg, "motion": {**cfg["motion"], "drop_shift_max_m": 0.0}}
    d._mc = cfg_off["motion"]
    sol0, *_, down0 = d._command(centre, None, "drop")
    assert sol0.ok and not down0  # without the shift: tilted at the centre
