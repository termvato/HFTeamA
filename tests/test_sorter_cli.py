"""`hfteama sort` command line, without hardware."""

import pytest

pytest.importorskip("numpy")

from hfteama.__main__ import main  # noqa: E402


def test_goto_dry_run_plans_without_hardware(capsys) -> None:
    assert main(["-n", "sort", "goto", "rakesh", "--tag", "2", "--z", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "IK ok=True" in out


def test_goto_needs_a_target() -> None:
    with pytest.raises(SystemExit):
        main(["-n", "sort", "goto", "rakesh"])


def test_run_refuses_saved_image_on_real_arms() -> None:
    with pytest.raises(SystemExit, match="dry-run"):
        main(["sort", "run", "--image", "x.jpg"])


def test_arm_not_plugged_in_is_refused_not_swapped(monkeypatch) -> None:
    # the config's fallback port may now belong to the other arm: never use it
    import hfteama.hardware.pairs as pairs
    from hfteama.sorter.geometry import load_config
    from hfteama.sorter.robot import ArmDriver

    monkeypatch.setattr(pairs, "find_port_by_serial", lambda serial: None)
    cfg = load_config()
    drv = ArmDriver("hacklab", cfg["arms"]["hacklab"], cfg["motion"])
    with pytest.raises(SystemExit, match="not connected"):
        drv.connect()


def test_jog_records_offset_and_fits(monkeypatch, capsys) -> None:
    """Jog with a fake arm and camera: piece 1, 2 steps forward, 1 left, record, quit."""
    import numpy as np

    import hfteama.sorter.robot as robot
    import hfteama.sorter.run as run
    from hfteama.arms.so101_ik import fk
    from hfteama.vision.detect import Piece, Scene

    class FakeEye:
        def __init__(self, cfg, image=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def look(self, arm_chains=()):
            p = Piece(x=0.65, y=0.15, length=0.06, width=0.02, yaw=0.0, area_cm2=12.0)
            return None, None, Scene(box=None, pieces=[p])

    class FakeDriver:
        def __init__(self, side, arm_cfg, motion, pairs_config=None):
            self.limits = None
            self._q = np.array([0.0, -1.5, 1.5, 1.0, 0.0])

        def connect(self):
            return self

        def disconnect(self):
            pass

        def q(self):
            return self._q

        def tcp(self):
            return fk(self._q)[:3, 3]

        def move_q(self, q, speed=None):
            self._q = np.asarray(q, float)

        def gripper(self, value, wait=None):
            return value

        def go_home(self):
            pass

    monkeypatch.setattr(run, "Eye", FakeEye)
    monkeypatch.setattr(robot, "ArmDriver", FakeDriver)
    answers = iter(["1", "f2", "l", "ok", "q", "n"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["sort", "jog", "rakesh", "--step", "10", "--no-viewer"]) == 0
    out = capsys.readouterr().out
    # rakesh faces the camera (+y): forward = table +y, its left = table -x
    assert "recorded: real = planned aim + (-1.0, +2.0) cm" in out
    assert "placement from 1 points" in out


def test_gripper_ramps_under_the_relative_cap_and_holds_where_it_stopped() -> None:
    import hfteama.sorter.robot as robot

    class FakeRobot:
        """Gripper follows commands but stops at 20 (a piece in the jaws)."""

        def __init__(self):
            self.g, self.sent = 60.0, []

        def get_observation(self):
            obs = {f"{j}.pos": 0.0 for j in robot.BODY}
            obs["gripper.pos"] = self.g
            return obs

        def send_action(self, act):
            assert abs(act["gripper.pos"] - self.g) <= 10.0, "LeRobot would clip this"
            self.sent.append(act["gripper.pos"])
            # a slow motor: covers half of each step
            self.g = max(20.0, self.g + 0.5 * (act["gripper.pos"] - self.g))

    import threading

    drv = robot.ArmDriver.__new__(robot.ArmDriver)
    drv.stop_event = threading.Event()
    drv.motion = {"max_relative_target": 10.0, "rate_hz": 1000}
    drv.robot = FakeRobot()
    end = drv.gripper(2.0, wait=3.0)
    assert end == pytest.approx(20.0, abs=0.5)
    assert drv.gripper_target == pytest.approx(12.0, abs=0.5)  # 20 squeezed by 8, not 2


def test_jog_log_saves_a_grasp_the_gripper_reading_missed(monkeypatch, capsys, tmp_path) -> None:
    """`g` reads an empty gripper (a thin piece); the operator says it held: `log` saves it,
    even after `box`."""
    import json

    import numpy as np

    import hfteama.sorter.geometry as geometry
    import hfteama.sorter.robot as robot
    import hfteama.sorter.run as run
    from hfteama.arms.so101_ik import fk
    from hfteama.vision.detect import Box, Piece, Scene

    class FakeEye:
        def __init__(self, cfg, image=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def look(self, arm_chains=()):
            p = Piece(x=0.65, y=0.15, length=0.06, width=0.015, yaw=0.0, area_cm2=8.0)
            return None, None, Scene(box=Box(0.5, 0.25, 0.09), pieces=[p])

    class FakeDriver:
        def __init__(self, side, arm_cfg, motion, pairs_config=None):
            self.limits = None
            self._q = np.array([0.0, -1.5, 1.5, 1.0, 0.0])

        def connect(self):
            return self

        def disconnect(self):
            pass

        def q(self):
            return self._q

        def tcp(self):
            return fk(self._q)[:3, 3]

        def move_q(self, q, speed=None):
            self._q = np.asarray(q, float)

        def move_line(self, q_from, target_to, yaw, steps=8, speed=None):
            return self._q

        def gripper(self, value, wait=None):
            return 2.5  # closes almost fully: reads as empty

        def lift(self, dz=None):
            pass

        def go_home(self):
            pass

    log = tmp_path / "grasps.json"
    monkeypatch.setattr(geometry, "GRASPS_FILE", log)
    monkeypatch.setattr(run, "Eye", FakeEye)
    monkeypatch.setattr(robot, "ArmDriver", FakeDriver)
    answers = iter(["1", "g", "box", "log", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["sort", "jog", "rakesh", "--no-viewer"]) == 0
    out = capsys.readouterr().out
    assert "empty" in out and "camera checks" in out
    entries = json.loads(log.read_text())["rakesh"]
    assert len(entries) == 1 and entries[0]["camera"] == [0.65, 0.15]


def test_jog_camera_confirms_a_grasp_after_box(monkeypatch, capsys, tmp_path) -> None:
    """The gripper reads empty, but after `box` the camera no longer sees the piece: the
    grasp is logged without the operator saying anything."""
    import json

    import numpy as np

    import hfteama.sorter.geometry as geometry
    import hfteama.sorter.robot as robot
    import hfteama.sorter.run as run
    from hfteama.arms.so101_ik import fk
    from hfteama.vision.detect import Box, Piece, Scene

    class FakeEye:
        cam = object()  # "live camera"
        looks = 0

        def __init__(self, cfg, image=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def look(self, arm_chains=()):
            FakeEye.looks += 1
            pieces = [Piece(x=0.65, y=0.15, length=0.06, width=0.02, yaw=0.0, area_cm2=8.0)]
            return None, None, Scene(box=Box(0.6, 0.25, 0.09), pieces=pieces[: FakeEye.looks == 1])

    class FakeDriver:
        def __init__(self, side, arm_cfg, motion, pairs_config=None):
            self.limits = None
            self._q = np.array([0.0, -1.5, 1.5, 1.0, 0.0])

        def connect(self):
            return self

        def disconnect(self):
            pass

        def q(self):
            return self._q

        def tcp(self):
            return fk(self._q)[:3, 3]

        def move_q(self, q, speed=None):
            self._q = np.asarray(q, float)

        def move_line(self, q_from, target_to, yaw, steps=8, speed=None):
            return self._q

        def gripper(self, value, wait=None):
            return 3.1  # a thin piece: reads like an empty grasp

        def lift(self, dz=None):
            pass

        def go_home(self):
            pass

    log = tmp_path / "grasps.json"
    monkeypatch.setattr(geometry, "GRASPS_FILE", log)
    monkeypatch.setattr(run, "Eye", FakeEye)
    monkeypatch.setattr(robot, "ArmDriver", FakeDriver)
    answers = iter(["1", "g", "box", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["sort", "jog", "rakesh", "--no-viewer"]) == 0
    out = capsys.readouterr().out
    assert "piece 1 is gone: it was taken" in out
    assert len(json.loads(log.read_text())["rakesh"]) == 1


def test_piece_still_there() -> None:
    from hfteama.sorter.run import piece_still_there
    from hfteama.vision.detect import Piece, Scene

    scene = Scene(box=None, pieces=[Piece(0.6, 0.2, 0.05, 0.02, 0.0, 8.0)])
    assert piece_still_there(scene, 0.61, 0.21)
    assert not piece_still_there(scene, 0.65, 0.2)


def test_fit_grasps_removes_the_width_part(monkeypatch, capsys, tmp_path) -> None:
    """Small pieces and boxes logged together fit one jaw offset once the width part
    (jaw_width_gain) is taken out."""
    import json

    import numpy as np

    import hfteama.sorter.geometry as geometry
    from hfteama.sorter.geometry import Placement, load_config
    from hfteama.sorter.plan import ArmModel

    cfg = load_config()
    a = cfg["arms"]["rakesh"]
    pl = Placement.from_config(a)
    arm = ArmModel(
        "rakesh",
        pl,
        None,
        (0.01, 0.03),
        (0.02, -0.005),
        jaw_ref_width=a["jaw_ref_width"],
        jaw_width_gain=a["jaw_width_gain"],
    )
    rng = np.random.default_rng(3)
    entries = []
    for i in range(6):
        cam = (0.65 + rng.uniform(-0.05, 0.05), 0.15 + rng.uniform(-0.05, 0.05))
        h, w = rng.uniform(-np.pi, np.pi), (0.025, 0.058)[i % 2]
        entries.append(
            {
                "time": f"2026-11-21 22:0{i}:00",
                "camera": list(cam),
                "aim": list(arm.aim(*cam, h, w)),
                "heading_deg": float(np.degrees(h)),
                "width": w,
            }
        )
    grasps, places = tmp_path / "g.json", tmp_path / "p.json"
    grasps.write_text(json.dumps({"rakesh": entries}))
    monkeypatch.setattr(geometry, "GRASPS_FILE", grasps)
    monkeypatch.setattr(geometry, "PLACEMENTS_FILE", places)
    assert main(["sort", "fit-grasps", "rakesh", "--save"]) == 0
    saved = json.loads(places.read_text())["rakesh"]
    assert saved["aim_offset"] == pytest.approx([0.01, 0.03], abs=1e-4)
    assert saved["jaw_offset"] == pytest.approx([0.02, -0.005], abs=1e-4)


def test_move_line_is_one_smooth_motion() -> None:
    """The waypoints of a straight line are driven without stopping at each one, and no
    joint step goes over the speed limit."""
    import numpy as np

    import hfteama.sorter.robot as robot
    from hfteama.arms.kinematics import ArmFrame
    from hfteama.arms.so101_ik import LOWER, UPPER, fk, solve_ik

    q0 = solve_ik([0.2, 0.0, 0.08]).q
    frame = ArmFrame()

    class FakeRobot:
        def __init__(self):
            self.pos = frame.to_lerobot(q0, clip=False)
            self.sent = []

        def get_observation(self):
            return {**self.pos, "gripper.pos": 50.0}

        def send_action(self, act):
            self.sent.append({k: v for k, v in act.items() if k != "gripper.pos"})
            self.pos = {k: v for k, v in act.items() if k != "gripper.pos"}

    import threading

    drv = robot.ArmDriver.__new__(robot.ArmDriver)
    drv.stop_event = threading.Event()
    drv.side, drv.frame, drv.limits = "test", frame, (LOWER, UPPER)
    drv.motion = {"max_joint_speed_deg_s": 45.0, "rate_hz": 1000}
    drv.gripper_target = 50.0
    drv.robot = FakeRobot()
    q = drv.move_line(q0, [0.2, 0.0, 0.0], yaw=None)
    assert fk(q)[:3, 3] == pytest.approx([0.2, 0.0, 0.0], abs=2e-3)
    path = np.array([[a[k] for k in sorted(a)] for a in drv.robot.sent])
    steps = np.abs(np.diff(path, axis=0)).max(axis=1)
    assert steps.max() <= 45.0 / 1000 * 1.01  # deg per tick at 45 deg/s
    moving = steps[: int(len(steps) * 0.9)]
    # after the start the arm keeps moving: no tick without motion before the end
    assert (moving[len(moving) // 10 :] > 1e-6).all()
