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
            p = Piece(x=0.30, y=0.22, length=0.06, width=0.02, yaw=0.0, area_cm2=12.0)
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

        def go_home(self):
            pass

    monkeypatch.setattr(run, "Eye", FakeEye)
    monkeypatch.setattr(robot, "ArmDriver", FakeDriver)
    answers = iter(["1", "f2", "l", "ok", "q", "n"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["sort", "jog", "rakesh", "--step", "10", "--no-viewer"]) == 0
    out = capsys.readouterr().out
    # rakesh faces the camera: forward = table -y, its left = table +x
    assert "recorded: real = camera + (+1.0, -2.0) cm" in out
    assert "placement from 1 points" in out
