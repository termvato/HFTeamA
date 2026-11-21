"""`hfteama sort` command line, without hardware."""

import pytest

pytest.importorskip("numpy")

from hfteama.__main__ import main  # noqa: E402


def test_goto_dry_run_plans_without_hardware(capsys) -> None:
    assert main(["-n", "sort", "goto", "left", "--tag", "2", "--z", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "IK ok=True" in out


def test_goto_needs_a_target() -> None:
    with pytest.raises(SystemExit):
        main(["-n", "sort", "goto", "left"])


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
    drv = ArmDriver("right", cfg["arms"]["right"], cfg["motion"])
    with pytest.raises(SystemExit, match="not connected"):
        drv.connect()
