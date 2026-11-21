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
