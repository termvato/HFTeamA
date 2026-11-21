"""Basic sanity tests for the package."""

import hfteama


def test_version_is_defined() -> None:
    assert isinstance(hfteama.__version__, str)
    assert hfteama.__version__
