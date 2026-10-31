"""Smoke test: the package imports and reports a version."""

import sinchole


def test_import_and_version():
    assert isinstance(sinchole.__version__, str)
