"""Smoke test for trainbench package import."""

import trainbench


def test_version() -> None:
  assert trainbench.__version__ == "0.1.0"
