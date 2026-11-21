"""Tests for PEP 562 lazy importer machinery."""

import pytest

from trainbench.lazy import create_lazy_importer


def test_lazy_importer_resolves_and_caches() -> None:
  mapping = {"create_lazy_importer": "lazy"}
  lazy_getattr = create_lazy_importer(mapping, package="trainbench")

  val = lazy_getattr("create_lazy_importer")
  assert callable(val)


def test_lazy_importer_raises_attribute_error() -> None:
  mapping = {"create_lazy_importer": "lazy"}
  lazy_getattr = create_lazy_importer(mapping, package="trainbench")

  with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
    lazy_getattr("nonexistent")
