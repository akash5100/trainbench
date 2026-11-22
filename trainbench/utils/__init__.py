"""General utilities for training, runtime, telemetry, and hardware management."""

from typing import TYPE_CHECKING

from trainbench.lazy import create_lazy_importer


if TYPE_CHECKING:
  from .logger import log_async, shutdown_logger

_LAZY_IMPORTS = {
  "log_async": "logger",
  "shutdown_logger": "logger",
}

__all__ = [
  "log_async",
  "shutdown_logger",
]

__getattr__ = create_lazy_importer(_LAZY_IMPORTS, __name__)
