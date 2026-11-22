"""Unit tests for asynchronous logger utility."""

from trainbench.utils import log_async, shutdown_logger


def test_log_async_no_active_run() -> None:
  # When wandb has no active run or is not imported, log_async should safely no-op
  log_async({"metric/loss": 0.5}, step=1)
  log_async({"metric/acc": 0.95})


def test_shutdown_logger() -> None:
  # Calling shutdown_logger should cleanly terminate the background executor
  shutdown_logger(wait=True)
