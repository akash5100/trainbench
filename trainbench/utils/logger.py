"""Asynchronous, non-blocking telemetry and metrics logging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any


# Background executor for non-blocking telemetry dispatch
_LOG_EXECUTOR = ThreadPoolExecutor(max_workers=1)


# ━━━━━━━━ Async Logging ━━━━━━━━


def log_async(metrics: dict[str, Any], step: int | None = None) -> None:
  """Dispatches metrics asynchronously to WandB if an active run exists."""
  try:
    import wandb

    if wandb.run is not None:
      if step is not None:
        _LOG_EXECUTOR.submit(wandb.log, metrics, step=step)
      else:
        _LOG_EXECUTOR.submit(wandb.log, metrics)
  except ImportError:
    pass


def shutdown_logger(wait: bool = False) -> None:
  """Shuts down the background logging thread pool executor."""
  _LOG_EXECUTOR.shutdown(wait=wait)


def init_wandb(
  project: str,
  name: str,
  config: dict[str, Any] | None = None,
  enabled: bool = True,
  rank: int = 0,
) -> None:
  """Initializes a WandB run on rank 0 when enabled."""
  if not enabled or rank != 0:
    return
  try:
    import wandb

    wandb.init(project=project, name=name, config=config)
  except ImportError:
    print("WandB requested but not installed. Skipping remote logging.", flush=True)


def finish_wandb(enabled: bool = True, rank: int = 0) -> None:
  """Flushes queued async logs and closes any active WandB run on rank 0."""
  if not enabled or rank != 0:
    return
  try:
    _LOG_EXECUTOR.submit(lambda: None).result()
    import wandb

    if wandb.run is not None:
      wandb.finish()
  except Exception:
    pass
