"""Distributed Data Parallel (DDP) initialization and rank utilities."""

from __future__ import annotations

import builtins
import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


# ━━━━━━━━ Distributed Setup ━━━━━━━━


def setup_for_distributed(is_master: bool) -> None:  # slop-ok: DDP process stdout filter
  """Disables built-in print statements on non-master worker processes."""
  original_print = builtins.print

  def filtered_print(*args: Any, **kwargs: Any) -> None:
    force = kwargs.pop("force", False)
    if is_master or force:
      original_print(*args, **kwargs)

  builtins.print = filtered_print


def is_dist_available_and_initialized() -> bool:
  """Returns True if distributed package is available and process group is initialized."""
  return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:  # slop-ok: distributed rank helper
  """Returns global world size across all distributed nodes."""
  if not is_dist_available_and_initialized():
    return 1
  return dist.get_world_size()


def get_rank() -> int:
  """Returns global rank of current process."""
  if not is_dist_available_and_initialized():
    return 0
  return dist.get_rank()


def is_main_process() -> bool:  # slop-ok: distributed rank helper
  """Returns True if current process is rank 0."""
  return get_rank() == 0


def print_rank0(*args: Any, **kwargs: Any) -> None:
  """Prints messages only from rank 0 (main process)."""
  if is_main_process():
    builtins.print(*args, **kwargs)


def setup_ddp(timeout_minutes: int = 30) -> tuple[int, bool]:
  """Initializes NCCL process group if launched under torchrun.

  Returns:
    tuple of (local_rank: int, is_ddp: bool).
  """
  if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
      backend="nccl",
      timeout=timedelta(minutes=timeout_minutes),
    )
    setup_for_distributed(dist.get_rank() == 0)
    return local_rank, True

  return 0, False
