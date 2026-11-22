"""PyTorch runtime utilities, device discovery, and distributed helpers."""

from typing import TYPE_CHECKING

from trainbench.lazy import create_lazy_importer


if TYPE_CHECKING:
  from .amp import configure_mixed_precision, get_autocast_dtype
  from .device import get_device
  from .distributed import (
    get_rank,
    get_world_size,
    is_dist_available_and_initialized,
    is_main_process,
    print_rank0,
    setup_ddp,
    setup_for_distributed,
  )
  from .gradient import clip_gradients
  from .seed import seed_everything

_LAZY_IMPORTS = {
  "get_device": "device",
  "get_autocast_dtype": "amp",
  "configure_mixed_precision": "amp",
  "seed_everything": "seed",
  "clip_gradients": "gradient",
  "setup_for_distributed": "distributed",
  "is_dist_available_and_initialized": "distributed",
  "get_world_size": "distributed",
  "get_rank": "distributed",
  "is_main_process": "distributed",
  "print_rank0": "distributed",
  "setup_ddp": "distributed",
}

__all__ = [
  "get_device",
  "get_autocast_dtype",
  "configure_mixed_precision",
  "seed_everything",
  "clip_gradients",
  "setup_for_distributed",
  "is_dist_available_and_initialized",
  "get_world_size",
  "get_rank",
  "is_main_process",
  "print_rank0",
  "setup_ddp",
]

__getattr__ = create_lazy_importer(_LAZY_IMPORTS, __name__)
