"""Hardware device discovery and diagnostic utilities for PyTorch."""

from __future__ import annotations

import torch


# ━━━━━━━━ Device Detection ━━━━━━━━


def get_device(verbose: bool = True) -> torch.device:
  """Resolves the preferred torch compute device (CUDA -> MPS -> CPU) with diagnostics."""
  if torch.cuda.is_available():
    device = torch.device("cuda")
    if verbose:
      gpu_count = torch.cuda.device_count()
      print(f"Using device: cuda ({gpu_count} GPU{'s' if gpu_count > 1 else ''} available)")
      for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB, Compute Capability {props.major}.{props.minor})")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
    if verbose:
      print("Using device: mps (Apple Silicon)")
  else:
    device = torch.device("cpu")
    if verbose:
      print("Using device: cpu")

  return device
