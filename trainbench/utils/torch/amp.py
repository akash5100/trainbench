"""Automatic Mixed Precision (AMP) and TF32 configuration utilities."""

from __future__ import annotations

import torch


# ━━━━━━━━ Precision Configuration ━━━━━━━━


def get_autocast_dtype(device: torch.device | None = None) -> torch.dtype:
  """Determines recommended autocast dtype based on accelerator compute capability.

  Ampere and newer GPUs (Compute Capability >= 8.0, e.g. A100, H100, RTX 3090/4090) use bfloat16.
  Pre-Ampere GPUs (Compute Capability < 8.0, e.g. Tesla T4, V100) use float16.
  MPS and CPU fall back to bfloat16 if supported, otherwise float32.
  """
  target_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if target_device.type == "cuda" and torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability(target_device)
    if major >= 8:
      return torch.bfloat16
    return torch.float16

  if target_device.type == "mps":
    return torch.bfloat16

  return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


def configure_mixed_precision(use_mixed_precision: bool, device: torch.device | None = None) -> torch.amp.GradScaler:
  """Configures TF32 matrix multiplication and returns a calibrated GradScaler."""
  target_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if not use_mixed_precision or target_device.type != "cuda" or not torch.cuda.is_available():
    return torch.amp.GradScaler("cuda" if target_device.type == "cuda" else "cpu", enabled=False)

  major, minor = torch.cuda.get_device_capability(target_device)
  if major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"TF32 matmul and cuDNN enabled (Compute Capability {major}.{minor}).")
  else:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"TF32 disabled for Compute Capability {major}.{minor} (Pre-Ampere GPU, e.g. Tesla T4).")

  dtype = get_autocast_dtype(target_device)
  print(f"Configuring AMP with dtype={dtype}.")
  return torch.amp.GradScaler(
    "cuda",
    enabled=True,
    init_scale=2**14,
    growth_factor=2.0,
    backoff_factor=0.5,
  )
