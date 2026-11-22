"""Gradient clipping and norm inspection utilities."""

from __future__ import annotations

import torch
from torch import nn


# ━━━━━━━━ Gradient Utilities ━━━━━━━━


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
  """Applies L2 gradient clipping to active parameters and returns scalar total norm."""
  params = [p for p in model.parameters() if p.grad is not None]
  if not params:
    return 0.0
  norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
  return float(norm.item() if isinstance(norm, torch.Tensor) else norm)
