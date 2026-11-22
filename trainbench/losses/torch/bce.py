"""Numerically stable Binary Cross Entropy with logits loss."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


# ━━━━━━━━ Loss Implementation ━━━━━━━━


class BCEWithLogitsLoss(nn.Module):
  """Numerically stable Binary Cross Entropy loss directly computed from logits in float32."""

  def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
    """Initializes BCE loss with reduction mode."""
    super().__init__()
    self.reduction = reduction

  def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes log-sum-exp BCE in float32 precision for AMP/FP16 stability."""
    z = logits.float()
    y = targets.float()

    # Stable formulation: max(z, 0) - z * y + log(1 + exp(-|z|))
    loss = torch.clamp(z, min=0.0) - z * y + torch.log1p(torch.exp(-z.abs()))

    if self.reduction == "mean":
      return loss.mean()
    if self.reduction == "sum":
      return loss.sum()
    return loss
