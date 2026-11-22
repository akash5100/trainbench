"""Sigmoid Knowledge Distillation loss for multi-label classification."""

from __future__ import annotations

import torch
from torch import nn

from .bce import BCEWithLogitsLoss


# ━━━━━━━━ Loss Implementation ━━━━━━━━


class SigmoidDistillationLoss(nn.Module):
  """Knowledge distillation loss combining multi-label sigmoid soft targets and ground-truth BCE."""

  def __init__(self, alpha: float = 0.5, temperature: float = 1.0) -> None:
    """Initializes distillation loss with balance coefficient alpha and temperature scaling."""
    super().__init__()
    if not (0.0 <= alpha <= 1.0):
      raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
    if temperature <= 0.0:
      raise ValueError(f"temperature must be positive, got {temperature}")

    self.alpha = float(alpha)
    self.temperature = float(temperature)
    self.bce = BCEWithLogitsLoss(reduction="mean")

  def forward(
    self,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Computes composite loss and decomposed components in float32 for numerical stability."""
    student_f32 = student_logits.float()
    teacher_f32 = teacher_logits.float()
    targets_f32 = targets.float()

    loss_gt = self.bce(student_f32, targets_f32)

    if self.alpha == 0.0:
      loss_kd = torch.zeros_like(loss_gt)
    elif self.temperature == 1.0:
      soft_targets = torch.sigmoid(teacher_f32).detach()
      loss_kd = self.bce(student_f32, soft_targets)
    else:
      soft_targets = torch.sigmoid(teacher_f32 / self.temperature).detach()
      loss_kd = self.bce(student_f32 / self.temperature, soft_targets) * self.temperature

    loss = self.alpha * loss_kd + (1.0 - self.alpha) * loss_gt
    return {
      "loss": loss,
      "loss_kd": loss_kd.detach(),
      "loss_gt": loss_gt.detach(),
    }
