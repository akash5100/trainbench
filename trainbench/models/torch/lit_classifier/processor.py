"""Image processor for LiT (Locked-image Tuning) Multi-Label Classifier."""

from __future__ import annotations

import torch
from torchvision import transforms

from .config import LitClassifierConfig


# ━━━━━━━━ Image Processor ━━━━━━━━


class LitClassifierProcessor:
  """Image preprocessing and normalization pipeline for LitClassifier."""

  def __init__(self, config: LitClassifierConfig | None = None) -> None:
    """Initializes preprocessing transforms based on model configuration."""
    self.config = config if config is not None else LitClassifierConfig()
    self.transform = transforms.Compose(
      [
        transforms.Resize(
          (self.config.image_size, self.config.image_size),
          interpolation=transforms.InterpolationMode.BILINEAR,
          antialias=True,
        ),
        transforms.Normalize(
          mean=self.config.image_mean,
          std=self.config.image_std,
        ),
      ]
    )

  def __call__(self, images: torch.Tensor) -> torch.Tensor:
    """Preprocesses and normalizes image tensor(s) for the model backbone."""
    return self.transform(images)
