"""Global configuration constants and normalization presets for trainbench."""

from __future__ import annotations


# ━━━━━━━━ Vision Normalization Presets ━━━━━━━━

# Standard ImageNet-1k RGB channel normalization statistics
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# SigLIP / Inception / CLIP symmetric RGB channel normalization [-1.0, 1.0]
SIGLIP_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
SIGLIP_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

__all__ = [
  "IMAGENET_MEAN",
  "IMAGENET_STD",
  "SIGLIP_MEAN",
  "SIGLIP_STD",
]
