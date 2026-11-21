"""Configuration for LiT (Locked-image Tuning) Multi-Label Classifier."""

from __future__ import annotations

from dataclasses import dataclass, field

from trainbench.constants import SIGLIP_MEAN, SIGLIP_STD


# ━━━━━━━━ Model Configuration ━━━━━━━━


@dataclass
class LitClassifierConfig:
  """Configuration for LitClassifier model."""

  model_id: str = "google/siglip2-base-patch16-256"
  image_size: int = 224
  image_mean: list[float] = field(default_factory=lambda: list(SIGLIP_MEAN))
  image_std: list[float] = field(default_factory=lambda: list(SIGLIP_STD))
  embed_dim: int = 768
  num_classes: int = 80
  dropout: float = 0.1
  freeze_backbone: bool = True

  @classmethod
  def from_pretrained(
    cls,
    model_id: str,
    num_classes: int = 80,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
  ) -> LitClassifierConfig:
    """Constructs LitClassifierConfig by resolving dimensions and normalization from HF pretrained config."""
    from transformers import AutoConfig, AutoImageProcessor

    cfg = AutoConfig.from_pretrained(model_id)
    vcfg = getattr(cfg, "vision_config", cfg)
    embed_dim = int(getattr(vcfg, "hidden_size", 768))
    image_size = int(getattr(vcfg, "image_size", 224))

    proc = AutoImageProcessor.from_pretrained(model_id)
    if proc.image_mean is None or proc.image_std is None:
      raise ValueError(f"Image processor for '{model_id}' does not provide image_mean/image_std.")
    image_mean = [float(x) for x in proc.image_mean]
    image_std = [float(x) for x in proc.image_std]

    return cls(
      model_id=model_id,
      image_size=image_size,
      image_mean=image_mean,
      image_std=image_std,
      embed_dim=embed_dim,
      num_classes=num_classes,
      dropout=dropout,
      freeze_backbone=freeze_backbone,
    )
