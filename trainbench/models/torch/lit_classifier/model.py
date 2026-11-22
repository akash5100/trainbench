"""LiT (Locked-image Tuning) Multi-Label Classifier Model."""

from __future__ import annotations

import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from .config import LitClassifierConfig


# ━━━━━━━━ Projection Head ━━━━━━━━


class MultiLabelHead(nn.Module):
  """Two-stage MLP projection head with LayerNorm, GELU, and Dropout."""

  def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.2, hidden_dim: int = 512) -> None:
    """Initializes multi-stage projection head with normalized intermediate layers."""
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(embed_dim, embed_dim),
      nn.LayerNorm(embed_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(embed_dim, hidden_dim),
      nn.LayerNorm(hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, num_classes),
    )
    self.apply(self._init_weights)

  def _init_weights(self, module: nn.Module) -> None:
    """Initializes linear weights with truncated normal distribution and LayerNorm with unit scale."""
    if isinstance(module, nn.Linear):
      nn.init.trunc_normal_(module.weight, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
      nn.init.zeros_(module.bias)
      nn.init.ones_(module.weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Computes logits from pooled embeddings."""
    return self.layers(x)


# ━━━━━━━━ Model Implementation ━━━━━━━━


class LitClassifierModel(nn.Module):
  """Multi-label image classifier with frozen or fine-tunable vision backbone and multi-stage head."""

  def __init__(self, config: LitClassifierConfig | None = None, pretrained: bool = True) -> None:
    """Initializes vision backbone (loading only the vision tower) and classification head."""
    super().__init__()
    self.config = config if config is not None else LitClassifierConfig()

    if pretrained:
      self.vision: nn.Module = SiglipVisionModel.from_pretrained(self.config.model_id)
    else:
      vcfg = SiglipVisionConfig(
        hidden_size=self.config.embed_dim,
        image_size=self.config.image_size,
      )
      self.vision = SiglipVisionModel(vcfg)

    self.is_frozen = self.config.freeze_backbone
    if self.is_frozen:
      self.vision.eval()
      for param in self.vision.parameters():
        param.requires_grad = False

    self.head = MultiLabelHead(
      embed_dim=self.config.embed_dim,
      num_classes=self.config.num_classes,
      dropout=self.config.dropout,
    )

  @classmethod
  def from_pretrained(
    cls,
    model_id: str,
    num_classes: int = 80,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
  ) -> LitClassifierModel:
    """Constructs LitClassifierModel and loads pretrained vision backbone weights."""
    config = LitClassifierConfig.from_pretrained(
      model_id=model_id,
      num_classes=num_classes,
      dropout=dropout,
      freeze_backbone=freeze_backbone,
    )
    return cls(config=config, pretrained=True)

  def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
    """Extracts post-MAP pooled embeddings [B, embed_dim] from vision backbone."""
    if self.is_frozen:
      with torch.no_grad():
        outputs = self.vision(pixel_values=pixel_values)
    else:
      outputs = self.vision(pixel_values=pixel_values)
    return outputs.pooler_output

  def forward_head(self, embeddings: torch.Tensor) -> torch.Tensor:
    """Computes multi-label logits from pooled embeddings via projection head."""
    return self.head(embeddings)

  def forward(self, pixel_values: torch.Tensor | None = None, embeddings: torch.Tensor | None = None) -> torch.Tensor:
    """Computes class logits from either pixel values or pre-extracted embeddings.

    Fails fast with ValueError if neither or both arguments are provided.
    """
    if pixel_values is not None and embeddings is None:
      features = self.forward_features(pixel_values)
    elif embeddings is not None and pixel_values is None:
      features = embeddings
    else:
      raise ValueError("Exactly one of `pixel_values` or `embeddings` must be provided to forward().")

    return self.forward_head(features)

  def train(self, mode: bool = True) -> LitClassifierModel:
    """Sets training mode while respecting backbone freeze invariant."""
    super().train(mode)
    if self.is_frozen:
      self.vision.eval()
    return self
