"""Unit tests for LiT Classifier configuration, processor, and head."""

import torch

from trainbench.models.torch.lit_classifier import LitClassifierConfig, LitClassifierModel, LitClassifierProcessor
from trainbench.models.torch.lit_classifier.model import MultiLabelHead


def test_lit_classifier_config_defaults() -> None:
  cfg = LitClassifierConfig()
  assert cfg.model_id == "google/siglip2-base-patch16-256"
  assert cfg.image_size == 256
  assert cfg.embed_dim == 768
  assert cfg.num_classes == 80
  assert cfg.image_mean == [0.5, 0.5, 0.5]
  assert cfg.image_std == [0.5, 0.5, 0.5]


def test_lit_classifier_processor_direct_warp() -> None:
  cfg = LitClassifierConfig(image_size=224)
  proc = LitClassifierProcessor(cfg)

  # Test single image tensor [3, 400, 500]
  img = torch.rand(3, 400, 500)
  out = proc(img)
  assert out.shape == (3, 224, 224)

  # Test batch tensor [4, 3, 300, 300]
  batch = torch.rand(4, 3, 300, 300)
  out_batch = proc(batch)
  assert out_batch.shape == (4, 3, 224, 224)


def test_multilabel_head() -> None:
  head = MultiLabelHead(embed_dim=768, num_classes=80, dropout=0.1)
  embeds = torch.randn(4, 768)
  logits = head(embeds)
  assert logits.shape == (4, 80)


def test_lit_classifier_model_forward() -> None:
  cfg = LitClassifierConfig(model_id="google/siglip2-base-patch16-256", image_size=256, num_classes=80)
  model = LitClassifierModel(cfg, pretrained=False)
  model.eval()

  dummy_pixels = torch.randn(2, 3, 256, 256)
  with torch.no_grad():
    logits_pixels = model(pixel_values=dummy_pixels)
  assert logits_pixels.shape == (2, 80)

  with torch.no_grad():
    embeds = model.forward_features(dummy_pixels)
    logits_embeds = model(embeddings=embeds)
  assert logits_embeds.shape == (2, 80)
  assert torch.allclose(logits_pixels, logits_embeds, atol=1e-5)


def test_lit_classifier_from_pretrained() -> None:
  model = LitClassifierModel.from_pretrained("google/siglip2-base-patch16-256", num_classes=80)
  assert model.config.embed_dim == 768
  assert model.config.freeze_backbone is True
