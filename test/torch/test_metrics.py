"""Unit tests for multi-label classification metrics."""

import torch

from trainbench.metrics.torch import compute_multi_label_metrics


def test_compute_multi_label_metrics_shapes_and_values() -> None:
  logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
  targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

  metrics = compute_multi_label_metrics(logits, targets)

  assert "loss" in metrics
  assert "macro_f1" in metrics
  assert "map" in metrics
  assert "neg_prob_mean" in metrics
  assert "majority_fp" in metrics

  assert metrics["macro_f1"] == 1.0
  assert metrics["map"] == 1.0
  assert metrics["majority_fp"] == 0
  assert metrics["neg_prob_mean"] < 1e-4
