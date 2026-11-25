"""Unit tests for multi-label classification metrics."""

import torch

from trainbench.metrics.torch import compute_multi_label_metrics, compute_per_class_confusion


def test_compute_multi_label_metrics_shapes_and_values() -> None:
  logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
  targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

  metrics = compute_multi_label_metrics(logits, targets)

  assert "loss" in metrics
  assert "macro_precision" in metrics
  assert "macro_recall" in metrics
  assert "macro_f1" in metrics
  assert "micro_precision" in metrics
  assert "micro_recall" in metrics
  assert "micro_f1" in metrics
  assert "map" in metrics
  assert "pos_prob_mean" in metrics
  assert "neg_prob_mean" in metrics
  assert "majority_fp" in metrics

  assert metrics["macro_precision"] == 1.0
  assert metrics["macro_recall"] == 1.0
  assert metrics["macro_f1"] == 1.0
  assert metrics["micro_precision"] == 1.0
  assert metrics["micro_recall"] == 1.0
  assert metrics["micro_f1"] == 1.0
  assert metrics["map"] == 1.0
  assert metrics["majority_fp"] == 0
  assert metrics["pos_prob_mean"] > 0.9999
  assert metrics["neg_prob_mean"] < 1e-4


def test_compute_per_class_confusion_returns_all_classes() -> None:
  logits = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]])
  targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

  rows = compute_per_class_confusion(logits, targets)
  assert len(rows) == 3
  assert rows[0] == {"class_id": 0, "tp": 1, "fp": 0, "fn": 0, "tn": 1, "support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0}
  assert rows[2]["fp"] == 1
  assert rows[2]["tn"] == 1
