"""Multi-label classification and ranking evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch

from trainbench.losses.torch import BCEWithLogitsLoss


# ━━━━━━━━ Evaluation Metrics ━━━━━━━━


def compute_multi_label_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
  """Computes multi-label evaluation metrics: BCE loss, macro F1, mAP, and negative calibration statistics.

  Args:
    logits: Unnormalized model predictions of shape [N, num_classes].
    targets: Binary ground-truth labels of shape [N, num_classes].

  Returns:
    Dictionary containing loss, macro_f1, map, neg_prob_mean, and majority_fp.
  """
  logits_f32 = logits.float()
  targets_f32 = targets.float()

  # Pure ground-truth BCE loss
  bce_fn = BCEWithLogitsLoss(reduction="mean")
  loss_val = bce_fn(logits_f32, targets_f32).item()

  # Sigmoid probabilities and thresholded binary predictions (threshold = 0.5)
  probs = torch.sigmoid(logits_f32)
  preds = (probs >= 0.5).float()

  # Macro F1 (averaged across all classes with at least 1 positive sample in validation)
  f1_list: list[float] = []
  num_classes = targets_f32.shape[1]

  for c in range(num_classes):
    c_targets = targets_f32[:, c]
    if c_targets.sum() > 0:
      c_preds = preds[:, c]
      tp = ((c_preds == 1.0) & (c_targets == 1.0)).sum().float()
      fp = ((c_preds == 1.0) & (c_targets == 0.0)).sum().float()
      fn = ((c_preds == 0.0) & (c_targets == 1.0)).sum().float()
      prec = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
      rec = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
      f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
      f1_list.append(f1)
  macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

  # Mean Average Precision (threshold-independent ranking metric across positive classes)
  ap_list: list[float] = []
  for c in range(num_classes):
    c_targets = targets_f32[:, c]
    n_pos = int(c_targets.sum().item())
    if n_pos > 0:
      order = torch.argsort(probs[:, c], descending=True)
      sorted_targets = c_targets[order]
      cum_tp = torch.cumsum(sorted_targets, dim=0)
      ranks = torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
      precisions = cum_tp / ranks
      ap = ((precisions * sorted_targets).sum() / float(n_pos)).item()
      ap_list.append(ap)
  map_score = float(np.mean(ap_list)) if ap_list else 0.0

  # Mean predicted probability on true-negative entries (calibration check)
  neg_mask = targets_f32 == 0.0
  neg_prob_mean = probs[neg_mask].mean().item() if neg_mask.any() else 0.0

  # False positive count on highest-frequency class
  maj_class = int(targets_f32.sum(dim=0).argmax().item())
  maj_preds = preds[:, maj_class]
  maj_targets = targets_f32[:, maj_class]
  majority_fp = int(((maj_preds == 1.0) & (maj_targets == 0.0)).sum().item())

  return {
    "loss": loss_val,
    "macro_f1": macro_f1,
    "map": map_score,
    "neg_prob_mean": neg_prob_mean,
    "majority_fp": float(majority_fp),
  }
