"""Multi-label classification and ranking evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch

from trainbench.losses.torch import BCEWithLogitsLoss


# ━━━━━━━━ Evaluation Metrics ━━━━━━━━


def compute_multi_label_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
  """Computes multi-label evaluation metrics: BCE loss, macro F1, micro F1, mAP, and negative calibration statistics.

  Args:
    logits: Unnormalized model predictions of shape [N, num_classes].
    targets: Binary ground-truth labels of shape [N, num_classes].

  Returns:
    Dictionary containing loss, macro_f1, micro_f1, map, neg_prob_mean, and majority_fp.
  """
  logits_f32 = logits.float()
  targets_f32 = targets.float()

  # Pure ground-truth BCE loss
  bce_fn = BCEWithLogitsLoss(reduction="mean")
  loss_val = bce_fn(logits_f32, targets_f32).item()

  # Sigmoid probabilities and thresholded binary predictions (threshold = 0.5)
  probs = torch.sigmoid(logits_f32)
  preds = (probs >= 0.5).float()

  # Micro Precision, Recall, and F1 (global TP, FP, FN across all classes and samples)
  total_tp = ((preds == 1.0) & (targets_f32 == 1.0)).sum().float().item()
  total_fp = ((preds == 1.0) & (targets_f32 == 0.0)).sum().float().item()
  total_fn = ((preds == 0.0) & (targets_f32 == 1.0)).sum().float().item()
  micro_precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0.0 else 0.0
  micro_recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0.0 else 0.0
  micro_denom = 2.0 * total_tp + total_fp + total_fn
  micro_f1 = (2.0 * total_tp / micro_denom) if micro_denom > 0.0 else 0.0

  # Macro Precision, Recall, and F1 (averaged across all classes with at least 1 positive sample in validation)
  prec_list: list[float] = []
  rec_list: list[float] = []
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
      prec_list.append(prec)
      rec_list.append(rec)
      f1_list.append(f1)
  macro_precision = float(np.mean(prec_list)) if prec_list else 0.0
  macro_recall = float(np.mean(rec_list)) if rec_list else 0.0
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

  # Mean predicted probability on true-positive and true-negative entries (calibration check)
  pos_mask = targets_f32 == 1.0
  pos_prob_mean = probs[pos_mask].mean().item() if pos_mask.any() else 0.0
  neg_mask = targets_f32 == 0.0
  neg_prob_mean = probs[neg_mask].mean().item() if neg_mask.any() else 0.0

  # False positive count on highest-frequency class
  maj_class = int(targets_f32.sum(dim=0).argmax().item())
  maj_preds = preds[:, maj_class]
  maj_targets = targets_f32[:, maj_class]
  majority_fp = int(((maj_preds == 1.0) & (maj_targets == 0.0)).sum().item())

  return {
    "loss": loss_val,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "micro_precision": micro_precision,
    "micro_recall": micro_recall,
    "micro_f1": micro_f1,
    "map": map_score,
    "pos_prob_mean": pos_prob_mean,
    "neg_prob_mean": neg_prob_mean,
    "majority_fp": float(majority_fp),
  }


def compute_per_class_confusion(logits: torch.Tensor, targets: torch.Tensor) -> list[dict[str, int | float]]:
  """Computes exact TP/FP/FN/TN confusion counts and precision/recall/F1 for every class.

  Args:
    logits: Unnormalized model predictions of shape [N, num_classes].
    targets: Binary ground-truth labels of shape [N, num_classes].

  Returns:
    Untruncated list of per-class metric dictionaries ordered by class index.
  """
  preds = (torch.sigmoid(logits.float()) >= 0.5).float()
  targets_f32 = targets.float()
  num_classes = targets_f32.shape[1]
  rows: list[dict[str, int | float]] = []

  for c in range(num_classes):
    c_preds = preds[:, c]
    c_targets = targets_f32[:, c]
    tp = int(((c_preds == 1.0) & (c_targets == 1.0)).sum().item())
    fp = int(((c_preds == 1.0) & (c_targets == 0.0)).sum().item())
    fn = int(((c_preds == 0.0) & (c_targets == 1.0)).sum().item())
    tn = int(((c_preds == 0.0) & (c_targets == 0.0)).sum().item())
    prec = (float(tp) / float(tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (float(tp) / float(tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    rows.append(
      {
        "class_id": c,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": tp + fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
      }
    )
  return rows
