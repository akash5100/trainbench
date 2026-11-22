"""Unit tests for Sigmoid Knowledge Distillation and BCE loss."""

import torch
import torch.nn.functional as F

from trainbench.losses.torch import BCEWithLogitsLoss, SigmoidDistillationLoss


def test_bce_with_logits_loss_reductions() -> None:
  logits = torch.randn(4, 10, requires_grad=True)
  targets = torch.randint(0, 2, (4, 10)).float()

  for reduction in ("mean", "sum", "none"):
    loss_custom = BCEWithLogitsLoss(reduction=reduction)(logits, targets)
    loss_torch = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction=reduction)
    assert torch.allclose(loss_custom, loss_torch, atol=1e-6)


def test_bce_with_logits_loss_extreme_values() -> None:
  loss_fn = BCEWithLogitsLoss(reduction="mean")
  logits = torch.tensor([[-100.0, 100.0], [100.0, -100.0]])
  targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

  loss = loss_fn(logits, targets)
  assert torch.isfinite(loss)
  assert loss.item() < 1e-4


def test_bce_with_logits_loss_fp16_gradients() -> None:
  loss_fn = BCEWithLogitsLoss(reduction="mean")
  logits = torch.randn(4, 8, dtype=torch.float16, requires_grad=True)
  targets = torch.randint(0, 2, (4, 8)).to(torch.float16)

  loss = loss_fn(logits, targets)
  loss.backward()

  assert logits.grad is not None
  assert torch.isfinite(logits.grad).all()


def test_sigmoid_distillation_alpha_zero() -> None:
  loss_fn = SigmoidDistillationLoss(alpha=0.0, temperature=1.0)
  student_logits = torch.randn(4, 10, requires_grad=True)
  teacher_logits = torch.randn(4, 10)
  targets = torch.randint(0, 2, (4, 10)).float()

  out = loss_fn(student_logits, teacher_logits, targets)
  expected_gt = F.binary_cross_entropy_with_logits(student_logits.float(), targets)

  assert torch.allclose(out["loss"], expected_gt, atol=1e-6)
  assert torch.allclose(out["loss_gt"], expected_gt, atol=1e-6)
  assert torch.equal(out["loss_kd"], torch.zeros_like(expected_gt))


def test_sigmoid_distillation_temperature_scaling() -> None:
  loss_fn_t1 = SigmoidDistillationLoss(alpha=1.0, temperature=1.0)
  loss_fn_t4 = SigmoidDistillationLoss(alpha=1.0, temperature=4.0)

  student_logits = torch.zeros(1, 1)
  teacher_logits = torch.tensor([[-5.0]])
  targets = torch.zeros(1, 1)

  out_t1 = loss_fn_t1(student_logits, teacher_logits, targets)
  out_t4 = loss_fn_t4(student_logits, teacher_logits, targets)

  assert out_t4["loss"].item() > out_t1["loss"].item()


def test_sigmoid_distillation_fp16_finite_gradients() -> None:
  loss_fn = SigmoidDistillationLoss(alpha=0.5, temperature=2.0)
  student_logits = torch.randn(2, 5, dtype=torch.float16, requires_grad=True)
  teacher_logits = torch.randn(2, 5, dtype=torch.float16)
  targets = torch.randint(0, 2, (2, 5)).to(torch.float16)

  out = loss_fn(student_logits, teacher_logits, targets)
  out["loss"].backward()

  assert student_logits.grad is not None
  assert torch.isfinite(student_logits.grad).all()
