"""Self-contained training and distillation pipeline for SigLIP2 on COCO 2017."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from trainbench.losses.torch import BCEWithLogitsLoss, SigmoidDistillationLoss
from trainbench.metrics.torch import compute_multi_label_metrics, compute_per_class_confusion
from trainbench.models.torch.lit_classifier import LitClassifierConfig, LitClassifierModel, LitClassifierProcessor, MultiLabelHead
from trainbench.utils import finish_wandb, init_wandb, log_async, shutdown_logger
from trainbench.utils.torch import clip_gradients, configure_mixed_precision, get_autocast_dtype, get_device, print_rank0, seed_everything, setup_ddp

from .data import build_dataloaders, load_coco_metadata


# ━━━━━━━━ Runtime Context ━━━━━━━━


@dataclass(frozen=True)
class RuntimeContext:
  """Bundles accelerator, mixed-precision, and distributed rank state."""

  device: torch.device
  scaler: torch.amp.GradScaler
  autocast_dtype: torch.dtype
  is_ddp: bool
  local_rank: int


# ━━━━━━━━ CLI Configuration ━━━━━━━━


def _model_short_tag(model_id: str) -> str:
  """Extracts a concise model architecture identifier for clean run naming."""
  name = Path(model_id).name.lower()
  for tag in ("so400m", "base", "large", "tiny", "small"):
    if tag in name:
      return tag
  return name


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments for SigLIP2 training and distillation."""
  parser = argparse.ArgumentParser(description="SigLIP2 Multi-Label Distillation on COCO 2017")

  # Model Backbones & Projections
  parser.add_argument("--teacher-model", type=str, default="google/siglip-so400m-patch14-384", help="Teacher vision backbone model ID")
  parser.add_argument("--student-model", type=str, default="google/siglip2-base-patch16-256", help="Student vision backbone model ID")
  parser.add_argument("--num-classes", type=int, default=80, help="Number of target classes")
  parser.add_argument("--dropout", type=float, default=0.1, help="Projection head dropout rate")
  parser.add_argument("--load-head", type=str, default=None, help="Optional checkpoint path to initialize student head")

  # Distillation Hyperparameters (Scalar)
  parser.add_argument(
    "--kd-alpha",
    "--alpha",
    dest="alpha",
    type=float,
    default=0.5,
    help="Knowledge distillation loss weight: loss = alpha * loss_kd + (1 - alpha) * loss_gt (0 = pure GT baseline)",
  )
  parser.add_argument(
    "--kd-temperature",
    "--temperature",
    dest="temperature",
    type=float,
    default=1.0,
    help="Knowledge distillation temperature scaling for soft target smoothing",
  )

  # Optimization Configuration
  parser.add_argument("--teacher-epochs", type=int, default=10, help="Epochs to train teacher projection head")
  parser.add_argument("--student-epochs", type=int, default=10, help="Epochs to train student projection head")
  parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for projection head")
  parser.add_argument("--eta-min-ratio", type=float, default=0.01, help="Minimum learning rate ratio: eta_min = lr * eta_min_ratio")
  parser.add_argument("--weight-decay", type=float, default=1e-2, help="Decoupled weight decay on 2D weights")
  parser.add_argument("--clip-grad", type=float, default=1.0, help="Maximum gradient L2 norm")
  parser.add_argument("--warmup-steps", type=int, default=50, help="Linear warmup steps")
  parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation micro-steps")

  # Data & Cache Configuration
  parser.add_argument("--data-root", type=str, default="data/coco2017", help="Path to COCO dataset root")
  parser.add_argument("--cache-dir", type=str, default=".cache", help="Directory to store extracted embeddings and teacher logits")
  parser.add_argument("--batch-size", type=int, default=64, help="Batch size per device")
  parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per device")
  parser.add_argument("--no-cache", action="store_true", default=False, help="Bypass and overwrite existing cache files")

  # Outputs & Telemetry
  parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save trained model heads")
  parser.add_argument("--run-name", type=str, default=None, help="Custom run name (defaults to student_T{temperature}_a{alpha})")
  parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
  parser.add_argument("--mixed-precision", action="store_true", default=True, help="Enable automatic mixed precision")
  parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile acceleration on projection head")
  parser.add_argument("--wandb", action="store_true", default=False, help="Enable asynchronous WandB logging")
  parser.add_argument("--project", type=str, default="sigmoid_distill", help="WandB project name")

  args = parser.parse_args()
  if args.grad_accum < 1:
    args.grad_accum = 1
  if args.run_name is None:
    student_tag = _model_short_tag(args.student_model)
    args.run_name = f"{student_tag}_T{args.temperature}_a{args.alpha}"
  return args


# ━━━━━━━━ Embedding Cache Helpers ━━━━━━━━


def forward_pass_and_cache_embeddings(
  model: LitClassifierModel,
  loader: DataLoader[Any],
  device: torch.device,
  cache_path: Path,
  autocast_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
  """Runs a forward extraction pass across a DataLoader and caches embeddings and image IDs to disk."""
  model.eval()
  all_embeds: list[torch.Tensor] = []
  all_targets: list[torch.Tensor] = []
  all_ids: list[int] = []

  total_batches = len(loader)
  overall_start = time.time()
  use_amp = autocast_dtype is not None and device.type == "cuda"
  effective_dtype = autocast_dtype if autocast_dtype is not None else torch.float32

  loader_iter = iter(loader)
  step = 0

  with torch.no_grad():
    while step < total_batches:
      t0 = time.time()
      try:
        images, targets, ids = next(loader_iter)
      except StopIteration:
        break
      step += 1
      t_data = time.time()

      images = images.to(device, non_blocking=True)
      with torch.amp.autocast(device_type=device.type, dtype=effective_dtype, enabled=use_amp):
        embeds = model.forward_features(images)

      if device.type == "cuda":
        torch.cuda.synchronize(device)
      t_fwd = time.time()

      all_embeds.append(embeds.float().cpu())
      all_targets.append(targets.cpu())
      if isinstance(ids, torch.Tensor):
        all_ids.extend(ids.tolist())
      else:
        all_ids.extend(list(ids))

      # step latency telemetry: data loader time vs GPU forward compute time
      if step % 20 == 0 or step == total_batches or step == 1:
        data_ms = (t_data - t0) * 1000.0
        fwd_ms = (t_fwd - t_data) * 1000.0
        dt_ms = (t_fwd - t0) * 1000.0
        batch_size = images.shape[0]
        img_s = batch_size / max(1e-5, (t_fwd - t0))
        elapsed_total = t_fwd - overall_start
        overall_speed = step / max(1e-5, elapsed_total)
        eta_s = (total_batches - step) / max(1e-5, overall_speed)
        rank_str = f"[GPU {device.index}] " if (device.type == "cuda" and device.index is not None) else ""
        print(
          f"  {rank_str}[Extraction] Batch {step:04d}/{total_batches:04d} | "
          f"data: {data_ms:6.1f}ms | fwd: {fwd_ms:6.1f}ms | dt: {dt_ms:6.1f}ms | "
          f"{img_s:6.1f} img/s | ETA: {eta_s:5.0f}s",
          flush=True,
        )

  embeds_cat = torch.cat(all_embeds, dim=0)
  targets_cat = torch.cat(all_targets, dim=0)
  ids_arr = np.array(all_ids, dtype=np.int64)

  cache_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    cache_path,
    embeddings=embeds_cat.numpy(),
    targets=targets_cat.numpy(),
    image_ids=ids_arr,
  )
  return embeds_cat, targets_cat, ids_arr


def load_or_extract_embeddings(
  prefix: str,
  split: str,
  cache_dir: Path,
  model: LitClassifierModel,
  loader: DataLoader[Any],
  ctx: RuntimeContext,
  no_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
  """Loads cached embeddings or extracts them across all available GPUs in parallel."""
  cache_path = cache_dir / f"{prefix}_embeds_{split}.npz"
  if cache_path.exists() and not no_cache:
    print_rank0(f"Loading cached {prefix} {split} embeddings from {cache_path}...")
    with np.load(cache_path) as data:
      embeds = torch.from_numpy(data["embeddings"])
      targets = torch.from_numpy(data["targets"])
      image_ids = data["image_ids"]
    return embeds, targets, image_ids

  world_size = dist.get_world_size() if (ctx.is_ddp and dist.is_initialized()) else 1

  if world_size > 1:
    dataset = loader.dataset
    total_samples = len(dataset)
    chunk = (total_samples + world_size - 1) // world_size
    start_idx = ctx.local_rank * chunk
    end_idx = min(total_samples, start_idx + chunk)

    shard_ds = Subset(dataset, range(start_idx, end_idx))
    shard_loader: DataLoader[Any] = DataLoader(
      shard_ds,
      batch_size=loader.batch_size,
      shuffle=False,
      num_workers=loader.num_workers,
      pin_memory=torch.cuda.is_available(),
    )
    shard_path = cache_dir / f"{prefix}_embeds_{split}_shard_{ctx.local_rank}.npz"
    print(f"[Rank {ctx.local_rank}] Extracting {prefix} {split} shard ({len(shard_ds)} samples)...", flush=True)
    forward_pass_and_cache_embeddings(model, shard_loader, ctx.device, shard_path, autocast_dtype=ctx.autocast_dtype)

    dist.barrier()

    if ctx.local_rank == 0:
      all_embeds: list[np.ndarray] = []
      all_targets: list[np.ndarray] = []
      all_ids: list[np.ndarray] = []

      for r in range(world_size):
        sp = cache_dir / f"{prefix}_embeds_{split}_shard_{r}.npz"
        with np.load(sp) as d:
          all_embeds.append(d["embeddings"])
          all_targets.append(d["targets"])
          all_ids.append(d["image_ids"])
        sp.unlink(missing_ok=True)

      np.savez_compressed(
        cache_path,
        embeddings=np.concatenate(all_embeds, axis=0),
        targets=np.concatenate(all_targets, axis=0),
        image_ids=np.concatenate(all_ids, axis=0),
      )
      print_rank0(f"Merged and saved {sum(len(x) for x in all_embeds)} {prefix} {split} embeddings to {cache_path}.")

    dist.barrier()

    with np.load(cache_path) as data:
      embeds = torch.from_numpy(data["embeddings"])
      targets = torch.from_numpy(data["targets"])
      image_ids = data["image_ids"]
    return embeds, targets, image_ids

  print_rank0(f"Extracting {prefix} {split} embeddings with frozen backbone...")
  embeds, targets, image_ids = forward_pass_and_cache_embeddings(model, loader, ctx.device, cache_path, autocast_dtype=ctx.autocast_dtype)
  print_rank0(f"Saved {len(embeds)} {prefix} {split} embeddings to {cache_path}.")
  return embeds, targets, image_ids


# ━━━━━━━━ Optimizer Parameter Filtering ━━━━━━━━


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
  """Configures AdamW with decoupled weight decay applied strictly to 2D+ projection kernels.

  Parameter Partitioning Rationale (Karpathy's nanoGPT ndim >= 2 heuristic):
    1. 2D+ tensors (weights, linear projections, conv kernels):
       Weight decay bounds the matrix Frobenius norm and Lipschitz constant, regularizing
       model capacity and reducing variance.
    2. 1D tensors (biases, LayerNorm/RMSNorm scale gamma and shift beta):
       - Biases: purely affine shifts (y = Wx + b). Decaying biases does not constrain model
         expressivity, but shifts activation centering away from optimal distributions.
       - Norm scales (gamma): Normalization layers scale variance to 1. Decaying gamma -> 0
         actively fights the normalization layer, artificially suppressing activation magnitude.
    3. Dimensionality check (p.ndim >= 2) is robust and architecture-agnostic compared to
       fragile layer name string-matching (e.g. 'bias', 'ln_1', 'layer_norm', 'scale').
  """
  decay_params: list[torch.nn.Parameter] = []
  no_decay_params: list[torch.nn.Parameter] = []

  for _, param in model.named_parameters():
    if not param.requires_grad:
      continue
    # 2D+ weights receive decoupled weight decay; 1D biases and norm scales receive 0.0 decay
    if param.ndim >= 2:
      decay_params.append(param)
    else:
      no_decay_params.append(param)

  param_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
  ]
  return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def unwrap_model(model: nn.Module) -> nn.Module:
  """Unwraps DDP and torch.compile wrappers to access the underlying module."""
  unwrapped = model.module if hasattr(model, "module") else model
  return getattr(unwrapped, "_orig_mod", unwrapped)


# ━━━━━━━━ Epoch Training & Validation Routines ━━━━━━━━


def train_one_epoch(
  model: nn.Module,
  dataloader: DataLoader[Any],
  optimizer: torch.optim.Optimizer,
  criterion: nn.Module,
  ctx: RuntimeContext,
  clip_grad: float = 1.0,
  accum_steps: int = 1,
  scheduler: Any = None,
  log_fn: Callable[[dict[str, Any], int], None] | None = None,
  global_step_start: int = 0,
) -> tuple[dict[str, float], int]:
  """Executes one training epoch with gradient accumulation, clipping, and telemetry."""
  model.train()
  total_loss = 0.0
  num_batches = 0
  num_updates = 0
  total_grad_norm = 0.0
  global_step = global_step_start
  grad_share: dict[str, float] = {}
  epoch_grad_share: dict[str, float] = {}

  for step, batch in enumerate(dataloader):
    # Transfer inputs to target accelerator device
    embeds, targets = batch[0].to(ctx.device), batch[1].to(ctx.device)

    # Forward pass and scaled loss calculation under mixed precision
    with torch.amp.autocast(device_type=ctx.device.type, dtype=ctx.autocast_dtype, enabled=ctx.scaler.is_enabled()):
      logits = model(embeds)
      if step == 0 and logits.requires_grad:
        logits.retain_grad()
      raw_loss = criterion(logits, batch[2].to(ctx.device), targets)["loss"] if len(batch) == 3 else criterion(logits, targets)
      loss = raw_loss / accum_steps

    # Backward pass with dynamic loss scaling
    ctx.scaler.scale(loss).backward()
    if step == 0 and logits.grad is not None:
      abs_grad = logits.grad.detach().float().abs()
      pos_grad = abs_grad[targets > 0.5].sum().item()
      neg_grad = abs_grad[targets <= 0.5].sum().item()
      total_logit_grad = pos_grad + neg_grad
      if total_logit_grad > 0.0:
        grad_share = {
          "train/pos_grad_share": pos_grad / total_logit_grad,
          "train/neg_grad_share": neg_grad / total_logit_grad,
        }
        epoch_grad_share = dict(grad_share)

    total_loss += loss.item() * accum_steps
    num_batches += 1

    # Optimizer step, gradient unscaling, and clipping at accumulation boundaries
    if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
      ctx.scaler.unscale_(optimizer)
      grad_norm = clip_gradients(model, max_norm=clip_grad)
      total_grad_norm += grad_norm
      ctx.scaler.step(optimizer)
      ctx.scaler.update()
      optimizer.zero_grad(set_to_none=True)
      if scheduler is not None:
        scheduler.step()

      global_step += 1
      num_updates += 1

      # Asynchronous step-level metric logging
      if log_fn:
        current_lr = optimizer.param_groups[0]["lr"]
        step_payload: dict[str, Any] = {
          "train/loss": loss.item() * accum_steps,
          "train/grad_norm": grad_norm,
          "train/lr": current_lr,
          **grad_share,
        }
        grad_share.clear()
        log_fn(step_payload, global_step)

  # Compile epoch-level loss and gradient norm summary
  metrics = {
    "loss": total_loss / max(1, num_batches),
    "grad_norm": total_grad_norm / max(1, num_updates),
    "pos_grad_share": epoch_grad_share.get("train/pos_grad_share", 0.0),
    "neg_grad_share": epoch_grad_share.get("train/neg_grad_share", 0.0),
  }
  return metrics, global_step


@torch.no_grad()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader[Any],
  ctx: RuntimeContext,
  epoch: int | None = None,
  train_loss: float | None = None,
  log_fn: Callable[[dict[str, Any], int], None] | None = None,
  step: int | None = None,
  report_confusion: bool = False,
  tag: str = "Eval",
  class_names: list[str] | None = None,
  use_wandb: bool = False,
) -> dict[str, float]:
  """Evaluates validation performance, logs unified epoch metrics, and optionally reports the full confusion matrix."""
  model.eval()
  all_logits_list: list[torch.Tensor] = []
  all_targets_list: list[torch.Tensor] = []

  with torch.amp.autocast(device_type=ctx.device.type, dtype=ctx.autocast_dtype, enabled=ctx.scaler.is_enabled()):
    for batch in dataloader:
      embeds, targets = batch[0].to(ctx.device), batch[1]
      logits = model(embeds)
      all_logits_list.append(logits.detach().cpu())
      all_targets_list.append(targets.cpu())

  all_logits = torch.cat(all_logits_list, dim=0)
  all_targets = torch.cat(all_targets_list, dim=0)
  metrics = compute_multi_label_metrics(all_logits, all_targets)

  if log_fn is not None and step is not None:
    payload: dict[str, Any] = {
      "val/loss": metrics["loss"],
      "val/macro_precision": metrics["macro_precision"],
      "val/macro_recall": metrics["macro_recall"],
      "val/macro_f1": metrics["macro_f1"],
      "val/precision": metrics["micro_precision"],
      "val/recall": metrics["micro_recall"],
      "val/f1": metrics["micro_f1"],
      "val/map": metrics["map"],
      "val/pos_prob_mean": metrics["pos_prob_mean"],
      "val/neg_prob_mean": metrics["neg_prob_mean"],
      "val/majority_fp": metrics["majority_fp"],
    }
    if epoch is not None:
      payload["train/epoch"] = epoch
    if train_loss is not None:
      payload["train/epoch_loss"] = train_loss
    log_fn(payload, step)

  if report_confusion and ctx.local_rank == 0:
    rows = compute_per_class_confusion(all_logits, all_targets)
    names = class_names if class_names is not None else []
    print_rank0(f"\n[{tag}] Full Per-Class Confusion Matrix ({len(rows)} classes, untruncated):")
    print_rank0(f"{'ID':>3} | {'Class':<18} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>5} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Supp':>5}")
    print_rank0("-" * 92)

    table_rows: list[list[Any]] = []
    for row in rows:
      cid = int(row["class_id"])
      cname = names[cid] if cid < len(names) else f"class_{cid}"
      print_rank0(
        f"{cid:>3} | {cname:<18} | {int(row['tp']):>5} | {int(row['fp']):>5} | {int(row['fn']):>5} | {int(row['tn']):>5} | "
        f"{float(row['precision']):>6.4f} | {float(row['recall']):>6.4f} | {float(row['f1']):>6.4f} | {int(row['support']):>5}"
      )
      table_rows.append(
        [
          cid,
          cname,
          int(row["tp"]),
          int(row["fp"]),
          int(row["fn"]),
          int(row["tn"]),
          float(row["precision"]),
          float(row["recall"]),
          float(row["f1"]),
          int(row["support"]),
        ]
      )

    if use_wandb:
      try:
        import wandb

        if wandb.run is not None:
          table = wandb.Table(
            columns=["class_id", "class_name", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "support"],
            data=table_rows,
          )
          wandb.log({"val/confusion_matrix": table}, step=step)
      except Exception:
        pass

  return metrics


def train_head(
  tag: str,
  head: nn.Module,
  train_ds: TensorDataset,
  val_ds: TensorDataset,
  criterion: nn.Module,
  epochs: int,
  ckpt_path: Path,
  args: argparse.Namespace,
  ctx: RuntimeContext,
  class_names: list[str],
) -> nn.Module:
  """Trains a projection head on cached embeddings, restores best Macro-F1 weights, and logs final confusion."""
  sampler = DistributedSampler(train_ds, shuffle=True) if ctx.is_ddp else None
  train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)
  val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

  if args.compile:
    head = torch.compile(head)  # type: ignore[assignment]
  if ctx.is_ddp:
    head = DDP(head, device_ids=[ctx.local_rank])

  optimizer = build_optimizer(head, lr=args.lr, weight_decay=args.weight_decay)
  total_steps = len(train_loader) * epochs // args.grad_accum
  warmup_iters = min(args.warmup_steps, max(1, total_steps))
  warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
  cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_iters), eta_min=args.lr * args.eta_min_ratio)
  scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])

  best_f1 = -1.0
  best_state: dict[str, Any] | None = None
  step = 0

  for epoch in range(1, epochs + 1):
    if sampler is not None:
      sampler.set_epoch(epoch)

    epoch_start = time.time()
    train_metrics, step = train_one_epoch(
      model=head,
      dataloader=train_loader,
      optimizer=optimizer,
      criterion=criterion,
      ctx=ctx,
      clip_grad=args.clip_grad,
      accum_steps=args.grad_accum,
      scheduler=scheduler,
      log_fn=log_async if args.wandb else None,
      global_step_start=step,
    )
    val_metrics = evaluate(
      model=head,
      dataloader=val_loader,
      ctx=ctx,
      epoch=epoch,
      train_loss=train_metrics["loss"],
      log_fn=log_async if args.wandb else None,
      step=step,
    )
    elapsed = time.time() - epoch_start

    print_rank0(
      f"[{tag}] Epoch {epoch:02d}/{epochs:02d} | "
      f"Train Loss: {train_metrics['loss']:.4f} | "
      f"Val Loss: {val_metrics['loss']:.4f} | "
      f"Val Macro-P/R/F1: {val_metrics['macro_precision']:.4f}/{val_metrics['macro_recall']:.4f}/{val_metrics['macro_f1']:.4f} | "
      f"Val P/R/F1: {val_metrics['micro_precision']:.4f}/{val_metrics['micro_recall']:.4f}/{val_metrics['micro_f1']:.4f} | "
      f"Val mAP: {val_metrics['map']:.4f} | "
      f"Pos/Neg Mean: {val_metrics['pos_prob_mean']:.4f}/{val_metrics['neg_prob_mean']:.4f} | "
      f"Grad Share +/-: {train_metrics['pos_grad_share']:.2%}/{train_metrics['neg_grad_share']:.2%} | "
      f"Maj FP: {int(val_metrics['majority_fp'])} | "
      f"Time: {elapsed:.2f}s"
    )

    if val_metrics["macro_f1"] > best_f1:
      best_f1 = val_metrics["macro_f1"]
      raw_head = unwrap_model(head)
      best_state = {k: v.cpu().clone() for k, v in raw_head.state_dict().items()}
      if ctx.local_rank == 0:
        torch.save(best_state, ckpt_path)
        print_rank0(f"  -> Saved best {tag.lower()} checkpoint to {ckpt_path} (macro_f1: {best_f1:.4f})")

  raw_head = unwrap_model(head)
  if best_state is not None:
    raw_head.load_state_dict(best_state)
  elif ckpt_path.exists():
    raw_head.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
  raw_head.eval()

  evaluate(
    model=raw_head,
    dataloader=val_loader,
    ctx=ctx,
    step=step,
    report_confusion=True,
    tag=tag,
    class_names=class_names,
    use_wandb=args.wandb,
  )
  return raw_head


# ━━━━━━━━ Main Entrypoint ━━━━━━━━


def main() -> None:
  """Executes end-to-end multi-label distillation: teacher caching, student caching, and student distillation."""
  args = parse_args()

  # ━━━━━━━━ Hardware & Distributed Runtime ━━━━━━━━
  # Seed random number generators for reproducible initialization
  seed_everything(args.seed)

  # Initialize distributed communication backend and resolve accelerator rank
  local_rank, is_ddp = setup_ddp()
  device = torch.device(f"cuda:{local_rank}") if (torch.cuda.is_available() and is_ddp) else get_device(verbose=(local_rank == 0))

  # Configure automatic mixed precision scaler and compute datatype
  scaler = configure_mixed_precision(args.mixed_precision, device=device)
  autocast_dtype = get_autocast_dtype(device)
  ctx = RuntimeContext(
    device=device,
    scaler=scaler,
    autocast_dtype=autocast_dtype,
    is_ddp=is_ddp,
    local_rank=local_rank,
  )

  # Ensure destination cache and checkpoint directories exist
  cache_dir = Path(args.cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)
  save_dir = Path(args.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)

  teacher_logits_path = cache_dir / "teacher_logits_train.npz"
  teacher_head_ckpt_path = save_dir / "teacher_head_best.pt"

  val_ann_path = Path(args.data_root) / "annotations" / "instances_val2017.json"
  if val_ann_path.exists():
    _, _, class_names = load_coco_metadata(Path(args.data_root), split="val2017")
  else:
    class_names = [f"class_{i}" for i in range(args.num_classes)]

  # ━━━━━━━━ Phase 1: Teacher Training & Logit Caching ━━━━━━━━
  if teacher_logits_path.exists() and not args.no_cache:
    print_rank0(f"Found cached teacher logits at {teacher_logits_path}. Skipping Phase 1.")
  else:
    init_wandb(args.project, f"{_model_short_tag(args.teacher_model)}_teacher", config=vars(args), enabled=args.wandb, rank=local_rank)

    print_rank0(f"Phase 1: Initializing teacher model ({args.teacher_model})...")
    teacher_cfg = LitClassifierConfig.from_pretrained(
      model_id=args.teacher_model,
      num_classes=args.num_classes,
      dropout=args.dropout,
      freeze_backbone=True,
    )
    teacher_proc = LitClassifierProcessor(teacher_cfg)
    teacher_model = LitClassifierModel(teacher_cfg, pretrained=True).to(device)

    print_rank0(f"Building deterministic loaders for teacher feature extraction from {args.data_root}...")
    raw_train_loader, raw_val_loader, _ = build_dataloaders(
      data_root=Path(args.data_root),
      batch_size=args.batch_size,
      processor=teacher_proc,
      num_workers=args.num_workers,
      shuffle=False,
    )

    t_train_embeds, t_train_targets, t_train_ids = load_or_extract_embeddings(
      prefix="teacher",
      split="train",
      cache_dir=cache_dir,
      model=teacher_model,
      loader=raw_train_loader,
      ctx=ctx,
      no_cache=args.no_cache,
    )
    t_val_embeds, t_val_targets, _ = load_or_extract_embeddings(
      prefix="teacher",
      split="val",
      cache_dir=cache_dir,
      model=teacher_model,
      loader=raw_val_loader,
      ctx=ctx,
      no_cache=args.no_cache,
    )

    print_rank0(f"Training teacher projection head for {args.teacher_epochs} epochs (DDP={is_ddp})...")
    raw_head = train_head(
      tag="Teacher",
      head=teacher_model.head,
      train_ds=TensorDataset(t_train_embeds, t_train_targets),
      val_ds=TensorDataset(t_val_embeds, t_val_targets),
      criterion=BCEWithLogitsLoss(reduction="mean"),
      epochs=args.teacher_epochs,
      ckpt_path=teacher_head_ckpt_path,
      args=args,
      ctx=ctx,
      class_names=class_names,
    )

    world_size = dist.get_world_size() if (is_ddp and dist.is_initialized()) else 1

    if world_size > 1:
      total_t = len(t_train_embeds)
      chunk_t = (total_t + world_size - 1) // world_size
      start_t = local_rank * chunk_t
      end_t = min(total_t, start_t + chunk_t)

      shard_t_embeds = t_train_embeds[start_t:end_t]
      shard_t_ids = t_train_ids[start_t:end_t]

      shard_logits: list[torch.Tensor] = []
      det_loader = DataLoader(TensorDataset(shard_t_embeds), batch_size=args.batch_size, shuffle=False)
      with torch.no_grad():
        for (batch_embeds,) in det_loader:
          batch_embeds = batch_embeds.to(device)
          logits = raw_head(batch_embeds)
          shard_logits.append(logits.cpu())

      shard_logits_arr = torch.cat(shard_logits, dim=0).numpy() if shard_logits else np.empty((0, args.num_classes), dtype=np.float32)
      t_shard_path = cache_dir / f"teacher_logits_shard_{local_rank}.npz"
      np.savez_compressed(t_shard_path, logits=shard_logits_arr, image_ids=shard_t_ids)

      dist.barrier()

      if local_rank == 0:
        all_t_logits = []
        all_t_ids = []
        for r in range(world_size):
          sp = cache_dir / f"teacher_logits_shard_{r}.npz"
          with np.load(sp) as d:
            all_t_logits.append(d["logits"])
            all_t_ids.append(d["image_ids"])
          sp.unlink(missing_ok=True)

        np.savez_compressed(
          teacher_logits_path,
          logits=np.concatenate(all_t_logits, axis=0),
          image_ids=np.concatenate(all_t_ids, axis=0),
        )
        print_rank0(f"Saved {sum(len(x) for x in all_t_logits)} teacher train logits to {teacher_logits_path}.")

      dist.barrier()
    else:
      print_rank0("Computing teacher logits on train split for distillation cache...")
      t_logits_list: list[torch.Tensor] = []
      deterministic_t_loader = DataLoader(
        TensorDataset(t_train_embeds),
        batch_size=args.batch_size,
        shuffle=False,
      )
      with torch.no_grad():
        for (batch_embeds,) in deterministic_t_loader:
          batch_embeds = batch_embeds.to(device)
          logits = raw_head(batch_embeds)
          t_logits_list.append(logits.cpu())

      t_logits_all = torch.cat(t_logits_list, dim=0).numpy()
      np.savez_compressed(
        teacher_logits_path,
        logits=t_logits_all,
        image_ids=t_train_ids,
      )
      print_rank0(f"Saved {len(t_logits_all)} teacher train logits to {teacher_logits_path}.")

    del teacher_model
    del raw_head
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

    # Finalize teacher WandB session so student starts fresh from step 0
    finish_wandb(enabled=args.wandb, rank=local_rank)

  # ━━━━━━━━ Phase 2: Student Embeddings Extraction & Alignment ━━━━━━━━
  student_embeds_train_path = cache_dir / "student_embeds_train.npz"
  student_embeds_val_path = cache_dir / "student_embeds_val.npz"

  student_cfg = LitClassifierConfig.from_pretrained(
    model_id=args.student_model,
    num_classes=args.num_classes,
    dropout=args.dropout,
    freeze_backbone=True,
  )

  if (not student_embeds_train_path.exists() or not student_embeds_val_path.exists()) or args.no_cache:
    print_rank0(f"Phase 2: Initializing student vision backbone ({args.student_model})...")
    student_proc = LitClassifierProcessor(student_cfg)
    student_model = LitClassifierModel(student_cfg, pretrained=True).to(device)

    print_rank0("Building deterministic loaders for student feature extraction...")
    raw_s_train_loader, raw_s_val_loader, _ = build_dataloaders(
      data_root=Path(args.data_root),
      batch_size=args.batch_size,
      processor=student_proc,
      num_workers=args.num_workers,
      shuffle=False,
    )

    s_train_embeds, s_train_targets, s_train_ids = load_or_extract_embeddings(
      prefix="student",
      split="train",
      cache_dir=cache_dir,
      model=student_model,
      loader=raw_s_train_loader,
      ctx=ctx,
      no_cache=args.no_cache,
    )
    s_val_embeds, s_val_targets, _ = load_or_extract_embeddings(
      prefix="student",
      split="val",
      cache_dir=cache_dir,
      model=student_model,
      loader=raw_s_val_loader,
      ctx=ctx,
      no_cache=args.no_cache,
    )

    del student_model
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  else:
    print_rank0(f"Loading cached student embeddings from {cache_dir}...")
    with np.load(student_embeds_train_path) as data:
      s_train_embeds = torch.from_numpy(data["embeddings"])
      s_train_targets = torch.from_numpy(data["targets"])
      s_train_ids = data["image_ids"]
    with np.load(student_embeds_val_path) as data:
      s_val_embeds = torch.from_numpy(data["embeddings"])
      s_val_targets = torch.from_numpy(data["targets"])

  with np.load(teacher_logits_path) as data:
    t_train_logits = torch.from_numpy(data["logits"])
    t_train_ids = data["image_ids"]

  if not np.array_equal(s_train_ids, t_train_ids):
    raise ValueError(f"1:1 Image ID alignment failure: student train IDs ({len(s_train_ids)}) do not match teacher train IDs ({len(t_train_ids)}).")
  print_rank0(f"Verified 1:1 image_id alignment across {len(s_train_ids)} train samples.")

  # ━━━━━━━━ Phase 3: Student Distillation Training ━━━━━━━━
  init_wandb(args.project, args.run_name, config=vars(args), enabled=args.wandb, rank=local_rank)

  print_rank0(f"Phase 3: Starting student distillation (alpha={args.alpha}, temperature={args.temperature}, epochs={args.student_epochs})...")
  student_head = MultiLabelHead(
    embed_dim=student_cfg.embed_dim,
    num_classes=args.num_classes,
    dropout=args.dropout,
  ).to(device)

  if args.load_head:
    print_rank0(f"Loading pretrained student head from {args.load_head}...")
    student_head.load_state_dict(torch.load(args.load_head, map_location=device))

  train_head(
    tag="Student",
    head=student_head,
    train_ds=TensorDataset(s_train_embeds, s_train_targets, t_train_logits),
    val_ds=TensorDataset(s_val_embeds, s_val_targets),
    criterion=SigmoidDistillationLoss(alpha=args.alpha, temperature=args.temperature),
    epochs=args.student_epochs,
    ckpt_path=save_dir / f"{args.run_name}_best.pt",
    args=args,
    ctx=ctx,
    class_names=class_names,
  )

  # ━━━━━━━━ Execution Teardown ━━━━━━━━
  finish_wandb(enabled=args.wandb, rank=local_rank)
  shutdown_logger(wait=True)
  if is_ddp:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
