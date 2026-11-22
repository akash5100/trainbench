"""Self-contained training and distillation pipeline for SigLIP2 on COCO 2017."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from trainbench.losses.torch import BCEWithLogitsLoss, SigmoidDistillationLoss
from trainbench.models.torch.lit_classifier import LitClassifierConfig, LitClassifierModel, LitClassifierProcessor, MultiLabelHead
from trainbench.utils import log_async, shutdown_logger
from trainbench.utils.torch import clip_gradients, configure_mixed_precision, get_autocast_dtype, get_device, print_rank0, seed_everything, setup_ddp

from .data import build_dataloaders


# ━━━━━━━━ CLI Configuration ━━━━━━━━


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
  parser.add_argument("--alpha", type=float, default=0.5, help="Distillation loss weight (0 = pure BCE, 1 = pure distillation)")
  parser.add_argument("--temperature", type=float, default=1.0, help="Distillation temperature scaling")

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
    args.run_name = f"student_T{args.temperature}_a{args.alpha}"
  return args


# ━━━━━━━━ Embedding Cache Helpers ━━━━━━━━


def extract_embeddings_to_cache(
  model: LitClassifierModel,
  loader: DataLoader[Any],
  device: torch.device,
  cache_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
  """Runs a forward extraction pass across a DataLoader and caches embeddings and image IDs to disk."""
  model.eval()
  all_embeds: list[torch.Tensor] = []
  all_targets: list[torch.Tensor] = []
  all_ids: list[int] = []

  with torch.no_grad():
    for images, targets, ids in loader:
      images = images.to(device)
      embeds = model.forward_features(images)
      all_embeds.append(embeds.cpu())
      all_targets.append(targets.cpu())
      if isinstance(ids, torch.Tensor):
        all_ids.extend(ids.tolist())
      else:
        all_ids.extend(list(ids))

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
  device: torch.device,
  no_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
  """Loads cached embeddings or extracts them if not present on disk."""
  cache_path = cache_dir / f"{prefix}_embeds_{split}.npz"
  if cache_path.exists() and not no_cache:
    print_rank0(f"Loading cached {prefix} {split} embeddings from {cache_path}...")
    with np.load(cache_path) as data:
      embeds = torch.from_numpy(data["embeddings"])
      targets = torch.from_numpy(data["targets"])
      image_ids = data["image_ids"]
    return embeds, targets, image_ids

  print_rank0(f"Extracting {prefix} {split} embeddings with frozen backbone...")
  embeds, targets, image_ids = extract_embeddings_to_cache(model, loader, device, cache_path)
  print_rank0(f"Saved {len(embeds)} {prefix} {split} embeddings to {cache_path}.")
  return embeds, targets, image_ids


# ━━━━━━━━ Optimizer Parameter Filtering ━━━━━━━━


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
  """Configures AdamW with weight decay decoupled from 1D biases and normalization layers."""
  decay_params: list[torch.nn.Parameter] = []
  no_decay_params: list[torch.nn.Parameter] = []

  for _, param in model.named_parameters():
    if not param.requires_grad:
      continue
    if param.ndim >= 2:
      decay_params.append(param)
    else:
      no_decay_params.append(param)

  param_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
  ]
  return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


# ━━━━━━━━ Epoch Training & Validation Routines ━━━━━━━━


def train_one_epoch(
  model: nn.Module,
  dataloader: DataLoader[Any],
  optimizer: torch.optim.Optimizer,
  criterion: nn.Module,
  device: torch.device,
  scaler: torch.amp.GradScaler,
  autocast_dtype: torch.dtype,
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

  for step, batch in enumerate(dataloader):
    if len(batch) == 3:
      embeds, targets, teacher_logits = batch
      embeds = embeds.to(device)
      targets = targets.to(device)
      teacher_logits = teacher_logits.to(device)
      with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=scaler.is_enabled()):
        logits = model(embeds)
        loss_dict = criterion(logits, teacher_logits, targets)
        loss = loss_dict["loss"] / accum_steps
    else:
      embeds, targets = batch
      embeds = embeds.to(device)
      targets = targets.to(device)
      with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=scaler.is_enabled()):
        logits = model(embeds)
        loss = criterion(logits, targets) / accum_steps

    scaler.scale(loss).backward()
    total_loss += loss.item() * accum_steps
    num_batches += 1

    if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
      scaler.unscale_(optimizer)
      grad_norm = clip_gradients(model, max_norm=clip_grad)
      total_grad_norm += grad_norm
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad(set_to_none=True)
      if scheduler is not None:
        scheduler.step()

      global_step += 1
      num_updates += 1

      if log_fn:
        current_lr = optimizer.param_groups[0]["lr"]
        log_fn(
          {
            "train/loss": loss.item() * accum_steps,
            "train/grad_norm": grad_norm,
            "train/lr": current_lr,
          },
          global_step,
        )

  metrics = {
    "loss": total_loss / max(1, num_batches),
    "grad_norm": total_grad_norm / max(1, num_updates),
  }
  return metrics, global_step


@torch.no_grad()
def validate(
  model: nn.Module,
  dataloader: DataLoader[Any],
  device: torch.device,
  autocast_dtype: torch.dtype,
  use_amp: bool = False,
) -> dict[str, float]:
  """Evaluates model performance across the validation dataset, returning ranking and calibration metrics."""
  model.eval()
  all_logits_list: list[torch.Tensor] = []
  all_targets_list: list[torch.Tensor] = []

  with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
    for batch in dataloader:
      embeds, targets = batch[0], batch[1]
      embeds = embeds.to(device)
      logits = model(embeds)
      all_logits_list.append(logits.detach().cpu())
      all_targets_list.append(targets.cpu())

  all_logits = torch.cat(all_logits_list, dim=0).float()
  all_targets = torch.cat(all_targets_list, dim=0).float()

  bce_fn = BCEWithLogitsLoss(reduction="mean")
  val_loss = bce_fn(all_logits, all_targets).item()

  probs = torch.sigmoid(all_logits)
  preds = (probs >= 0.5).float()

  # Macro F1 (at threshold 0.5, averaged over classes with >= 1 positive in val)
  f1_list: list[float] = []
  for c in range(all_targets.shape[1]):
    c_targets = all_targets[:, c]
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

  # Mean Average Precision across classes with >= 1 positive in val
  ap_list: list[float] = []
  for c in range(all_targets.shape[1]):
    c_targets = all_targets[:, c]
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

  # Mean predicted probability on true-negative entries
  neg_mask = all_targets == 0.0
  neg_prob_mean = probs[neg_mask].mean().item() if neg_mask.any() else 0.0

  # False positive count on highest-frequency class
  maj_class = int(all_targets.sum(dim=0).argmax().item())
  maj_preds = preds[:, maj_class]
  maj_targets = all_targets[:, maj_class]
  majority_fp = int(((maj_preds == 1.0) & (maj_targets == 0.0)).sum().item())

  return {
    "loss": val_loss,
    "macro_f1": macro_f1,
    "map": map_score,
    "neg_prob_mean": neg_prob_mean,
    "majority_fp": float(majority_fp),
  }


# ━━━━━━━━ Main Entrypoint ━━━━━━━━


def main() -> None:
  """Executes end-to-end multi-label distillation: teacher caching, student caching, and student distillation."""
  args = parse_args()

  # ━━━━━━━━ Hardware & Distributed Runtime ━━━━━━━━
  seed_everything(args.seed)
  local_rank, is_ddp = setup_ddp()
  if torch.cuda.is_available() and is_ddp:
    device = torch.device(f"cuda:{local_rank}")
  else:
    device = get_device(verbose=(local_rank == 0))

  scaler = configure_mixed_precision(args.mixed_precision, device=device)
  autocast_dtype = get_autocast_dtype(device)

  cache_dir = Path(args.cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)
  save_dir = Path(args.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)

  teacher_logits_path = cache_dir / "teacher_logits_train.npz"
  teacher_head_ckpt_path = save_dir / "teacher_head_best.pt"

  if args.wandb and (local_rank == 0):
    try:
      import wandb

      wandb.init(project=args.project, name=args.run_name, config=vars(args))
    except ImportError:
      print_rank0("WandB requested but not installed. Skipping remote logging.")

  # ━━━━━━━━ Phase 1: Teacher Training & Logit Caching ━━━━━━━━
  if teacher_logits_path.exists() and not args.no_cache:
    print_rank0(f"Found cached teacher logits at {teacher_logits_path}. Skipping Phase 1.")
  else:
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

    if is_ddp:
      if local_rank == 0:
        load_or_extract_embeddings(
          prefix="teacher", split="train", cache_dir=cache_dir, model=teacher_model, loader=raw_train_loader, device=device, no_cache=args.no_cache
        )
        load_or_extract_embeddings(
          prefix="teacher", split="val", cache_dir=cache_dir, model=teacher_model, loader=raw_val_loader, device=device, no_cache=args.no_cache
        )
      dist.barrier()
      t_train_embeds, t_train_targets, t_train_ids = load_or_extract_embeddings(
        prefix="teacher", split="train", cache_dir=cache_dir, model=teacher_model, loader=raw_train_loader, device=device, no_cache=False
      )
      t_val_embeds, t_val_targets, _ = load_or_extract_embeddings(
        prefix="teacher", split="val", cache_dir=cache_dir, model=teacher_model, loader=raw_val_loader, device=device, no_cache=False
      )
    else:
      t_train_embeds, t_train_targets, t_train_ids = load_or_extract_embeddings(
        prefix="teacher", split="train", cache_dir=cache_dir, model=teacher_model, loader=raw_train_loader, device=device, no_cache=args.no_cache
      )
      t_val_embeds, t_val_targets, _ = load_or_extract_embeddings(
        prefix="teacher", split="val", cache_dir=cache_dir, model=teacher_model, loader=raw_val_loader, device=device, no_cache=args.no_cache
      )

    print_rank0(f"Training teacher projection head for {args.teacher_epochs} epochs (DDP={is_ddp})...")
    teacher_train_ds = TensorDataset(t_train_embeds, t_train_targets)
    teacher_sampler = DistributedSampler(teacher_train_ds, shuffle=True) if is_ddp else None
    teacher_train_loader = DataLoader(
      teacher_train_ds,
      batch_size=args.batch_size,
      shuffle=(teacher_sampler is None),
      sampler=teacher_sampler,
    )
    teacher_val_loader = DataLoader(
      TensorDataset(t_val_embeds, t_val_targets),
      batch_size=args.batch_size,
      shuffle=False,
    )

    teacher_head = teacher_model.head
    if args.compile:
      teacher_head = torch.compile(teacher_head)  # type: ignore[assignment]
    if is_ddp:
      teacher_head = DDP(teacher_head, device_ids=[local_rank])

    t_optimizer = build_optimizer(teacher_head, lr=args.lr, weight_decay=args.weight_decay)
    t_total_steps = len(teacher_train_loader) * args.teacher_epochs // args.grad_accum
    t_warmup = LinearLR(t_optimizer, start_factor=0.01, end_factor=1.0, total_iters=min(args.warmup_steps, max(1, t_total_steps)))
    t_cosine = CosineAnnealingLR(t_optimizer, T_max=max(1, t_total_steps - args.warmup_steps), eta_min=args.lr * args.eta_min_ratio)
    t_scheduler = SequentialLR(t_optimizer, schedulers=[t_warmup, t_cosine], milestones=[min(args.warmup_steps, max(1, t_total_steps))])
    t_criterion = BCEWithLogitsLoss(reduction="mean")

    best_teacher_f1 = -1.0
    best_teacher_state: dict[str, Any] | None = None
    t_step = 0

    for epoch in range(1, args.teacher_epochs + 1):
      if teacher_sampler is not None:
        teacher_sampler.set_epoch(epoch)

      t_metrics, t_step = train_one_epoch(
        model=teacher_head,
        dataloader=teacher_train_loader,
        optimizer=t_optimizer,
        criterion=t_criterion,
        device=device,
        scaler=scaler,
        autocast_dtype=autocast_dtype,
        clip_grad=args.clip_grad,
        accum_steps=args.grad_accum,
        scheduler=t_scheduler,
        log_fn=log_async if args.wandb else None,
        global_step_start=t_step,
      )
      t_val_metrics = validate(
        model=teacher_head,
        dataloader=teacher_val_loader,
        device=device,
        autocast_dtype=autocast_dtype,
        use_amp=scaler.is_enabled(),
      )
      print_rank0(
        f"[Teacher] Epoch {epoch:02d}/{args.teacher_epochs:02d} | "
        f"Train Loss: {t_metrics['loss']:.4f} | "
        f"Val Loss: {t_val_metrics['loss']:.4f} | "
        f"Val Macro-F1: {t_val_metrics['macro_f1']:.4f} | "
        f"Val mAP: {t_val_metrics['map']:.4f}"
      )
      if t_val_metrics["macro_f1"] > best_teacher_f1:
        best_teacher_f1 = t_val_metrics["macro_f1"]
        raw_head = teacher_head.module if hasattr(teacher_head, "module") else getattr(teacher_head, "_orig_mod", teacher_head)
        best_teacher_state = {k: v.cpu().clone() for k, v in raw_head.state_dict().items()}
        if local_rank == 0:
          torch.save(best_teacher_state, teacher_head_ckpt_path)

    raw_head = teacher_head.module if hasattr(teacher_head, "module") else getattr(teacher_head, "_orig_mod", teacher_head)
    if best_teacher_state is not None:
      raw_head.load_state_dict(best_teacher_state)
    elif teacher_head_ckpt_path.exists():
      raw_head.load_state_dict(torch.load(teacher_head_ckpt_path, map_location="cpu"))

    if local_rank == 0:
      print_rank0("Computing teacher logits on train split for distillation cache...")
      raw_head.eval()
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

    if is_ddp:
      dist.barrier()

    del teacher_model
    del teacher_head
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

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

    if is_ddp:
      if local_rank == 0:
        load_or_extract_embeddings(
          prefix="student", split="train", cache_dir=cache_dir, model=student_model, loader=raw_s_train_loader, device=device, no_cache=args.no_cache
        )
        load_or_extract_embeddings(
          prefix="student", split="val", cache_dir=cache_dir, model=student_model, loader=raw_s_val_loader, device=device, no_cache=args.no_cache
        )
      dist.barrier()
      s_train_embeds, s_train_targets, s_train_ids = load_or_extract_embeddings(
        prefix="student", split="train", cache_dir=cache_dir, model=student_model, loader=raw_s_train_loader, device=device, no_cache=False
      )
      s_val_embeds, s_val_targets, _ = load_or_extract_embeddings(
        prefix="student", split="val", cache_dir=cache_dir, model=student_model, loader=raw_s_val_loader, device=device, no_cache=False
      )
    else:
      s_train_embeds, s_train_targets, s_train_ids = load_or_extract_embeddings(
        prefix="student", split="train", cache_dir=cache_dir, model=student_model, loader=raw_s_train_loader, device=device, no_cache=args.no_cache
      )
      s_val_embeds, s_val_targets, _ = load_or_extract_embeddings(
        prefix="student", split="val", cache_dir=cache_dir, model=student_model, loader=raw_s_val_loader, device=device, no_cache=args.no_cache
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
  print_rank0(f"Phase 3: Starting student distillation (alpha={args.alpha}, temperature={args.temperature}, epochs={args.student_epochs})...")
  student_train_ds = TensorDataset(s_train_embeds, s_train_targets, t_train_logits)
  student_sampler = DistributedSampler(student_train_ds, shuffle=True) if is_ddp else None
  student_train_loader = DataLoader(
    student_train_ds,
    batch_size=args.batch_size,
    shuffle=(student_sampler is None),
    sampler=student_sampler,
  )
  student_val_ds = TensorDataset(s_val_embeds, s_val_targets)
  student_val_loader = DataLoader(
    student_val_ds,
    batch_size=args.batch_size,
    shuffle=False,
  )

  student_head = MultiLabelHead(
    embed_dim=student_cfg.embed_dim,
    num_classes=args.num_classes,
    dropout=args.dropout,
  ).to(device)

  if args.load_head:
    print_rank0(f"Loading pretrained student head from {args.load_head}...")
    student_head.load_state_dict(torch.load(args.load_head, map_location=device))

  if args.compile:
    student_head = torch.compile(student_head)  # type: ignore[assignment]
  if is_ddp:
    student_head = DDP(student_head, device_ids=[local_rank])

  s_optimizer = build_optimizer(student_head, lr=args.lr, weight_decay=args.weight_decay)
  s_total_steps = len(student_train_loader) * args.student_epochs // args.grad_accum
  s_warmup = LinearLR(s_optimizer, start_factor=0.01, end_factor=1.0, total_iters=min(args.warmup_steps, max(1, s_total_steps)))
  s_cosine = CosineAnnealingLR(s_optimizer, T_max=max(1, s_total_steps - args.warmup_steps), eta_min=args.lr * args.eta_min_ratio)
  s_scheduler = SequentialLR(s_optimizer, schedulers=[s_warmup, s_cosine], milestones=[min(args.warmup_steps, max(1, s_total_steps))])

  s_criterion = SigmoidDistillationLoss(alpha=args.alpha, temperature=args.temperature)

  best_student_f1 = -1.0
  s_step = 0
  student_best_ckpt_path = save_dir / f"{args.run_name}_best.pt"

  for epoch in range(1, args.student_epochs + 1):
    if student_sampler is not None:
      student_sampler.set_epoch(epoch)

    epoch_start = time.time()
    s_metrics, s_step = train_one_epoch(
      model=student_head,
      dataloader=student_train_loader,
      optimizer=s_optimizer,
      criterion=s_criterion,
      device=device,
      scaler=scaler,
      autocast_dtype=autocast_dtype,
      clip_grad=args.clip_grad,
      accum_steps=args.grad_accum,
      scheduler=s_scheduler,
      log_fn=log_async if args.wandb else None,
      global_step_start=s_step,
    )
    s_val_metrics = validate(
      model=student_head,
      dataloader=student_val_loader,
      device=device,
      autocast_dtype=autocast_dtype,
      use_amp=scaler.is_enabled(),
    )
    elapsed = time.time() - epoch_start

    log_payload = {
      "train/epoch": epoch,
      "train/epoch_loss": s_metrics["loss"],
      "val/loss": s_val_metrics["loss"],
      "val/macro_f1": s_val_metrics["macro_f1"],
      "val/map": s_val_metrics["map"],
      "val/neg_prob_mean": s_val_metrics["neg_prob_mean"],
      "val/majority_fp": s_val_metrics["majority_fp"],
    }
    if args.wandb:
      log_async(log_payload, step=s_step)

    print_rank0(
      f"Epoch {epoch:02d}/{args.student_epochs:02d} | "
      f"Train Loss: {s_metrics['loss']:.4f} | "
      f"Val Loss: {s_val_metrics['loss']:.4f} | "
      f"Val Macro-F1: {s_val_metrics['macro_f1']:.4f} | "
      f"Val mAP: {s_val_metrics['map']:.4f} | "
      f"Neg Mean: {s_val_metrics['neg_prob_mean']:.4f} | "
      f"Maj FP: {int(s_val_metrics['majority_fp'])} | "
      f"Time: {elapsed:.2f}s"
    )

    if s_val_metrics["macro_f1"] > best_student_f1:
      best_student_f1 = s_val_metrics["macro_f1"]
      raw_head = student_head.module if hasattr(student_head, "module") else getattr(student_head, "_orig_mod", student_head)
      if local_rank == 0:
        torch.save(raw_head.state_dict(), student_best_ckpt_path)
        print_rank0(f"  -> Saved best student head checkpoint to {student_best_ckpt_path} (macro_f1: {best_student_f1:.4f})")

  # ━━━━━━━━ Execution Teardown ━━━━━━━━
  shutdown_logger(wait=True)
  if args.wandb and (local_rank == 0):
    try:
      import wandb

      wandb.finish()
    except Exception:
      pass
  if is_ddp:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
