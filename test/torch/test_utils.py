"""Unit tests for PyTorch runtime utilities."""

import torch
from torch import nn

from trainbench.utils.torch import (
  clip_gradients,
  configure_mixed_precision,
  get_autocast_dtype,
  get_device,
  get_rank,
  get_world_size,
  is_dist_available_and_initialized,
  is_main_process,
  print_rank0,
  seed_everything,
  setup_ddp,
  setup_for_distributed,
)


def test_device_discovery() -> None:
  device = get_device(verbose=False)
  assert isinstance(device, torch.device)
  assert device.type in {"cuda", "mps", "cpu"}


def test_autocast_dtype_and_scaler() -> None:
  cpu_device = torch.device("cpu")
  dtype = get_autocast_dtype(cpu_device)
  assert isinstance(dtype, torch.dtype)

  scaler_disabled = configure_mixed_precision(use_mixed_precision=False, device=cpu_device)
  assert not scaler_disabled.is_enabled()

  # On CPU or non-CUDA, scaler remains disabled even if requested
  scaler_cpu = configure_mixed_precision(use_mixed_precision=True, device=cpu_device)
  assert not scaler_cpu.is_enabled()


def test_seed_everything() -> None:
  seed_everything(1234)
  t1 = torch.randn(5)
  seed_everything(1234)
  t2 = torch.randn(5)
  assert torch.equal(t1, t2)


def test_clip_gradients() -> None:
  model = nn.Linear(10, 2)
  x = torch.randn(4, 10)
  out = model(x).sum()
  out.backward()

  norm = clip_gradients(model, max_norm=0.5)
  assert isinstance(norm, float)
  assert norm > 0.0

  empty_model = nn.Linear(5, 1)
  assert clip_gradients(empty_model, max_norm=1.0) == 0.0


def test_distributed_single_node_defaults() -> None:
  assert not is_dist_available_and_initialized()
  assert get_world_size() == 1
  assert get_rank() == 0
  assert is_main_process()

  local_rank, is_ddp = setup_ddp()
  assert local_rank == 0
  assert not is_ddp

  # Smoke test print filters
  setup_for_distributed(is_master=True)
  print_rank0("Test print rank 0 output")
