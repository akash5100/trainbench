# trainbench

A minimal, from-scratch deep learning training and experimentation monorepo (PyTorch on GPU, JAX/Flax on TPU) built around self-contained, reproducible experiments.

## Experiments

| Experiment | Summary | Writeup & Logs |
| :--- | :--- | :--- |
| [`experiments/sigmoid_distill`](experiments/sigmoid_distill/README.md) | Controlled MS-COCO 2017 reproduction showing why $T > 1$ degrades sparse multi-label Sigmoid/BCE distillation (Negative-Tail Gradient Hijacking) across a 7-temperature sweep ($T = 0.5$ to $6.0$). | [Blog Post](https://akash5100.github.io/posts/2026-11-20-why_t_greater_than_1_breaks_sigmoid_distillation/) · [W&B Report](https://forge.coreweave.com/wandb/akash5100-trainbench/siglip2-distill-coco/reports/Why-T-1-Breaks-Sigmoid-Distillation--VmlldzoxODM2MjUyOA?accessToken=s5ab1vpq61ov4buzfkzkqu5s46dvi7pzqj76wun868zwg8ul0ffpwfh18kbs9wm0) |

## Structure

```text
trainbench/
├── trainbench/
│   ├── losses/        # From-scratch loss modules (torch/ and jax/ isolated)
│   ├── metrics/       # Evaluation metrics and per-class confusion diagnostics
│   ├── models/        # Model definitions and projection heads
│   └── utils/         # DDP, mixed precision, seeding, and async W&B logging
├── experiments/
│   └── sigmoid_distill/
└── test/
```

## Installation

Requires Python 3.11+ (uses [uv](https://docs.astral.sh/uv/) if available, or falls back to `pip`):

```bash
# PyTorch (GPU)
make setup-gpu    # or: uv sync --extra gpu

# JAX / Flax (TPU)
make setup-tpu    # or: uv sync --extra tpu
```

## License

See [LICENSE](LICENSE).
