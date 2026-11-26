# Why T > 1 Breaks Sigmoid Distillation

Controlled MS-COCO 2017 reproduction of multi-label Sigmoid/BCE knowledge distillation across temperatures $T \in [0.5, 6.0]$ and teacher weights $\alpha \in [0.0, 1.0]$.

- **Blog Post:** [Why T > 1 Breaks Sigmoid Distillation](https://akash5100.github.io/posts/2026-11-20-why_t_greater_than_1_breaks_sigmoid_distillation/)
- **W&B Report:** [siglip2-distill-coco](https://forge.coreweave.com/wandb/akash5100-trainbench/siglip2-distill-coco/reports/Why-T-1-Breaks-Sigmoid-Distillation--VmlldzoxODM2MjUyOA?accessToken=s5ab1vpq61ov4buzfkzkqu5s46dvi7pzqj76wun868zwg8ul0ffpwfh18kbs9wm0)
- **Kaggle Notebook:** [`sigmoid-distill.ipynb`](sigmoid-distill.ipynb)

## Summary

In Softmax distillation, raising temperature $T > 1$ exposes relative similarities across competing classes. In sparse multi-label Sigmoid/BCE distillation (no sum-to-1 cap), raising $T > 1$ flattens ~77 absent classes toward 0.5 independently. Instead of causing false positives (Precision stays flat around ~88%), those inflated absent targets hijack **90.30%–92.54%** of the gradient budget at $T = 4.0$–$6.0$, starving present classes down to 7.46%–9.70% and hurting Recall. Going too cold ($T = 0.5$) triggers the opposite failure: absent-class probability leakage $E[p \mid y = 0]$ doubles (0.93% to 1.77%). $T = 1.0$ sits at the minimum validation BCE (0.0376).

## Benchmark Setup

- **Dataset:** [MS-COCO 2017](https://arxiv.org/abs/1405.0312) (80 classes, 118,287 train / 5,000 val, ~96.4% negative sparsity).
- **Teacher:** Frozen `google/siglip2-so400m-patch14-384` + trained 2-layer `MultiLabelHead`.
- **Student:** Frozen `google/siglip2-base-patch16-256` + trained 2-layer `MultiLabelHead`.
- **Training:** 10 epochs, AdamW (`lr = 1e-3`, decoupled weight decay `1e-2` on 2D+ weights), cosine schedule with 50-step linear warmup, batch size 64 per GPU, mixed precision, with fail-fast 1:1 `image_id` alignment between cached student embeddings and teacher logits.
- **Loss (`trainbench/losses/torch/sigmoid_kd.py`):**

$$\mathcal{L} = \alpha \cdot T \cdot \text{BCE}\!\left(\sigma(t / T),\, \sigma(s / T)\right) + (1 - \alpha) \cdot \text{BCE}\!\left(y,\, \sigma(s)\right)$$

## Results

### 1. $\alpha \times T$ Isolation Grid (Validation Macro-F1)

| $\alpha$ (Teacher Weight) | $T = 1.0$ Macro-F1 | $T = 4.0$ Macro-F1 | Delta ($T = 4$ vs $T = 1$) |
| :--- | :---: | :---: | :---: |
| **Teacher (`SO400M-384`)** | 81.72% | — | Ceiling (87.01% mAP) |
| **0.0** (Pure GT baseline) | 78.84% | 78.84% | — |
| **0.5** (50% teacher / 50% GT) | 79.22% | 78.08% | **-1.14%** |
| **0.9** (90% teacher / 10% GT) | 78.73% | 77.64% | **-1.09%** |
| **1.0** (100% teacher / 0% GT) | 78.70% | 77.50% | **-1.20%** |

### 2. Full 7-Temperature Sweep ($\alpha = 1.0$)

| Run ($\alpha = 1.0$) | $y = 0$ Grad | $y = 1$ Grad | Val BCE | Val mAP | Macro-P | Macro-R | Macro-F1 | $E[p \mid y = 1]$ | $E[p \mid y = 0]$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`so400m_teacher`** | 54.69% | 45.31% | 0.0363 | 87.01% | 87.95% | 76.90% | 81.72% | 77.73% | 0.73% |
| **$T = 0.5$** | 65.53% | 34.47% | 0.0411 | 84.77% | 88.09% | **72.47%** | **78.72%** | 72.83% | **1.77%** |
| **$T = 1.0$** | **69.67%** | **30.33%** | **0.0376** | 85.02% | 88.77% | 72.28% | 78.70% | **73.57%** | 0.93% |
| **$T = 1.5$** | 74.93% | 25.07% | 0.0384 | **85.22%** | 88.86% | 71.69% | 78.16% | 73.50% | 0.80% |
| **$T = 2.0$** | 81.11% | 18.89% | 0.0392 | 85.06% | **89.14%** | 71.78% | 78.46% | 73.21% | 0.75% |
| **$T = 3.0$** | 87.28% | 12.72% | 0.0402 | 84.81% | 88.03% | 70.96% | 77.67% | 72.77% | 0.70% |
| **$T = 4.0$** | 90.30% | 9.70% | 0.0408 | 84.43% | 88.34% | 70.54% | 77.50% | 72.50% | 0.69% |
| **$T = 6.0$** | **92.54%** | **7.46%** | 0.0414 | 83.98% | 87.99% | **69.99%** | **76.99%** | **72.15%** | 0.67% |

## Reproduction

Install the GPU stack from the repo root:

```bash
make setup-gpu
```

Embeddings and teacher logits are extracted once and cached under `--cache-dir`, so subsequent temperature/alpha sweeps only train the 2-layer head (~1–2 minutes per run on 2× T4).

```bash
# Single run (2 GPUs via DDP; swap torchrun for `python -m` on 1 GPU)
torchrun --nproc_per_node=2 -m experiments.sigmoid_distill.train \
  --data-root data/coco2017 \
  --cache-dir .cache \
  --save-dir checkpoints \
  --teacher-model google/siglip2-so400m-patch14-384 \
  --student-model google/siglip2-base-patch16-256 \
  --teacher-epochs 10 \
  --student-epochs 10 \
  --batch-size 64 \
  --lr 1e-3 \
  --kd-alpha 1.0 \
  --kd-temperature 1.0 \
  --mixed-precision \
  --wandb \
  --project siglip2-distill-coco
```

To run the full 7-temperature sweep at $\alpha = 1.0$:

```bash
for T in 0.5 1.0 1.5 2.0 3.0 4.0 6.0; do
  torchrun --nproc_per_node=2 -m experiments.sigmoid_distill.train \
    --data-root data/coco2017 \
    --teacher-model google/siglip2-so400m-patch14-384 \
    --student-model google/siglip2-base-patch16-256 \
    --kd-alpha 1.0 \
    --kd-temperature "$T" \
    --mixed-precision
done
```

## Files

- [`train.py`](train.py) — End-to-end pipeline: teacher embedding/logit caching, student embedding caching with 1:1 `image_id` verification, distillation training, gradient-share telemetry, and full 80-class confusion reporting.
- [`data.py`](data.py) — MS-COCO 2017 multi-label dataset loader and metadata helpers.
- [`sigmoid-distill.ipynb`](sigmoid-distill.ipynb) — Kaggle notebook used to execute the sweep.

## Citation

```bibtex
@article{verma2026whytgt1breakssigmoiddistillation,
  title   = {Why T > 1 Breaks Sigmoid Distillation},
  author  = {Verma, Akash},
  journal = {akash5100.github.io},
  year    = {2026},
  month   = {November},
  url     = {https://akash5100.github.io/posts/2026-11-20-why_t_greater_than_1_breaks_sigmoid_distillation/}
}
```
