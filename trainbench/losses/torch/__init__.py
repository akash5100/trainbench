"""PyTorch loss primitives."""

from typing import TYPE_CHECKING

from trainbench.lazy import create_lazy_importer


if TYPE_CHECKING:
  from .bce import BCEWithLogitsLoss
  from .sigmoid_kd import SigmoidDistillationLoss

_LAZY_IMPORTS = {
  "BCEWithLogitsLoss": "bce",
  "SigmoidDistillationLoss": "sigmoid_kd",
}

__all__ = [
  "BCEWithLogitsLoss",
  "SigmoidDistillationLoss",
]

__getattr__ = create_lazy_importer(_LAZY_IMPORTS, __name__)
