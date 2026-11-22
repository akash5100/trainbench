"""PyTorch evaluation metrics primitives."""

from typing import TYPE_CHECKING

from trainbench.lazy import create_lazy_importer


if TYPE_CHECKING:
  from .multi_label import compute_multi_label_metrics

_LAZY_IMPORTS = {
  "compute_multi_label_metrics": "multi_label",
}

__all__ = [
  "compute_multi_label_metrics",
]

__getattr__ = create_lazy_importer(_LAZY_IMPORTS, __name__)
