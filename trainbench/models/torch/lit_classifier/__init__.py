"""LiT Multi-Label Classifier Package."""

from typing import TYPE_CHECKING

from trainbench.lazy import create_lazy_importer


if TYPE_CHECKING:
  from .config import LitClassifierConfig
  from .model import LitClassifierModel
  from .processor import LitClassifierProcessor

_LAZY_IMPORTS = {
  "LitClassifierConfig": "config",
  "LitClassifierModel": "model",
  "LitClassifierProcessor": "processor",
}

__all__ = [
  "LitClassifierConfig",
  "LitClassifierModel",
  "LitClassifierProcessor",
]

__getattr__ = create_lazy_importer(_LAZY_IMPORTS, __name__)
