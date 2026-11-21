"""Reusable lazy import utilities (PEP 562 compliant).

Enables instant module import times and prevents memory/accelerator conflicts between
PyTorch and JAX by deferring heavy runtime imports until symbol access.
"""

# ━━━━━━━━ Lazy Import Machinery ━━━━━━━━

import importlib
import sys
from collections.abc import Callable
from typing import Any


def create_lazy_importer(mapping: dict[str, str], package: str | None) -> Callable[[str], Any]:
  """Create a cached, self-healing lazy-import __getattr__ function.

  Args:
    mapping: Dictionary mapping exported symbol names to relative or absolute module paths.
    package: The parent package name (__name__ from the calling module).

  Returns:
    A PEP 562 compliant __getattr__ function for module-level dynamic loading.
  """

  def __getattr__(name: str) -> Any:
    """Dynamically import symbols on first access and cache them on the parent module.

    Args:
      name: Attribute name requested from the module.

    Returns:
      The resolved attribute or module object.

    Raises:
      AttributeError: If the requested attribute is not present in the lazy mapping.
    """
    if name in mapping:
      target = mapping[name]

      if package and (target.startswith(f"{package}.") or target == package):
        module = importlib.import_module(target)
      else:
        module = importlib.import_module(f".{target.lstrip('.')}", package)

      val = getattr(module, name)

      # Attach to parent module so subsequent lookups hit C-level dict at full speed
      if package and package in sys.modules:
        setattr(sys.modules[package], name, val)

      return val

    raise AttributeError(f"module '{package}' has no attribute '{name}'")

  return __getattr__
