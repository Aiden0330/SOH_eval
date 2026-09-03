"""Estimator implementations of the SOH kernel.

Registration is explicit: :func:`register_all` imports the estimator modules (their
``@register_estimator`` decorators run on first import) and re-registers any class the
registry has since dropped, so it is safe to call repeatedly and after
``registry.clear()``.  The torch-backed estimators (``deep``, ``pinn``) are included
only when torch is importable, or when ``include_torch=True`` is passed explicitly.
"""
from __future__ import annotations

import importlib
from importlib.util import find_spec

from .. import registry as _registry

CORE_MODULES = ("baselines", "mechanism", "physics", "neural", "byo")
TORCH_MODULES = ("deep", "pinn")


def torch_available() -> bool:
    try:
        return find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def _estimator_classes(module):
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and value.__module__ == module.__name__
            and getattr(value, "name", None)
            and callable(getattr(value, "fit", None))
        ):
            yield value


def register_all(include_torch: bool | None = None) -> list[str]:
    """Register every built-in estimator; returns the registered names in module order."""
    if include_torch is None:
        include_torch = torch_available()
    names: list[str] = []
    for module_name in CORE_MODULES + (TORCH_MODULES if include_torch else ()):
        module = importlib.import_module(f"{__name__}.{module_name}")
        for cls in _estimator_classes(module):
            _registry._ESTIMATORS.setdefault(cls.name, cls)
            names.append(cls.name)
    return names
