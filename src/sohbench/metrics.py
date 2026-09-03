"""Compatibility shim: the implementation moved to ``sohkernel.metrics`` (ADR-001, extraction plan Task 2).

The workbench keeps importing ``sohbench.metrics``; every public name and the names listed
below resolve to the kernel objects, so identities (classes, registries) are shared.
"""
from sohkernel.metrics import *  # noqa: F401,F403
from sohkernel.metrics import (  # noqa: F401
    mae,
    max_error,
    mean_bias_error,
    monotonicity_violation_rate,
    per_cell_monotonicity_violation_rate,
    picp,
    rmse,
    value_range_violation,
)
