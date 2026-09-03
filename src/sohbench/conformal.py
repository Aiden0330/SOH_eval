"""Compatibility shim: the implementation moved to ``sohkernel.conformal`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.conformal``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.conformal import *  # noqa: F401,F403
from sohkernel.conformal import (  # noqa: F401
    np,
    feature_columns,
    ConformalWrapper,
    WeightedConformalWrapper,
    _weighted_quantile_1d,
)
