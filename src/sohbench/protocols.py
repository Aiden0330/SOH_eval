"""Compatibility shim: the implementation moved to ``sohkernel.protocols`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.protocols``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.protocols import *  # noqa: F401,F403
from sohkernel.protocols import (  # noqa: F401
    warnings,
    np,
    pd,
    Iterator,
    Fold,
    _folds_by_group,
    LeaveOneCellOut,
    LeaveOneDatasetOut,
    LeaveOneChemistryOut,
)
