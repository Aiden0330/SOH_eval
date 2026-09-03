"""Compatibility shim: the implementation moved to ``sohkernel.contracts`` (ADR-001, extraction plan Task 2).

The workbench keeps importing ``sohbench.contracts``; every public name and the names listed
below resolve to the kernel objects, so identities (classes, registries) are shared.
"""
from sohkernel.contracts import *  # noqa: F401,F403
from sohkernel.contracts import (  # noqa: F401
    CellMeta,
    Estimator,
    Fold,
    ID_COLUMNS,
    QCReport,
    Verdict,
    feature_columns,
)
