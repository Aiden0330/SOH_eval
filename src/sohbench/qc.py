"""Compatibility shim: the implementation moved to ``sohkernel.qc`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.qc``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.qc import *  # noqa: F401,F403
from sohkernel.qc import (  # noqa: F401
    np,
    pd,
    QCReport,
    ID_COLUMNS,
    qc_gate,
)
