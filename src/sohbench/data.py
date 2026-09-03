"""Compatibility shim: the implementation moved to ``sohkernel.data`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.data``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.data import *  # noqa: F401,F403
from sohkernel.data import (  # noqa: F401
    Path,
    pd,
    build_meta,
    qc_gate,
    REQUIRED_COLUMNS,
    ALL_DATASETS,
    DatasetLoader,
)
