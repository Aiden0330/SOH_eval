"""Compatibility shim: the implementation moved to ``sohkernel.features`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.features``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.features import *  # noqa: F401,F403
from sohkernel.features import (  # noqa: F401
    pd,
    feature_columns,
    unified_features,
)
