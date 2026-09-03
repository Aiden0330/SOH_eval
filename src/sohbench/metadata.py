"""Compatibility shim: the implementation moved to ``sohkernel.metadata`` (ADR-001, extraction plan Task 2).

The workbench keeps importing ``sohbench.metadata``; every public name and the names listed
below resolve to the kernel objects, so identities (classes, registries) are shared.
"""
from sohkernel.metadata import *  # noqa: F401,F403
from sohkernel.metadata import (  # noqa: F401
    DATASET_CHEMISTRY,
    DATASET_C_RATED_AH,
    DATASET_LABEL_SOURCE,
    build_meta,
)
