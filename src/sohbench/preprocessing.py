"""Compatibility shim: the implementation moved to ``sohkernel.preprocessing`` (ADR-001, extraction plan Task 2).

The workbench keeps importing ``sohbench.preprocessing``; every public name and the names listed
below resolve to the kernel objects, so identities (classes, registries) are shared.
"""
from sohkernel.preprocessing import *  # noqa: F401,F403
from sohkernel.preprocessing import (  # noqa: F401
    FeatureMedianImputer,
)
