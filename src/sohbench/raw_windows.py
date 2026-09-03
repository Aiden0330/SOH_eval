"""Compatibility shim: the implementation moved to ``sohkernel.raw_windows`` (ADR-001, extraction plan Task 2).

The workbench keeps importing ``sohbench.raw_windows``; every public name and the names listed
below resolve to the kernel objects, so identities (classes, registries) are shared.
"""
from sohkernel.raw_windows import *  # noqa: F401,F403
from sohkernel.raw_windows import (  # noqa: F401
    iter_charge_curves,
)
