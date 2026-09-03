"""Compatibility shim: the implementation moved to ``sohkernel.method_contracts`` (ADR-001, extraction plan Task 2).

The workbench keeps importing ``sohbench.method_contracts``; every public name and the names listed
below resolve to the kernel objects, so identities (classes, registries) are shared.
"""
from sohkernel.method_contracts import *  # noqa: F401,F403
from sohkernel.method_contracts import (  # noqa: F401
    incompatible_pairing_reason,
    method_contract,
)
