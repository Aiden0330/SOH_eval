"""Compatibility shim: the implementation moved to ``sohkernel.window_bank`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.window_bank``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.window_bank import *  # noqa: F401,F403
from sohkernel.window_bank import (  # noqa: F401
    Path,
    np,
    pd,
    iter_charge_curves,
    DATASET_LATTICE,
    LATTICE_STEP,
    lattice_edges,
    bank_path,
    _edge_values,
    bank_from_curves,
    build_bank,
    load_bank,
    select_bounds,
    apply_bank_window,
)
