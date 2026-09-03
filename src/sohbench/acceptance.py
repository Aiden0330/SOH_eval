"""Compatibility shim: the implementation moved to ``sohkernel.acceptance`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.acceptance``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.acceptance import *  # noqa: F401,F403
from sohkernel.acceptance import (  # noqa: F401
    Verdict,
    evaluate_verdicts,
)
