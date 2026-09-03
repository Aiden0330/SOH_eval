"""Compatibility shim: the implementation moved to ``sohkernel.tasks`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.tasks``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.tasks import *  # noqa: F401,F403
from sohkernel.tasks import (  # noqa: F401
    dataclass,
    field,
    register_task,
    SOHEstimationTask,
)
