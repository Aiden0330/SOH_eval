"""Compatibility shim: the implementation moved to ``sohkernel.registry`` (ADR-001, extraction plan Task 2).

The workbench keeps importing ``sohbench.registry``; every public name and the names listed
below resolve to the kernel objects, so identities (classes, registries) are shared.
"""
from sohkernel.registry import *  # noqa: F401,F403
from sohkernel.registry import (  # noqa: F401
    register_estimator,
    get_estimator,
    register_dataset,
    get_dataset,
    register_task,
    get_task,
    register_featureset,
    get_featureset,
    clear,
    _register,
    _ESTIMATORS,
    _DATASETS,
    _TASKS,
    _FEATURESETS,
)
