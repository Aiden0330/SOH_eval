"""Compatibility shim: the implementation moved to ``sohkernel.runner`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.runner``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.runner import *  # noqa: F401,F403
from sohkernel.runner import (  # noqa: F401
    np,
    pd,
    M,
    ConformalWrapper,
    WeightedConformalWrapper,
    feature_columns,
    is_label_source_feature,
    unified_features,
    project,
    method_contract,
    get_featureset,
    _featuresets,
    BenchmarkRunner,
)
