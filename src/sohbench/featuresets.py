"""Compatibility shim: the implementation moved to ``sohkernel.featuresets`` (ADR-001, extraction plan Task 3).

The workbench keeps importing ``sohbench.featuresets``; every module-level name resolves to the
kernel object, so classes, registries and helpers keep a single identity.
"""
from sohkernel.featuresets import *  # noqa: F401,F403
from sohkernel.featuresets import (  # noqa: F401
    warnings,
    dataclass,
    field,
    Callable,
    List,
    np,
    pd,
    ID_COLUMNS,
    feature_columns,
    is_label_source_feature,
    register_featureset,
    get_featureset,
    _MECHANISM_DERIVED_COLUMNS,
    FeatureSet,
    project,
    _FullFeatureSet,
    _R,
    _full_fs,
    _PARTIAL_WINDOW_DERIVED_COLUMNS,
    _partial_window_norm_transform,
    _numeric,
    _mechanism_proxy_transform,
)
