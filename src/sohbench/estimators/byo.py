"""Compatibility shim: the implementation moved to ``sohkernel.estimators.byo`` (ADR-001, Task 4).

Every module-level name resolves to the kernel object, so registrations, class identities
and pickled artifacts that name ``sohbench.estimators.byo`` keep working.
"""
from sohkernel.estimators.byo import *  # noqa: F401,F403
from sohkernel.estimators.byo import (  # noqa: F401
    np,
    SelectKBest,
    f_regression,
    Ridge,
    StandardScaler,
    Pipeline,
    feature_columns,
    is_label_source_feature,
    FeatureMedianImputer,
    register_estimator,
    SOH_LOW,
    SOH_HIGH,
    ByoSelectKBestRidge,
)
