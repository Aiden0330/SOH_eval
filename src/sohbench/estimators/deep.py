"""Compatibility shim: the implementation moved to ``sohkernel.estimators.deep`` (ADR-001, Task 4).

Every module-level name resolves to the kernel object, so registrations, class identities
and pickled artifacts that name ``sohbench.estimators.deep`` keep working.
"""
from sohkernel.estimators.deep import *  # noqa: F401,F403
from sohkernel.estimators.deep import (  # noqa: F401
    math,
    Optional,
    np,
    pd,
    StandardScaler,
    feature_columns,
    FeatureMedianImputer,
    register_estimator,
    SOH_LOW,
    SOH_HIGH,
    build_windows,
    clamp,
    set_seed,
    subsample_idx,
    train_mlp_like,
    _DeepBase,
    LSTMEstimator,
    CNN1DEstimator,
    TransformerEstimator,
)
