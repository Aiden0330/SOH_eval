"""Compatibility shim: the implementation moved to ``sohkernel.estimators.physics`` (ADR-001, Task 4).

Every module-level name resolves to the kernel object, so registrations, class identities
and pickled artifacts that name ``sohbench.estimators.physics`` keep working.
"""
from sohkernel.estimators.physics import *  # noqa: F401,F403
from sohkernel.estimators.physics import (  # noqa: F401
    np,
    pd,
    HistGradientBoostingRegressor,
    feature_columns,
    FeatureMedianImputer,
    register_estimator,
    SOH_LOW,
    SOH_HIGH,
    _subsample,
    _THROUGHPUT_PREFERRED,
    _THROUGHPUT_FALLBACK,
    _THROUGHPUT_ID_FALLBACK,
    PhysicsResidualEstimator,
)
