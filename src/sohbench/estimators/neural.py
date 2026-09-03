"""Compatibility shim: the implementation moved to ``sohkernel.estimators.neural`` (ADR-001, Task 4).

Every module-level name resolves to the kernel object, so registrations, class identities
and pickled artifacts that name ``sohbench.estimators.neural`` keep working.
"""
from sohkernel.estimators.neural import *  # noqa: F401,F403
from sohkernel.estimators.neural import (  # noqa: F401
    np,
    pd,
    MLPRegressor,
    StandardScaler,
    feature_columns,
    FeatureMedianImputer,
    register_estimator,
    SOH_LOW,
    SOH_HIGH,
    _subsample,
    MLPEstimator,
)
