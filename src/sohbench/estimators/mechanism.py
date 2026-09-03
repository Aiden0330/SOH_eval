"""Compatibility shim: the implementation moved to ``sohkernel.estimators.mechanism`` (ADR-001, Task 4).

Every module-level name resolves to the kernel object, so registrations, class identities
and pickled artifacts that name ``sohbench.estimators.mechanism`` keep working.
"""
from sohkernel.estimators.mechanism import *  # noqa: F401,F403
from sohkernel.estimators.mechanism import (  # noqa: F401
    np,
    pd,
    nnls,
    PLSRegression,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    ElasticNet,
    StandardScaler,
    feature_columns,
    FeatureMedianImputer,
    register_estimator,
    SOH_HIGH,
    SOH_LOW,
    _subsample,
    ElasticNetEstimator,
    ExtraTreesEstimator,
    PLSEstimator,
    _numeric,
    StressResidualEstimator,
)
