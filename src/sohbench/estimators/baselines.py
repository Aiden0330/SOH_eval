"""Compatibility shim: the implementation moved to ``sohkernel.estimators.baselines`` (ADR-001, Task 4).

Every module-level name resolves to the kernel object, so registrations, class identities
and pickled artifacts that name ``sohbench.estimators.baselines`` keep working.
"""
from sohkernel.estimators.baselines import *  # noqa: F401,F403
from sohkernel.estimators.baselines import (  # noqa: F401
    np,
    Ridge,
    StandardScaler,
    HistGradientBoostingRegressor,
    SVR,
    GaussianProcessRegressor,
    RBF,
    WhiteKernel,
    ConstantKernel,
    norm,
    feature_columns,
    FeatureMedianImputer,
    register_estimator,
    _subsample,
    RidgeEstimator,
    GBDTEstimator,
    SVREstimator,
    GPREstimator,
)
