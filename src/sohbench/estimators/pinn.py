"""Compatibility shim: the implementation moved to ``sohkernel.estimators.pinn`` (ADR-001, Task 4).

Every module-level name resolves to the kernel object, so registrations, class identities
and pickled artifacts that name ``sohbench.estimators.pinn`` keep working.
"""
from sohkernel.estimators.pinn import *  # noqa: F401,F403
from sohkernel.estimators.pinn import (  # noqa: F401
    np,
    pd,
    StandardScaler,
    feature_columns,
    FeatureMedianImputer,
    register_estimator,
    SOH_HIGH,
    SOH_LOW,
    clamp,
    set_seed,
    subsample_idx,
    _trajectory_pairs,
    PINNEstimator,
)
