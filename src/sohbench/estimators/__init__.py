"""Compatibility shim package: the estimators moved to ``sohkernel.estimators`` (ADR-001, Task 4).

Importing this package registers every built-in estimator through the kernel's explicit
``register_all()``; the submodule shims (``sohbench.estimators.baselines`` ...) re-export
the classes so old import paths and pickled models keep resolving.
"""
from sohkernel.estimators import register_all, torch_available  # noqa: F401

register_all()

from sohkernel.estimators.baselines import (  # noqa: E402,F401
    SOH_HIGH,
    SOH_LOW,
    GBDTEstimator,
    GPREstimator,
    RidgeEstimator,
    SVREstimator,
)
from sohkernel.estimators.byo import ByoSelectKBestRidge  # noqa: E402,F401
from sohkernel.estimators.mechanism import (  # noqa: E402,F401
    ElasticNetEstimator,
    ExtraTreesEstimator,
    PLSEstimator,
    StressResidualEstimator,
)
from sohkernel.estimators.neural import MLPEstimator  # noqa: E402,F401
from sohkernel.estimators.physics import PhysicsResidualEstimator  # noqa: E402,F401

if torch_available():
    from sohkernel.estimators.deep import (  # noqa: F401
        CNN1DEstimator,
        LSTMEstimator,
        TransformerEstimator,
    )
    from sohkernel.estimators.pinn import PINNEstimator  # noqa: F401
