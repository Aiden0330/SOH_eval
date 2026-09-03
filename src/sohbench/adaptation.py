"""Compatibility shim: the implementation moved to ``sohkernel.adaptation`` (ADR-001, Task 5).

Every module-level name resolves to the kernel object.
"""
from sohkernel.adaptation import *  # noqa: F401,F403
from sohkernel.adaptation import (  # noqa: F401
    np,
    pd,
    ADAPTATION_OPERATORS,
    OPERATOR_CHANNELS,
    CALIBRATION_SPLITS,
    INTERVAL_MODES,
    _COV_RIDGE,
    _MIN_EVALUATION_ROWS,
    _feature_matrix,
    _matrix_sqrt,
    fit_coral,
    apply_coral,
    _spread_order,
    calibration_split,
    _RECAL_SLOPE_BOUNDS,
    _RECAL_MIN_TRUTH_RANGE,
    _RECAL_MIN_ABS_CORRELATION,
    fit_recalibration,
    apply_recalibration,
    fit_residual,
    fit_conformal_interval,
    apply_conformal_interval,
    wide_residual_columns,
    apply_monotone,
    _column_medians,
    _impute_columns,
    apply_residual,
    _loo_predictions,
    recalibration_loo_predictions,
    residual_loo_predictions,
    InferenceAdapter,
)
