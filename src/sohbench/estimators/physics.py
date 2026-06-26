"""Physics-residual estimator for the sohbench benchmark.

``PhysicsResidualEstimator`` fits a semi-empirical degradation prior
``soh ≈ 1 - a·√t - b·t`` by ordinary least squares (where *t* is a
throughput proxy column), then fits a ``HistGradientBoostingRegressor``
on the residual ``soh - prior`` using all feature columns.  At predict
time the two components are summed and clamped to the physical SOH range
``[0.0, 1.3]``.

The throughput proxy is resolved at fit time:

1. ``cycle_index_feature`` — preferred (absolute cycle number, correlates
   with throughput in most datasets).
2. ``cycle_fraction`` — fallback (normalised cycle index).

The chosen column is stored in ``_throughput_col`` for diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..contracts import feature_columns
from ..registry import register_estimator
from .baselines import SOH_LOW, SOH_HIGH

__all__ = ["PhysicsResidualEstimator"]

_THROUGHPUT_PREFERRED = "cycle_index_feature"
_THROUGHPUT_FALLBACK = "cycle_fraction"


@register_estimator
class PhysicsResidualEstimator:
    """Semi-empirical prior + GBDT residual correction.

    Parameters
    ----------
    max_iter:
        Maximum boosting iterations for the residual ``HistGradientBoostingRegressor``.
    learning_rate:
        Learning rate for the residual model.
    random_state:
        Seed for the residual model.
    """

    name: str = "phys_resid"
    feature_mode: str = "unified"
    backend: str = "sklearn"

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        random_state: int = 0,
    ) -> None:
        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self._random_state = random_state

        # Set after fit
        self._throughput_col: str | None = None
        self._prior_coeffs: np.ndarray | None = None  # [a, b]
        self._residual_model: HistGradientBoostingRegressor | None = None
        self._feature_cols: list[str] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_throughput_col(self, X: pd.DataFrame) -> str:
        """Return the throughput proxy column present in *X*."""
        if _THROUGHPUT_PREFERRED in X.columns:
            return _THROUGHPUT_PREFERRED
        if _THROUGHPUT_FALLBACK in X.columns:
            return _THROUGHPUT_FALLBACK
        raise ValueError(
            f"PhysicsResidualEstimator requires either {_THROUGHPUT_PREFERRED!r} or "
            f"{_THROUGHPUT_FALLBACK!r} to be present in X; got columns: {list(X.columns)}"
        )

    def _prior(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the fitted prior ``1 - a·√t - b·t``."""
        a, b = self._prior_coeffs
        return 1.0 - a * np.sqrt(np.maximum(t, 0.0)) - b * t

    # ------------------------------------------------------------------
    # Estimator API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray, meta=None) -> "PhysicsResidualEstimator":
        """Fit the prior by OLS and the residual by GBDT.

        Parameters
        ----------
        X:
            Feature table including ID columns.
        y:
            SOH target values aligned with ``X``.
        meta:
            Unused; accepted for API compatibility.
        """
        y = np.asarray(y, dtype=float)

        # Resolve throughput proxy
        self._throughput_col = self._resolve_throughput_col(X)
        t = X[self._throughput_col].to_numpy(dtype=float)

        # Fit prior: soh - 1 ≈ -a·√t - b·t  →  [√t, t] @ [a, b] ≈ -(soh-1)
        t_safe = np.maximum(t, 0.0)
        A_mat = np.column_stack([np.sqrt(t_safe), t_safe])
        rhs = 1.0 - y  # we want A @ coeffs ≈ rhs
        coeffs, *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)
        self._prior_coeffs = coeffs  # [a, b]

        # Compute residuals
        prior_pred = self._prior(t)
        residual = y - prior_pred

        # Fit residual model on feature columns
        self._feature_cols = feature_columns(X)
        self._residual_model = HistGradientBoostingRegressor(
            max_iter=self._max_iter,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
        )
        Z = X[self._feature_cols].to_numpy()
        self._residual_model.fit(Z, residual)

        return self

    def predict(self, X: pd.DataFrame, meta=None) -> np.ndarray:
        """Predict SOH as ``prior(t) + residual_model(features)``, clamped to ``[0, 1.3]``."""
        if self._throughput_col is None or self._residual_model is None:
            raise RuntimeError("Call fit() before predict().")

        # Throughput proxy — if the column used at fit time is missing,
        # fall back to zeros (safe for clamp, better than crash).
        if self._throughput_col in X.columns:
            t = X[self._throughput_col].to_numpy(dtype=float)
        else:
            t = np.zeros(len(X))

        prior_pred = self._prior(t)

        Z = X[self._feature_cols].to_numpy()
        residual_pred = self._residual_model.predict(Z)

        return np.clip(prior_pred + residual_pred, SOH_LOW, SOH_HIGH)
