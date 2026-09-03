"""MLP (multilayer perceptron) estimator for the sohbench benchmark.

``MLPEstimator`` wraps ``sklearn.neural_network.MLPRegressor`` with a
``StandardScaler`` preprocessing step and output clamping to the physical
SOH range ``[0.0, 1.3]``.  It uses ``early_stopping=True`` so it does not
need a manually tuned number of epochs.

This is the Phase-1 neural-network comparison entry; it requires only
``scikit-learn`` and introduces no new heavy dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..contracts import feature_columns
from ..preprocessing import FeatureMedianImputer
from ..registry import register_estimator
from .baselines import SOH_LOW, SOH_HIGH, _subsample

__all__ = ["MLPEstimator"]


@register_estimator
class MLPEstimator:
    """StandardScaler → MLPRegressor(64, 32) with output clamping.

    Parameters
    ----------
    hidden_layer_sizes:
        Tuple of hidden layer widths.  Defaults to ``(64, 32)`` per spec.
    max_iter:
        Maximum training epochs.
    random_state:
        Seed for reproducibility.
    n_max:
        Maximum rows sent into MLPRegressor.fit. Large real-data folds are
        subsampled for tractability.
    """

    name: str = "mlp"
    feature_mode: str = "unified"
    backend: str = "sklearn"

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 300,
        random_state: int = 0,
        n_max: int = 5000,
    ) -> None:
        self._hidden_layer_sizes = hidden_layer_sizes
        self._max_iter = max_iter
        self._random_state = random_state
        self._n_max = n_max

        self._scaler = StandardScaler()
        self._imputer = FeatureMedianImputer()
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            early_stopping=True,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, meta=None) -> "MLPEstimator":
        """Fit scaler and MLP on feature columns of *X*."""
        self._cols = feature_columns(X)
        X_arr = self._imputer.fit(X, self._cols).transform_array(X)
        X_sub, y_sub = _subsample(X_arr, np.asarray(y, dtype=float), self._n_max, self._random_state)
        Z = self._scaler.fit_transform(X_sub)
        self._model.fit(Z, y_sub)
        return self

    def predict(self, X: pd.DataFrame, meta=None) -> np.ndarray:
        """Predict SOH; output clamped to ``[0.0, 1.3]``."""
        if self._cols is None:
            raise RuntimeError("Call fit() before predict().")
        Z = self._scaler.transform(self._imputer.transform_array(X))
        return np.clip(self._model.predict(Z), SOH_LOW, SOH_HIGH)
