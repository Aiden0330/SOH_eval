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
from ..registry import register_estimator
from .baselines import SOH_LOW, SOH_HIGH

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
    """

    name: str = "mlp"
    feature_mode: str = "unified"
    backend: str = "sklearn"

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 300,
        random_state: int = 0,
    ) -> None:
        self._hidden_layer_sizes = hidden_layer_sizes
        self._max_iter = max_iter
        self._random_state = random_state

        self._scaler = StandardScaler()
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
        Z = self._scaler.fit_transform(X[self._cols].to_numpy())
        self._model.fit(Z, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame, meta=None) -> np.ndarray:
        """Predict SOH; output clamped to ``[0.0, 1.3]``."""
        if self._cols is None:
            raise RuntimeError("Call fit() before predict().")
        Z = self._scaler.transform(X[self._cols].to_numpy())
        return np.clip(self._model.predict(Z), SOH_LOW, SOH_HIGH)
