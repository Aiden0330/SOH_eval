from __future__ import annotations
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ..contracts import feature_columns
from ..feature_safety import is_label_source_feature
from ..preprocessing import FeatureMedianImputer
from ..registry import register_estimator
from .baselines import SOH_LOW, SOH_HIGH


@register_estimator
class ByoSelectKBestRidge:
    """BYO-track estimator: SelectKBest(f_regression) → StandardScaler → Ridge.

    Demonstrates self-selected features; receives the full feature table
    (never projected to a unified featureset).
    """
    name = "byo_skb_ridge"
    feature_mode = "byo"
    backend = "sklearn"

    def __init__(self, k: int = 6, alpha: float = 1.0):
        self._k = k
        self._alpha = alpha
        self._pipeline: Pipeline | None = None
        self._cols: list[str] | None = None
        self._imputer = FeatureMedianImputer()

    def fit(self, X, y, meta=None):
        # BYO receives the full table; label-source columns (the quantities
        # SOH is computed from) are banned as inputs on every dataset.
        self._cols = [c for c in feature_columns(X) if not is_label_source_feature(c)]
        n_features = len(self._cols)
        k = min(self._k, n_features)
        self._pipeline = Pipeline([
            ("selector", SelectKBest(score_func=f_regression, k=k)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self._alpha)),
        ])
        X_arr = self._imputer.fit(X, self._cols).transform_array(X)
        self._pipeline.fit(X_arr, np.asarray(y))
        return self

    def predict(self, X, meta=None):
        Xarr = self._imputer.transform_array(X)
        preds = self._pipeline.predict(Xarr)
        return np.clip(preds, SOH_LOW, SOH_HIGH)
