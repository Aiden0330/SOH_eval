from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureMedianImputer:
    """Fit train-fold medians and apply them to feature matrices."""

    def __init__(self) -> None:
        self.columns_: list[str] = []
        self.fill_values_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, columns: list[str]) -> "FeatureMedianImputer":
        self.columns_ = list(columns)
        self.fill_values_ = {col: _median_or_zero(X[col]) for col in self.columns_}
        return self

    def transform_array(self, X: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
        cols = self.columns_ if columns is None else list(columns)
        if not cols:
            return np.empty((len(X), 0), dtype=float)

        values = []
        for col in cols:
            if col in X.columns:
                arr = _numeric_array(X[col])
            else:
                arr = np.full(len(X), np.nan, dtype=float)
            fill = self.fill_values_.get(col, 0.0)
            values.append(np.where(np.isfinite(arr), arr, fill))
        return np.column_stack(values).astype(float, copy=False)

    def transform_frame(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        cols = self.columns_ if columns is None else list(columns)
        out = X.copy()
        arr = self.transform_array(X, cols)
        for idx, col in enumerate(cols):
            out[col] = arr[:, idx]
        return out


def _numeric_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _median_or_zero(series: pd.Series) -> float:
    arr = _numeric_array(series)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    return median if np.isfinite(median) else 0.0
