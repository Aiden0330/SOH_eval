from __future__ import annotations
import numpy as np
from .contracts import feature_columns


class ConformalWrapper:
    """Split-conformal. Calibration carved from the TRAIN TAIL — never the test fold."""
    ci_source = "conformal"

    def __init__(self, base, alpha: float = 0.1, calib_frac: float = 0.2):
        self.base = base; self.alpha = alpha; self.calib_frac = calib_frac
        self._q = None; self._n_calib = 0; self._n_fit = 0

    def fit(self, X, y, meta):
        n = len(X); n_calib = max(1, int(round(n * self.calib_frac)))
        fit_slice = slice(0, n - n_calib); cal_slice = slice(n - n_calib, n)
        self._n_fit = n - n_calib; self._n_calib = n_calib
        Xf = X.iloc[fit_slice] if hasattr(X, "iloc") else X[fit_slice]
        yf = np.asarray(y)[fit_slice]
        self.base.fit(Xf, yf, meta)
        Xc = X.iloc[cal_slice] if hasattr(X, "iloc") else X[cal_slice]
        yc = np.asarray(y)[cal_slice]
        resid = np.abs(yc - self.base.predict(Xc, meta))
        # finite-sample conformal quantile
        level = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
        self._q = float(np.quantile(resid, min(level, 1.0), method="higher"))
        return self

    def predict(self, X, meta):
        return self.base.predict(X, meta)

    def predict_interval(self, X, meta, alpha: float | None = None):
        center = self.base.predict(X, meta)
        return center - self._q, center + self._q


class WeightedConformalWrapper:
    """Covariate-shift-aware split-conformal.

    Calibration is carved from the TRAIN TAIL (same as ConformalWrapper).
    At prediction time a logistic-regression density-ratio weights the
    calibration residuals toward the test distribution, then the weighted
    quantile is used for the interval width.

    Falls back to the plain unweighted quantile when the classifier
    degenerates (one class present, fewer than 5 calibration points,
    or any other failure).
    """
    ci_source = "weighted_conformal"

    def __init__(self, base, alpha: float = 0.1, calib_frac: float = 0.2,
                 density_ratio_max_samples: int | None = 5000,
                 random_state: int = 0):
        self.base = base
        self.alpha = alpha
        self.calib_frac = calib_frac
        self.density_ratio_max_samples = density_ratio_max_samples
        self.random_state = random_state
        self._q_plain: float | None = None
        self._calib_resid: np.ndarray | None = None
        self._calib_features: np.ndarray | None = None
        self._n_calib: int = 0
        self._n_fit: int = 0

    def fit(self, X, y, meta):
        n = len(X)
        n_calib = max(1, int(round(n * self.calib_frac)))
        fit_slice = slice(0, n - n_calib)
        cal_slice = slice(n - n_calib, n)
        self._n_fit = n - n_calib
        self._n_calib = n_calib

        Xf = X.iloc[fit_slice] if hasattr(X, "iloc") else X[fit_slice]
        yf = np.asarray(y)[fit_slice]
        self.base.fit(Xf, yf, meta)

        Xc = X.iloc[cal_slice] if hasattr(X, "iloc") else X[cal_slice]
        yc = np.asarray(y)[cal_slice]
        resid = np.abs(yc - self.base.predict(Xc, meta))
        self._calib_resid = resid

        # Store calib feature matrix (numeric feature columns only)
        cols = feature_columns(Xc) if hasattr(Xc, "columns") else None
        if cols is not None:
            self._calib_features = Xc[cols].to_numpy(dtype=float)
        else:
            self._calib_features = np.asarray(Xc, dtype=float)

        # Plain unweighted quantile (fallback)
        level = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
        self._q_plain = float(np.quantile(resid, min(level, 1.0), method="higher"))
        return self

    def predict(self, X, meta):
        return self.base.predict(X, meta)

    def predict_interval(self, X, meta, alpha: float | None = None):
        a = alpha if alpha is not None else self.alpha
        center = self.base.predict(X, meta)
        q_w = self._weighted_quantile(X, a)
        return center - q_w, center + q_w

    def _weighted_quantile(self, X_test, alpha: float) -> float:
        """Return the importance-weighted quantile of calibration residuals."""
        resid = self._calib_resid
        if resid is None or self._n_calib < 5:
            return self._q_plain

        # Build test feature matrix
        try:
            cols = feature_columns(X_test) if hasattr(X_test, "columns") else None
            if cols is not None:
                X_test_arr = X_test[cols].to_numpy(dtype=float)
            else:
                X_test_arr = np.asarray(X_test, dtype=float)

            X_calib = self._calib_features
            rng = np.random.default_rng(self.random_state)
            X_calib_fit = self._sample_rows(X_calib, rng)
            X_test_fit = self._sample_rows(X_test_arr, rng)
            n_calib = len(X_calib_fit)
            n_test = len(X_test_fit)

            # Labels: 0 = calibration, 1 = test
            X_lr = np.vstack([X_calib_fit, X_test_fit])
            y_lr = np.concatenate([np.zeros(n_calib), np.ones(n_test)])

            # Check degenerate: only one class
            if len(np.unique(y_lr)) < 2:
                return self._q_plain

            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=200)
            lr.fit(X_lr, y_lr)

            # Probability of being test (class 1) for calibration points
            proba = lr.predict_proba(X_calib)
            class_labels = list(lr.classes_)
            if 1 not in class_labels and 1.0 not in class_labels:
                return self._q_plain
            idx1 = class_labels.index(1) if 1 in class_labels else class_labels.index(1.0)
            p1 = lr.predict_proba(X_calib)[:, idx1]

            # Density ratio w = p(test) / p(calib) = p1 / (1 - p1)
            p1 = np.clip(p1, 1e-6, 1 - 1e-6)
            w = p1 / (1.0 - p1)
            w = np.clip(w, 1e-3, 1e3)
            w = w / w.sum()  # normalize

            return float(_weighted_quantile_1d(resid, w, 1.0 - alpha))

        except Exception:
            return self._q_plain

    def _sample_rows(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        max_samples = self.density_ratio_max_samples
        if max_samples is None or len(X) <= max_samples:
            return X
        idx = rng.choice(len(X), int(max_samples), replace=False)
        return X[idx]


def _weighted_quantile_1d(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile via sorted cumulative-weight method."""
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order]
    cdf = np.cumsum(sw)
    # Find first index where cumulative weight >= q
    idx = np.searchsorted(cdf, q)
    idx = min(idx, len(sv) - 1)
    return float(sv[idx])
