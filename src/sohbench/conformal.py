from __future__ import annotations
import numpy as np

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
