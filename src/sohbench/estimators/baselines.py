from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.stats import norm
from ..contracts import feature_columns
from ..registry import register_estimator

# Physical SOH bounds (D2): capacity reserve allows >1.0, capped below 1.3.
SOH_LOW, SOH_HIGH = 0.0, 1.3


def _subsample(X: np.ndarray, y: np.ndarray, n_max: int, seed: int = 0):
    """Cap training size for O(n^2)/O(n^3) learners (GPR/SVR) on large datasets."""
    if len(X) <= n_max:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), n_max, replace=False)
    return X[idx], y[idx]

@register_estimator
class RidgeEstimator:
    name = "ridge"; feature_mode = "unified"; backend = "sklearn"

    def __init__(self, alpha: float = 1.0):
        self._scaler = StandardScaler(); self._model = Ridge(alpha=alpha)
        self._cols = None

    def fit(self, X, y, meta=None):
        self._cols = feature_columns(X)
        Z = self._scaler.fit_transform(X[self._cols].to_numpy())
        self._model.fit(Z, np.asarray(y)); return self

    def predict(self, X, meta=None):
        # Clamp to the physical SOH range. A reasonable SOH estimator must emit
        # physically-valid values; this also caps out-of-distribution feature-
        # scaling blow-ups under leave-one-domain splits (a near-constant
        # training feature gives a tiny scaler scale_ -> exploded held-out z).
        Z = self._scaler.transform(X[self._cols].to_numpy())
        return np.clip(self._model.predict(Z), SOH_LOW, SOH_HIGH)

    def export_cpp(self, path: str) -> None:
        coef = self._model.coef_; intercept = float(self._model.intercept_)
        mean = self._scaler.mean_; scale = self._scaler.scale_
        with open(path, "w") as f:
            f.write("// auto-generated ridge inference\n")
            f.write(f"// features={self._cols}\n")
            f.write("inline float soh_infer(const float* x){float z;float acc=%r;" % intercept)
            for i in range(len(coef)):
                f.write(f"z=(x[{i}]-{mean[i]!r}f)/{scale[i]!r}f;acc+={coef[i]!r}f*z;")
            f.write(f"if(acc<{SOH_LOW!r}f)acc={SOH_LOW!r}f;if(acc>{SOH_HIGH!r}f)acc={SOH_HIGH!r}f;return acc;}}\n")


@register_estimator
class GBDTEstimator:
    """Gradient-boosted trees (histogram). Fast on large data, NaN-robust."""
    name = "gbdt"; feature_mode = "unified"; backend = "sklearn"

    def __init__(self, max_iter: int = 200, learning_rate: float = 0.1):
        self._model = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate)
        self._cols = None

    def fit(self, X, y, meta=None):
        self._cols = feature_columns(X)
        self._model.fit(X[self._cols].to_numpy(), np.asarray(y)); return self

    def predict(self, X, meta=None):
        return np.clip(self._model.predict(X[self._cols].to_numpy()), SOH_LOW, SOH_HIGH)


@register_estimator
class SVREstimator:
    """RBF-kernel SVR on standardized features; training subsampled for tractability."""
    name = "svr"; feature_mode = "unified"; backend = "sklearn"

    def __init__(self, n_max: int = 3000, C: float = 10.0, gamma="scale"):
        self._n_max = n_max; self._scaler = StandardScaler()
        self._model = SVR(kernel="rbf", C=C, gamma=gamma); self._cols = None

    def fit(self, X, y, meta=None):
        self._cols = feature_columns(X)
        Xs, ys = _subsample(X[self._cols].to_numpy(), np.asarray(y), self._n_max)
        self._model.fit(self._scaler.fit_transform(Xs), ys); return self

    def predict(self, X, meta=None):
        Z = self._scaler.transform(X[self._cols].to_numpy())
        return np.clip(self._model.predict(Z), SOH_LOW, SOH_HIGH)


@register_estimator
class GPREstimator:
    """Gaussian Process with NATIVE predictive intervals (return_std). Subsampled; fixed
    kernel (optimizer disabled) keeps per-fold fits fast across the benchmark matrix."""
    name = "gpr"; feature_mode = "unified"; backend = "sklearn"

    def __init__(self, n_max: int = 800):
        self._n_max = n_max; self._scaler = StandardScaler()
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
        self._model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                               optimizer=None, alpha=1e-6)
        self._cols = None

    def fit(self, X, y, meta=None):
        self._cols = feature_columns(X)
        Xs, ys = _subsample(X[self._cols].to_numpy(), np.asarray(y), self._n_max)
        self._model.fit(self._scaler.fit_transform(Xs), ys); return self

    def predict(self, X, meta=None):
        Z = self._scaler.transform(X[self._cols].to_numpy())
        return np.clip(self._model.predict(Z), SOH_LOW, SOH_HIGH)

    def predict_interval(self, X, meta=None, alpha: float = 0.1):
        Z = self._scaler.transform(X[self._cols].to_numpy())
        mean, std = self._model.predict(Z, return_std=True)
        z = float(norm.ppf(1 - alpha / 2))
        lo = np.clip(mean - z * std, SOH_LOW, SOH_HIGH)
        hi = np.clip(mean + z * std, SOH_LOW, SOH_HIGH)
        return lo, hi
