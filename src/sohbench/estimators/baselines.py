from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from ..contracts import feature_columns
from ..registry import register_estimator

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
        Z = self._scaler.transform(X[self._cols].to_numpy())
        return self._model.predict(Z)

    def export_cpp(self, path: str) -> None:
        coef = self._model.coef_; intercept = float(self._model.intercept_)
        mean = self._scaler.mean_; scale = self._scaler.scale_
        with open(path, "w") as f:
            f.write("// auto-generated ridge inference\n")
            f.write(f"// features={self._cols}\n")
            f.write("inline float soh_infer(const float* x){float z;float acc=%r;" % intercept)
            for i in range(len(coef)):
                f.write(f"z=(x[{i}]-{mean[i]!r}f)/{scale[i]!r}f;acc+={coef[i]!r}f*z;")
            f.write("return acc;}\n")
