"""Additional tabular baselines and a gray-box SOH degradation method.

The first three estimators remain valid members of the ordinary
``estimator x feature-set`` benchmark.  ``StressResidualEstimator`` is
different: it couples a non-negative semi-empirical degradation prior with a
data-driven residual, so its catalog metadata limits it to feature profiles
that carry degradation or operating-stress meaning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from ..contracts import feature_columns
from ..preprocessing import FeatureMedianImputer
from ..registry import register_estimator
from .baselines import SOH_HIGH, SOH_LOW, _subsample

__all__ = [
    "ElasticNetEstimator",
    "ExtraTreesEstimator",
    "PLSEstimator",
    "StressResidualEstimator",
]


@register_estimator
class ElasticNetEstimator:
    """Sparse linear baseline for broad, correlated feature catalogs."""

    name = "elastic_net"
    feature_mode = "unified"
    backend = "sklearn"
    method_family = "data_driven_interpretable"
    pairing_mode = "free"
    description = "L1/L2 联合正则的稀疏线性基线，便于检查新增机理代理特征是否提供稳定增益。"

    def __init__(self, alpha: float = 1e-3, l1_ratio: float = 0.25) -> None:
        self._imputer = FeatureMedianImputer()
        self._scaler = StandardScaler()
        self._model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=5000,
            selection="cyclic",
            random_state=0,
        )
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y, meta=None):
        del meta
        self._cols = feature_columns(X)
        values = self._imputer.fit(X, self._cols).transform_array(X)
        self._model.fit(self._scaler.fit_transform(values), np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame, meta=None):
        del meta
        values = self._imputer.transform_array(X)
        return np.clip(self._model.predict(self._scaler.transform(values)), SOH_LOW, SOH_HIGH)


@register_estimator
class ExtraTreesEstimator:
    """Nonlinear ensemble baseline with feature-importance diagnostics."""

    name = "extra_trees"
    feature_mode = "unified"
    backend = "sklearn"
    method_family = "data_driven_ensemble"
    pairing_mode = "free"
    description = "极端随机树非线性基线，用于检验机理代理特征的非线性交互与鲁棒性。"

    def __init__(self, n_estimators: int = 240, n_max: int = 16_000, random_state: int = 0) -> None:
        self._n_max = n_max
        self._random_state = random_state
        self._imputer = FeatureMedianImputer()
        self._model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=random_state,
        )
        self._cols: list[str] = []

    def set_seed(self, seed: int) -> None:
        """Re-seed the subsample and the forest (evaluation seeds must reach the model)."""
        self._random_state = int(seed)
        self._model.set_params(random_state=int(seed))

    def fit(self, X: pd.DataFrame, y, meta=None):
        del meta
        self._cols = feature_columns(X)
        values = self._imputer.fit(X, self._cols).transform_array(X)
        values, target = _subsample(values, np.asarray(y, dtype=float), self._n_max, self._random_state)
        self._model.fit(values, target)
        return self

    def predict(self, X: pd.DataFrame, meta=None):
        del meta
        return np.clip(self._model.predict(self._imputer.transform_array(X)), SOH_LOW, SOH_HIGH)


@register_estimator
class PLSEstimator:
    """Partial least-squares baseline for collinear electrochemical descriptors."""

    name = "pls"
    feature_mode = "unified"
    backend = "sklearn"
    method_family = "latent_variable"
    pairing_mode = "recommended"
    preferred_featureset = "electrochemical_diagnostics"
    description = "面向 IC/DVA、局部窗口等强共线诊断量的低维潜变量回归。"

    def __init__(self, n_components: int = 4) -> None:
        self._requested_components = n_components
        self._imputer = FeatureMedianImputer()
        self._model: PLSRegression | None = None
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y, meta=None):
        del meta
        self._cols = feature_columns(X)
        values = self._imputer.fit(X, self._cols).transform_array(X)
        if values.shape[1] == 0:
            raise ValueError("PLSEstimator requires at least one feature column")
        components = max(1, min(self._requested_components, values.shape[1], max(1, len(values) - 1)))
        self._model = PLSRegression(n_components=components, scale=True, max_iter=1000)
        self._model.fit(values, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame, meta=None):
        del meta
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        prediction = np.asarray(self._model.predict(self._imputer.transform_array(X))).reshape(-1)
        return np.clip(prediction, SOH_LOW, SOH_HIGH)


def _numeric(X: pd.DataFrame, name: str) -> np.ndarray:
    if name not in X.columns:
        return np.full(len(X), np.nan, dtype=float)
    values = pd.to_numeric(X[name], errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


@register_estimator
class StressResidualEstimator:
    """Non-negative aging/stress prior plus boosted residual correction.

    The prior uses square-root and linear cycle exposure, a fixed Arrhenius
    temperature factor, and an RMS-current stress proxy.  It is deliberately a
    gray-box method: coefficients are identified from training data while
    their signs are constrained to represent non-negative degradation.
    """

    name = "stress_resid"
    feature_mode = "unified"
    backend = "sklearn"
    method_family = "gray_box_degradation"
    pairing_mode = "method_package"
    compatible_featuresets = (
        "degradation_kinetics",
        "operating_stress",
        "electrochemical_diagnostics",
        "full",
    )
    preferred_featureset = "operating_stress"
    data_granularity = "cycle_table"
    required_signals = ("cycle_index",)
    physics_constraints = (
        "non_negative_degradation_coefficients",
        "arrhenius_temperature_stress",
        "trajectory_monotonicity",
        "soh_range",
    )
    description = "非负退化先验（sqrt-cycle、线性老化、温度与电流应力）叠加树模型残差。"

    def __init__(
        self,
        max_iter: int = 180,
        learning_rate: float = 0.06,
        n_max: int = 14_000,
        random_state: int = 0,
        enforce_monotone: bool = True,
    ) -> None:
        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self._n_max = n_max
        self._random_state = random_state
        self._enforce_monotone = enforce_monotone
        self._imputer = FeatureMedianImputer()
        self._model: HistGradientBoostingRegressor | None = None
        self._cols: list[str] = []
        self._cycle_scale = 1.0
        self._current_scale = 1.0
        self._baseline_soh = 1.0

    def set_seed(self, seed: int) -> None:
        """Re-seed the subsample and the residual booster built in ``fit``."""
        self._random_state = int(seed)
        if self._model is not None:
            self._model.set_params(random_state=int(seed))
        self._prior_coeffs = np.zeros(4, dtype=float)

    def _cycle(self, X: pd.DataFrame) -> np.ndarray:
        values = _numeric(X, "cycle_index_feature")
        if not np.isfinite(values).any():
            values = _numeric(X, "cycle_index")
        return np.maximum(np.where(np.isfinite(values), values, 0.0), 0.0)

    def _thermal(self, X: pd.DataFrame) -> np.ndarray:
        factor = _numeric(X, "arrhenius_stress_factor")
        if np.isfinite(factor).any():
            fill = float(np.nanmedian(factor))
            return np.where(np.isfinite(factor), factor, fill)
        temperature = _numeric(X, "temperature_mean")
        kelvin = temperature + 273.15
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            exponent = (24_000.0 / 8.314462618) * (1.0 / 298.15 - 1.0 / kelvin)
            factor = np.exp(np.clip(exponent, -30.0, 30.0))
        return np.where(np.isfinite(factor), factor, 1.0)

    def _current(self, X: pd.DataFrame) -> np.ndarray:
        rms = _numeric(X, "current_rms_proxy_a")
        if not np.isfinite(rms).any():
            mean = _numeric(X, "current_mean")
            std = _numeric(X, "current_std")
            rms = np.sqrt(np.square(mean) + np.square(std))
        if not np.isfinite(rms).any():
            rms = _numeric(X, "abs_current_mean")
        return np.where(np.isfinite(rms), np.maximum(rms, 0.0), 0.0)

    def _prior_design(self, X: pd.DataFrame, *, fitting: bool) -> np.ndarray:
        cycle = self._cycle(X)
        current = self._current(X)
        if fitting:
            finite_cycle = cycle[np.isfinite(cycle) & (cycle > 0)]
            self._cycle_scale = float(np.quantile(finite_cycle, 0.9)) if finite_cycle.size else 1.0
            finite_current = current[np.isfinite(current) & (current > 0)]
            self._current_scale = float(np.median(finite_current)) if finite_current.size else 1.0
        q = cycle / max(self._cycle_scale, 1e-12)
        i_stress = current / max(self._current_scale, 1e-12)
        thermal = self._thermal(X)
        return np.column_stack([np.sqrt(q), q, q * thermal, q * i_stress])

    def _prior(self, X: pd.DataFrame) -> np.ndarray:
        loss = self._prior_design(X, fitting=False) @ self._prior_coeffs
        return self._baseline_soh - loss

    def fit(self, X: pd.DataFrame, y, meta=None):
        del meta
        target = np.asarray(y, dtype=float)
        design = self._prior_design(X, fitting=True)
        cycle = self._cycle(X)
        early = target[cycle <= np.nanquantile(cycle, 0.1)] if len(cycle) else target
        finite_early = early[np.isfinite(early)]
        self._baseline_soh = float(np.median(finite_early)) if finite_early.size else 1.0
        self._baseline_soh = float(np.clip(self._baseline_soh, 0.8, SOH_HIGH))
        observed_loss = np.maximum(self._baseline_soh - target, 0.0)
        self._prior_coeffs, _ = nnls(design, observed_loss)
        residual = target - self._prior(X)

        self._cols = feature_columns(X)
        values = self._imputer.fit(X, self._cols).transform_array(X)
        values, residual = _subsample(values, residual, self._n_max, self._random_state)
        self._model = HistGradientBoostingRegressor(
            max_iter=self._max_iter,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
        )
        self._model.fit(values, residual)
        return self

    def predict(self, X: pd.DataFrame, meta=None):
        del meta
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        prediction = self._prior(X) + self._model.predict(self._imputer.transform_array(X))
        prediction = np.clip(prediction, SOH_LOW, SOH_HIGH)
        if self._enforce_monotone and {"dataset", "cell_id", "cycle_index"}.issubset(X.columns):
            projected = prediction.copy()
            identity = X.reset_index(drop=True)
            for _, group in identity.groupby(["dataset", "cell_id"], sort=False):
                order = group.sort_values("cycle_index").index.to_numpy()
                projected[order] = np.minimum.accumulate(projected[order])
            prediction = projected
        return prediction
