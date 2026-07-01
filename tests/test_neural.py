"""Tests for sohbench.estimators.neural — TDD: written before implementation."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from sohbench.contracts import Estimator
from sohbench.estimators.neural import MLPEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth_trend_table(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Smooth linear-decay SOH with two feature columns (no noise)."""
    rng = np.random.default_rng(seed)
    f = np.linspace(0, 1, n)
    soh = 1.0 - 0.3 * f
    X = pd.DataFrame({
        "dataset": ["synth"] * n,
        "cell_id": ["c1"] * n,
        "cycle_index": np.arange(n),
        "soh": soh,
        "f1": f,
        "f2": f ** 2,
        "f3": rng.random(n),
    })
    return X, soh.copy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLPEstimator:

    def test_is_estimator_protocol(self):
        est = MLPEstimator()
        assert isinstance(est, Estimator)

    def test_name_feature_mode_backend(self):
        est = MLPEstimator()
        assert est.name == "mlp"
        assert est.feature_mode == "unified"
        assert est.backend == "sklearn"

    def test_fit_predict_returns_correct_shape(self):
        X, y = _smooth_trend_table()
        est = MLPEstimator()
        est.fit(X, y, meta=None)
        p = est.predict(X, meta=None)
        assert p.shape == (len(X),)

    def test_predictions_are_float_array(self):
        X, y = _smooth_trend_table()
        est = MLPEstimator().fit(X, y, meta=None)
        p = est.predict(X, meta=None)
        assert isinstance(p, np.ndarray)
        assert p.dtype.kind == "f"

    def test_fits_smooth_trend_mae_under_threshold(self):
        """On a smooth synthetic trend the MLP should achieve MAE < 0.1 on training data."""
        X, y = _smooth_trend_table(n=300)
        est = MLPEstimator()
        est.fit(X, y, meta=None)
        p = est.predict(X, meta=None)
        mae = float(np.mean(np.abs(p - y)))
        assert mae < 0.1, f"Training MAE {mae:.4f} exceeds 0.1 threshold"

    def test_predictions_clamp_to_soh_range_ood(self):
        """OOD feature value 1e6 must not produce predictions outside [0.0, 1.3]."""
        X, y = _smooth_trend_table()
        est = MLPEstimator().fit(X, y, meta=None)

        ood = X.copy()
        ood["f1"] = 1e6
        ood["f2"] = 1e6
        ood["f3"] = 1e6
        p = est.predict(ood, meta=None)
        assert np.all(p >= 0.0), f"Min prediction {p.min():.4f} < 0"
        assert np.all(p <= 1.3), f"Max prediction {p.max():.4f} > 1.3"

    def test_predictions_in_range_on_training_data(self):
        X, y = _smooth_trend_table()
        est = MLPEstimator().fit(X, y, meta=None)
        p = est.predict(X, meta=None)
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.3)

    def test_fit_returns_self(self):
        X, y = _smooth_trend_table(n=50)
        est = MLPEstimator()
        ret = est.fit(X, y, meta=None)
        assert ret is est

    def test_meta_argument_accepted(self):
        """fit/predict must accept a meta argument (may be None or a DataFrame)."""
        X, y = _smooth_trend_table(n=50)
        est = MLPEstimator()
        est.fit(X, y, meta=pd.DataFrame())
        p = est.predict(X, meta=pd.DataFrame())
        assert p.shape == (len(X),)

    def test_subsampling_caps_large_fit_matrix(self):
        """Large real-data folds should not send every row into MLPRegressor.fit."""
        X, y = _smooth_trend_table(n=100)
        est = MLPEstimator(n_max=25, random_state=7)

        class FitSpy:
            def __init__(self):
                self.n_rows = None
                self.n_targets = None

            def fit(self, Z, yy):
                self.n_rows = len(Z)
                self.n_targets = len(yy)
                return self

        spy = FitSpy()
        est._model = spy

        est.fit(X, y, meta=None)

        assert spy.n_rows == 25
        assert spy.n_targets == 25
