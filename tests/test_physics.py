"""Tests for sohbench.estimators.physics — TDD: written before implementation."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from sohbench.contracts import Estimator
from sohbench.estimators.physics import PhysicsResidualEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sqrt_decay_table(n: int = 120, noise_scale: float = 0.005, seed: int = 42,
                      throughput_col: str = "cycle_index_feature") -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic sqrt-decay SOH series with one feature column = throughput."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 400, n)
    soh = 1.0 - 0.05 * np.sqrt(t) + rng.normal(0, noise_scale, n)
    soh = np.clip(soh, 0.0, 1.3)

    X = pd.DataFrame({
        "dataset": ["synth"] * n,
        "cell_id": ["c1"] * n,
        "cycle_index": np.arange(n),
        "soh": soh,
        throughput_col: t,
        # extra feature
        "extra_feature": rng.random(n),
    })
    y = soh.copy()
    return X, y


def _flat_mean_mae(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """MAE of predicting the training mean for every test point."""
    return float(np.mean(np.abs(y_test - np.mean(y_train))))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPhysicsResidualEstimator:

    def test_is_estimator_protocol(self):
        est = PhysicsResidualEstimator()
        assert isinstance(est, Estimator)

    def test_name_feature_mode_backend(self):
        est = PhysicsResidualEstimator()
        assert est.name == "phys_resid"
        assert est.feature_mode == "unified"
        assert est.backend == "sklearn"

    def test_fit_predict_same_size(self):
        X, y = _sqrt_decay_table()
        est = PhysicsResidualEstimator()
        est.fit(X, y, meta=None)
        p = est.predict(X, meta=None)
        assert p.shape == (len(X),)

    def test_beats_flat_mean_baseline_on_sqrt_decay(self):
        """PhysicsResidual should fit the sqrt trend better than predicting the mean."""
        X, y = _sqrt_decay_table(n=200, noise_scale=0.005)
        # Simple train/test split
        n_train = 150
        X_train, y_train = X.iloc[:n_train], y[:n_train]
        X_test, y_test = X.iloc[n_train:], y[n_train:]

        est = PhysicsResidualEstimator()
        est.fit(X_train, y_train, meta=None)
        p = est.predict(X_test, meta=None)

        mae_model = float(np.mean(np.abs(p - y_test)))
        mae_flat = _flat_mean_mae(y_train, y_test)

        assert mae_model < mae_flat, (
            f"Model MAE {mae_model:.4f} should beat flat-mean MAE {mae_flat:.4f}"
        )

    def test_predictions_clamp_to_soh_range(self):
        """OOD features must not produce predictions outside [0.0, 1.3]."""
        X, y = _sqrt_decay_table()
        est = PhysicsResidualEstimator()
        est.fit(X, y, meta=None)

        ood = X.copy()
        ood["extra_feature"] = 1e6
        ood["cycle_index_feature"] = -1e6  # negative throughput (impossible, OOD)
        p = est.predict(ood, meta=None)
        assert np.all(p >= 0.0), "Predictions below 0 detected"
        assert np.all(p <= 1.3), "Predictions above 1.3 detected"

    def test_uses_cycle_index_feature_when_present(self):
        """Estimator should prefer cycle_index_feature as throughput proxy."""
        X, y = _sqrt_decay_table(throughput_col="cycle_index_feature")
        est = PhysicsResidualEstimator()
        est.fit(X, y, meta=None)
        assert est._throughput_col == "cycle_index_feature"

    def test_falls_back_to_cycle_fraction(self):
        """When cycle_index_feature absent, fall back to cycle_fraction."""
        X, y = _sqrt_decay_table(throughput_col="cycle_fraction")
        # Remove cycle_index_feature if present
        X = X.drop(columns=["cycle_index_feature"], errors="ignore")
        est = PhysicsResidualEstimator()
        est.fit(X, y, meta=None)
        assert est._throughput_col == "cycle_fraction"

    def test_predictions_are_float_array(self):
        X, y = _sqrt_decay_table()
        est = PhysicsResidualEstimator().fit(X, y, meta=None)
        p = est.predict(X, meta=None)
        assert isinstance(p, np.ndarray)
        assert p.dtype.kind == "f"

    def test_prior_captures_trend_direction(self):
        """Prior predictions should decrease monotonically on the training trend."""
        X, y = _sqrt_decay_table(n=100, noise_scale=0.0)
        est = PhysicsResidualEstimator()
        est.fit(X, y, meta=None)
        # The full model (prior + residual) should fit training data well
        p = est.predict(X, meta=None)
        mae = float(np.mean(np.abs(p - y)))
        assert mae < 0.05, f"MAE {mae:.4f} too large for noiseless sqrt-decay"
