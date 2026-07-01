"""Tests for sohbench.estimators.deep — LSTM, CNN1D, Transformer estimators.

Gated with pytest.importorskip("torch") so the suite stays green without PyTorch.

Test goals (per spec):
  1. isinstance(est, Estimator)
  2. fit then predict → shape (len(X),)
  3. Training-fit MAE < 0.12 on a smooth synthetic trend
  4. Predictions in [0, 1.3] on OOD input (features = 1e6)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sohbench.contracts import Estimator
from sohbench.estimators.deep import (
    CNN1DEstimator,
    LSTMEstimator,
    TransformerEstimator,
)
from sohbench.estimators.baselines import SOH_LOW, SOH_HIGH
from sohbench.registry import get_estimator


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def _multi_cell_smooth_frame(
    n_cells: int = 4,
    n_cycles: int = 30,
    n_features: int = 3,
    seed: int = 42,
    feature_noise: float = 0.005,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic frame: smooth linear SOH decay, features correlated with SOH.

    Uses low feature noise so that nets can learn the trend within ~60 epochs.
    Returns ``(X, y)`` where y == X["soh"].
    """
    rng = np.random.default_rng(seed)
    records = []
    for ci in range(n_cells):
        cycles = np.arange(n_cycles)
        # Each cell starts at slightly different SOH and decays at different rate
        start_soh = rng.uniform(0.90, 1.05)
        rate = rng.uniform(0.005, 0.012)
        soh = np.clip(start_soh - rate * cycles, 0.0, 1.3)

        for t, c in enumerate(cycles):
            # Features are strongly correlated with soh + tiny noise
            feat = {
                f"f{k}": soh[t] * (0.9 + 0.1 * k / max(n_features - 1, 1))
                         + rng.normal(0, feature_noise)
                for k in range(n_features)
            }
            records.append({
                "dataset": "synth",
                "cell_id": f"cell_{ci}",
                "cycle_index": int(c),
                "soh": float(soh[t]),
                **feat,
            })

    df = pd.DataFrame(records)
    y = df["soh"].to_numpy()
    return df, y


# ---------------------------------------------------------------------------
# OOD frame
# ---------------------------------------------------------------------------

def _ood_frame(X: pd.DataFrame, val: float = 1e6) -> pd.DataFrame:
    """Copy of X with all feature columns set to ``val``."""
    from sohbench.contracts import feature_columns
    ood = X.copy()
    for col in feature_columns(X):
        ood[col] = val
    return ood


# ---------------------------------------------------------------------------
# Parametrise over all three estimators
# ---------------------------------------------------------------------------

ESTIMATOR_CLASSES = [LSTMEstimator, CNN1DEstimator, TransformerEstimator]
ESTIMATOR_IDS = ["lstm", "cnn1d", "transformer"]


@pytest.fixture(params=ESTIMATOR_CLASSES, ids=ESTIMATOR_IDS)
def fitted_estimator(request):
    """Returns (estimator_class, X, y, fitted_estimator) on synthetic data."""
    cls = request.param
    # Low-noise data (feature_noise=0.005) so nets converge within 60 epochs.
    # 6 cells × 40 cycles = 240 rows (well under n_max=5000 subsample cap).
    X, y = _multi_cell_smooth_frame(n_cells=6, n_cycles=40, feature_noise=0.005)
    # Use lr=3e-3 so CNN1D/Transformer converge faster
    est = cls(W=4, epochs=60, n_max=5000, seed=0, lr=3e-3)
    est.fit(X, y)
    return est, X, y


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class TestEstimatorProtocol:
    @pytest.mark.parametrize("cls, eid", list(zip(ESTIMATOR_CLASSES, ESTIMATOR_IDS)))
    def test_is_estimator_protocol(self, cls, eid):
        est = cls()
        assert isinstance(est, Estimator), f"{cls.__name__} does not satisfy Estimator protocol"

    @pytest.mark.parametrize("cls, eid", list(zip(ESTIMATOR_CLASSES, ESTIMATOR_IDS)))
    def test_name(self, cls, eid):
        assert cls.name == eid

    @pytest.mark.parametrize("cls", ESTIMATOR_CLASSES)
    def test_feature_mode(self, cls):
        assert cls.feature_mode == "unified"

    @pytest.mark.parametrize("cls", ESTIMATOR_CLASSES)
    def test_backend(self, cls):
        assert cls.backend == "pytorch"


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

class TestPredictShape:
    def test_predict_shape(self, fitted_estimator):
        est, X, y = fitted_estimator
        preds = est.predict(X)
        assert preds.shape == (len(X),), (
            f"Expected ({len(X)},) but got {preds.shape}"
        )

    def test_predict_returns_ndarray(self, fitted_estimator):
        est, X, y = fitted_estimator
        preds = est.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_float_dtype(self, fitted_estimator):
        est, X, y = fitted_estimator
        preds = est.predict(X)
        assert preds.dtype.kind == "f"


# ---------------------------------------------------------------------------
# MAE on training data
# ---------------------------------------------------------------------------

class TestTrainingFit:
    def test_training_mae_under_012(self, fitted_estimator):
        """Training-fit MAE must be < 0.12 on a smooth synthetic trend."""
        est, X, y = fitted_estimator
        preds = est.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
        assert mae < 0.12, (
            f"{est.__class__.__name__} training MAE {mae:.4f} exceeds 0.12 threshold"
        )


# ---------------------------------------------------------------------------
# OOD clamp
# ---------------------------------------------------------------------------

class TestOODClamp:
    def test_ood_predictions_in_soh_range(self, fitted_estimator):
        """Predictions on OOD (features=1e6) must remain in [SOH_LOW, SOH_HIGH]."""
        est, X, y = fitted_estimator
        ood = _ood_frame(X, val=1e6)
        preds = est.predict(ood)
        assert np.all(preds >= SOH_LOW), (
            f"Min OOD prediction {preds.min():.4f} < SOH_LOW={SOH_LOW}"
        )
        assert np.all(preds <= SOH_HIGH), (
            f"Max OOD prediction {preds.max():.4f} > SOH_HIGH={SOH_HIGH}"
        )

    def test_negative_ood_predictions_clamped(self, fitted_estimator):
        """Predictions on OOD (features=-1e6) must remain >= SOH_LOW."""
        est, X, y = fitted_estimator
        ood = _ood_frame(X, val=-1e6)
        preds = est.predict(ood)
        assert np.all(preds >= SOH_LOW)
        assert np.all(preds <= SOH_HIGH)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistration:
    @pytest.mark.parametrize("name, cls", list(zip(ESTIMATOR_IDS, ESTIMATOR_CLASSES)))
    def test_registered_in_registry(self, name, cls):
        """Each estimator must be retrievable by name from the registry."""
        retrieved = get_estimator(name)
        assert retrieved is cls


# ---------------------------------------------------------------------------
# fit returns self
# ---------------------------------------------------------------------------

class TestFitReturnsSelf:
    @pytest.mark.parametrize("cls", ESTIMATOR_CLASSES)
    def test_fit_returns_self(self, cls):
        X, y = _multi_cell_smooth_frame(n_cells=2, n_cycles=15)
        est = cls(W=3, epochs=5, n_max=500, seed=0)
        ret = est.fit(X, y)
        assert ret is est


# ---------------------------------------------------------------------------
# meta argument accepted (None or DataFrame)
# ---------------------------------------------------------------------------

class TestMetaArgument:
    @pytest.mark.parametrize("cls", ESTIMATOR_CLASSES)
    def test_meta_none_accepted(self, cls):
        X, y = _multi_cell_smooth_frame(n_cells=2, n_cycles=15)
        est = cls(W=3, epochs=5, n_max=500, seed=0)
        est.fit(X, y, meta=None)
        preds = est.predict(X, meta=None)
        assert preds.shape == (len(X),)

    @pytest.mark.parametrize("cls", ESTIMATOR_CLASSES)
    def test_meta_dataframe_accepted(self, cls):
        X, y = _multi_cell_smooth_frame(n_cells=2, n_cycles=15)
        est = cls(W=3, epochs=5, n_max=500, seed=0)
        est.fit(X, y, meta=pd.DataFrame())
        preds = est.predict(X, meta=pd.DataFrame())
        assert preds.shape == (len(X),)


# ---------------------------------------------------------------------------
# Predict before fit raises
# ---------------------------------------------------------------------------

class TestPredictBeforeFit:
    @pytest.mark.parametrize("cls", ESTIMATOR_CLASSES)
    def test_predict_before_fit_raises(self, cls):
        X, _ = _multi_cell_smooth_frame(n_cells=1, n_cycles=5)
        est = cls()
        with pytest.raises(RuntimeError):
            est.predict(X)
