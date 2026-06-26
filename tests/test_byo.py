import numpy as np, pandas as pd
import pytest
from sohbench.contracts import Estimator
from sohbench.estimators.baselines import SOH_LOW, SOH_HIGH
# Import byo module at module level so @register_estimator fires before registry snapshot.
import sohbench.estimators.byo as _byo_mod
from sohbench.estimators.byo import ByoSelectKBestRidge


def _synthetic_table(n=120, n_features=8, seed=0):
    """Return (X_train, y_train, X_test, y_test, meta) with smooth SOH trend."""
    rng = np.random.default_rng(seed)
    cycles = np.linspace(0, 1, n)
    soh = 1.0 - 0.3 * cycles + rng.normal(0, 0.01, n)
    feats = {f"f{i}": cycles * (i + 1) + rng.normal(0, 0.02, n) for i in range(n_features)}
    df = pd.DataFrame({"dataset": "nasa", "cell_id": "c1",
                       "cycle_index": np.arange(n), "soh": soh, **feats})
    mid = n // 2
    X_train, y_train = df.iloc[:mid], soh[:mid]
    X_test, y_test = df.iloc[mid:], soh[mid:]
    meta = pd.DataFrame({"dataset": ["nasa"], "cell_id": ["c1"],
                          "chemistry": ["LCO"], "c_rated_ah": [2.0],
                          "label_source": ["calibration_capacity"]})
    return X_train, y_train, X_test, y_test, meta


# ── protocol tests ────────────────────────────────────────────────────────────

def test_byo_is_estimator():
    est = ByoSelectKBestRidge()
    assert isinstance(est, Estimator)

def test_byo_attrs():
    est = ByoSelectKBestRidge()
    assert est.name == "byo_skb_ridge"
    assert est.feature_mode == "byo"
    assert est.backend == "sklearn"

def test_byo_registered():
    from sohbench.registry import get_estimator
    est_cls = get_estimator("byo_skb_ridge")
    assert est_cls is not None

# ── functional tests ──────────────────────────────────────────────────────────

def test_byo_fits_smooth_trend():
    X_train, y_train, X_test, y_test, meta = _synthetic_table()
    est = ByoSelectKBestRidge()
    est.fit(X_train, y_train, meta)
    preds = est.predict(X_test, meta)
    mae = float(np.mean(np.abs(preds - y_test)))
    assert mae < 0.1, f"MAE {mae:.4f} is too high (expected < 0.1)"

def test_byo_clamps_ood_high():
    X_train, y_train, X_test, y_test, meta = _synthetic_table()
    est = ByoSelectKBestRidge()
    est.fit(X_train, y_train, meta)
    # Extreme features that would blow up linear prediction
    X_ood = X_test.copy()
    feat_cols = [c for c in X_ood.columns if c.startswith("f")]
    X_ood[feat_cols] = 1e6
    preds = est.predict(X_ood, meta)
    assert (preds <= SOH_HIGH).all(), f"Predictions exceed SOH_HIGH={SOH_HIGH}: {preds.max()}"
    assert (preds >= SOH_LOW).all(), f"Predictions below SOH_LOW={SOH_LOW}: {preds.min()}"

def test_byo_clamps_ood_low():
    X_train, y_train, X_test, y_test, meta = _synthetic_table()
    est = ByoSelectKBestRidge()
    est.fit(X_train, y_train, meta)
    X_ood = X_test.copy()
    feat_cols = [c for c in X_ood.columns if c.startswith("f")]
    X_ood[feat_cols] = -1e6
    preds = est.predict(X_ood, meta)
    assert (preds >= SOH_LOW).all(), f"Predictions below SOH_LOW={SOH_LOW}: {preds.min()}"
    assert (preds <= SOH_HIGH).all(), f"Predictions exceed SOH_HIGH={SOH_HIGH}: {preds.max()}"

def test_byo_handles_fewer_features_than_k():
    """k=min(6, n_features): if only 2 features present, SelectKBest uses k=2."""
    rng = np.random.default_rng(99)
    n = 60
    cycles = np.linspace(0, 1, n)
    soh = 1.0 - 0.2 * cycles
    df = pd.DataFrame({
        "dataset": "nasa", "cell_id": "c1",
        "cycle_index": np.arange(n), "soh": soh,
        "f0": cycles, "f1": cycles * 2,  # only 2 features
    })
    meta = pd.DataFrame({"dataset": ["nasa"], "cell_id": ["c1"],
                          "chemistry": ["LCO"], "c_rated_ah": [2.0],
                          "label_source": ["calibration_capacity"]})
    mid = n // 2
    est = ByoSelectKBestRidge()
    est.fit(df.iloc[:mid], soh[:mid], meta)
    preds = est.predict(df.iloc[mid:], meta)
    assert len(preds) == mid
    assert (preds >= SOH_LOW).all()
    assert (preds <= SOH_HIGH).all()
