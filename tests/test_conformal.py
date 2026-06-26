import numpy as np, pandas as pd
from sohbench.conformal import ConformalWrapper, WeightedConformalWrapper

class ConstResidEstimator:
    name = "const"; feature_mode = "unified"; backend = "sklearn"
    def fit(self, X, y, meta): self._mean = float(np.mean(y))
    def predict(self, X, meta): return np.full(len(X), self._mean)

def _xy(n):
    X = pd.DataFrame({"dataset":["a"]*n,"cell_id":["c"]*n,
                      "cycle_index":range(n),"soh":np.linspace(1.0,0.8,n),"f":np.arange(n)})
    return X, X["soh"].to_numpy()

def test_calibration_uses_train_tail_only():
    X, y = _xy(10)
    w = ConformalWrapper(ConstResidEstimator(), alpha=0.2, calib_frac=0.3)
    w.fit(X, y, meta=None)
    # last 30% (3 rows) are calibration; base must have been fit on first 7
    assert w._n_calib == 3 and w._n_fit == 7

def test_interval_covers_and_orders():
    X, y = _xy(50)
    w = ConformalWrapper(ConstResidEstimator(), alpha=0.1, calib_frac=0.3)
    w.fit(X, y, meta=None)
    lo, hi = w.predict_interval(X, meta=None)
    assert np.all(hi >= lo)
    assert w.ci_source == "conformal"


# ── WeightedConformalWrapper tests ────────────────────────────────────────────

def test_weighted_conformal_ci_source():
    X, y = _xy(50)
    w = WeightedConformalWrapper(ConstResidEstimator(), alpha=0.1, calib_frac=0.3)
    w.fit(X, y, meta=None)
    assert w.ci_source == "weighted_conformal"


def test_weighted_conformal_calibration_uses_train_tail_only():
    """Calibration is carved from the tail; fit sees only the head rows."""
    X, y = _xy(20)
    w = WeightedConformalWrapper(ConstResidEstimator(), alpha=0.1, calib_frac=0.3)
    w.fit(X, y, meta=None)
    # round(20 * 0.3) = 6 calib, 14 fit
    assert w._n_calib == 6 and w._n_fit == 14


def test_weighted_conformal_interval_valid():
    """lo <= hi always, even with small data."""
    X, y = _xy(50)
    w = WeightedConformalWrapper(ConstResidEstimator(), alpha=0.1, calib_frac=0.2)
    w.fit(X, y, meta=None)
    lo, hi = w.predict_interval(X, meta=None)
    assert np.all(hi >= lo)


def test_weighted_conformal_wider_under_shift():
    """When the test distribution is covariate-shifted from calibration,
    q_w (weighted) >= q_plain (unweighted).

    Train: f in [0, 40) — calibration tail is f in [32, 40).
    Test: f in [200, 250) — far from calibration; logistic regression
    easily separates them and assigns high density-ratio to calib points
    with higher f, up-weighting the larger residuals → q_w >= q_plain.
    """
    rng = np.random.default_rng(42)
    n_train = 40
    n_test = 20

    # Train data: f ~ [0, n_train)
    f_train = np.arange(n_train, dtype=float)
    y_train = 1.0 - 0.005 * f_train + rng.normal(0, 0.02, n_train)

    X_train = pd.DataFrame({
        "dataset": ["a"] * n_train,
        "cell_id": ["c"] * n_train,
        "cycle_index": np.arange(n_train),
        "soh": y_train,
        "f": f_train,
    })

    # Test data: heavily shifted
    f_test = np.linspace(200, 250, n_test)
    y_test = 0.5 * np.ones(n_test)
    X_test = pd.DataFrame({
        "dataset": ["a"] * n_test,
        "cell_id": ["c"] * n_test,
        "cycle_index": np.arange(n_test),
        "soh": y_test,
        "f": f_test,
    })

    w = WeightedConformalWrapper(ConstResidEstimator(), alpha=0.1, calib_frac=0.2)
    w.fit(X_train, y_train, meta=None)

    q_plain = w._q_plain
    q_w = w._weighted_quantile(X_test, alpha=0.1)
    # Under shift, the weighted quantile should be >= the plain quantile
    assert q_w >= q_plain, f"expected q_w ({q_w:.4f}) >= q_plain ({q_plain:.4f})"


def test_weighted_conformal_degenerate_fallback():
    """With fewer than 5 calib points, falls back to plain quantile without error."""
    n = 10  # 10 * 0.3 = 3 calib → < 5 → fallback
    X, y = _xy(n)
    w = WeightedConformalWrapper(ConstResidEstimator(), alpha=0.1, calib_frac=0.3)
    w.fit(X, y, meta=None)
    assert w._n_calib == 3  # confirms < 5 path
    lo, hi = w.predict_interval(X, meta=None)
    assert np.all(hi >= lo)  # no crash, valid interval
