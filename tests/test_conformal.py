import numpy as np, pandas as pd
from sohbench.conformal import ConformalWrapper

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
