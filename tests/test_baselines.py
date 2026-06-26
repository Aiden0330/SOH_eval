import numpy as np, pandas as pd
from sohbench.estimators.baselines import RidgeEstimator
from sohbench.contracts import Estimator

def _xy(n=30):
    rng = np.random.default_rng(0)
    f = np.linspace(0,1,n)
    X = pd.DataFrame({"dataset":["a"]*n,"cell_id":["c"]*n,"cycle_index":range(n),
                      "soh":1.0-0.2*f,"f1":f,"f2":f**2})
    return X, X["soh"].to_numpy()

def test_ridge_is_estimator_and_predicts():
    X, y = _xy()
    est = RidgeEstimator()
    assert isinstance(est, Estimator)
    est.fit(X, y, meta=None)
    p = est.predict(X, meta=None)
    assert p.shape == (len(X),)
    assert np.mean(np.abs(p - y)) < 0.05
