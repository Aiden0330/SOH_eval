import numpy as np, pandas as pd
from sohkernel.estimators.baselines import RidgeEstimator
from sohkernel.contracts import Estimator

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


def test_ridge_clamps_out_of_distribution_predictions():
    # Train on a normal range, then predict on wildly out-of-distribution
    # features (the leave-one-domain scaling blow-up). Output must stay in [0,1.3].
    X, y = _xy()
    est = RidgeEstimator().fit(X, y, meta=None)
    ood = X.copy()
    ood["f1"] = 1e6
    ood["f2"] = -1e6
    p = est.predict(ood, meta=None)
    assert np.all(p >= 0.0) and np.all(p <= 1.3)


def test_ridge_imputes_structural_missing_features_from_training_fold():
    X, y = _xy()
    X.loc[X.index[:10], "f2"] = np.nan
    est = RidgeEstimator().fit(X, y, meta=None)

    held_out = X.copy()
    held_out["f1"] = np.nan
    p = est.predict(held_out, meta=None)

    assert p.shape == (len(X),)
    assert np.isfinite(p).all()
    assert np.all(p >= 0.0) and np.all(p <= 1.3)
