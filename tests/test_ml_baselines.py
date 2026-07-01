import numpy as np, pandas as pd
import pytest
from sohbench.contracts import Estimator
from sohbench.estimators.baselines import GBDTEstimator, SVREstimator, GPREstimator


def _xy(n=60):
    f = np.linspace(0, 1, n)
    X = pd.DataFrame({"dataset": ["a"] * n, "cell_id": ["c"] * n, "cycle_index": range(n),
                      "soh": 1.0 - 0.2 * f, "f1": f, "f2": f ** 2})
    return X, X["soh"].to_numpy()


@pytest.mark.parametrize("cls", [GBDTEstimator, SVREstimator, GPREstimator])
def test_is_estimator_and_predicts_in_range(cls):
    X, y = _xy()
    est = cls()
    assert isinstance(est, Estimator)
    est.fit(X, y, meta=None)
    p = est.predict(X, meta=None)
    assert p.shape == (len(X),)
    assert np.all(p >= 0.0) and np.all(p <= 1.3)
    assert np.mean(np.abs(p - y)) < 0.1  # should at least roughly fit training trend


@pytest.mark.parametrize("cls", [GBDTEstimator, SVREstimator, GPREstimator])
def test_clamps_out_of_distribution(cls):
    X, y = _xy()
    est = cls().fit(X, y, meta=None)
    ood = X.copy(); ood["f1"] = 1e6; ood["f2"] = -1e6
    p = est.predict(ood, meta=None)
    assert np.all(p >= 0.0) and np.all(p <= 1.3)


def test_gpr_has_native_interval():
    X, y = _xy()
    est = GPREstimator().fit(X, y, meta=None)
    assert hasattr(est, "predict_interval")
    lo, hi = est.predict_interval(X, meta=None, alpha=0.1)
    assert lo.shape == hi.shape == (len(X),)
    assert np.all(hi >= lo)
    assert np.all(lo >= 0.0) and np.all(hi <= 1.3)


def test_subsampling_keeps_large_fit_tractable():
    # 20k rows must fit quickly because large learners subsample internally
    X, y = _xy(20000)
    for cls in (GBDTEstimator, SVREstimator, GPREstimator):
        cls().fit(X, y, meta=None)  # must not hang / raise


def test_gbdt_subsample_cap(monkeypatch):
    fit_shapes = []

    class SpyHGBR:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            fit_shapes.append(X.shape)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr("sohbench.estimators.baselines.HistGradientBoostingRegressor", SpyHGBR)
    X, y = _xy(100)

    GBDTEstimator(n_max=12).fit(X, y, meta=None)

    assert fit_shapes == [(12, 2)]
