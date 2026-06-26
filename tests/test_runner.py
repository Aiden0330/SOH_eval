import numpy as np, pandas as pd
from sohbench.runner import BenchmarkRunner
from sohbench.protocols import LeaveOneCellOut
from sohbench.tasks import SOHEstimationTask
from sohbench.estimators.baselines import RidgeEstimator

def _table():
    rows = []
    for cell in ["c1","c2","c3"]:
        for k in range(15):
            rows.append(("ds", cell, k, 1.0-0.01*k, float(k), float(k*k)))
    return pd.DataFrame(rows, columns=["dataset","cell_id","cycle_index","soh","f1","f2"])

def test_runner_produces_results_with_ci_and_seeds():
    from sohbench.metadata import build_meta
    t = _table(); meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0,1,2,3,4])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"loco": LeaveOneCellOut()},
                    estimators=[RidgeEstimator])
    assert set(rs.columns) >= {"estimator","protocol","held_out_key","fold","seed","metric","value"}
    assert (rs["metric"] == "picp").any()                 # CI guaranteed (conformal)
    assert rs["seed"].nunique() == 5

def test_failed_estimator_isolated():
    from sohbench.metadata import build_meta
    class Boom:
        name="boom"; feature_mode="unified"; backend="sklearn"
        def fit(self,X,y,meta): raise RuntimeError("boom")
        def predict(self,X,meta): ...
    t=_table(); meta=build_meta(t)
    rs = BenchmarkRunner(seeds=[0]).run(SOHEstimationTask(), table=t, meta=meta,
                                        protocols={"loco":LeaveOneCellOut()},
                                        estimators=[Boom])
    assert rs["value"].isna().any()                       # failed fold recorded, run completed


def test_runner_featureset_loop():
    """Passing featuresets=["full","voltage"] yields records for both featuresets."""
    import sohbench.featuresets  # noqa: F401 — registers featuresets
    from sohbench.metadata import build_meta
    # Table needs voltage_* cols so the "voltage" featureset has something to select
    rows = []
    for cell in ["c1", "c2", "c3"]:
        for k in range(15):
            rows.append(("ds", cell, k, 1.0 - 0.01 * k,
                         float(k), float(k * k),
                         float(k) * 0.1, float(k) * 0.2,   # voltage_mean, voltage_std
                         float(k) * 0.05, float(k) * 0.3))  # voltage_min, voltage_max
    t = pd.DataFrame(rows, columns=[
        "dataset", "cell_id", "cycle_index", "soh",
        "f1", "f2", "voltage_mean", "voltage_std", "voltage_min", "voltage_max"
    ])
    meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"loco": LeaveOneCellOut()},
                    estimators=[RidgeEstimator],
                    featuresets=["full", "voltage"])
    assert "featureset" in rs.columns, "featureset column missing"
    assert set(rs["featureset"]) == {"full", "voltage"}, \
        f"expected both featuresets, got {set(rs['featureset'])}"


def test_runner_featureset_default_is_full():
    """Calling run() without featuresets= still produces featureset='full' records."""
    import sohbench.featuresets  # noqa: F401
    from sohbench.metadata import build_meta
    t = _table(); meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"loco": LeaveOneCellOut()},
                    estimators=[RidgeEstimator])
    assert "featureset" in rs.columns
    assert set(rs["featureset"]) == {"full"}
