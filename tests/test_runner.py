import numpy as np, pandas as pd
from sohbench.runner import BenchmarkRunner
from sohbench.protocols import LeaveOneCellOut, LeaveOneDatasetOut
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


# ── H2b: BYO track tests ──────────────────────────────────────────────────────

def _table_multi_dataset():
    """Two datasets so LeaveOneDatasetOut has >=2 folds."""
    rows = []
    for ds in ["ds1", "ds2"]:
        for cell in ["c1", "c2"]:
            for k in range(10):
                rows.append((ds, cell, k, 1.0 - 0.01 * k, float(k), float(k * k)))
    return pd.DataFrame(rows, columns=["dataset", "cell_id", "cycle_index", "soh", "f1", "f2"])


def test_byo_estimator_track_is_byo_and_not_projected():
    """ByoSelectKBestRidge runs with track='byo', featureset='(byo)',
    even if featuresets=['voltage'] is passed — BYO ignores it."""
    import sohbench.featuresets  # noqa: F401
    import sohbench.estimators.byo  # noqa: F401 — registers byo_skb_ridge
    from sohbench.estimators.byo import ByoSelectKBestRidge
    from sohbench.metadata import build_meta
    t = _table(); meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"loco": LeaveOneCellOut()},
                    estimators=[ByoSelectKBestRidge],
                    featuresets=["voltage"])  # should be ignored for BYO
    assert "track" in rs.columns
    assert set(rs["track"]) == {"byo"}, f"expected only byo, got {set(rs['track'])}"
    assert set(rs["featureset"]) == {"(byo)"}, f"expected (byo), got {set(rs['featureset'])}"


def test_unified_estimator_has_track_unified():
    """A unified estimator (RidgeEstimator) gets track='unified' on all records."""
    import sohbench.featuresets  # noqa: F401
    from sohbench.metadata import build_meta
    t = _table(); meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"loco": LeaveOneCellOut()},
                    estimators=[RidgeEstimator])
    assert "track" in rs.columns
    assert set(rs["track"]) == {"unified"}


# ── H1b: cross_domain → WeightedConformalWrapper ─────────────────────────────

def test_cross_domain_protocol_uses_weighted_conformal():
    """Under LeaveOneDatasetOut (cross_domain=True), non-native estimators get
    ci_source='weighted_conformal'."""
    from sohbench.metadata import build_meta
    t = _table_multi_dataset(); meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"lodo": LeaveOneDatasetOut()},
                    estimators=[RidgeEstimator])
    ci_sources = set(rs["ci_source"])
    assert "weighted_conformal" in ci_sources, \
        f"expected weighted_conformal, got {ci_sources}"
    assert "conformal" not in ci_sources


def test_loco_protocol_uses_plain_conformal():
    """Under LeaveOneCellOut (cross_domain=False), non-native estimators get
    ci_source='conformal'."""
    from sohbench.metadata import build_meta
    t = _table(); meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"loco": LeaveOneCellOut()},
                    estimators=[RidgeEstimator])
    ci_sources = set(rs["ci_source"])
    assert "conformal" in ci_sources, f"expected conformal, got {ci_sources}"
    assert "weighted_conformal" not in ci_sources


def test_backward_compat_records_have_track_unified():
    """Existing callers that don't pass featuresets still get track='unified'
    and featureset='full' on every record."""
    from sohbench.metadata import build_meta
    t = _table(); meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"loco": LeaveOneCellOut()},
                    estimators=[RidgeEstimator])
    assert set(rs["track"]) == {"unified"}
    assert set(rs["featureset"]) == {"full"}


def test_runner_calls_on_model_fitted_after_successful_fit():
    from sohbench.contracts import Fold

    class TinyProtocol:
        def generate(self, table, meta):
            yield Fold(
                train_idx=np.array([0, 1, 2]),
                test_idx=np.array([3, 4]),
                held_out_key="demo",
            )

    class TinyEstimator:
        name = "tiny_estimator"
        feature_mode = "unified"
        backend = "external"

        def fit(self, X, y, meta=None):
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X, meta=None):
            return np.full(len(X), self.mean_)

        def predict_interval(self, X, meta=None):
            center = self.predict(X, meta)
            return center, center

    table = pd.DataFrame(
        {
            "dataset": ["demo"] * 5,
            "cell_id": ["c1"] * 5,
            "cycle_index": [0, 1, 2, 3, 4],
            "soh": [1.0, 0.99, 0.98, 0.97, 0.96],
            "voltage_mean": [3.5, 3.49, 3.48, 3.47, 3.46],
        }
    )
    seen = []

    BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(),
        table=table,
        meta=pd.DataFrame(),
        protocols={"demo": TinyProtocol()},
        estimators=[TinyEstimator],
        featuresets=["full"],
        on_model_fitted=lambda **kwargs: seen.append(kwargs),
    )

    assert len(seen) == 1
    assert seen[0]["record"]["estimator"] == "tiny_estimator"
    assert np.isclose(seen[0]["estimator"].mean_, 0.99)
    assert list(seen[0]["train_table"]["cycle_index"]) == [0, 1, 2]


def test_estimators_never_see_the_label_inside_x():
    from sohbench.metadata import build_meta
    seen = {}
    class Spy:
        name = "spy"; feature_mode = "unified"; backend = "sklearn"
        def fit(self, X, y, meta): seen["fit"] = list(X.columns)
        def predict(self, X, meta):
            seen["predict"] = list(X.columns)
            return np.full(len(X), 0.9)
    t = _table(); meta = build_meta(t)
    BenchmarkRunner(seeds=[0]).run(SOHEstimationTask(), table=t, meta=meta,
                                   protocols={"loco": LeaveOneCellOut()},
                                   estimators=[Spy])
    assert "soh" not in seen["fit"] and "soh" not in seen["predict"]
    assert {"f1", "f2"} <= set(seen["predict"])


def test_multi_cell_fold_monotonicity_is_scored_per_cell():
    """LODO test folds hold many cells; a predictor that is perfectly
    monotone within every cell must score zero violations even though the
    cells sit at different levels (a global cycle sort interleaves them)."""
    from sohbench.metadata import build_meta
    rows = []
    for ds, offset in (("a", 0.0), ("b", 0.2)):
        for cell in ("c1", "c2"):
            for k in range(15):
                rows.append((ds, f"{ds}-{cell}", k, 1.0 - 0.01 * k - offset,
                             float(k), float(k * k)))
    t = pd.DataFrame(rows, columns=["dataset", "cell_id", "cycle_index", "soh", "f1", "f2"])
    class LevelShift:
        name = "level_shift"; feature_mode = "unified"; backend = "sklearn"
        def fit(self, X, y, meta): pass
        def predict(self, X, meta):
            return np.array([
                1.0 - 0.01 * k - (0.3 if str(c).endswith("c2") else 0.0)
                for c, k in zip(X["cell_id"], X["cycle_index"])
            ])
    meta = build_meta(t)
    rs = BenchmarkRunner(seeds=[0]).run(SOHEstimationTask(), table=t, meta=meta,
                                        protocols={"lodo": LeaveOneDatasetOut()},
                                        estimators=[LevelShift])
    mono = rs.loc[rs["metric"] == "monotonicity_violation_rate", "value"]
    assert len(mono) == 2 and (mono == 0.0).all()
