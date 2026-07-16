import random
import threading

import numpy as np, pandas as pd, pytest
from sohbench import metrics as M
from sohbench.contracts import Fold
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


class _SingleFoldProtocol:
    def generate(self, table, meta):
        yield Fold(
            train_idx=np.array([0, 1, 2]),
            test_idx=np.array([3, 4]),
            held_out_key="demo",
        )


def _single_fold_table():
    return pd.DataFrame(
        {
            "dataset": ["demo"] * 5,
            "cell_id": ["c1", "c1", "c1", "c2", "c2"],
            "cycle_index": [0, 1, 2, 3, 4],
            "soh": [1.0, 0.99, 0.98, 0.97, 0.96],
            "voltage_mean": [3.5, 3.49, 3.48, 3.47, 3.46],
        }
    )


def test_concurrent_legacy_candidates_keep_seeded_random_streams_isolated():
    first_drawn = {
        "seed-111": threading.Event(),
        "seed-222": threading.Event(),
    }
    release_first = threading.Event()
    draws = {}
    errors = []

    class InterleavingGlobalRandomEstimator:
        name = "interleaving_global_random"
        feature_mode = "unified"
        backend = "external"

        def fit(self, X, y, meta=None):
            job = threading.current_thread().name
            numpy_values = [float(np.random.random())]
            python_values = [random.random()]
            first_drawn[job].set()
            if job == "seed-111":
                if not release_first.wait(timeout=5):
                    raise TimeoutError("second candidate was not released")
            numpy_values.append(float(np.random.random()))
            python_values.append(random.random())
            draws[job] = {
                "numpy": numpy_values,
                "python": python_values,
            }
            return self

        def predict(self, X, meta=None):
            return np.full(len(X), 0.9)

        def predict_interval(self, X, meta=None):
            center = self.predict(X, meta)
            return center, center

    table = _single_fold_table()
    original_numpy_state = np.random.get_state()
    original_python_state = random.getstate()

    def execute(seed):
        try:
            BenchmarkRunner(seeds=[seed]).run(
                SOHEstimationTask(),
                table=table,
                meta=pd.DataFrame(),
                protocols={"demo": _SingleFoldProtocol()},
                estimators=[InterleavingGlobalRandomEstimator],
            )
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=execute, args=(111,), name="seed-111")
    second = threading.Thread(target=execute, args=(222,), name="seed-222")
    try:
        first.start()
        assert first_drawn["seed-111"].wait(timeout=5)
        second.start()
        first_drawn["seed-222"].wait(timeout=1)
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
    finally:
        release_first.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
        np.random.set_state(original_numpy_state)
        random.setstate(original_python_state)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    for seed in (111, 222):
        job = f"seed-{seed}"
        expected_numpy = np.random.RandomState(seed).random_sample(2)
        expected_python_rng = random.Random(seed)
        expected_python = [
            expected_python_rng.random(),
            expected_python_rng.random(),
        ]
        np.testing.assert_allclose(draws[job]["numpy"], expected_numpy)
        np.testing.assert_allclose(draws[job]["python"], expected_python)


def test_concurrent_private_seed_candidates_still_isolate_global_random_streams():
    first_drawn = {
        "seed-111": threading.Event(),
        "seed-222": threading.Event(),
    }
    release_first = threading.Event()
    draws = {}
    errors = []

    class PrivateAndGlobalRandomEstimator:
        name = "private_and_global_random"
        feature_mode = "unified"
        backend = "external"

        def __init__(self, random_state=None):
            self.random_state = random_state

        def fit(self, X, y, meta=None):
            job = threading.current_thread().name
            numpy_values = [float(np.random.random())]
            python_values = [random.random()]
            first_drawn[job].set()
            if job == "seed-111" and not release_first.wait(timeout=5):
                raise TimeoutError("second candidate was not released")
            numpy_values.append(float(np.random.random()))
            python_values.append(random.random())
            draws[job] = {"numpy": numpy_values, "python": python_values}
            return self

        def predict(self, X, meta=None):
            return np.full(len(X), 0.9)

        def predict_interval(self, X, meta=None):
            center = self.predict(X, meta)
            return center, center

    table = _single_fold_table()
    original_numpy_state = np.random.get_state()
    original_python_state = random.getstate()

    def execute(seed):
        try:
            BenchmarkRunner(seeds=[seed]).run(
                SOHEstimationTask(),
                table=table,
                meta=pd.DataFrame(),
                protocols={"demo": _SingleFoldProtocol()},
                estimators=[PrivateAndGlobalRandomEstimator],
            )
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=execute, args=(111,), name="seed-111")
    second = threading.Thread(target=execute, args=(222,), name="seed-222")
    try:
        first.start()
        assert first_drawn["seed-111"].wait(timeout=5)
        second.start()
        first_drawn["seed-222"].wait(timeout=1)
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
    finally:
        release_first.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
        np.random.set_state(original_numpy_state)
        random.setstate(original_python_state)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    for seed in (111, 222):
        job = f"seed-{seed}"
        np.testing.assert_allclose(
            draws[job]["numpy"], np.random.RandomState(seed).random_sample(2)
        )
        expected_python = random.Random(seed)
        np.testing.assert_allclose(
            draws[job]["python"],
            [expected_python.random(), expected_python.random()],
        )


@pytest.mark.parametrize(
    "prediction",
    [
        pytest.param(np.array([[0.9], [0.8]]), id="two-dimensional"),
        pytest.param(np.array([0.9]), id="wrong-length"),
        pytest.param(np.array([0.9, np.nan]), id="nan"),
        pytest.param(np.array([0.9, np.inf]), id="infinity"),
    ],
)
def test_runner_rejects_invalid_prediction_arrays(prediction):
    class InvalidPredictionEstimator:
        name = "invalid_prediction"
        feature_mode = "unified"
        backend = "external"

        def fit(self, X, y, meta=None):
            return self

        def predict(self, X, meta=None):
            return prediction

        def predict_interval(self, X, meta=None):
            return np.full(len(X), 0.8), np.full(len(X), 1.0)

    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(),
        table=_single_fold_table(),
        meta=pd.DataFrame(),
        protocols={"demo": _SingleFoldProtocol()},
        estimators=[InvalidPredictionEstimator],
    )

    assert results["unit_status"].eq("failed").all()
    assert results["error_category"].eq("ALGORITHM").all()
    assert results["error_message"].str.contains("predictions").all()


@pytest.mark.parametrize("bound", ["interval_low", "interval_high"])
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(np.array([[0.8], [0.8]]), id="two-dimensional"),
        pytest.param(np.array([0.8]), id="wrong-length"),
        pytest.param(np.array([0.8, np.nan]), id="nan"),
        pytest.param(np.array([0.8, np.inf]), id="infinity"),
    ],
)
def test_runner_rejects_invalid_interval_arrays(bound, invalid):
    class InvalidIntervalEstimator:
        name = "invalid_interval"
        feature_mode = "unified"
        backend = "external"

        def fit(self, X, y, meta=None):
            return self

        def predict(self, X, meta=None):
            return np.full(len(X), 0.9)

        def predict_interval(self, X, meta=None):
            interval = {
                "interval_low": np.full(len(X), 0.8),
                "interval_high": np.full(len(X), 1.0),
            }
            interval[bound] = invalid
            return interval["interval_low"], interval["interval_high"]

    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(),
        table=_single_fold_table(),
        meta=pd.DataFrame(),
        protocols={"demo": _SingleFoldProtocol()},
        estimators=[InvalidIntervalEstimator],
    )

    assert results["unit_status"].eq("failed").all()
    assert results["error_category"].eq("ALGORITHM").all()
    assert results["error_message"].str.contains(bound).all()


def test_runner_rejects_inverted_native_intervals():
    class InvertedIntervalEstimator:
        name = "inverted_interval"
        feature_mode = "unified"
        backend = "external"

        def fit(self, X, y, meta=None):
            return self

        def predict(self, X, meta=None):
            return np.full(len(X), 0.9)

        def predict_interval(self, X, meta=None):
            return np.array([0.8, 1.1]), np.array([1.0, 1.0])

    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(),
        table=_single_fold_table(),
        meta=pd.DataFrame(),
        protocols={"demo": _SingleFoldProtocol()},
        estimators=[InvertedIntervalEstimator],
    )

    assert results["unit_status"].eq("failed").all()
    assert results["error_category"].eq("ALGORITHM").all()
    assert results["error_message"].str.contains(
        "interval_low must not exceed interval_high"
    ).all()

def test_runner_produces_results_with_ci_and_seeds():
    from sohbench.metadata import build_meta
    t = _table()
    t["soh"] = 1.0 - 0.02 * t["cycle_index"]
    meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0,1,2,3,4])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"loco": LeaveOneCellOut()},
                    estimators=[RidgeEstimator])
    assert set(rs.columns) >= {
        "estimator", "protocol", "held_out_key", "fold", "seed", "metric", "value",
        "unit_status", "error_category", "error_type", "error_message",
    }
    assert (rs["metric"] == "picp").any()                 # CI guaranteed (conformal)
    assert rs["seed"].nunique() == 5
    assert rs["unit_status"].eq("succeeded").all()
    assert rs[["error_category", "error_type", "error_message"]].eq("").all().all()

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
    assert rs["unit_status"].eq("failed").all()
    assert rs["error_category"].eq("ALGORITHM").all()
    assert rs["error_type"].eq("RuntimeError").all()
    assert rs["error_message"].eq("boom").all()


@pytest.mark.parametrize(
    ("stage", "exception_type", "message"),
    [
        ("fit", TypeError, "fit rejected dtype"),
        ("predict", ValueError, "predict rejected values"),
        ("predict_interval", TypeError, "interval rejected shape"),
    ],
)
def test_estimator_type_and_value_errors_are_algorithm_failures(stage, exception_type, message):
    from sohbench.metadata import build_meta

    class StageFailure:
        name = "stage_failure"
        feature_mode = "unified"
        backend = "sklearn"

        def fit(self, X, y, meta):
            if stage == "fit":
                raise exception_type(message)
            return self

        def predict(self, X, meta):
            if stage == "predict":
                raise exception_type(message)
            return np.full(len(X), 0.9)

        def predict_interval(self, X, meta):
            if stage == "predict_interval":
                raise exception_type(message)
            center = self.predict(X, meta)
            return center, center

    table = _table()
    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), table=table, meta=build_meta(table),
        protocols={"loco": LeaveOneCellOut()}, estimators=[StageFailure],
    )

    assert results["unit_status"].eq("failed").all()
    assert results["error_category"].eq("ALGORITHM").all()
    assert results["error_type"].eq(exception_type.__name__).all()
    assert results["error_message"].eq(message).all()


def test_metric_value_error_is_a_metric_failure(monkeypatch):
    from sohbench.metadata import build_meta

    def fail_mae(y_true, y_pred):
        raise ValueError("mae calculation failed")

    monkeypatch.setattr(M, "mae", fail_mae)
    table = _table()
    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), table=table, meta=build_meta(table),
        protocols={"loco": LeaveOneCellOut()}, estimators=[RidgeEstimator],
    )

    assert results["unit_status"].eq("failed").all()
    assert results["error_category"].eq("METRIC").all()
    assert results["error_type"].eq("ValueError").all()
    assert results["error_message"].eq("mae calculation failed").all()


def test_non_finite_metric_output_is_a_metric_failure():
    class FiniteExtremeEstimator:
        name = "finite_extreme"
        feature_mode = "unified"
        backend = "external"

        def fit(self, X, y, meta=None):
            return self

        def predict(self, X, meta=None):
            return np.full(len(X), np.finfo(float).max)

        def predict_interval(self, X, meta=None):
            center = self.predict(X, meta)
            return center, center

    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(),
        table=_single_fold_table(),
        meta=pd.DataFrame(),
        protocols={"demo": _SingleFoldProtocol()},
        estimators=[FiniteExtremeEstimator],
    )

    assert results["unit_status"].eq("failed").all()
    assert results["error_category"].eq("METRIC").all()
    assert results["error_message"].str.contains("metrics must be finite").all()


@pytest.mark.parametrize(
    "invalid_metric",
    [None, np.array([0.1]), 0.1 + 0j],
    ids=["none", "one_element_array", "complex"],
)
def test_non_real_scalar_metric_output_is_a_metric_failure(
    monkeypatch, invalid_metric
):
    from sohbench.metadata import build_meta

    monkeypatch.setattr(M, "mae", lambda _y_true, _y_pred: invalid_metric)
    table = _table()

    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(point_metrics=("mae",)),
        table=table,
        meta=build_meta(table),
        protocols={"loco": LeaveOneCellOut()},
        estimators=[RidgeEstimator],
    )

    assert results["unit_status"].eq("failed").all()
    assert results["error_category"].eq("METRIC").all()
    assert results["error_message"].str.contains(
        "finite non-boolean real scalar"
    ).all()


def test_feature_contract_value_error_is_an_input_contract_failure():
    from sohbench.metadata import build_meta

    class MalformedFeaturePipeline:
        def clone_for_fold(self):
            return MalformedFeaturePipeline()

        def fit(self, train_x, train_y, meta):
            from sohbench.fabric import FeatureState

            return FeatureState.create({"kind": "malformed"})

        def transform(self, frame, state, meta):
            return frame.to_numpy()

    table = _table()
    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), table=table, meta=build_meta(table),
        protocols={"loco": LeaveOneCellOut()}, estimators=[RidgeEstimator],
        featuresets=["custom"], feature_pipelines={"custom": MalformedFeaturePipeline()},
    )

    assert results["unit_status"].eq("failed").all()
    assert results["error_category"].eq("INPUT_CONTRACT").all()
    assert results["error_type"].eq("ValueError").all()
    assert results["error_message"].eq("feature pipeline did not return a DataFrame").all()


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
            "cell_id": ["c1", "c1", "c1", "c2", "c2"],
            "cycle_index": [0, 1, 2, 3, 4],
            "soh": [1.0, 0.9, 0.8, 0.79, 0.78],
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
    assert np.isclose(seen[0]["estimator"].mean_, 0.9)
    assert list(seen[0]["train_table"]["cycle_index"]) == [0, 1, 2]
