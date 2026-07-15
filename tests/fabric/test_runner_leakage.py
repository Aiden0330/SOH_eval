import numpy as np
import pandas as pd
import pytest

import sohbench.runner as runner_module
from sohbench.runner import BenchmarkRunner
from sohbench.protocols import LeaveOneCellOut
from sohbench.tasks import SOHEstimationTask
from sohbench.contracts import Fold
from sohbench.fabric.views import FoldViews
from sohbench.fabric import FeatureState


MODULE_PIPELINE_CACHE = []


@pytest.fixture
def simple_table():
    rows = []
    for cell in ["c1", "c2", "c3"]:
        for cycle in range(5):
            rows.append(("ds", cell, cycle, 1.0 - 0.01 * cycle, float(cycle)))
    return pd.DataFrame(rows, columns=["dataset", "cell_id", "cycle_index", "soh", "feature"])


@pytest.fixture
def simple_meta(simple_table):
    return pd.DataFrame()


class SpyEstimator:
    name = "spy"
    feature_mode = "unified"
    backend = "external"
    seen_random_states = []
    seen_fit_random_states = []

    def __init__(self, random_state=0):
        self.random_state = random_state
        self.seen_random_states.append(random_state)

    def fit(self, X, y, meta=None):
        assert "soh" not in X.columns
        self.seen_fit_random_states.append(self.random_state)
        self.mean = float(np.mean(y))
        return self

    def predict(self, X, meta=None):
        assert "soh" not in X.columns
        return np.full(len(X), self.mean)

    def predict_interval(self, X, meta=None):
        center = self.predict(X, meta)
        return center, center


class MutatingEstimator:
    name = "mutating"
    feature_mode = "unified"
    backend = "external"
    seen_train_values = []
    seen_test_values = []

    def fit(self, X, y, meta=None):
        assert "soh" not in X.columns
        assert "soh" not in meta.columns
        self.seen_train_values.append(float(X.iloc[0]["feature"]))
        X.iloc[0, X.columns.get_loc("feature")] = -999.0
        y[0] = -999.0
        return self

    def predict(self, X, meta=None):
        assert "soh" not in X.columns
        assert "soh" not in meta.columns
        self.seen_test_values.append(float(X.iloc[0]["feature"]))
        X.iloc[0, X.columns.get_loc("feature")] = -888.0
        return np.zeros(len(X))

    def predict_interval(self, X, meta=None):
        center = self.predict(X, meta)
        return center, center


class SetSeedOnlyEstimator:
    name = "set_seed_only"
    feature_mode = "unified"
    backend = "external"

    def __init__(self):
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed


def test_runner_strips_target_and_propagates_seed(simple_table, simple_meta):
    SpyEstimator.seen_random_states.clear()
    BenchmarkRunner(seeds=[3, 9]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
    )
    assert set(SpyEstimator.seen_random_states) == {3, 9}


def test_runner_deepcopies_instance_and_sets_seed_without_pollution(simple_table, simple_meta):
    SpyEstimator.seen_random_states.clear()
    SpyEstimator.seen_fit_random_states.clear()
    original = SpyEstimator(random_state=17)
    BenchmarkRunner(seeds=[3, 9]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [original],
    )
    assert original.random_state == 17
    assert set(SpyEstimator.seen_fit_random_states) == {3, 9}


def test_class_with_only_set_seed_receives_evaluation_seed():
    estimator = runner_module._instantiate_estimator(SetSeedOnlyEstimator, 23)
    assert estimator.seed == 23


def test_each_seed_gets_unpolluted_algorithm_views(simple_table, simple_meta):
    MutatingEstimator.seen_train_values.clear()
    MutatingEstimator.seen_test_values.clear()
    BenchmarkRunner(seeds=[0, 1]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [MutatingEstimator],
    )
    assert set(MutatingEstimator.seen_train_values) == {0.0}
    assert set(MutatingEstimator.seen_test_values[::2]) == {0.0}


def test_model_callback_meta_isolated_from_algorithm_and_caller(simple_table):
    class MetaEstimator(SpyEstimator):
        seen_meta_columns = []

        def fit(self, X, y, meta=None):
            assert "soh" not in meta.columns
            return super().fit(X, y, meta)

        def predict(self, X, meta=None):
            assert "soh" not in meta.columns
            self.seen_meta_columns.append(tuple(meta.columns))
            return super().predict(X, meta)

    caller_meta = pd.DataFrame({"soh": [1.0], "meta_feature": [2.0]})
    seen = []

    def on_model_fitted(**kwargs):
        seen.append(kwargs["meta"])
        kwargs["meta"]["soh"] = -1.0
        kwargs["meta"]["callback_only"] = 7.0

    BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, caller_meta,
        {"loco": LeaveOneCellOut()}, [MetaEstimator],
        on_model_fitted=on_model_fitted,
    )

    assert seen
    assert all("soh" not in columns for columns in MetaEstimator.seen_meta_columns)
    assert list(caller_meta.columns) == ["soh", "meta_feature"]
    assert caller_meta["soh"].tolist() == [1.0]
    assert caller_meta["meta_feature"].tolist() == [2.0]


def test_prediction_callback_has_stable_sample_ids_but_algorithm_never_sees_truth(simple_table, simple_meta):
    captured = []
    BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        on_predictions=lambda **value: captured.append(value),
    )
    assert captured
    value = captured[0]
    assert {"dataset", "cell_id", "cycle_index", "soh"} <= set(value["sample_frame"].columns)
    assert len(value["y_pred"]) == len(value["sample_frame"])
    assert {"record", "sample_frame", "y_pred", "interval_low", "interval_high"} == set(value)


def test_runner_builds_one_safe_view_per_fold_across_seeds(
    simple_table, simple_meta, monkeypatch
):
    calls = []
    original = runner_module.make_fold_views

    def spy_make_fold_views(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "make_fold_views", spy_make_fold_views)
    BenchmarkRunner(seeds=[0, 1, 2]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
    )

    assert len(calls) == 3


def test_runner_fits_feature_pipeline_only_on_fold_training_rows(simple_table, simple_meta):
    seen = []

    class MeanPipeline:
        def fit(self, train_x, train_y, meta):
            seen.append(train_x["cell_id"].tolist())
            return FeatureState.create({"mean": float(np.mean(train_y))})

        def transform(self, frame, state, meta):
            out = frame.copy()
            out["train_mean_feature"] = state.payload["mean"]
            return out

    BenchmarkRunner(seeds=[0, 1]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        featuresets=["full"], feature_pipelines={"full": MeanPipeline()},
    )

    assert len(seen) == 3
    assert all(len(set(cells)) == 2 for cells in seen)
    assert {frozenset(cells) for cells in seen} == {
        frozenset({"c1", "c2"}),
        frozenset({"c1", "c3"}),
        frozenset({"c2", "c3"}),
    }


def test_byo_estimator_receives_fold_scoped_transformed_unprojected_features(simple_table, simple_meta):
    class ByoEstimator(SpyEstimator):
        name = "byo_with_pipeline"
        feature_mode = "byo"
        seen_columns = []

        def fit(self, X, y, meta=None):
            self.seen_columns.append(tuple(X.columns))
            assert "derived" in X.columns
            assert "feature" in X.columns
            return super().fit(X, y, meta)

    class ByoPipeline:
        def fit(self, train_x, train_y, meta):
            return FeatureState.create({"kind": "byo"})

        def transform(self, frame, state, meta):
            out = frame.copy()
            out["derived"] = out["feature"] + 1
            return out

    ByoEstimator.seen_columns.clear()
    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [ByoEstimator],
        featuresets=["full"], feature_pipelines={"full": ByoPipeline()},
    )

    assert ByoEstimator.seen_columns
    assert all("derived" in columns and "feature" in columns for columns in ByoEstimator.seen_columns)
    assert "error" not in results.columns
    assert set(results["featureset"]) == {"(byo)"}


def test_callback_runs_after_metrics_and_receives_no_target_or_pipeline(simple_table, simple_meta):
    callbacks = []

    class CallbackEstimator(SpyEstimator):
        name = "callback_order"

        def fit(self, X, y, meta=None):
            self.value = 0.0
            return self

        def predict(self, X, meta=None):
            return np.full(len(X), self.value)

    def malicious_callback(**payload):
        assert "soh" not in payload["train_table"].columns
        assert "soh" not in payload["test_table"].columns
        assert "feature_pipeline" not in payload
        payload["estimator"].value = 0.97
        callbacks.append(payload)

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [CallbackEstimator],
        on_model_fitted=malicious_callback,
    )

    assert callbacks
    assert result.loc[result["metric"] == "mae", "value"].gt(0).all()


def test_callbacks_are_deferred_until_every_fold_prediction_is_complete(simple_table, simple_meta):
    callbacks = []

    class CallbackContaminatingEstimator(SpyEstimator):
        name = "callback_matrix_order"
        use_features = False

        def fit(self, X, y, meta=None):
            return self

        def predict(self, X, meta=None):
            if type(self).use_features:
                return 1.0 - 0.01 * X["feature"].to_numpy()
            return np.zeros(len(X))

    def callback(**payload):
        type(payload["estimator"]).use_features = True
        callbacks.append(payload)

    try:
        result = BenchmarkRunner(seeds=[0, 1]).run(
            SOHEstimationTask(), simple_table, simple_meta,
            {"loco": LeaveOneCellOut()}, [CallbackContaminatingEstimator],
            on_model_fitted=callback,
        )
    finally:
        CallbackContaminatingEstimator.use_features = False

    assert len(callbacks) == 6
    assert result.loc[result["metric"] == "mae", "value"].gt(0).all()


def test_model_callback_receives_pre_prediction_snapshot(simple_table, simple_meta):
    observed_predict_calls = []

    class PredictMutatingEstimator(SpyEstimator):
        name = "snapshot"

        def fit(self, X, y, meta=None):
            self.predict_calls = 0
            return self

        def predict(self, X, meta=None):
            self.predict_calls += 1
            return np.zeros(len(X))

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [PredictMutatingEstimator],
        on_model_fitted=lambda **payload: observed_predict_calls.append(payload["estimator"].predict_calls),
    )

    assert result.loc[result["metric"] == "mae", "value"].notna().all()
    assert observed_predict_calls == [0, 0, 0]


def test_callback_failures_are_diagnostic_and_do_not_erase_metrics(simple_table, simple_meta):
    def broken_model_callback(**payload):
        raise RuntimeError("model evidence unavailable")

    def broken_prediction_callback(**payload):
        raise RuntimeError("prediction evidence unavailable")

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        on_model_fitted=broken_model_callback,
        on_predictions=broken_prediction_callback,
    )

    assert result.loc[result["metric"] == "mae", "value"].notna().all()
    assert result["model_callback_error"].eq("model evidence unavailable").all()
    assert result["callback_error"].eq("prediction evidence unavailable").all()


def test_runner_scopes_meta_to_each_partition_and_strips_target(simple_table):
    meta = pd.DataFrame(
        {
            "dataset": ["ds", "ds", "ds", "outside"],
            "cell_id": ["c1", "c2", "c3", "z"],
            "soh": [1.0, 0.9, 0.8, 0.0],
            "meta_value": [1, 2, 3, 4],
        }
    )

    class MetaPipeline:
        def fit(self, train_x, train_y, seen_meta):
            assert "soh" not in seen_meta.columns
            assert set(seen_meta["cell_id"]) == set(train_x["cell_id"])
            return FeatureState.create({})

        def transform(self, frame, state, seen_meta):
            assert "soh" not in seen_meta.columns
            assert set(seen_meta["cell_id"]) == set(frame["cell_id"])
            return frame.copy()

    class MetaEstimator(SpyEstimator):
        name = "meta_scoped"

        def fit(self, X, y, seen_meta=None):
            assert set(seen_meta["cell_id"]) == set(X["cell_id"])
            return super().fit(X, y, seen_meta)

        def predict(self, X, seen_meta=None):
            assert set(seen_meta["cell_id"]) == set(X["cell_id"])
            return super().predict(X, seen_meta)

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, meta,
        {"loco": LeaveOneCellOut()}, [MetaEstimator],
        feature_pipelines={"full": MetaPipeline()},
    )

    assert "error" not in result.columns


def test_runner_fits_pipeline_once_per_fold_across_estimators(simple_table, simple_meta):
    fit_events = []

    class CountingPipeline:
        def fit(self, train_x, train_y, meta):
            fit_events.append(tuple(sorted(train_x["cell_id"].unique())))
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    result = BenchmarkRunner(seeds=[0, 1]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator, SpyEstimator],
        feature_pipelines={"full": CountingPipeline()},
    )

    assert "error" not in result.columns
    assert len(fit_events) == 3


def test_pipeline_fit_cache_is_shared_between_unified_and_byo_tracks(simple_table, simple_meta):
    fit_events = []

    class CountingPipeline:
        def fit(self, train_x, train_y, meta):
            fit_events.append(tuple(sorted(train_x["cell_id"].unique())))
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    class ByoEstimator(SpyEstimator):
        name = "cache_byo"
        feature_mode = "byo"

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator, ByoEstimator],
        feature_pipelines={"full": CountingPipeline()},
    )

    assert "error" not in result.columns
    assert len(fit_events) == 3


def test_pipeline_class_cache_is_isolated_per_fold(simple_table, simple_meta):
    class ClassCachePipeline:
        cache = []

        def fit(self, train_x, train_y, meta):
            if type(self).cache:
                raise AssertionError("class cache crossed fold boundary")
            type(self).cache.append(float(np.mean(train_y)))
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    ClassCachePipeline.cache = []
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": ClassCachePipeline()},
    )

    assert "error" not in result.columns


def test_pipeline_with_mutable_module_global_is_rejected(simple_table, simple_meta):
    class GlobalCachePipeline:
        def fit(self, train_x, train_y, meta):
            MODULE_PIPELINE_CACHE.append(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    MODULE_PIPELINE_CACHE.clear()
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": GlobalCachePipeline()},
    )

    assert result["error"].str.contains("mutable module global").all()


def test_pipeline_with_direct_original_class_cache_is_rejected(simple_table, simple_meta):
    class OriginalClassCachePipeline:
        cache = []

        def fit(self, train_x, train_y, meta):
            OriginalClassCachePipeline.cache.append(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    OriginalClassCachePipeline.cache = []
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": OriginalClassCachePipeline()},
    )

    assert result["error"].str.contains("mutable original class state").all()


def test_pipeline_transform_cannot_reuse_instance_fields_set_during_fit(simple_table, simple_meta):
    class FieldLeakingPipeline:
        def fit(self, train_x, train_y, meta):
            self.labels = train_y.copy()
            return FeatureState.create({"kind": "field_leak"})

        def transform(self, frame, state, meta):
            assert not hasattr(self, "labels")
            return frame.copy()

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": FieldLeakingPipeline()},
    )

    assert "error" not in result.columns


def test_invalid_fold_isolated_for_each_seed_without_control_frame_slice(
    simple_table, simple_meta
):
    class InvalidProtocol:
        def generate(self, table, meta):
            yield Fold(np.array([0]), np.array([len(table) + 1]), "invalid")

    result = BenchmarkRunner(seeds=[0, 1]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"invalid": InvalidProtocol()}, [SpyEstimator],
    )

    assert len(result) > 0
    assert result["value"].isna().all()
    assert result["seed"].nunique() == 2


def test_without_callbacks_runner_does_not_slice_control_frames(simple_table, simple_meta, monkeypatch):
    class NoIlocDataFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return NoIlocDataFrame

        @property
        def iloc(self):
            raise AssertionError("control frame slicing should be callback-driven")

    def fake_views(data_table, fold, target):
        plain_table = pd.DataFrame(data_table)
        return FoldViews(
            train_x=plain_table.drop(columns=[target]).iloc[:10].copy(),
            train_y=np.ones(10),
            test_x=plain_table.drop(columns=[target]).iloc[:5].copy(),
            test_y=np.ones(5),
        )

    monkeypatch.setattr(runner_module, "make_fold_views", fake_views)
    table = NoIlocDataFrame(simple_table)
    BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
    )
