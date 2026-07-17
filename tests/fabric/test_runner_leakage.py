from collections import deque

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
MODULE_PIPELINE_DEQUE = deque()


class _MutablePipelineBox:
    def __init__(self):
        self.values = []

    def remember(self, labels):
        self.values.append(labels.copy())


MODULE_PIPELINE_BOX = _MutablePipelineBox()
MODULE_PIPELINE_BOUND_HELPER = MODULE_PIPELINE_BOX.remember


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


def test_runner_omits_unavailable_eol_metric_and_still_runs_callbacks(
    simple_table, simple_meta
):
    model_records = []
    prediction_records = []

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        on_model_fitted=lambda **value: model_records.append(value["record"]),
        on_predictions=lambda **value: prediction_records.append(value["record"]),
    )

    expected_metrics = {
        *SOHEstimationTask().point_metrics,
        *SOHEstimationTask().interval_metrics,
        *SOHEstimationTask().trajectory_metrics,
    } - {"eol_region_error"}
    assert result["unit_status"].eq("succeeded").all(), result[
        "error_message"
    ].unique().tolist()
    assert set(result["metric"]) == expected_metrics
    assert np.isfinite(result["value"]).all()
    assert len(model_records) == len(prediction_records) == 3
    assert all(record["unit_status"] == "succeeded" for record in model_records)
    assert all(record["unit_status"] == "succeeded" for record in prediction_records)


def test_runner_emits_finite_eol_metric_when_eol_samples_are_present(
    simple_table, simple_meta
):
    table = simple_table.copy()
    table.loc[table["cycle_index"] == 4, "soh"] = 0.8

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
    )

    eol_records = result.loc[result["metric"] == "eol_region_error"]
    assert len(eol_records) == 3
    assert eol_records["unit_status"].eq("succeeded").all()
    assert np.isfinite(eol_records["value"]).all()


def test_runner_rejects_non_finite_eol_metric_when_eol_samples_are_present(
    simple_table, simple_meta, monkeypatch
):
    table = simple_table.copy()
    table.loc[table["cycle_index"] == 4, "soh"] = 0.8
    monkeypatch.setattr(
        runner_module.M,
        "eol_region_error",
        lambda _y_true, _y_pred: float("nan"),
    )

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
    )

    assert result["unit_status"].eq("failed").all()
    assert result["error_category"].eq("METRIC").all()
    assert result["error_message"].str.contains("eol_region_error").all()


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


def test_runner_fits_feature_pipeline_only_on_fold_training_rows(simple_table, simple_meta, tmp_path):
    event_path = str(tmp_path / "fit-assets.txt")

    class MeanPipeline:
        def fit(self, train_x, train_y, meta):
            with open(event_path, "a", encoding="utf-8") as handle:
                handle.write(",".join(train_x["cell_id"].tolist()) + "\n")
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

    seen = [line.split(",") for line in (tmp_path / "fit-assets.txt").read_text(encoding="utf-8").splitlines()]
    assert len(seen) == 6
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


def test_model_snapshot_does_not_trust_estimator_deepcopy(simple_table, simple_meta):
    observed_predict_calls = []

    class AliasingDeepcopyEstimator(SpyEstimator):
        name = "aliasing_snapshot"

        def __deepcopy__(self, memo):
            return self

        def fit(self, X, y, meta=None):
            self.predict_calls = 0
            return self

        def predict(self, X, meta=None):
            self.predict_calls += 1
            return np.zeros(len(X))

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [AliasingDeepcopyEstimator],
        on_model_fitted=lambda **payload: observed_predict_calls.append(payload["estimator"].predict_calls),
    )

    assert result.loc[result["metric"] == "mae", "value"].notna().all()
    assert observed_predict_calls == [0, 0, 0]


def test_deferred_model_snapshots_are_disk_spooled_before_callbacks(
    simple_table, simple_meta, monkeypatch
):
    events = []
    base_spool = runner_module._DeferredCallbackSpool

    class TrackingSpool(base_spool):
        def dump(self, kind, value):
            events.append(("dump", kind))
            return super().dump(kind, value)

        def load(self, path):
            events.append(("load", path.name.split("-", 1)[0]))
            return super().load(path)

    monkeypatch.setattr(runner_module, "_DeferredCallbackSpool", TrackingSpool)

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        on_model_fitted=lambda **payload: None,
    )

    dump_positions = [index for index, event in enumerate(events) if event == ("dump", "model")]
    prepared_dump_positions = [index for index, event in enumerate(events) if event == ("dump", "prepared")]
    load_positions = [index for index, event in enumerate(events) if event == ("load", "model")]
    assert result.loc[result["metric"] == "mae", "value"].notna().all()
    assert len(dump_positions) == len(load_positions) == 3
    assert len(prepared_dump_positions) == 3
    assert max(dump_positions + prepared_dump_positions) < min(load_positions)


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


def test_runner_fits_pipeline_once_per_seed_candidate(simple_table, simple_meta, tmp_path):
    event_path = str(tmp_path / "fit-events.txt")

    class CountingPipeline:
        def fit(self, train_x, train_y, meta):
            with open(event_path, "a", encoding="utf-8") as handle:
                handle.write("fit\n")
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    result = BenchmarkRunner(seeds=[0, 1]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator, SpyEstimator],
        feature_pipelines={"full": CountingPipeline()},
    )

    assert "error" not in result.columns
    assert len((tmp_path / "fit-events.txt").read_text(encoding="utf-8").splitlines()) == 12


def test_pipeline_fit_state_is_not_shared_between_unified_and_byo_candidates(simple_table, simple_meta, tmp_path):
    event_path = str(tmp_path / "mixed-fit-events.txt")

    class CountingPipeline:
        def fit(self, train_x, train_y, meta):
            with open(event_path, "a", encoding="utf-8") as handle:
                handle.write("fit\n")
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
    assert len((tmp_path / "mixed-fit-events.txt").read_text(encoding="utf-8").splitlines()) == 6


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


def test_pipeline_with_deque_module_global_is_rejected(simple_table, simple_meta):
    class GlobalDequePipeline:
        def fit(self, train_x, train_y, meta):
            MODULE_PIPELINE_DEQUE.append(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    MODULE_PIPELINE_DEQUE.clear()
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": GlobalDequePipeline()},
    )

    assert result["error"].str.contains("mutable module global").all()


def test_pipeline_with_custom_mutable_module_global_is_rejected(simple_table, simple_meta):
    class GlobalObjectPipeline:
        def fit(self, train_x, train_y, meta):
            MODULE_PIPELINE_BOX.values.append(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    MODULE_PIPELINE_BOX.values.clear()
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": GlobalObjectPipeline()},
    )

    assert result["error"].str.contains("mutable module global").all()


def test_pipeline_with_mutable_closure_is_rejected(simple_table, simple_meta):
    closure_cache = deque()

    class ClosureCachePipeline:
        def fit(self, train_x, train_y, meta):
            closure_cache.append(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": ClosureCachePipeline()},
    )

    assert result["error"].str.contains("mutable closure state").all()


def test_pipeline_with_mutable_method_default_is_rejected(simple_table, simple_meta):
    class DefaultCachePipeline:
        def fit(self, train_x, train_y, meta, cache=[]):
            cache.append(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": DefaultCachePipeline()},
    )

    assert result["error"].str.contains("mutable callable default state").all()


def test_pipeline_with_mutable_method_keyword_default_is_rejected(simple_table, simple_meta):
    class KeywordDefaultCachePipeline:
        def fit(self, train_x, train_y, meta, *, cache=None):
            return FeatureState.create({})

        def transform(self, frame, state, meta, *, cache={}):
            cache["last"] = frame.index.tolist()
            return frame.copy()

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": KeywordDefaultCachePipeline()},
    )

    assert result["error"].str.contains("mutable callable default state").all()


def test_pipeline_with_mutable_function_attribute_is_rejected(simple_table, simple_meta):
    class FunctionAttributePipeline:
        def fit(self, train_x, train_y, meta):
            type(self).fit.cache.append(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    FunctionAttributePipeline.fit.cache = []
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": FunctionAttributePipeline()},
    )

    assert result["error"].str.contains("mutable callable attribute state").all()


def test_pipeline_with_instance_attached_fit_callable_is_rejected(simple_table, simple_meta):
    closure_cache = deque()

    class InstanceCallablePipeline:
        def transform(self, frame, state, meta):
            return frame.copy()

    def fit(train_x, train_y, meta):
        closure_cache.append(train_y.copy())
        return FeatureState.create({})

    pipeline = InstanceCallablePipeline()
    pipeline.fit = fit
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": pipeline},
    )

    assert result["error"].str.contains("class-defined fit.*transform methods").all()


def test_pipeline_with_instance_attached_clone_callable_is_rejected(simple_table, simple_meta):
    retained_clones = []

    class InstanceClonePipeline:
        def fit(self, train_x, train_y, meta):
            self.labels = train_y.copy()
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    pipeline = InstanceClonePipeline()

    def clone_for_fold():
        clone = InstanceClonePipeline()
        retained_clones.append(clone)
        return clone

    pipeline.clone_for_fold = clone_for_fold
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": pipeline},
    )

    assert result["error"].str.contains("class-defined clone_for_fold method").all()


def test_pipeline_with_instance_helper_closure_is_rejected(simple_table, simple_meta):
    retained_labels = deque()

    def remember(labels):
        retained_labels.append(labels.copy())

    class InstanceHelperPipeline:
        def __init__(self):
            self.remember = remember

        def fit(self, train_x, train_y, meta):
            self.remember(train_y)
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": InstanceHelperPipeline()},
    )

    assert result["error"].str.contains("mutable closure state").all()


def test_pipeline_with_instance_bound_builtin_helper_is_rejected(simple_table, simple_meta):
    retained_labels = deque()

    class BoundBuiltinHelperPipeline:
        def fit(self, train_x, train_y, meta):
            self.remember(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    pipeline = BoundBuiltinHelperPipeline()
    pipeline.remember = retained_labels.append
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": pipeline},
    )

    assert result["error"].str.contains("mutable instance callable state").all()


def test_pipeline_with_module_global_bound_method_is_rejected(simple_table, simple_meta):
    class GlobalBoundMethodPipeline:
        def fit(self, train_x, train_y, meta):
            MODULE_PIPELINE_BOUND_HELPER(train_y)
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    MODULE_PIPELINE_BOX.values.clear()
    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": GlobalBoundMethodPipeline()},
    )

    assert result["error"].str.contains("mutable module global").all()


def test_pipeline_with_bound_method_default_is_rejected(simple_table, simple_meta):
    retained_labels = deque()

    class BoundDefaultPipeline:
        def fit(self, train_x, train_y, meta, remember=retained_labels.append):
            remember(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": BoundDefaultPipeline()},
    )

    assert result["error"].str.contains("mutable callable default state").all()


def test_pipeline_with_bound_method_closure_is_rejected(simple_table, simple_meta):
    box = _MutablePipelineBox()
    remember = box.remember

    class BoundClosurePipeline:
        def fit(self, train_x, train_y, meta):
            remember(train_y)
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": BoundClosurePipeline()},
    )

    assert result["error"].str.contains("mutable closure state").all()


def test_pipeline_with_stateful_primitive_bound_owner_is_rejected(simple_table, simple_meta):
    class StatefulInt(int):
        def __new__(cls, value):
            instance = super().__new__(cls, value)
            instance.values = []
            return instance

        def remember(self, labels):
            self.values.append(labels.copy())

    owner = StatefulInt(1)

    class StatefulPrimitivePipeline:
        def fit(self, train_x, train_y, meta, remember=owner.remember):
            remember(train_y)
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": StatefulPrimitivePipeline()},
    )

    assert result["error"].str.contains("mutable callable default state").all()


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


def test_pipeline_with_direct_original_class_deque_is_rejected(simple_table, simple_meta):
    class OriginalClassDequePipeline:
        cache = deque()

        def fit(self, train_x, train_y, meta):
            OriginalClassDequePipeline.cache.append(train_y.copy())
            return FeatureState.create({})

        def transform(self, frame, state, meta):
            return frame.copy()

    result = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        feature_pipelines={"full": OriginalClassDequePipeline()},
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
