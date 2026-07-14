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
