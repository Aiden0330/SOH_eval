import numpy as np
import pandas as pd
import pytest

from sohbench.runner import BenchmarkRunner
from sohbench.protocols import LeaveOneCellOut
from sohbench.tasks import SOHEstimationTask


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
