import numpy as np
import pandas as pd
import pytest

from sohbench.fabric.failures import MetricEvaluationError
from sohbench.runner import BenchmarkRunner
from sohbench.tasks import SOHEstimationTask


def test_trajectory_metrics_do_not_join_independent_cells():
    runner = BenchmarkRunner(seeds=[0])
    task = SOHEstimationTask()
    test_x = pd.DataFrame(
        {
            "dataset": ["d"] * 4,
            "cell_id": ["a", "a", "b", "b"],
            "cycle_index": [0, 1, 0, 1],
        }
    )
    predictions = np.array([1.0, 0.9, 0.8, 0.7])

    values = runner._trajectory_metrics(task, test_x, predictions)

    shuffled = test_x.iloc[[3, 0, 2, 1]].reset_index(drop=True)
    shuffled_predictions = predictions[[3, 0, 2, 1]]
    shuffled_values = runner._trajectory_metrics(task, shuffled, shuffled_predictions)

    assert values["monotonicity_violation_rate"] == 0.0
    assert shuffled_values == values


def test_trajectory_metrics_average_each_eligible_asset_equally():
    test_x = pd.DataFrame(
        {
            "dataset": ["d"] * 6,
            "cell_id": ["short", "short", "long", "long", "long", "long"],
            "cycle_index": [0, 1, 0, 1, 2, 3],
        }
    )
    predictions = np.array([1.4, 1.5, 1.0, 0.9, 0.8, 0.7])

    values = BenchmarkRunner(seeds=[0])._trajectory_metrics(
        SOHEstimationTask(), test_x, predictions
    )

    assert values["monotonicity_violation_rate"] == 0.5
    assert values["value_range_violation"] == 0.5


def test_trajectory_metrics_rejects_when_no_asset_has_two_cycles():
    test_x = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "cell_id": ["a", "b"],
            "cycle_index": [0, 0],
        }
    )

    with pytest.raises(MetricEvaluationError, match="at least two cycles"):
        BenchmarkRunner(seeds=[0])._trajectory_metrics(
            SOHEstimationTask(), test_x, np.array([1.0, 0.9])
        )


def test_trajectory_metrics_rejects_duplicate_asset_cycle_identities_regardless_of_input_order():
    runner = BenchmarkRunner(seeds=[0])
    task = SOHEstimationTask()
    test_x = pd.DataFrame(
        {
            "dataset": ["d", "d", "d"],
            "cell_id": ["a", "a", "a"],
            "cycle_index": [0, 1, 0],
        }
    )
    predictions = np.array([1.0, 0.9, 0.95])
    shuffled = test_x.iloc[[2, 0, 1]].reset_index(drop=True)
    shuffled_predictions = predictions[[2, 0, 1]]

    for frame, values in ((test_x, predictions), (shuffled, shuffled_predictions)):
        with pytest.raises(MetricEvaluationError, match="duplicate cycle identity"):
            runner._trajectory_metrics(task, frame, values)


def test_trajectory_metrics_distinguish_same_cell_id_across_datasets():
    test_x = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d2", "d2"],
            "cell_id": ["shared"] * 4,
            "cycle_index": [0, 1, 0, 1],
        }
    )
    predictions = np.array([1.0, 0.9, 0.8, 0.7])

    values = BenchmarkRunner(seeds=[0])._trajectory_metrics(
        SOHEstimationTask(), test_x, predictions
    )

    assert values["monotonicity_violation_rate"] == 0.0
