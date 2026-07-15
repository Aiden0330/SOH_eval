import json

import numpy as np
import pandas as pd
import pytest

from sohbench.fabric.statistics import (
    asset_metric_table,
    bootstrap_mean_ci,
    paired_delta,
)


def test_asset_metric_table_uses_one_unweighted_value_per_sorted_asset():
    predictions = pd.DataFrame(
        {
            "dataset": ["z", "a", "a", "a", "a"],
            "cell_id": ["c", "b", "a", "a", "a"],
            "y_true": [1.0, 1.0, 1.0, 1.0, 1.0],
            "y_pred": [0.5, 1.0, 0.0, 0.0, 0.0],
        }
    )

    assets = asset_metric_table(predictions, "mae")

    assert assets.to_dict("records") == [
        {"dataset": "a", "cell_id": "a", "value": 1.0},
        {"dataset": "a", "cell_id": "b", "value": 0.0},
        {"dataset": "z", "cell_id": "c", "value": 0.5},
    ]


def test_asset_metric_table_supports_rmse():
    predictions = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "cell_id": ["a", "a"],
            "y_true": [0.0, 0.0],
            "y_pred": [3.0, 4.0],
        }
    )

    assets = asset_metric_table(predictions, "rmse")

    assert assets["value"].tolist() == [pytest.approx(np.sqrt(12.5))]


@pytest.mark.parametrize("metric", ["mape", "MAE", "", None])
def test_asset_metric_table_rejects_unsupported_metrics(metric):
    predictions = pd.DataFrame(
        {"dataset": ["d"], "cell_id": ["a"], "y_true": [1.0], "y_pred": [1.0]}
    )

    with pytest.raises(ValueError, match="mae|rmse|metric"):
        asset_metric_table(predictions, metric)


@pytest.mark.parametrize("missing", ["dataset", "cell_id", "y_true", "y_pred"])
def test_asset_metric_table_requires_core_columns(missing):
    predictions = pd.DataFrame(
        {"dataset": ["d"], "cell_id": ["a"], "y_true": [1.0], "y_pred": [1.0]}
    ).drop(columns=[missing])

    with pytest.raises(ValueError, match=missing):
        asset_metric_table(predictions)


def test_asset_metric_table_rejects_empty_predictions_and_null_asset_keys():
    empty = pd.DataFrame(columns=["dataset", "cell_id", "y_true", "y_pred"])
    with pytest.raises(ValueError, match="empty|non-empty"):
        asset_metric_table(empty)

    null_key = pd.DataFrame(
        {"dataset": ["d"], "cell_id": [None], "y_true": [1.0], "y_pred": [1.0]}
    )
    with pytest.raises(ValueError, match="cell_id|asset"):
        asset_metric_table(null_key)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("y_true", np.nan),
        ("y_true", np.inf),
        ("y_pred", -np.inf),
        ("y_pred", "0.9"),
        ("y_pred", True),
    ],
)
def test_asset_metric_table_rejects_nonfinite_or_nonnumeric_values(column, value):
    data = {"dataset": ["d"], "cell_id": ["a"], "y_true": [1.0], "y_pred": [1.0]}
    data[column] = [value]
    predictions = pd.DataFrame(data)

    with pytest.raises(ValueError, match=column):
        asset_metric_table(predictions)


def test_asset_metric_table_rejects_duplicate_sample_identity_when_present():
    row = {
        "run_id": "RUN-one",
        "candidate_id": "ridge",
        "featureset": "full",
        "protocol": "loco",
        "fold": 0,
        "seed": 0,
        "dataset": "d",
        "cell_id": "a",
        "cycle_index": 1,
        "sample_id": "d:a:1",
        "y_true": 1.0,
        "y_pred": 0.9,
    }

    with pytest.raises(ValueError, match="duplicate.*sample|sample.*duplicate"):
        asset_metric_table(pd.DataFrame([row, row]))


def test_asset_metric_table_allows_same_sample_in_distinct_execution_units():
    rows = []
    for seed, y_pred in [(0, 0.9), (1, 0.8)]:
        rows.append(
            {
                "run_id": "RUN-one",
                "candidate_id": "ridge",
                "featureset": "full",
                "protocol": "loco",
                "fold": 0,
                "seed": seed,
                "dataset": "d",
                "cell_id": "a",
                "cycle_index": 1,
                "sample_id": "d:a:1",
                "y_true": 1.0,
                "y_pred": y_pred,
            }
        )

    assets = asset_metric_table(pd.DataFrame(rows))

    assert assets["value"].tolist() == [pytest.approx(0.15)]


def test_asset_metric_table_rejects_nonfinite_computed_results():
    predictions = pd.DataFrame(
        {
            "dataset": ["d"],
            "cell_id": ["a"],
            "y_true": [-1.0e308],
            "y_pred": [1.0e308],
        }
    )

    with pytest.raises(ValueError, match="finite|overflow|result"):
        asset_metric_table(predictions, "rmse")


def test_bootstrap_mean_ci_is_reproducible_unweighted_and_json_safe():
    values = np.array([0.0, 0.1, 0.2, 0.3])

    first = bootstrap_mean_ci(values, n_resamples=500, seed=7)
    second = bootstrap_mean_ci(values, n_resamples=500, seed=7)

    assert first == second
    assert first["mean"] == pytest.approx(0.15)
    assert first["low"] <= first["mean"] <= first["high"]
    assert first["confidence"] == 0.95
    assert first["n_assets"] == 4
    assert all(type(first[key]) is float for key in ("mean", "low", "high", "confidence"))
    assert type(first["n_assets"]) is int
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize(
    "values",
    [
        [],
        np.array(1.0),
        np.array([[1.0, 2.0]]),
        [1.0, np.nan],
        [1.0, np.inf],
        [1.0, "2.0"],
        [True, False],
    ],
)
def test_bootstrap_mean_ci_rejects_invalid_values(values):
    with pytest.raises((TypeError, ValueError), match="values|one-dimensional|finite|numeric|non-empty"):
        bootstrap_mean_ci(values)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1, np.nan, True, "0.95"])
def test_bootstrap_mean_ci_validates_confidence(confidence):
    with pytest.raises((TypeError, ValueError), match="confidence"):
        bootstrap_mean_ci([1.0], confidence=confidence)


@pytest.mark.parametrize("n_resamples", [0, -1, 1.5, True, "10"])
def test_bootstrap_mean_ci_requires_positive_integer_resamples(n_resamples):
    with pytest.raises((TypeError, ValueError), match="n_resamples|resamples"):
        bootstrap_mean_ci([1.0], n_resamples=n_resamples)


@pytest.mark.parametrize("seed", [-1, 1.5, True, "7", None])
def test_bootstrap_mean_ci_requires_a_nonnegative_integer_seed(seed):
    with pytest.raises((TypeError, ValueError), match="seed"):
        bootstrap_mean_ci([1.0], seed=seed)


def test_paired_delta_is_left_minus_right_inner_joined_and_row_order_independent():
    left = pd.DataFrame(
        {
            "dataset": ["left-only", "b", "a"],
            "cell_id": ["x", "2", "1"],
            "value": [99.0, 0.7, 0.5],
        }
    )
    right = pd.DataFrame(
        {
            "dataset": ["a", "right-only", "b"],
            "cell_id": ["1", "x", "2"],
            "value": [0.2, -99.0, 0.4],
        }
    )

    first = paired_delta(left, right, ["dataset", "cell_id"], "value")
    second = paired_delta(
        left.sample(frac=1.0, random_state=3),
        right.sample(frac=1.0, random_state=4),
        ["dataset", "cell_id"],
        "value",
    )

    assert first == second
    assert first["direction"] == "left-minus-right"
    assert first["mean"] == pytest.approx(0.3)
    assert first["n_pairs"] == 2
    assert first["low"] <= first["mean"] <= first["high"]
    assert "n_assets" not in first
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("side", ["left", "right"])
def test_paired_delta_rejects_duplicate_pairing_keys(side):
    unique = pd.DataFrame({"asset": ["a", "b"], "value": [1.0, 2.0]})
    duplicate = pd.DataFrame({"asset": ["a", "a"], "value": [1.0, 2.0]})
    left, right = (duplicate, unique) if side == "left" else (unique, duplicate)

    with pytest.raises(ValueError, match=f"{side}.*unique|duplicate"):
        paired_delta(left, right, ["asset"], "value")


@pytest.mark.parametrize(
    ("left", "right", "key_columns", "value_column", "message"),
    [
        (
            pd.DataFrame({"asset": ["a"], "value": [1.0]}),
            pd.DataFrame({"asset": ["b"], "value": [1.0]}),
            ["asset"],
            "value",
            "pair|overlap",
        ),
        (
            pd.DataFrame({"asset": ["a"], "value": [np.nan]}),
            pd.DataFrame({"asset": ["a"], "value": [1.0]}),
            ["asset"],
            "value",
            "finite|value",
        ),
        (
            pd.DataFrame({"asset": ["a"], "value": [1.0]}),
            pd.DataFrame({"asset": ["a"], "value": ["1.0"]}),
            ["asset"],
            "value",
            "numeric|value",
        ),
        (
            pd.DataFrame({"asset": ["a"], "value": [1.0]}),
            pd.DataFrame({"asset": ["a"], "value": [1.0]}),
            [],
            "value",
            "key_columns|pairing",
        ),
        (
            pd.DataFrame({"asset": ["a"], "value": [1.0]}),
            pd.DataFrame({"asset": ["a"], "value": [1.0]}),
            ["missing"],
            "value",
            "missing",
        ),
    ],
)
def test_paired_delta_rejects_invalid_pairing_inputs(
    left, right, key_columns, value_column, message,
):
    with pytest.raises((TypeError, ValueError), match=message):
        paired_delta(left, right, key_columns, value_column)
