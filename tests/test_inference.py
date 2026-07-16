import json
import pickle

import numpy as np
import pandas as pd
import pytest

from sohbench import metrics as M
from sohbench.inference import run_inference_benchmark
from sohbench.model_store import ModelStore


class PredictOnlyModel:
    def fit(self, X, y=None, meta=None):
        raise AssertionError("fit must not be called")

    def predict(self, X, meta=None):
        return np.asarray(X["voltage_mean"], dtype=float) * 0.0 + 0.95


class ConstantModel:
    def fit(self, X, y=None, meta=None):
        raise AssertionError("fit must not be called")

    def predict(self, X, meta=None):
        return np.full(len(X), 0.95)


class ColumnModel:
    def fit(self, X, y=None, meta=None):
        raise AssertionError("fit must not be called")

    def predict(self, X, meta=None):
        return np.asarray(X["voltage_mean"], dtype=float)


class FixedOutputLoadedModel:
    load_ms = 0.0
    manifest = {"featureset": "full", "estimator": "fixed_output"}

    def __init__(self, prediction, interval=None):
        self.prediction = prediction
        self.interval = interval

    def predict(self, table, meta=None):
        return self.prediction

    def predict_interval(self, table, meta=None):
        return self.interval


class FixedOutputStore:
    def __init__(self, loaded):
        self.loaded = loaded

    def load_model(self, name):
        return self.loaded


def _manifest():
    return {
        "model_name": "predict_only",
        "model_kind": "pickle",
        "estimator": "predict_only",
        "featureset": "full",
        "feature_columns": ["voltage_mean"],
        "target": "soh",
        "trained_on": ["train"],
        "created_by_run": "unit",
        "predict_contract": "predict(X, meta=None)",
    }


def _table():
    return pd.DataFrame(
        {
            "dataset": ["new_a", "new_a", "new_b", "new_b"],
            "cell_id": ["c1", "c1", "c2", "c2"],
            "cycle_index": [0, 1, 0, 1],
            "soh": [0.96, 0.94, 0.90, 0.88],
            "voltage_mean": [3.5, 3.4, 3.3, 3.2],
        }
    )


def test_inference_benchmark_predicts_by_dataset_without_fit(tmp_path):
    store = ModelStore(tmp_path / "models")
    store.upload_model(
        "predict_only",
        "predict_only.pkl",
        pickle.dumps(PredictOnlyModel()),
        "model_manifest.json",
        json.dumps(_manifest()).encode("utf-8"),
    )

    records = run_inference_benchmark(store, _table(), pd.DataFrame(), ["predict_only"])

    assert {row["dataset"] for row in records} == {"new_a", "new_b"}
    assert all(row["status"] == "succeeded" for row in records)
    assert all(row["predict_ms"] >= 0 for row in records)
    assert all(row["ms_per_1000_samples"] >= 0 for row in records)
    assert records[0]["mae"] is not None


def test_inference_benchmark_records_feature_mismatch(tmp_path):
    store = ModelStore(tmp_path / "models")
    manifest = _manifest()
    manifest["feature_columns"] = ["missing_signal"]
    store.save_training_model("bad_model", ConstantModel(), manifest, replace=True)

    records = run_inference_benchmark(store, _table(), pd.DataFrame(), ["bad_model"])

    assert records[0]["status"] == "failed"
    assert "missing feature columns" in records[0]["error"]


def test_inference_benchmark_scores_only_fully_labeled_dataset_groups(tmp_path):
    store = ModelStore(tmp_path / "models")
    store.upload_model(
        "predict_only",
        "predict_only.pkl",
        pickle.dumps(PredictOnlyModel()),
        "model_manifest.json",
        json.dumps(_manifest()).encode("utf-8"),
    )
    labeled = _table().loc[lambda frame: frame["dataset"].eq("new_a")]
    unlabeled = (
        _table()
        .loc[lambda frame: frame["dataset"].eq("new_b")]
        .drop(columns=["soh"])
    )
    mixed = pd.concat([labeled, unlabeled], ignore_index=True, sort=False)

    records = run_inference_benchmark(
        store, mixed, pd.DataFrame(), ["predict_only"],
    )
    by_dataset = {record["dataset"]: record for record in records}

    assert np.isfinite(by_dataset["new_a"]["mae"])
    assert by_dataset["new_b"]["mae"] is None
    assert by_dataset["new_b"]["rmse"] is None
    assert by_dataset["new_b"]["mean_bias_error"] is None
    assert by_dataset["new_b"]["status"] == "succeeded"


def test_inference_benchmark_treats_object_pd_na_labels_as_unscored(tmp_path):
    store = ModelStore(tmp_path / "models")
    store.upload_model(
        "predict_only",
        "predict_only.pkl",
        pickle.dumps(PredictOnlyModel()),
        "model_manifest.json",
        json.dumps(_manifest()).encode("utf-8"),
    )
    table = _table().loc[lambda frame: frame["dataset"].eq("new_a")].copy()
    table["soh"] = table["soh"].astype(object)
    table.loc[table.index[0], "soh"] = pd.NA

    records = run_inference_benchmark(
        store, table, pd.DataFrame(), ["predict_only"],
    )

    assert records[0]["status"] == "succeeded"
    assert records[0]["mae"] is None
    assert records[0]["rmse"] is None
    assert records[0]["mean_bias_error"] is None


def test_inference_trajectory_metrics_do_not_join_independent_cells(tmp_path):
    store = ModelStore(tmp_path / "models")
    store.save_training_model("trajectory", ColumnModel(), _manifest())
    table = pd.DataFrame(
        {
            "dataset": ["d"] * 4,
            "cell_id": ["a", "a", "b", "b"],
            "cycle_index": [0, 1, 0, 1],
            "soh": [1.0, 0.9, 0.8, 0.7],
            "voltage_mean": [1.0, 0.9, 0.8, 0.7],
        }
    )

    record = run_inference_benchmark(
        store, table, pd.DataFrame(), ["trajectory"]
    )[0]
    shuffled = table.iloc[[3, 0, 2, 1]].reset_index(drop=True)
    shuffled_record = run_inference_benchmark(
        store, shuffled, pd.DataFrame(), ["trajectory"]
    )[0]

    assert record["monotonicity_violation_rate"] == 0.0
    assert shuffled_record["monotonicity_violation_rate"] == 0.0


def test_inference_trajectory_metrics_average_eligible_assets_equally(tmp_path):
    store = ModelStore(tmp_path / "models")
    store.save_training_model("trajectory", ColumnModel(), _manifest())
    table = pd.DataFrame(
        {
            "dataset": ["d"] * 6,
            "cell_id": ["short", "short", "long", "long", "long", "long"],
            "cycle_index": [0, 1, 0, 1, 2, 3],
            "soh": [1.0, 0.9, 1.0, 0.9, 0.8, 0.7],
            "voltage_mean": [1.4, 1.5, 1.0, 0.9, 0.8, 0.7],
        }
    )

    record = run_inference_benchmark(
        store, table, pd.DataFrame(), ["trajectory"]
    )[0]

    assert record["monotonicity_violation_rate"] == 0.5
    assert record["value_range_violation"] == 0.5


def test_inference_trajectory_metrics_exclude_singleton_assets(tmp_path):
    store = ModelStore(tmp_path / "models")
    store.save_training_model("trajectory", ColumnModel(), _manifest())
    table = pd.DataFrame(
        {
            "dataset": ["d", "d", "d"],
            "cell_id": ["singleton", "eligible", "eligible"],
            "cycle_index": [0, 0, 1],
            "soh": [1.0, 1.0, 0.9],
            "voltage_mean": [1.4, 1.0, 0.9],
        }
    )

    record = run_inference_benchmark(
        store, table, pd.DataFrame(), ["trajectory"]
    )[0]

    assert record["status"] == "succeeded"
    assert record["monotonicity_violation_rate"] == 0.0
    assert record["value_range_violation"] == 0.0


def test_inference_fails_when_no_asset_has_two_cycles(tmp_path):
    store = ModelStore(tmp_path / "models")
    store.save_training_model("trajectory", ColumnModel(), _manifest())
    table = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "cell_id": ["a", "b"],
            "cycle_index": [0, 0],
            "soh": [1.0, 0.9],
            "voltage_mean": [1.0, 0.9],
        }
    )

    record = run_inference_benchmark(
        store, table, pd.DataFrame(), ["trajectory"]
    )[0]

    assert record["status"] == "failed"
    assert record["monotonicity_violation_rate"] is None
    assert "trajectory metrics require an asset with at least two cycles" in record[
        "error"
    ]


def test_inference_rejects_non_finite_trajectory_metrics(tmp_path, monkeypatch):
    store = ModelStore(tmp_path / "models")
    store.save_training_model("trajectory", ColumnModel(), _manifest())
    table = _table().loc[lambda frame: frame["dataset"].eq("new_a")]
    monkeypatch.setattr(M, "monotonicity_violation_rate", lambda values: np.nan)

    record = run_inference_benchmark(
        store, table, pd.DataFrame(), ["trajectory"]
    )[0]

    assert record["status"] == "failed"
    assert record["monotonicity_violation_rate"] is None
    assert "trajectory metrics must be finite" in record["error"]


def test_inference_rejects_duplicate_asset_cycle_identities(tmp_path):
    store = ModelStore(tmp_path / "models")
    store.save_training_model("trajectory", ColumnModel(), _manifest())
    table = pd.DataFrame(
        {
            "dataset": ["d", "d", "d"],
            "cell_id": ["a", "a", "a"],
            "cycle_index": [0, 1, 0],
            "soh": [1.0, 0.9, 0.95],
            "voltage_mean": [1.0, 0.9, 0.95],
        }
    )

    record = run_inference_benchmark(
        store, table, pd.DataFrame(), ["trajectory"]
    )[0]

    assert record["status"] == "failed"
    assert "duplicate cycle identity" in record["error"]


@pytest.mark.parametrize(
    "prediction",
    [
        pytest.param(np.array([[0.9], [0.8]]), id="two-dimensional"),
        pytest.param(np.array([0.9]), id="wrong-length"),
        pytest.param(np.array([0.9, np.nan]), id="nan"),
        pytest.param(np.array([0.9, np.inf]), id="infinity"),
    ],
)
def test_inference_rejects_invalid_prediction_arrays(prediction):
    table = _table().loc[lambda frame: frame["dataset"].eq("new_a")]
    store = FixedOutputStore(FixedOutputLoadedModel(prediction))

    record = run_inference_benchmark(
        store, table, pd.DataFrame(), ["fixed_output"]
    )[0]

    assert record["status"] == "failed"
    assert record["mae"] is None
    assert "predictions" in record["error"]


@pytest.mark.parametrize("bound", ["interval_low", "interval_high"])
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(np.array([[0.7], [0.7]]), id="two-dimensional"),
        pytest.param(np.array([0.7]), id="wrong-length"),
        pytest.param(np.array([0.7, np.nan]), id="nan"),
        pytest.param(np.array([0.7, np.inf]), id="infinity"),
    ],
)
def test_inference_rejects_invalid_interval_arrays(bound, invalid):
    table = _table().loc[lambda frame: frame["dataset"].eq("new_a")]
    interval = {
        "interval_low": np.array([0.7, 0.7]),
        "interval_high": np.array([1.0, 1.0]),
    }
    interval[bound] = invalid
    store = FixedOutputStore(
        FixedOutputLoadedModel(
            np.array([0.9, 0.8]),
            (interval["interval_low"], interval["interval_high"]),
        )
    )

    record = run_inference_benchmark(
        store, table, pd.DataFrame(), ["fixed_output"]
    )[0]

    assert record["status"] == "failed"
    assert record["mae"] is None
    assert bound in record["error"]
