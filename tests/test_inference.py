import json
import pickle

import numpy as np
import pandas as pd

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
