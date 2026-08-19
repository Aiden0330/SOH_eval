import json
import pickle

import numpy as np
import pandas as pd
import pytest

from sohbench.inference import run_inference_benchmark, run_inference_benchmark_with_evidence
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


def test_inference_benchmark_retains_per_cycle_prediction_evidence(tmp_path):
    store = ModelStore(tmp_path / "models")
    stored = store.upload_model(
        "predict_only",
        "predict_only.pkl",
        pickle.dumps(PredictOnlyModel()),
        "model_manifest.json",
        json.dumps(_manifest()).encode("utf-8"),
    )

    records, evidence = run_inference_benchmark_with_evidence(
        store,
        _table(),
        pd.DataFrame(index=_table().index),
        ["predict_only"],
        job_id="infer-01",
        dataset_snapshot_ids={"new_a": "dataset-snapshot@sha256:a"},
    )

    assert len(records) == 2
    assert len(evidence["prediction_records"]) == 4
    first = evidence["prediction_records"][0]
    assert first["prediction_record_id"].startswith("pred:infer-01:")
    assert first["candidate_id"].startswith("artifact:")
    assert first["model_artifact_id"] == stored["artifact_id"]
    assert first["dataset_snapshot_id"] == "dataset-snapshot@sha256:a"
    assert first["feature_values"] == {"voltage_mean": 3.5}
    assert first["metric_contributions"]["mae_contribution"] == pytest.approx(0.005)
    assert len(evidence["metric_contributions"]) == 4


def test_safe_inference_uses_only_matching_loco_holdout(tmp_path):
    store = ModelStore(tmp_path / "models")
    manifest = _manifest()
    manifest.update({
        "trained_on": ["new_a", "new_b"],
        "protocol": "loco",
        "held_out_key": "new_a/c1",
    })
    store.save_training_model("heldout_model", ConstantModel(), manifest, replace=True)

    records, evidence = run_inference_benchmark_with_evidence(
        store,
        _table(),
        pd.DataFrame(index=_table().index),
        ["heldout_model"],
        job_id="infer-heldout",
        evaluation_policy="held_out_same_chemistry",
    )

    new_a = next(row for row in records if row["dataset"] == "new_a")
    new_b = next(row for row in records if row["dataset"] == "new_b")
    assert new_a["status"] == "succeeded"
    assert new_a["evaluation_scope"] == "held_out_test"
    assert new_a["n_samples"] == 2
    assert new_b["status"] == "skipped"
    assert new_b["evaluation_scope"] == "training_overlap"
    assert {row["dataset"] for row in evidence["prediction_records"]} == {"new_a"}
    assert {row["evaluation_scope"] for row in evidence["prediction_records"]} == {"held_out_test"}


def test_safe_inference_allows_same_chemistry_and_skips_cross_chemistry(tmp_path):
    store = ModelStore(tmp_path / "models")
    manifest = _manifest()
    manifest.update({
        "trained_on": ["calce_cs2"],
        "protocol": "loco",
        "held_out_key": "calce_cs2/CS2_33",
    })
    store.save_training_model("transfer_model", ConstantModel(), manifest, replace=True)
    table = _table().copy()
    table["dataset"] = ["nasa", "nasa", "hust", "hust"]

    records, _evidence = run_inference_benchmark_with_evidence(
        store,
        table,
        pd.DataFrame(index=table.index),
        ["transfer_model"],
        job_id="infer-chemistry",
        evaluation_policy="held_out_same_chemistry",
    )

    nasa = next(row for row in records if row["dataset"] == "nasa")
    hust = next(row for row in records if row["dataset"] == "hust")
    assert nasa["status"] == "succeeded"
    assert nasa["evaluation_scope"] == "same_chemistry_transfer"
    assert nasa["target_chemistry"] == "LCO"
    assert hust["status"] == "skipped"
    assert hust["evaluation_scope"] == "cross_chemistry"
