import json
import pickle

import numpy as np
import pandas as pd
import pytest

from sohbench.model_store import ModelStore, ModelStoreError


class PredictOnlyModel:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        raise AssertionError("inference path must not call fit")

    def predict(self, X, meta=None):
        return np.full(len(X), 0.91)


class RejectTargetModel:
    def predict(self, X, meta=None):
        assert "soh" not in X.columns
        return np.full(len(X), 0.9)


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X, meta=None):
        return np.full(len(X), self.value)


def _table():
    return pd.DataFrame(
        {
            "dataset": ["demo"] * 4,
            "cell_id": ["c1"] * 4,
            "cycle_index": [0, 1, 2, 3],
            "soh": [1.0, 0.98, 0.96, 0.94],
            "voltage_mean": [3.5, 3.48, 3.46, 3.44],
            "resistance_proxy": [0.01, 0.02, 0.03, 0.04],
        }
    )


def _manifest(name="demo_model"):
    return {
        "model_name": name,
        "model_kind": "pickle",
        "estimator": "predict_only",
        "featureset": "full",
        "feature_columns": ["voltage_mean", "resistance_proxy"],
        "target": "soh",
        "trained_on": ["train_demo"],
        "created_by_run": "unit",
        "predict_contract": "predict(X, meta=None)",
    }


def test_save_training_model_writes_pickle_manifest_and_catalog(tmp_path):
    store = ModelStore(tmp_path / "models")

    resource = store.save_training_model("demo_model", PredictOnlyModel(), _manifest())

    model_dir = tmp_path / "models" / "demo_model"
    assert resource["name"] == "demo_model"
    assert resource["source"] == "training"
    assert (model_dir / "model.pkl").exists()
    assert (model_dir / "model_manifest.json").exists()
    assert resource["artifact_id"].startswith("model@sha256:")
    assert resource["contract_version"] == "1.0.0"
    assert resource["validation"]["predict_smoke_test"] == "passed"
    assert store.catalog()[0]["name"] == "demo_model"


def test_load_artifact_verifies_immutable_model_digest(tmp_path):
    store = ModelStore(tmp_path / "models")
    resource = store.save_training_model("demo_model", PredictOnlyModel(), _manifest())

    loaded = store.load_artifact(resource["artifact_id"])

    assert loaded.name == "demo_model"
    (tmp_path / "models" / "demo_model" / "model_manifest.json").write_text(
        json.dumps({**_manifest(), "created_by_run": "tampered"}), encoding="utf-8"
    )
    with pytest.raises(ModelStoreError, match="unknown or changed model artifact"):
        store.load_artifact(resource["artifact_id"])


def test_replacing_logical_name_preserves_previous_artifact_bytes(tmp_path):
    store = ModelStore(tmp_path / "models")
    first = store.save_training_model("demo_model", ConstantModel(0.81), _manifest())
    second_manifest = {**_manifest(), "created_by_run": "second"}

    second = store.save_training_model(
        "demo_model", ConstantModel(0.87), second_manifest, replace=True
    )

    assert first["artifact_id"] != second["artifact_id"]
    assert store.load_model("demo_model").predict(_table()).tolist() == [0.87] * 4
    assert store.load_artifact(first["artifact_id"]).predict(_table()).tolist() == [0.81] * 4
    digest = first["artifact_id"].split(":", 1)[1]
    assert (tmp_path / "models" / "_artifacts" / digest / "model.pkl").exists()


def test_upload_model_loads_and_predicts_without_fit(tmp_path):
    store = ModelStore(tmp_path / "models")
    model_payload = pickle.dumps(PredictOnlyModel())
    manifest_payload = json.dumps(_manifest("uploaded_model")).encode("utf-8")

    store.upload_model(
        "uploaded_model",
        "uploaded_model.pkl",
        model_payload,
        "model_manifest.json",
        manifest_payload,
    )

    loaded = store.load_model("uploaded_model")
    pred = loaded.predict(_table(), pd.DataFrame())

    assert pred.tolist() == [0.91, 0.91, 0.91, 0.91]
    assert loaded.compatibility(_table())["ok"] is True


def test_loaded_model_rejects_missing_manifest_features(tmp_path):
    store = ModelStore(tmp_path / "models")
    manifest = _manifest("bad_model")
    manifest["feature_columns"] = ["not_present"]
    store.save_training_model("bad_model", PredictOnlyModel(), manifest)

    loaded = store.load_model("bad_model")

    with pytest.raises(ModelStoreError, match="missing feature columns"):
        loaded.predict(_table(), pd.DataFrame())


def test_loaded_model_never_passes_manifest_target_to_predictor(tmp_path):
    store = ModelStore(tmp_path / "models")
    manifest = _manifest("target_guard")
    manifest["input_columns"] = [
        "dataset", "cell_id", "cycle_index", "soh", "voltage_mean", "resistance_proxy"
    ]
    store.save_training_model("target_guard", RejectTargetModel(), manifest)

    loaded = store.load_model("target_guard")
    prediction = loaded.predict(_table(), pd.DataFrame())

    assert "soh" not in loaded.input_columns(_table())
    assert prediction.tolist() == [0.9, 0.9, 0.9, 0.9]


def test_invalidated_artifact_is_persisted_hidden_from_use_and_kept_for_audit(tmp_path):
    store = ModelStore(tmp_path / "models")
    resource = store.save_training_model("demo_model", PredictOnlyModel(), _manifest())

    invalidation = store.invalidate_artifact(
        resource["artifact_id"],
        reason="label_leakage",
        details={"feature_columns": ["fade_from_full"]},
        source="unit-test",
    )

    assert invalidation["reason"] == "label_leakage"
    assert store.is_artifact_invalidated(resource["artifact_id"]) is True
    item = next(row for row in store.catalog() if row["artifact_id"] == resource["artifact_id"])
    assert item["status"] == "invalidated"
    assert item["usable"] is False
    assert item["validation"]["label_safety"] == "blocked"
    with pytest.raises(ModelStoreError, match="invalidated"):
        store.load_artifact(resource["artifact_id"])

    reopened = ModelStore(tmp_path / "models")
    assert reopened.is_artifact_invalidated(resource["artifact_id"]) is True


def test_loaded_model_rejects_renamed_exact_label_copy_at_inference(tmp_path):
    store = ModelStore(tmp_path / "models")
    manifest = _manifest("renamed_label_copy")
    manifest["feature_columns"] = ["innocent_looking_signal"]
    store.save_training_model("renamed_label_copy", PredictOnlyModel(), manifest)
    table = pd.concat([_table(), _table()], ignore_index=True)
    table["innocent_looking_signal"] = table["soh"]

    loaded = store.load_model("renamed_label_copy")

    with pytest.raises(ModelStoreError, match="inference blocked"):
        loaded.predict(table)


def test_loaded_model_rejects_renamed_exact_label_complement_at_inference(tmp_path):
    store = ModelStore(tmp_path / "models")
    manifest = _manifest("renamed_label_complement")
    manifest["feature_columns"] = ["innocent_looking_signal"]
    store.save_training_model("renamed_label_complement", PredictOnlyModel(), manifest)
    table = pd.concat([_table(), _table()], ignore_index=True)
    table["innocent_looking_signal"] = 1.0 - table["soh"]

    loaded = store.load_model("renamed_label_complement")

    with pytest.raises(ModelStoreError, match="inference blocked"):
        loaded.predict(table)
