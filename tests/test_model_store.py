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
    assert store.catalog()[0]["name"] == "demo_model"


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
