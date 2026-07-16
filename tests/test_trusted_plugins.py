import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohbench import registry as R
from sohbench.data import DatasetLoader
from sohbench.registry import get_estimator, get_featureset
from sohbench.trusted_plugins import TrustedPluginStore, _load_module


def _dataset_frame(name="public_demo", cells=("c1", "c2", "c3"), n=8):
    rows = []
    for cell in cells:
        for cycle in range(n):
            soh = 1.0 - 0.01 * cycle
            rows.append(
                {
                    "dataset": name,
                    "cell_id": cell,
                    "cycle_index": cycle,
                    "soh": soh,
                    "voltage_mean": 3.4 - 0.01 * cycle,
                    "resistance_proxy": 0.02 + 0.001 * cycle,
                }
            )
    return pd.DataFrame(rows)


def test_upload_dataset_validates_and_persists_normalized_pickle(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    payload = _dataset_frame("ignored_name").to_csv(index=False).encode("utf-8")

    registered = store.upload_dataset(
        name="public_demo",
        filename="demo.csv",
        payload=payload,
    )

    saved_path = tmp_path / "data" / "processed" / "public_demo_features.pkl"
    saved = pd.read_pickle(saved_path)
    assert registered["name"] == "public_demo"
    assert registered["rows"] == len(saved)
    assert saved["dataset"].unique().tolist() == ["public_demo"]
    assert DatasetLoader(tmp_path / "data").discover_datasets() == ["public_demo"]


def test_upload_and_catalog_preserve_unlabeled_inference_dataset(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    payload = _dataset_frame("ignored_name").drop(columns=["soh"]).to_csv(index=False).encode("utf-8")

    registered = store.upload_dataset(
        name="unlabeled_demo",
        filename="unlabeled.csv",
        payload=payload,
    )
    catalog_item = next(
        item for item in store.catalog()["datasets"] if item["name"] == "unlabeled_demo"
    )

    assert registered["has_labels"] is False
    assert catalog_item["has_labels"] is False
    assert DatasetLoader(tmp_path / "data").discover_datasets() == []
    assert DatasetLoader(tmp_path / "data").discover_datasets(require_labels=False) == [
        "unlabeled_demo"
    ]


def test_upload_estimator_plugin_smoke_tests_and_registers(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    code = b'''
import numpy as np

class Estimator:
    name = "trusted_mean_estimator"
    feature_mode = "unified"
    backend = "external"

    def fit(self, X, y, meta=None):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X, meta=None):
        return np.full(len(X), self.mean_, dtype=float)
'''

    registered = store.upload_estimator_plugin(
        name="mean_estimator",
        filename="mean_estimator.py",
        payload=code,
    )

    cls = get_estimator("trusted_mean_estimator")
    est = cls()
    sample = pd.DataFrame({"dataset": ["d"] * 2, "cell_id": ["c"] * 2, "cycle_index": [0, 1], "soh": [1.0, 0.9], "f": [1, 2]})
    est.fit(sample, sample["soh"].to_numpy(), None)
    assert registered["name"] == "trusted_mean_estimator"
    assert est.predict(sample, None).shape == (2,)
    assert any(item["name"] == "trusted_mean_estimator" for item in store.catalog()["estimators"])


def test_upload_featureset_plugin_registers_transforming_feature_set(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    code = b'''
FEATURESET_NAME = "trusted_trend_features"
FEATURE_COLUMNS = ["cycle_feature"]

def transform(table, meta=None):
    out = table.copy()
    out["cycle_feature"] = out["cycle_index"].astype(float)
    return out
'''

    registered = store.upload_featureset_plugin(
        name="trend_features",
        filename="trend_features.py",
        payload=code,
    )
    transformed = store.apply_feature_transforms(
        _dataset_frame(),
        meta=pd.DataFrame(),
        featuresets=["trusted_trend_features"],
    )

    fs = get_featureset("trusted_trend_features")
    assert registered["name"] == "trusted_trend_features"
    assert fs.columns(transformed) == ["cycle_feature"]
    assert transformed["cycle_feature"].equals(transformed["cycle_index"].astype(float))
    assert any(item["name"] == "trusted_trend_features" for item in store.catalog()["featuresets"])


def test_legacy_pipeline_rejects_a_state_from_changed_plugin_source(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    store.upload_featureset_plugin(
        name="stable_source",
        filename="stable_source.py",
        payload=b'''
FEATURESET_NAME = "stable_source"
FEATURE_COLUMNS = ["stable_feature"]

def transform(table, meta=None):
    out = table.copy()
    assert "soh" not in out.columns
    out["stable_feature"] = out["cycle_index"]
    return out
''',
    )
    pipeline = store.feature_pipeline("stable_source")
    frame = _dataset_frame().drop(columns=["soh"])
    state = pipeline.fit(frame, np.empty(len(frame)), pd.DataFrame())
    path = tmp_path / "plugins" / "featuresets" / "stable_source.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="plugin digest"):
        pipeline.transform(frame, state, pd.DataFrame())


def test_legacy_pipeline_reloads_module_globals_for_each_transform(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    store.upload_featureset_plugin(
        name="fresh_globals",
        filename="fresh_globals.py",
        payload=b'''
FEATURESET_NAME = "fresh_globals"
FEATURE_COLUMNS = ["fresh_counter"]
COUNTER = 0

def transform(table, meta=None):
    global COUNTER
    COUNTER += 1
    out = table.copy()
    out["fresh_counter"] = COUNTER
    return out
''',
    )
    pipeline = store.feature_pipeline("fresh_globals")
    frame = _dataset_frame().drop(columns=["soh"])
    state = pipeline.fit(frame, np.empty(len(frame)), pd.DataFrame())

    first = pipeline.transform(frame, state, pd.DataFrame())
    second = pipeline.transform(frame, state, pd.DataFrame())

    assert first["fresh_counter"].eq(1).all()
    assert second["fresh_counter"].eq(1).all()


def test_plugin_registration_executes_replacement_source_with_same_size_and_mtime(tmp_path):
    path = tmp_path / "registration.py"
    old_source = "REGISTRATION_MARKER = 1\n"
    new_source = "REGISTRATION_MARKER = 2\n"
    assert len(old_source.encode("utf-8")) == len(new_source.encode("utf-8"))
    path.write_text(old_source, encoding="utf-8")
    original_mtime = path.stat().st_mtime_ns

    assert _load_module(path, "featureset").REGISTRATION_MARKER == 1
    path.write_text(new_source, encoding="utf-8")
    os.utime(path, ns=(original_mtime, original_mtime))

    assert _load_module(path, "featureset").REGISTRATION_MARKER == 2


def test_catalog_is_persisted_for_uploaded_resources(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    store.upload_dataset("persisted_demo", "demo.csv", _dataset_frame().to_csv(index=False).encode("utf-8"))

    catalog_path = tmp_path / "plugins" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert catalog["datasets"][0]["name"] == "persisted_demo"


def test_plugin_catalog_paths_reload_from_different_working_directory(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    store = TrustedPluginStore(base_dir="plugins", data_dir="data")
    store.upload_featureset_plugin(
        name="reload_features",
        filename="reload_features.py",
        payload=b'''
FEATURESET_NAME = "reload_features"
FEATURE_COLUMNS = ["reload_cycle"]

def transform(table, meta=None):
    out = table.copy()
    out["reload_cycle"] = out["cycle_index"]
    return out
''',
    )
    catalog = json.loads((project / "plugins" / "catalog.json").read_text(encoding="utf-8"))
    assert Path(catalog["featuresets"][0]["path"]).is_absolute()

    R._FEATURESETS.pop("reload_features", None)
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    TrustedPluginStore(base_dir=project / "plugins", data_dir=project / "data")

    assert get_featureset("reload_features").candidates == ["reload_cycle"]
