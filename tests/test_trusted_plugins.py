import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohbench import registry as R
from sohbench.data import DatasetLoader
from sohbench.registry import get_estimator, get_featureset
from sohbench.trusted_plugins import TrustedPluginError, TrustedPluginStore


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
import numpy as np

FEATURE_COLUMNS = ["cycle_norm", "cycle_log1p"]

def transform(table, meta=None):
    out = table.copy()
    max_cycle = out.groupby(["dataset", "cell_id"])["cycle_index"].transform("max").replace(0, 1)
    out["cycle_norm"] = out["cycle_index"] / max_cycle
    out["cycle_log1p"] = np.log1p(out["cycle_index"].clip(lower=0))
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
    assert fs.columns(transformed) == ["cycle_norm", "cycle_log1p"]
    assert transformed["cycle_norm"].between(0, 1).all()
    assert transformed["cycle_log1p"].iloc[0] == 0
    assert any(item["name"] == "trusted_trend_features" for item in store.catalog()["featuresets"])


def test_upload_featureset_plugin_rejects_target_dependent_transform(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    code = b'''
FEATURESET_NAME = "leaking_features"
FEATURE_COLUMNS = ["fade_from_full"]

def transform(table, meta=None):
    out = table.copy()
    out["fade_from_full"] = 1.0 - out["soh"]
    return out
'''

    with pytest.raises(TrustedPluginError, match="protected label"):
        store.upload_featureset_plugin(
            name="leaking_features",
            filename="leaking_features.py",
            payload=code,
        )

    assert not (tmp_path / "plugins" / "featuresets" / "leaking_features.py").exists()


def test_catalog_is_persisted_for_uploaded_resources(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    store.upload_dataset("persisted_demo", "demo.csv", _dataset_frame().to_csv(index=False).encode("utf-8"))

    catalog_path = tmp_path / "plugins" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert catalog["datasets"][0]["name"] == "persisted_demo"


def test_catalog_reports_dataset_scale_chemistry_and_capacity(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    frame = _dataset_frame("nasa", cells=("B1", "B2"), n=3)
    frame["capacity_ah"] = [1.8, 1.7, 1.6, 1.9, 1.8, 1.7]
    store.upload_dataset("nasa", "nasa.csv", frame.to_csv(index=False).encode("utf-8"))

    item = next(row for row in store.catalog()["datasets"] if row["name"] == "nasa")

    assert item["record_count"] == 6
    assert item["cycle_count"] == 6
    assert item["battery_count"] == 2
    assert item["chemistry"] == "LCO"
    assert item["max_capacity_ah"] == 1.9
    assert item["capacity_source"] == "measured_capacity"
    assert item["held_out_ready"] is True


def test_catalog_enriches_datasets_from_local_data_manifest(tmp_path):
    data_dir = tmp_path / "data"
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=data_dir)
    store.upload_dataset(
        "nasa",
        "nasa.csv",
        _dataset_frame("nasa", cells=("B1", "B2"), n=2).to_csv(index=False).encode("utf-8"),
    )
    (data_dir / "manifest.json").write_text(
        json.dumps({
            "schema_version": "1.0.0",
            "migration_strategy": "B-curated-supervised",
            "verification": {"all_inventory_checks_passed": True},
            "summary": {"processed_datasets": 1},
            "datasets": [{
                "name": "nasa",
                "status": "ready",
                "roles": ["training", "inference", "audit"],
                "raw_local": True,
                "curve_path": "curves/nasa_curves.json",
                "source": "NASA PCoE",
                "license": "public",
                "processed": {"sha256": "abc123"},
            }],
        }),
        encoding="utf-8",
    )

    catalog = store.catalog()
    item = next(row for row in catalog["datasets"] if row["name"] == "nasa")

    assert catalog["data_manifest"]["verified"] is True
    assert catalog["data_manifest"]["migration_strategy"] == "B-curated-supervised"
    assert item["raw_local"] is True
    assert item["curve_available"] is True
    assert item["roles"] == ["training", "inference", "audit"]
    assert item["source"] == "NASA PCoE"
    assert item["processed_sha256"] == "abc123"


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
