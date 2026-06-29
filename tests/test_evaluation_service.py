import json
from pathlib import Path

import pandas as pd

from sohbench.evaluation_service import EvaluationService


def _dataset_csv(name="plugin_eval_demo"):
    rows = []
    for cell in ["c1", "c2", "c3"]:
        for cycle in range(12):
            rows.append(
                {
                    "dataset": name,
                    "cell_id": cell,
                    "cycle_index": cycle,
                    "soh": 1.0 - 0.012 * cycle,
                    "voltage_mean": 3.5 - 0.015 * cycle,
                    "resistance_proxy": 0.01 + 0.002 * cycle,
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


ESTIMATOR_PLUGIN = b'''
import numpy as np
from sohbench.contracts import feature_columns

class Estimator:
    name = "trusted_linear_decay"
    feature_mode = "unified"
    backend = "external"

    def fit(self, X, y, meta=None):
        cols = feature_columns(X)
        self.col_ = cols[0]
        self.offset_ = float(np.mean(y))
        self.slope_ = -0.01
        return self

    def predict(self, X, meta=None):
        centered = X[self.col_].to_numpy(dtype=float) - float(X[self.col_].mean())
        return np.clip(self.offset_ + self.slope_ * centered, 0.0, 1.3)
'''


FEATURE_PLUGIN = b'''
FEATURESET_NAME = "trusted_cycle_shape"
FEATURE_COLUMNS = ["cycle_norm", "fade_from_full"]

def transform(table, meta=None):
    out = table.copy()
    max_cycle = out.groupby(["dataset", "cell_id"])["cycle_index"].transform("max").replace(0, 1)
    out["cycle_norm"] = out["cycle_index"] / max_cycle
    out["fade_from_full"] = 1.0 - out["soh"]
    return out
'''


def test_inline_job_runs_uploaded_dataset_algorithm_and_featureset(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.store.upload_estimator_plugin("trusted_linear_decay", "trusted_linear_decay.py", ESTIMATOR_PLUGIN)
    service.store.upload_featureset_plugin("trusted_cycle_shape", "trusted_cycle_shape.py", FEATURE_PLUGIN)

    job = service.create_job(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["trusted_linear_decay"],
            "featuresets": ["trusted_cycle_shape"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    finished = service.run_job_inline(job["id"])

    assert finished["status"] == "succeeded"
    run_dir = Path(finished["run_dir"])
    assert (run_dir / "leaderboard.json").exists()
    assert (tmp_path / "frontend_data" / "leaderboard.json").exists()
    cfg = json.loads((tmp_path / "frontend_data" / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["datasets"] == ["plugin_eval_demo"]
    assert cfg["estimators"] == ["trusted_linear_decay"]
    assert cfg["featuresets"] == ["trusted_cycle_shape"]


def test_service_catalog_lists_uploaded_and_builtin_resources(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    catalog = service.catalog()

    assert any(item["name"] == "plugin_eval_demo" for item in catalog["datasets"])
    assert any(item["name"] == "ridge" for item in catalog["estimators"])
    assert any(item["name"] == "full" for item in catalog["featuresets"])


def test_training_job_publishes_model_catalog(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    job = service.create_job(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    finished = service.run_job_inline(job["id"])

    assert finished["status"] == "succeeded"
    assert (Path(finished["run_dir"]) / "model_catalog.json").exists()
    frontend_catalog = tmp_path / "frontend_data" / "model_catalog.json"
    assert frontend_catalog.exists()
    models = json.loads(frontend_catalog.read_text(encoding="utf-8"))["models"]
    assert any(item["estimator"] == "ridge" for item in models)
