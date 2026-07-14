import json
import pickle
from pathlib import Path

import numpy as np
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
FEATURE_COLUMNS = ["cycle_feature"]

def transform(table, meta=None):
    out = table.copy()
    out["cycle_feature"] = out["cycle_index"].astype(float)
    return out
'''


class PickledPredictOnly:
    def fit(self, X, y=None, meta=None):
        raise AssertionError("fit must not be called")

    def predict(self, X, meta=None):
        return np.full(len(X), 0.95)


def _model_manifest(name="api_model"):
    return {
        "model_name": name,
        "model_kind": "pickle",
        "estimator": "pickled_predict_only",
        "featureset": "full",
        "feature_columns": ["voltage_mean"],
        "target": "soh",
        "trained_on": ["train"],
        "created_by_run": "unit",
        "predict_contract": "predict(X, meta=None)",
    }


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


def test_service_catalog_can_return_limited_preview_with_total_counts(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.model_store._catalog["models"] = [
        {"name": f"model_{idx}", "feature_columns": ["voltage_mean"]}
        for idx in range(5)
    ]
    for idx in range(3):
        service._save_job(
            {
                "id": f"job_{idx}",
                "status": "succeeded",
                "created_at": f"2026-07-01 00:0{idx}:00",
                "request": {"datasets": ["demo"], "estimators": ["ridge"], "featuresets": ["full"]},
                "saved_models": [f"model_{idx}_{item}" for item in range(20)],
            }
        )

    catalog = service.catalog(model_limit=2, job_limit=1, compact_jobs=True)

    assert len(catalog["models"]) == 2
    assert catalog["model_count"] == 5
    assert catalog["models_limited"] is True
    assert len(catalog["jobs"]) == 1
    assert catalog["job_count"] == 3
    assert catalog["jobs_limited"] is True
    assert "saved_models" not in catalog["jobs"][0]
    assert len(service.catalog()["models"]) == 5


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


def test_saved_model_manifest_excludes_control_target_from_model_inputs(tmp_path):
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

    completed = service.run_job_inline(job["id"])
    manifest = service.model_store.load_model(completed["saved_models"][0]).manifest

    assert "soh" not in manifest["input_columns"]
    assert manifest["feature_columns"] == ["voltage_mean", "resistance_proxy"]


def test_inference_job_runs_uploaded_pickle_model(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
    )

    job = service.create_inference_job(
        {"datasets": ["plugin_eval_demo"], "models": ["api_model"]},
        start=False,
    )
    finished = service.run_inference_job_inline(job["id"])

    assert finished["status"] == "succeeded"
    assert (Path(finished["run_dir"]) / "inference_benchmark.json").exists()
    assert (tmp_path / "frontend_data" / "inference_benchmark.json").exists()


def test_inference_fails_when_manifest_pipeline_or_state_is_missing(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    manifest = _model_manifest("explicit_pipeline")
    manifest["feature_pipeline"] = "missing_pipeline"
    service.model_store.upload_model(
        "explicit_pipeline",
        "explicit_pipeline.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(manifest).encode("utf-8"),
    )

    job = service.create_inference_job(
        {"datasets": ["plugin_eval_demo"], "models": ["explicit_pipeline"]}, start=False
    )
    finished = service.run_inference_job_inline(job["id"])

    assert finished["status"] == "failed"
    assert "missing_pipeline" in finished["error"]


def test_inference_fails_when_explicit_pipeline_state_is_missing(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.store.upload_featureset_plugin(
        "explicit_features",
        "explicit_features.py",
        b'''
FEATURESET_NAME = "explicit_features"
FEATURE_COLUMNS = ["extra"]
def transform(table, meta=None):
    out = table.copy()
    out["extra"] = out["cycle_index"]
    return out
''',
    )
    manifest = _model_manifest("missing_state")
    manifest["feature_pipeline"] = "explicit_features"
    service.model_store.upload_model(
        "missing_state", "missing_state.pkl", pickle.dumps(PickledPredictOnly()),
        "model_manifest.json", json.dumps(manifest).encode("utf-8"),
    )

    job = service.create_inference_job({"datasets": ["plugin_eval_demo"], "models": ["missing_state"]}, start=False)
    finished = service.run_inference_job_inline(job["id"])

    assert finished["status"] == "failed"
    assert "feature state" in finished["error"]


def test_inference_job_persists_launch_strategy_audit_fields(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
    )

    job = service.create_inference_job(
        {
            "datasets": ["plugin_eval_demo"],
            "models": ["api_model"],
            "launch_strategy": "compatible_only",
            "selected_datasets": ["plugin_eval_demo", "risk_demo"],
            "submitted_datasets": ["plugin_eval_demo"],
            "skipped_datasets": ["risk_demo"],
        },
        start=False,
    )
    assert job["request"]["launch_strategy"] == "compatible_only"
    assert job["request"]["selected_datasets"] == ["plugin_eval_demo", "risk_demo"]
    assert job["request"]["submitted_datasets"] == ["plugin_eval_demo"]
    assert job["request"]["skipped_datasets"] == ["risk_demo"]

    finished = service.run_inference_job_inline(job["id"])
    config = json.loads((Path(finished["run_dir"]) / "inference_run_config.json").read_text(encoding="utf-8"))
    assert config["launch_strategy"] == "compatible_only"
    assert config["selected_datasets"] == ["plugin_eval_demo", "risk_demo"]
    assert config["submitted_datasets"] == ["plugin_eval_demo"]
    assert config["skipped_datasets"] == ["risk_demo"]
    published_config = json.loads((tmp_path / "frontend_data" / "inference_run_config.json").read_text(encoding="utf-8"))
    assert published_config["job_id"] == job["id"]
    assert published_config["launch_strategy"] == "compatible_only"
    assert published_config["skipped_datasets"] == ["risk_demo"]


def test_list_inference_jobs_sorts_by_created_at_desc(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    older = {
        "id": "z-old-name",
        "kind": "inference",
        "status": "succeeded",
        "created_at": "2026-06-30 06:00:00",
        "request": {"datasets": ["nasa"], "models": ["m1"]},
    }
    newer = {
        "id": "a-new-name",
        "kind": "inference",
        "status": "succeeded",
        "created_at": "2026-06-30 12:00:00",
        "request": {"datasets": ["nasa"], "models": ["m1"], "launch_strategy": "compatible_only"},
    }
    (service.inference_jobs_dir / "z-old-name.json").write_text(json.dumps(older), encoding="utf-8")
    (service.inference_jobs_dir / "a-new-name.json").write_text(json.dumps(newer), encoding="utf-8")

    jobs = service.list_inference_jobs()

    assert [job["id"] for job in jobs] == ["a-new-name", "z-old-name"]
