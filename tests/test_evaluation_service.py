import json
import pickle
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sohbench.evaluation_service as evaluation_service_module
from sohbench.evaluation_service import EvaluationService
from sohbench.fabric.evidence import verify_evidence_pack


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


FAILING_ESTIMATOR_PLUGIN = b'''
class Estimator:
    name = "always_fails"

    def fit(self, X, y, meta=None):
        if len(X) == 8:
            return self
        raise RuntimeError("intentional estimator failure")

    def predict(self, X, meta=None):
        return [1.0] * len(X)
'''


REPLACEMENT_ESTIMATOR_PLUGIN = b'''
import numpy as np

class Estimator:
    name = "trusted_linear_decay"
    feature_mode = "unified"
    backend = "external"

    def fit(self, X, y, meta=None):
        return self

    def predict(self, X, meta=None):
        return np.full(len(X), 0.50)
'''


class PickledPredictOnly:
    def fit(self, X, y=None, meta=None):
        raise AssertionError("fit must not be called")

    def predict(self, X, meta=None):
        return np.full(len(X), 0.95)


class PickledPredictOnlyV2(PickledPredictOnly):
    def predict(self, X, meta=None):
        return np.full(len(X), 0.94)


class PickledPredictOnlyReplacement(PickledPredictOnly):
    def predict(self, X, meta=None):
        return np.full(len(X), 0.50)


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


def test_training_job_exposes_verified_evidence_contract(tmp_path):
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

    assert finished["run_id"].startswith("RUN-")
    assert finished["execution_state"] == "REVIEW"
    assert finished["outcome"] in {"complete", "partial", "failed"}
    assert Path(finished["evidence_path"]).name == "evidence-pack.json"
    assert verify_evidence_pack(finished["evidence_path"])["ok"] is True


def test_training_lifecycle_persists_contract_before_loader_failure(tmp_path, monkeypatch):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    queued = service.create_job(
        {"datasets": ["missing"], "estimators": ["ridge"]}, start=False,
    )

    assert queued["run_id"].startswith("RUN-")
    assert queued["execution_state"] == "QUEUED"
    assert queued["outcome"] is None
    assert queued["verdict"] == "not_reviewed"
    assert queued["evidence_path"] is None

    def fail_load(_loader, _datasets, **_kwargs):
        running = service.get_job(queued["id"])
        assert running["run_id"] == queued["run_id"]
        assert running["execution_state"] == "RUNNING"
        assert running["outcome"] is None
        assert running["evidence_path"] is None
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(evaluation_service_module.DatasetLoader, "load", fail_load)
    failed = service.run_job_inline(queued["id"])

    assert failed["run_id"] == queued["run_id"]
    assert failed["status"] == "failed"
    assert failed["execution_state"] == "FAILED"
    assert failed["outcome"] == "failed"
    assert failed["verdict"] == "fail"
    assert failed["evidence_path"] is None


def test_inference_lifecycle_persists_contract_before_loader_failure(tmp_path, monkeypatch):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.model_store.upload_model(
        "api_model", "api_model.pkl", pickle.dumps(PickledPredictOnly()),
        "model_manifest.json", json.dumps(_model_manifest()).encode("utf-8"),
    )
    queued = service.create_inference_job(
        {"datasets": ["missing"], "models": ["api_model"]}, start=False,
    )

    assert queued["run_id"].startswith("RUN-")
    assert queued["execution_state"] == "QUEUED"
    assert queued["outcome"] is None
    assert queued["verdict"] == "not_reviewed"
    assert queued["evidence_path"] is None

    def fail_load(_loader, _datasets, **_kwargs):
        running = service.get_inference_job(queued["id"])
        assert running["run_id"] == queued["run_id"]
        assert running["execution_state"] == "RUNNING"
        assert running["outcome"] is None
        assert running["evidence_path"] is None
        raise RuntimeError("loader exploded")

    monkeypatch.setattr(evaluation_service_module.DatasetLoader, "load", fail_load)
    failed = service.run_inference_job_inline(queued["id"])

    assert failed["run_id"] == queued["run_id"]
    assert failed["status"] == "failed"
    assert failed["execution_state"] == "FAILED"
    assert failed["outcome"] == "failed"
    assert failed["verdict"] == "fail"
    assert failed["evidence_path"] is None


def test_all_failed_training_job_has_failed_outcome_without_leaderboard_verdict(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.store.upload_estimator_plugin(
        "always_fails", "always_fails.py", FAILING_ESTIMATOR_PLUGIN,
    )

    job = service.create_job(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["always_fails"],
            "featuresets": ["full"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    finished = service.run_job_inline(job["id"])

    assert finished["status"] == "failed"
    assert finished["outcome"] == "failed"
    assert finished["execution_state"] == "FAILED"
    assert finished["verdict"] == "fail"
    assert all(not entry["verdicts"] for entry in json.loads(
        (Path(finished["raw_run_dir"]) / "leaderboard.json").read_text(encoding="utf-8")
    ))


def test_inference_job_keeps_config_digest_valid_after_evidence_creation(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.model_store.upload_model(
        "api_model", "api_model.pkl", pickle.dumps(PickledPredictOnly()),
        "model_manifest.json", json.dumps(_model_manifest()).encode("utf-8"),
    )

    job = service.create_inference_job(
        {"datasets": ["plugin_eval_demo"], "models": ["api_model"]}, start=False,
    )
    finished = service.run_inference_job_inline(job["id"])

    assert finished["run_id"].startswith("RUN-")
    assert finished["execution_state"] == "REVIEW"
    assert finished["outcome"] == "complete"
    assert verify_evidence_pack(finished["evidence_path"])["ok"] is True


def test_final_rebuilt_evidence_is_verified_before_publish(tmp_path, monkeypatch):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    original_write = evaluation_service_module.write_evidence_pack

    def write_then_corrupt(run_dir, manifest, artifact_paths):
        path = original_write(run_dir, manifest, artifact_paths)
        catalog = Path(run_dir) / "model_catalog.json"
        catalog.write_text(catalog.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return path

    monkeypatch.setattr(
        evaluation_service_module, "write_evidence_pack", write_then_corrupt,
    )
    job = service.create_job(
        {"datasets": ["plugin_eval_demo"], "estimators": ["ridge"]}, start=False,
    )
    finished = service.run_job_inline(job["id"])

    assert finished["status"] == "failed"
    assert finished["execution_state"] == "FAILED"
    assert finished["outcome"] == "failed"
    assert finished["evidence_path"] is None
    assert "mismatch" in finished["error"]
    assert not (tmp_path / "frontend_data" / "leaderboard.json").exists()


def test_prediction_sample_ids_are_intrinsic_and_order_independent(tmp_path):
    source = pd.read_csv(BytesIO(_dataset_csv()))
    observed = []
    for label, frame in (
        ("ordered", source),
        ("shuffled", source.sample(frac=1.0, random_state=17)),
    ):
        root = tmp_path / label
        service = EvaluationService(
            data_dir=root / "data",
            work_dir=root / "work",
            runs_dir=root / "runs",
            frontend_data_dir=root / "frontend_data",
        )
        service.store.upload_dataset(
            "plugin_eval_demo", "demo.csv", frame.to_csv(index=False).encode("utf-8"),
        )
        job = service.create_job(
            {"datasets": ["plugin_eval_demo"], "estimators": ["ridge"]}, start=False,
        )
        finished = service.run_job_inline(job["id"])
        predictions = pd.read_parquet(Path(finished["raw_run_dir"]) / "predictions.parquet")
        observed.append({
            (row.dataset, row.cell_id, int(row.cycle_index)): row.sample_id
            for row in predictions.itertuples()
        })

    assert observed[0] == observed[1]


@pytest.mark.parametrize(
    "bad_cycle",
    [1.2, np.nan, np.inf, -np.inf, True, -1, 2**63, float(2**63)],
)
def test_prediction_identity_rejects_non_integral_cycle_index(bad_cycle):
    frame = pd.read_csv(BytesIO(_dataset_csv()))
    frame["cycle_index"] = frame["cycle_index"].astype(object)
    frame.loc[0, "cycle_index"] = bad_cycle

    with pytest.raises(ValueError, match="cycle_index must contain finite integers"):
        evaluation_service_module._intrinsic_sample_ids(frame)


def test_duplicate_intrinsic_prediction_identity_fails_closed(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    frame = pd.read_csv(BytesIO(_dataset_csv()))
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    service.store.upload_dataset(
        "plugin_eval_demo", "demo.csv", frame.to_csv(index=False).encode("utf-8"),
    )
    job = service.create_job(
        {"datasets": ["plugin_eval_demo"], "estimators": ["ridge"]}, start=False,
    )

    finished = service.run_job_inline(job["id"])

    assert finished["status"] == "failed"
    assert finished["evidence_path"] is None
    assert "duplicate intrinsic sample identity" in finished["error"]


def test_job_json_persistence_does_not_expose_partial_documents(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    for job_id, save, get in (
        ("polling-training", service._save_job, service.get_job),
        ("polling-inference", service._save_inference_job, service.get_inference_job),
    ):
        job = {"id": job_id, "status": "queued", "payload": "x" * 100_000}
        save(job)
        errors = []
        done = threading.Event()

        def writer():
            for index in range(50):
                save({**job, "status": str(index)})
            done.set()

        def reader():
            while not done.is_set():
                try:
                    get(job_id)
                except (json.JSONDecodeError, OSError) as exc:
                    errors.append(exc)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        writer_thread.start()
        reader_thread.start()
        writer_thread.join()
        done.set()
        reader_thread.join()

        assert errors == []


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


def test_identical_training_jobs_publish_immutable_run_scoped_models(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset(
        "plugin_eval_demo", "demo.csv", _dataset_csv()
    )
    payload = {
        "datasets": ["plugin_eval_demo"],
        "estimators": ["ridge"],
        "featuresets": ["full"],
        "protocols": ["loco"],
        "seeds": [0],
    }

    first_job = service.create_job(payload, start=False)
    first = service.run_job_inline(first_job["id"])
    first_names = set(first["saved_models"])
    first_catalog = {
        item["name"]: item for item in service.model_store.catalog()
        if item["name"] in first_names
    }
    first_bytes = {
        name: Path(item["model_path"]).read_bytes()
        for name, item in first_catalog.items()
    }

    second_job = service.create_job(payload, start=False)
    second = service.run_job_inline(second_job["id"])
    second_names = set(second["saved_models"])
    catalog = {item["name"]: item for item in service.model_store.catalog()}

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first_names
    assert second_names
    assert first_names.isdisjoint(second_names)
    assert first_names | second_names <= set(catalog)
    for name in first_names:
        assert Path(catalog[name]["model_path"]).read_bytes() == first_bytes[name]
        assert service.model_store.load_model(name).manifest["created_by_run"] == first[
            "run_id"
        ]
    for name in second_names:
        assert service.model_store.load_model(name).manifest["created_by_run"] == second[
            "run_id"
        ]


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


def test_inference_job_keeps_object_pd_na_labels_unscored_and_successful(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    table = pd.read_csv(BytesIO(_dataset_csv("partial_labels")))
    table["soh"] = table["soh"].astype(object)
    table.loc[0, "soh"] = pd.NA
    service.store.upload_dataset(
        "partial_labels",
        "partial.pkl",
        pickle.dumps(table),
    )
    service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
    )
    job = service.create_inference_job(
        {"datasets": ["partial_labels"], "models": ["api_model"]},
        start=False,
    )

    finished = service.run_inference_job_inline(job["id"])
    payload = json.loads(
        (Path(finished["raw_run_dir"]) / "inference_benchmark.json").read_text(
            encoding="utf-8"
        )
    )

    assert finished["status"] == "succeeded"
    assert finished["outcome"] == "complete"
    assert payload["records"][0]["status"] == "succeeded"
    assert payload["records"][0]["mae"] is None
    assert verify_evidence_pack(finished["evidence_path"])["ok"] is True


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
    assert finished["outcome"] == "failed"
    assert verify_evidence_pack(finished["evidence_path"])["ok"] is True
    records = json.loads(
        (Path(finished["raw_run_dir"]) / "inference_benchmark.json").read_text(encoding="utf-8")
    )["records"]
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert "missing_pipeline" in records[0]["error"]


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
    assert finished["outcome"] == "failed"
    assert verify_evidence_pack(finished["evidence_path"])["ok"] is True
    records = json.loads(
        (Path(finished["raw_run_dir"]) / "inference_benchmark.json").read_text(encoding="utf-8")
    )["records"]
    assert records[0]["status"] == "failed"
    assert "feature state" in records[0]["error"]


def test_inference_preparation_failure_isolated_per_model_and_dataset(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.store.upload_dataset("plugin_eval_second", "second.csv", _dataset_csv())
    service.model_store.upload_model(
        "valid_model", "valid.pkl", pickle.dumps(PickledPredictOnly()),
        "model_manifest.json", json.dumps(_model_manifest("valid_model")).encode("utf-8"),
    )
    broken_manifest = _model_manifest("broken_model")
    broken_manifest["feature_pipeline"] = "missing_pipeline"
    service.model_store.upload_model(
        "broken_model", "broken.pkl", pickle.dumps(PickledPredictOnly()),
        "model_manifest.json", json.dumps(broken_manifest).encode("utf-8"),
    )
    job = service.create_inference_job(
        {
            "datasets": ["plugin_eval_demo", "plugin_eval_second"],
            "models": ["valid_model", "broken_model"],
        },
        start=False,
    )

    finished = service.run_inference_job_inline(job["id"])

    assert finished["status"] == "partial"
    assert finished["outcome"] == "partial"
    assert finished["execution_state"] == "REVIEW"
    assert verify_evidence_pack(finished["evidence_path"])["ok"] is True
    records = json.loads(
        (Path(finished["raw_run_dir"]) / "inference_benchmark.json").read_text(encoding="utf-8")
    )["records"]
    assert [(row["model"], row["dataset"], row["status"]) for row in records] == [
        ("valid_model", "plugin_eval_demo", "succeeded"),
        ("valid_model", "plugin_eval_second", "succeeded"),
        ("broken_model", "plugin_eval_demo", "failed"),
        ("broken_model", "plugin_eval_second", "failed"),
    ]


def test_unlabeled_run_spec_digest_is_stable_without_label_qc(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    table = pd.read_csv(BytesIO(_dataset_csv())).drop(columns=["soh"])
    request = {"datasets": ["plugin_eval_demo"], "models": ["api_model"]}

    first = service._build_run_spec(
        run_type="inference", request=request, datasets=["plugin_eval_demo"],
        table=table, qc={},
    )
    second = service._build_run_spec(
        run_type="inference", request=request, datasets=["plugin_eval_demo"],
        table=table.sample(frac=1.0, random_state=11), qc={},
    )

    assert first.dataset.digest == second.dataset.digest


def test_training_run_spec_canonicalizes_equivalent_protocol_sets(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    table = pd.read_csv(BytesIO(_dataset_csv()))
    request = {
        "datasets": ["plugin_eval_demo"],
        "estimators": ["ridge"],
        "featuresets": ["full"],
        "seeds": [0],
    }


def test_inference_run_spec_identity_binds_model_and_manifest_bytes(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset(
        "plugin_eval_demo", "demo.csv", _dataset_csv()
    )
    service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
    )
    table = pd.read_csv(BytesIO(_dataset_csv()))
    request = service._normalize_inference_request(
        {"datasets": ["plugin_eval_demo"], "models": ["api_model"]}
    )

    first = service._build_run_spec(
        run_type="inference",
        request=request,
        datasets=["plugin_eval_demo"],
        table=table,
        qc={},
    )
    artifact_kinds = {item.kind for item in first.candidates[0].artifacts}

    changed_manifest = _model_manifest()
    changed_manifest["created_by_run"] = "replacement"
    service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnlyV2()),
        "model_manifest.json",
        json.dumps(changed_manifest).encode("utf-8"),
        replace=True,
    )
    second = service._build_run_spec(
        run_type="inference",
        request=request,
        datasets=["plugin_eval_demo"],
        table=table,
        qc={},
    )

    assert {"model_pickle", "model_manifest", "feature_definition"} <= artifact_kinds
    assert first.candidates[0].spec_id == second.candidates[0].spec_id
    assert first.candidates[0].digest != second.candidates[0].digest
    assert first.digest != second.digest
    assert first.comparison_key == second.comparison_key


def test_inference_evidence_executes_content_bound_model_snapshot(
    tmp_path, monkeypatch
):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset(
        "plugin_eval_demo", "demo.csv", _dataset_csv()
    )
    service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
    )
    original_build_run_spec = service._build_run_spec
    replacement_started = threading.Event()
    replacement_finished = threading.Event()
    replacement_errors: list[BaseException] = []
    replacement_threads: list[threading.Thread] = []
    bound_specs = []

    def replace_model() -> None:
        replacement_started.set()
        replacement_manifest = _model_manifest()
        replacement_manifest["created_by_run"] = "replacement"
        try:
            service.model_store.upload_model(
                "api_model",
                "api_model.pkl",
                pickle.dumps(PickledPredictOnlyReplacement()),
                "model_manifest.json",
                json.dumps(replacement_manifest).encode("utf-8"),
                replace=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            replacement_errors.append(exc)
        finally:
            replacement_finished.set()

    def build_then_replace(**kwargs):
        run_spec = original_build_run_spec(**kwargs)
        bound_specs.append(run_spec)
        thread = threading.Thread(target=replace_model)
        replacement_threads.append(thread)
        thread.start()
        assert replacement_started.wait(timeout=2)
        replacement_finished.wait(timeout=1)
        return run_spec

    monkeypatch.setattr(service, "_build_run_spec", build_then_replace)
    job = service.create_inference_job(
        {"datasets": ["plugin_eval_demo"], "models": ["api_model"]},
        start=False,
    )

    finished = service.run_inference_job_inline(job["id"])
    for thread in replacement_threads:
        thread.join(timeout=5)

    assert replacement_finished.is_set()
    assert replacement_errors == []
    assert finished["status"] == "succeeded"
    benchmark = json.loads(
        (Path(finished["raw_run_dir"]) / "inference_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    expected_mae = float(
        np.mean(
            np.abs(
                pd.read_csv(BytesIO(_dataset_csv()))["soh"].to_numpy()
                - 0.95
            )
        )
    )
    assert benchmark["records"][0]["mae"] == pytest.approx(expected_mae)
    evidence = json.loads(Path(finished["evidence_path"]).read_text(encoding="utf-8"))
    assert evidence["run_spec_digest"] == bound_specs[0].digest
    assert (
        evidence["run_spec"]["candidates"][0]["digest"]
        == bound_specs[0].candidates[0].digest
    )
    assert verify_evidence_pack(finished["evidence_path"])["ok"] is True
    assert (
        service.model_store.load_model("api_model").manifest["created_by_run"]
        == "replacement"
    )


def test_training_evidence_executes_content_bound_trusted_plugin_snapshot(
    tmp_path, monkeypatch
):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset(
        "plugin_eval_demo", "demo.csv", _dataset_csv()
    )
    service.store.upload_estimator_plugin(
        "trusted_linear_decay",
        "trusted_linear_decay.py",
        ESTIMATOR_PLUGIN,
    )
    original_build_run_spec = service._build_run_spec
    replacement_started = threading.Event()
    replacement_finished = threading.Event()
    replacement_errors: list[BaseException] = []
    replacement_threads: list[threading.Thread] = []

    def replace_plugin() -> None:
        replacement_started.set()
        try:
            service.store.upload_estimator_plugin(
                "trusted_linear_decay",
                "trusted_linear_decay.py",
                REPLACEMENT_ESTIMATOR_PLUGIN,
                replace=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            replacement_errors.append(exc)
        finally:
            replacement_finished.set()

    def build_then_replace(**kwargs):
        run_spec = original_build_run_spec(**kwargs)
        thread = threading.Thread(target=replace_plugin)
        replacement_threads.append(thread)
        thread.start()
        assert replacement_started.wait(timeout=2)
        replacement_finished.wait(timeout=1)
        return run_spec

    monkeypatch.setattr(service, "_build_run_spec", build_then_replace)
    job = service.create_job(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["trusted_linear_decay"],
            "featuresets": ["full"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )

    finished = service.run_job_inline(job["id"])
    for thread in replacement_threads:
        thread.join(timeout=5)

    assert replacement_finished.is_set()
    assert replacement_errors == []
    assert finished["status"] == "succeeded"
    predictions = pd.read_parquet(
        Path(finished["raw_run_dir"]) / "predictions.parquet"
    )
    assert not np.allclose(predictions["y_pred"].to_numpy(), 0.50)
    assert verify_evidence_pack(finished["evidence_path"])["ok"] is True


def test_training_run_spec_identity_binds_estimator_and_feature_plugin_bytes(
    tmp_path,
):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset(
        "plugin_eval_demo", "demo.csv", _dataset_csv()
    )
    service.store.upload_estimator_plugin(
        "trusted_linear_decay", "trusted_linear_decay.py", ESTIMATOR_PLUGIN
    )
    service.store.upload_featureset_plugin(
        "trusted_cycle_shape", "trusted_cycle_shape.py", FEATURE_PLUGIN
    )
    table = pd.read_csv(BytesIO(_dataset_csv()))
    request = service._normalize_request(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["trusted_linear_decay"],
            "featuresets": ["trusted_cycle_shape"],
            "protocols": ["loco"],
            "seeds": [0],
        }
    )

    first = service._build_run_spec(
        run_type="training",
        request=request,
        datasets=["plugin_eval_demo"],
        table=table,
        qc={},
    )
    artifact_kinds = {item.kind for item in first.candidates[0].artifacts}

    service.store.upload_featureset_plugin(
        "trusted_cycle_shape",
        "trusted_cycle_shape.py",
        FEATURE_PLUGIN + b"\n# replacement bytes\n",
        replace=True,
    )
    second = service._build_run_spec(
        run_type="training",
        request=request,
        datasets=["plugin_eval_demo"],
        table=table,
        qc={},
    )

    assert {"estimator_plugin", "feature_plugin"} <= artifact_kinds
    assert first.candidates[0].digest != second.candidates[0].digest
    assert first.digest != second.digest
    specs = [
        service._build_run_spec(
            run_type="training",
            request={**request, "protocols": protocols},
            datasets=["plugin_eval_demo"],
            table=table,
            qc={},
        )
        for protocols in (
            ["loco", "lodo"],
            ["lodo", "loco"],
            ["loco", "loco", "lodo"],
        )
    ]

    assert {spec.protocol.spec_id for spec in specs} == {"loco,lodo"}
    assert len({spec.comparison_key for spec in specs}) == 1
    assert len({spec.idempotency_key for spec in specs}) == 1


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
