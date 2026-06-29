import base64
import json
import pickle
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd

from sohbench.api_server import create_handler
from sohbench.evaluation_service import EvaluationService


def _dataset_bytes():
    rows = []
    for cell in ["c1", "c2", "c3"]:
        for cycle in range(10):
            rows.append(
                {
                    "dataset": "api_demo",
                    "cell_id": cell,
                    "cycle_index": cycle,
                    "soh": 1.0 - 0.01 * cycle,
                    "voltage_mean": 3.4 - 0.01 * cycle,
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


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


def _post_json(base_url, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(base_url, path):
    with urllib.request.urlopen(base_url + path, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _serve(service):
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_api_upload_catalog_and_job_flow(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        health = _get_json(base_url, "/api/health")
        assert health["ok"] is True

        uploaded = _post_json(
            base_url,
            "/api/upload/dataset",
            {
                "name": "api_demo",
                "filename": "api_demo.csv",
                "contentBase64": base64.b64encode(_dataset_bytes()).decode("ascii"),
            },
        )
        assert uploaded["resource"]["name"] == "api_demo"

        catalog = _get_json(base_url, "/api/catalog")
        assert any(item["name"] == "api_demo" for item in catalog["datasets"])

        job = _post_json(
            base_url,
            "/api/jobs",
            {
                "datasets": ["api_demo"],
                "estimators": ["ridge"],
                "featuresets": ["full"],
                "protocols": ["loco"],
                "seeds": [0],
            },
        )
        assert job["status"] in {"queued", "running", "succeeded"}

        deadline = time.time() + 20
        status = job
        while time.time() < deadline and status["status"] not in {"succeeded", "failed"}:
            time.sleep(0.2)
            status = _get_json(base_url, f"/api/jobs/{job['id']}")

        assert status["status"] == "succeeded"
    finally:
        server.shutdown()


def test_api_returns_400_for_invalid_dataset_upload(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        try:
            _post_json(
                base_url,
                "/api/upload/dataset",
                {
                    "name": "bad",
                    "filename": "bad.csv",
                    "contentBase64": base64.b64encode(b"dataset,soh\nbad,1.0\n").decode("ascii"),
                },
            )
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read().decode("utf-8"))
            assert "missing required columns" in body["error"]
        else:
            raise AssertionError("invalid upload unexpectedly succeeded")
    finally:
        server.shutdown()


def test_api_upload_model_and_start_inference_job(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        _post_json(
            base_url,
            "/api/upload/dataset",
            {
                "name": "api_demo",
                "filename": "api_demo.csv",
                "contentBase64": base64.b64encode(_dataset_bytes()).decode("ascii"),
            },
        )
        manifest = _model_manifest("api_model")
        uploaded = _post_json(
            base_url,
            "/api/upload/model",
            {
                "name": "api_model",
                "modelFilename": "api_model.pkl",
                "modelContentBase64": base64.b64encode(pickle.dumps(PickledPredictOnly())).decode("ascii"),
                "manifestFilename": "model_manifest.json",
                "manifestContentBase64": base64.b64encode(
                    json.dumps(manifest).encode("utf-8")
                ).decode("ascii"),
            },
        )
        assert uploaded["resource"]["name"] == "api_model"

        models = _get_json(base_url, "/api/models")
        assert any(item["name"] == "api_model" for item in models["models"])

        job = _post_json(
            base_url,
            "/api/inference-jobs",
            {"datasets": ["api_demo"], "models": ["api_model"]},
        )
        deadline = time.time() + 20
        status = job
        while time.time() < deadline and status["status"] not in {"succeeded", "failed"}:
            time.sleep(0.2)
            status = _get_json(base_url, f"/api/inference-jobs/{job['id']}")
        assert status["status"] == "succeeded"
    finally:
        server.shutdown()
