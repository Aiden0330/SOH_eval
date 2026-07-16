import base64
import json
import pickle
import threading
import time
import urllib.parse
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


def _write_native_dashboard_fixture(root):
    dashboard = root / "dashboard"
    data = dashboard / "data"
    data.mkdir(parents=True)
    (dashboard / "index.html").write_text(
        """
<!doctype html>
<html><head><style>footer { color:var(--dim); font-size:12px; margin-top:14px; }</style></head>
<body>
<span class="modetabs"><button data-mode="soh">SOH</button></span>
<script>
const $ = id => document.getElementById(id);
(async () => {
  await setMode("soh");
})();
</script>
</body></html>
""".strip(),
        encoding="utf-8",
    )
    (data / "manifest.json").write_text(
        json.dumps({"scenarios": [{"mode": "xjtu", "title": "XJTU"}]}),
        encoding="utf-8",
    )
    (data / "overview.json").write_text(json.dumps({"scenarios": []}), encoding="utf-8")
    (data / "xjtu.json").write_text(json.dumps({"points": []}), encoding="utf-8")
    (data / "xjtu_curves.json").write_text(
        json.dumps({"cell-1|1": {"t_s": [0], "voltage_v": [4.1]}}),
        encoding="utf-8",
    )
    (data / "ess_detection.json").write_text(json.dumps({"fleet": []}), encoding="utf-8")
    (data / "ess_detection_curves.json").write_text(json.dumps({"module": {}}), encoding="utf-8")
    return dashboard


def _write_minimal_frontend_app(root):
    app = root / "app"
    data = app / "data"
    data.mkdir(parents=True)
    (app / "index.html").write_text(
        """
<!doctype html>
<html><head><link rel="stylesheet" href="styles.css" /></head>
<body>
D:\\battery_SOH\\data
<section id="package-operations-card"></section>
<section id="package-report-card"></section>
<script type="module" src="app.js"></script>
</body></html>
""".strip(),
        encoding="utf-8",
    )
    (app / "styles.css").write_text("body{font-family:sans-serif}", encoding="utf-8")
    (app / "dashboard-core.mjs").write_text("export function datasetOptions(){ return []; }", encoding="utf-8")
    (app / "app.js").write_text(
        (
            "import { datasetOptions } from \"./dashboard-core.mjs\";\n"
            "console.log(datasetOptions(), 'ess_detection source guard');"
        ),
        encoding="utf-8",
    )
    (data / "leaderboard.json").write_text(json.dumps([]), encoding="utf-8")
    profile = data / "profiles" / "battery-soh-diagnostics"
    profile.mkdir(parents=True)
    (profile / "battery_dashboard_ess_detection.json").write_text(
        json.dumps({"dataset": "ess_detection"}),
        encoding="utf-8",
    )
    (profile / "battery_curve_previews.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"dataset": "ess_detection", "preview_curves": []},
                    {"dataset": "xjtu", "preview_curves": [{"id": "xjtu"}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    native = app / "battery-dashboard"
    native_data = native / "data"
    native_data.mkdir(parents=True)
    (native / "index.html").write_text("<html><body>SOH</body></html>", encoding="utf-8")
    (native_data / "manifest.json").write_text(json.dumps({"scenarios": [{"mode": "xjtu"}]}), encoding="utf-8")
    (native_data / "overview.json").write_text(json.dumps({"scenarios": []}), encoding="utf-8")
    (native_data / "xjtu.json").write_text(json.dumps({"points": []}), encoding="utf-8")
    (native_data / "xjtu_curves.json").write_text(json.dumps({"cell": {}}), encoding="utf-8")
    (native_data / "ess_detection.json").write_text(json.dumps({"fleet": []}), encoding="utf-8")
    (native_data / "ess_detection_curves.json").write_text(json.dumps({"module": {}}), encoding="utf-8")
    return app


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
        assert status["run_id"].startswith("RUN-")
        assert status["execution_state"] == "REVIEW"
        assert status["outcome"] in {"complete", "partial"}
        assert status["verdict"] == "not_reviewed"
        assert status["evidence_path"].endswith("evidence-pack.json")
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


def test_api_catalog_supports_lightweight_preview_query(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.model_store._catalog["models"] = [
        {"name": f"model_{idx}", "feature_columns": ["voltage_mean"]}
        for idx in range(4)
    ]
    for idx in range(2):
        service._save_job(
            {
                "id": f"job_{idx}",
                "status": "succeeded",
                "created_at": f"2026-07-01 00:0{idx}:00",
                "request": {"datasets": ["api_demo"], "estimators": ["ridge"], "featuresets": ["full"]},
                "saved_models": [f"model_{idx}_{item}" for item in range(10)],
            }
        )
    server, base_url = _serve(service)
    try:
        catalog = _get_json(base_url, "/api/catalog?model_limit=2&job_limit=1&compact_jobs=1")

        assert len(catalog["models"]) == 2
        assert catalog["model_count"] == 4
        assert catalog["models_limited"] is True
        assert len(catalog["jobs"]) == 1
        assert catalog["job_count"] == 2
        assert "saved_models" not in catalog["jobs"][0]
    finally:
        server.shutdown()


def test_api_serves_battery_curve_dataset_from_profile_manifest(tmp_path):
    frontend_data = tmp_path / "frontend_data"
    profile_dir = frontend_data / "profiles" / "battery-soh-diagnostics"
    curve_source = tmp_path / "battery" / "xjtu_curves.json"
    curve_source.parent.mkdir(parents=True, exist_ok=True)
    curve_source.write_text(
        json.dumps(
            {
                "cell-1|1": {"t_s": [0, 1], "voltage_v": [3.0, 3.1]},
                "cell-1|2": {"t_s": [0, 1], "voltage_v": [3.2, 3.3]},
            }
        ),
        encoding="utf-8",
    )
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "manifest.json").write_text(
        json.dumps(
            {
                "profile": "battery-soh-diagnostics",
                "curve_artifacts": [
                    {
                        "dataset": "xjtu",
                        "filename": "battery_dashboard_xjtu_curves.json",
                        "source": str(curve_source),
                        "bytes": curve_source.stat().st_size,
                        "copied": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=frontend_data,
    )
    server, base_url = _serve(service)
    try:
        payload = _get_json(base_url, "/api/battery-curves?profile=battery-soh-diagnostics&dataset=xjtu&limit=1")

        assert payload["ok"] is True
        assert payload["profile"] == "battery-soh-diagnostics"
        assert payload["dataset"] == "xjtu"
        assert payload["total_curves"] == 2
        assert payload["returned_curves"] == 1
        assert payload["truncated"] is True
        assert list(payload["curves"]) == ["cell-1|1"]
    finally:
        server.shutdown()


def test_api_syncs_native_dashboard_and_builds_offline_packages(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    source_dashboard = _write_native_dashboard_fixture(tmp_path / "battery")
    target_dashboard = tmp_path / "frontend" / "app" / "battery-dashboard"
    app_dir = _write_minimal_frontend_app(tmp_path / "frontend")
    portable_output = tmp_path / "dist" / "soh-eval-portable"
    standalone_output = tmp_path / "dist" / "soh-eval-standalone.html"
    server, base_url = _serve(service)
    try:
        sync = _post_json(
            base_url,
            "/api/sync/native-dashboard",
            {
                "source": str(source_dashboard),
                "target": str(target_dashboard),
                "include_curves": True,
            },
        )
        assert sync["ok"] is True
        assert sync["action"] == "sync_native_dashboard"
        assert sync["manifest"]["scenario_count"] == 1
        assert sync["manifest"]["include_soh_curves"] is True
        assert sync["manifest"]["excluded_detection_assets"] is True
        assert (target_dashboard / "data" / "xjtu_curves.json").exists()
        assert not (target_dashboard / "data" / "ess_detection.json").exists()

        portable = _post_json(
            base_url,
            "/api/package/portable",
            {
                "app_dir": str(app_dir),
                "output": str(portable_output),
            },
        )
        assert portable["ok"] is True
        assert portable["action"] == "package_portable"
        assert portable["manifest"]["portable"] is True
        assert portable["manifest"]["include_soh_curves"] is True
        assert portable["manifest"]["excluded_detection_assets"] is True
        assert (portable_output / "MANIFEST.json").exists()
        assert not (portable_output / "battery-dashboard" / "data" / "ess_detection.json").exists()
        assert not (portable_output / "battery-dashboard" / "data" / "ess_detection_curves.json").exists()
        assert not (
            portable_output
            / "data"
            / "profiles"
            / "battery-soh-diagnostics"
            / "battery_dashboard_ess_detection.json"
        ).exists()

        standalone = _post_json(
            base_url,
            "/api/package/standalone",
            {
                "app_dir": str(app_dir),
                "output": str(standalone_output),
            },
        )
        assert standalone["ok"] is True
        assert standalone["action"] == "package_standalone"
        assert standalone["manifest"]["output"] == str(standalone_output)
        assert standalone["manifest"]["artifact_count"] >= 2
        assert standalone["manifest"]["excluded_detection_assets"] is True
        html = standalone_output.read_text(encoding="utf-8")
        assert "window.__SOH_STANDALONE__ = true" in html
        assert "ess_detection source guard" in html
        assert "battery_dashboard_ess_detection.json" not in html
        assert "ess_detection_curves" not in html
    finally:
        server.shutdown()


def test_api_package_status_reports_delivery_readiness(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    source_dashboard = _write_native_dashboard_fixture(tmp_path / "battery")
    target_dashboard = tmp_path / "frontend" / "app" / "battery-dashboard"
    app_dir = _write_minimal_frontend_app(tmp_path / "frontend")
    portable_output = tmp_path / "dist" / "soh-eval-portable"
    standalone_output = tmp_path / "dist" / "soh-eval-standalone.html"
    server, base_url = _serve(service)
    try:
        _post_json(
            base_url,
            "/api/sync/native-dashboard",
            {
                "source": str(source_dashboard),
                "target": str(target_dashboard),
                "include_curves": True,
            },
        )
        _post_json(
            base_url,
            "/api/package/portable",
            {
                "app_dir": str(app_dir),
                "output": str(portable_output),
            },
        )
        _post_json(
            base_url,
            "/api/package/standalone",
            {
                "app_dir": str(app_dir),
                "output": str(standalone_output),
            },
        )
        query = urllib.parse.urlencode(
            {
                "native_dashboard": str(target_dashboard),
                "portable_output": str(portable_output),
                "standalone_output": str(standalone_output),
            }
        )

        payload = _get_json(base_url, f"/api/package/status?{query}")

        assert payload["ok"] is True
        report = payload["report"]
        assert report["summary"]["ready"] is True
        assert report["summary"]["failed"] == 0
        checks = {check["id"]: check for check in report["checks"]}
        assert checks["native_dashboard"]["passed"] is True
        assert checks["native_curves"]["passed"] is True
        assert checks["portable_package"]["passed"] is True
        assert checks["standalone_html"]["passed"] is True
        assert checks["detection_excluded"]["passed"] is True
        assert report["outputs"]["native_dashboard"]["scenario_count"] == 1
        assert report["outputs"]["portable"]["file_count"] >= 1
        assert report["outputs"]["standalone"]["artifact_count"] >= 2
        assert report["outputs"]["standalone"]["has_package_operations"] is True
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
        assert status["run_id"].startswith("RUN-")
        assert status["execution_state"] == "REVIEW"
        assert status["outcome"] == "complete"
        assert status["verdict"] == "not_reviewed"
        assert status["evidence_path"].endswith("evidence-pack.json")
    finally:
        server.shutdown()
