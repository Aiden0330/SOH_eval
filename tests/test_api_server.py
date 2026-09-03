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
import pytest

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


def _request_json(base_url, path, *, method="GET", payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(
        base_url + path, data=data, method=method, headers=request_headers
    )
    # Return the HTTPMessage itself so header lookups stay case-insensitive
    # (the legacy server title-cases header names, uvicorn lowercases them).
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.headers, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, json.loads(exc.read().decode("utf-8"))


def _get_json(base_url, path):
    with urllib.request.urlopen(base_url + path, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


# Every test in this module runs against both API implementations: the legacy
# stdlib server (sohbench.api_server) and its FastAPI replacement
# (sohbench.api).  The FastAPI server must stay a drop-in replacement — the V1
# frontend runs unmodified against it — so the whole suite doubles as the
# parity gate.
_API_IMPL = "legacy"


@pytest.fixture(autouse=True, params=["legacy", "fastapi"])
def api_impl(request):
    global _API_IMPL
    _API_IMPL = request.param
    yield request.param
    _API_IMPL = "legacy"


def _start_server(service, **kwargs):
    if _API_IMPL == "fastapi":
        from sohbench.api import create_app
        from sohbench.api.testing import ThreadedApiServer

        server = ThreadedApiServer(create_app(service, **kwargs))
        return server, server.base_url
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), create_handler(service, **kwargs)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _serve(service):
    return _start_server(service, allow_dangerous_operations=True)


def _serve_secure(service, *, token="test-token", dangerous=False):
    return _start_server(
        service,
        access_token=token,
        allow_dangerous_operations=dangerous,
    )


def _prepare_phase21_api_case(service, tmp_path):
    frontend_data = service.frontend_data_dir
    profile_dir = frontend_data / "profiles" / "phase21-api"
    profile_dir.mkdir(parents=True)
    (frontend_data / "run_config.json").write_text(
        json.dumps({"datasets": ["xjtu"], "protocols": ["loco"], "seeds": [0]}),
        encoding="utf-8",
    )
    (frontend_data / "leaderboard.json").write_text(
        json.dumps([{
            "estimator": "ridge",
            "featureset": "voltage",
            "metrics": {
                "mae": {"mean": 0.01},
                "picp": {"mean": 0.95},
                "monotonicity_violation_rate": {"mean": 0.0},
                "value_range_violation": {"mean": 0.0},
            },
        }]),
        encoding="utf-8",
    )
    (profile_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (profile_dir / "battery_dashboard_xjtu.json").write_text(
        json.dumps({
            "config_brief": {"model": "ridge voltage"},
            "points": [
                {
                    "cell_id": "b2c9", "cycle_index": cycle,
                    "true_soh": 0.81 - (cycle - 438) * 0.001,
                    "pred_soh": 0.809 - (cycle - 438) * 0.001,
                    "lower": 0.80 - (cycle - 438) * 0.001,
                    "upper": 0.82 - (cycle - 438) * 0.001,
                }
                for cycle in range(436, 441)
            ],
        }),
        encoding="utf-8",
    )
    curve_dir = frontend_data.parent / "battery-dashboard" / "data"
    curve_dir.mkdir(parents=True)
    (curve_dir / "xjtu_curves.json").write_text(
        json.dumps({"b2c9|438": {
            "t_s": [0.0, 360.0, 720.0, 1080.0, 1440.0],
            "voltage_v": [3.5, 3.7, 3.9, 4.1, 4.2],
            "current_a": [1.0, 1.0, 1.0, 1.0, 1.0],
        }}),
        encoding="utf-8",
    )
    (curve_dir / "xjtu.json").write_text(
        json.dumps({
            "feature_window": [3.7, 4.1],
            "feature_window_approx": True,
            "feature_columns": ["voltage_mean"],
        }),
        encoding="utf-8",
    )
    service.store.upload_dataset(
        "xjtu",
        "xjtu.csv",
        pd.DataFrame([
            {
                "dataset": "xjtu", "cell_id": "b2c9", "cycle_index": cycle,
                "soh": 0.81 - (cycle - 438) * 0.001,
                "voltage_mean": 3.50 - (cycle - 438) * 0.001,
            }
            for cycle in range(436, 441)
        ]).to_csv(index=False).encode("utf-8"),
    )
    model = service.model_store.upload_model(
        "phase21_api_model",
        "phase21_api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest("phase21_api_model")).encode("utf-8"),
    )
    case = service.create_evaluation_case({
        "profile_name": "phase21-api",
        "idempotency_key": "phase21:api-retest",
        "source_class": "measured",
        "execution": {"state": "completed", "run_id": "run-original"},
        "candidates": [{
            "candidate_id": "legacy:ridge:voltage", "model": "ridge", "featureset": "voltage",
            "model_artifact_id": model["artifact_id"],
        }],
    })
    return service.update_evaluation_case_evidence(case["id"], {
        "expected_version": case["version"],
        "expected_context_binding_id": case["context_binding"]["binding_id"],
        "criterion_set_id": "strict",
        "selection": {
            "candidate_id": "legacy:ridge:voltage", "dataset": "xjtu",
            "asset_id": "b2c9", "cycle_id": 438, "evidence_tab": "raw",
        },
    })


def test_api_exposes_canonical_prediction_record_detail(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    case = _prepare_phase21_api_case(service, tmp_path)
    profile_dir = service.frontend_data_dir / "profiles" / "phase21-api"
    record = {
        "prediction_record_id": "pred:xjtu:b2c9:438",
        "sample_id": "xjtu:b2c9:438",
        "candidate_id": case["candidates"][0]["candidate_id"],
        "model_artifact_id": case["candidates"][0]["model_artifact_id"],
        "dataset": "xjtu",
        "cell_id": "b2c9",
        "cycle_index": 438,
        "true_soh": 0.81,
        "pred_soh": 0.809,
        "featureset": "voltage",
        "feature_values": {"voltage_mean": 3.84},
    }
    (profile_dir / "prediction_records.json").write_text(
        json.dumps({"run_id": "run-original", "records": [record]}), encoding="utf-8"
    )
    (profile_dir / "metric_contributions.json").write_text(
        json.dumps({"records": [{"prediction_record_id": record["prediction_record_id"], "mae": 0.001}]}),
        encoding="utf-8",
    )
    (profile_dir / "curve_segment_refs.json").write_text(
        json.dumps({"records": [{"prediction_record_id": record["prediction_record_id"], "curve_key": "b2c9|438"}]}),
        encoding="utf-8",
    )
    server, base_url = _serve(service)
    try:
        missing_status, _, missing_body = _request_json(
            base_url,
            f"/api/v2/cases/{case['id']}/prediction-record?sample_id=xjtu%3Ab2c9%3A438",
        )
        assert missing_status == 409
        assert "expected_context_binding_id is required" in missing_body["error"]
        review_status, _, review_body = _request_json(
            base_url,
            f"/api/v2/cases/{case['id']}/review-workspace",
        )
        assert review_status == 409
        assert "expected_context_binding_id is required" in review_body["error"]
        review_workspace = _get_json(
            base_url,
            f"/api/v2/cases/{case['id']}/review-workspace"
            f"?context_binding_id={case['context_binding']['binding_id']}",
        )
        assert review_workspace["context_binding"]["binding_id"] == case["context_binding"]["binding_id"]
        detail = _get_json(
            base_url,
            f"/api/v2/cases/{case['id']}/prediction-record?sample_id=xjtu%3Ab2c9%3A438"
            f"&context_binding_id={case['context_binding']['binding_id']}",
        )
        assert detail["prediction_record"]["prediction_record_id"] == record["prediction_record_id"]
        assert detail["metric_contribution"]["mae"] == 0.001
        assert detail["curve_segment_ref"]["curve_key"] == "b2c9|438"
        curves = _get_json(
            base_url,
            f"/api/v2/prediction-records/pred%3Axjtu%3Ab2c9%3A438/curves"
            f"?case_id={case['id']}&signals=voltage,current&resolution=viewport"
            f"&context_binding_id={case['context_binding']['binding_id']}",
        )
        assert curves["prediction_record"]["sample_id"] == "xjtu:b2c9:438"
        assert curves["available_signals"] == ["voltage", "current"]
        assert curves["signals"]["voltage"]["source_points"] == 5
        assert curves["sampling"]["resolution"] == "viewport"
        assert curves["feature_window"]["cycle_time_s"] == [360.0, 1080.0]
        assert curves["feature_curves"][0]["name"] == "voltage_mean"
        assert curves["feature_curves"][0]["highlight"]["cycle_time_s"] == [0.0, 1440.0]
        trajectory = _get_json(
            base_url,
            f"/api/v2/cases/{case['id']}/trajectory?dataset=xjtu&asset_id=b2c9"
            f"&candidate_id={urllib.parse.quote(case['candidates'][0]['candidate_id'])}"
            f"&model_artifact_id={urllib.parse.quote(case['candidates'][0]['model_artifact_id'])}"
            f"&context_binding_id={case['context_binding']['binding_id']}",
        )
        assert trajectory["case_id"] == case["id"]
        assert trajectory["asset_id"] == "b2c9"
        assert trajectory["candidate_id"] == case["candidates"][0]["candidate_id"]
        assert trajectory["model_artifact_id"] == case["candidates"][0]["model_artifact_id"]
        assert trajectory["context_binding_id"] == case["context_binding"]["binding_id"]
        assert trajectory["points"][0]["prediction_record_id"] == record["prediction_record_id"]
        compact_trajectory = _get_json(
            base_url,
            f"/api/v2/cases/{case['id']}/trajectory?dataset=xjtu&asset_id=b2c9&compact=1"
            f"&candidate_id={urllib.parse.quote(case['candidates'][0]['candidate_id'])}"
            f"&model_artifact_id={urllib.parse.quote(case['candidates'][0]['model_artifact_id'])}"
            f"&context_binding_id={case['context_binding']['binding_id']}",
        )
        assert compact_trajectory["compact"] is True
        assert compact_trajectory["points"][0]["prediction_record_id"] == record["prediction_record_id"]
        assert "feature_values" not in compact_trajectory["points"][0]
    finally:
        server.shutdown()


def test_api_rejects_untrusted_browser_origin_and_requires_session_token(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve_secure(service)
    try:
        status, headers, body = _request_json(
            base_url,
            "/api/session",
            headers={"Origin": "http://127.0.0.1:8765"},
        )
        assert status == 200
        assert headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:8765"
        assert body == {
            "access_token": "test-token",
            "legacy_execution_enabled": False,
        }

        status, headers, body = _request_json(
            base_url,
            "/api/v2/cases",
            method="POST",
            payload={"profile_name": "p", "idempotency_key": "run:1"},
            headers={"Origin": "https://evil.example"},
        )
        assert status == 403
        assert "Access-Control-Allow-Origin" not in headers
        assert body["error"] == "origin not allowed"

        status, _, body = _request_json(
            base_url,
            "/api/v2/cases",
            method="POST",
            payload={"profile_name": "p", "idempotency_key": "run:1"},
            headers={"Origin": "http://127.0.0.1:8765"},
        )
        assert status == 401
        assert body["error"] == "invalid local access token"

        status, headers, body = _request_json(
            base_url,
            "/api/v2/cases",
            method="POST",
            payload={"profile_name": "p", "idempotency_key": "run:1"},
            headers={
                "Origin": "http://127.0.0.1:8765",
                "X-SOHBench-Token": "test-token",
            },
        )
        assert status == 201
        assert headers["Access-Control-Allow-Origin"] == "http://127.0.0.1:8765"
        assert body["idempotency_key"] == "run:1"
    finally:
        server.shutdown()


def test_api_disables_legacy_code_execution_endpoints_by_default(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve_secure(service)
    try:
        status, _, body = _request_json(
            base_url,
            "/api/upload/algorithm",
            method="POST",
            payload={"name": "unsafe", "filename": "unsafe.py", "contentBase64": "cHJpbnQoMSk="},
            headers={
                "Origin": "http://127.0.0.1:8765",
                "X-SOHBench-Token": "test-token",
            },
        )
        assert status == 403
        assert "legacy execution endpoint is disabled" in body["error"]
    finally:
        server.shutdown()


def test_secure_service_startup_does_not_import_uploaded_python(tmp_path):
    work = tmp_path / "work"
    estimators = work / "estimators"
    estimators.mkdir(parents=True)
    marker = tmp_path / "plugin-imported.txt"
    plugin = estimators / "unsafe.py"
    plugin.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('loaded')\n",
        encoding="utf-8",
    )
    (work / "catalog.json").write_text(
        json.dumps(
            {
                "datasets": [],
                "estimators": [{"name": "unsafe", "path": str(plugin)}],
                "featuresets": [],
            }
        ),
        encoding="utf-8",
    )

    EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=work,
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
        allow_unsafe_plugins=False,
    )

    assert not marker.exists()


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

        request = {
            "datasets": ["api_demo"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["loco"],
            "seeds": [0],
        }
        preflight = _post_json(base_url, "/api/training/preflight", request)
        assert preflight["ready"] is True
        assert preflight["dataset_snapshots"][0]["snapshot_id"].startswith("dataset@sha256:")

        case = _post_json(base_url, "/api/v2/cases", {
            "profile_name": "api-job-flow",
            "idempotency_key": "api-job-flow:case",
            "source_class": "measured",
        })
        job = _post_json(
            base_url,
            "/api/jobs",
            {
                **request,
                "preflight_id": preflight["preflight_id"],
                "case_id": case["id"],
                "expected_context_binding_id": case["context_binding"]["binding_id"],
            },
        )
        assert job["status"] in {"queued", "running", "succeeded"}
        assert job["preflight"]["preflight_id"] == preflight["preflight_id"]

        deadline = time.time() + 20
        status = job
        while time.time() < deadline and status["status"] not in {"succeeded", "failed"}:
            time.sleep(0.2)
            status = _get_json(base_url, f"/api/jobs/{job['id']}")

        assert status["status"] == "succeeded"
        assert status["result_summary"]["source_class"] == "measured"
        artifacts = _get_json(base_url, f"/api/jobs/{job['id']}/artifacts")
        assert artifacts["job_id"] == job["id"]
        assert artifacts["kind"] == "training"
        assert artifacts["artifacts"]["run_config"]["job_id"] == job["id"]
        assert artifacts["artifacts"]["leaderboard"]
    finally:
        server.shutdown()


def test_api_persists_evaluation_case_and_evidence_context(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        payload = {
            "profile_name": "battery-soh-diagnostics",
            "idempotency_key": "profile:battery-soh-diagnostics|run:run-01",
            "title": "电池 SOH 诊断案例",
            "source_class": "measured",
            "execution": {"state": "completed", "run_id": "run-01"},
            "candidates": [
                {
                    "candidate_id": "legacy:ridge:voltage",
                    "model": "ridge",
                    "featureset": "voltage",
                }
            ],
        }
        resolved = _post_json(base_url, "/api/v2/cases/resolve", payload)
        resolved_again = _post_json(base_url, "/api/v2/cases/resolve", payload)
        case = resolved["case"]

        assert resolved["created"] is True
        assert resolved_again["created"] is False
        assert resolved_again["case"]["id"] == case["id"]

        updated = _post_json(
            base_url,
            f"/api/v2/cases/{case['id']}/evidence-selection",
            {
                "expected_version": case["version"],
                "expected_context_binding_id": case["context_binding"]["binding_id"],
                "selection": {
                    "candidate_id": "legacy:ridge:voltage",
                    "dataset": "xjtu",
                    "asset_id": "batch-1:cell-1",
                    "sample_id": "batch-1:cell-1|236",
                    "cycle_id": 236,
                    "evidence_tab": "raw",
                },
                "criterion_set_id": "strict",
            },
        )
        assert updated["version"] == case["version"] + 1
        assert updated["evidence_selection"]["sample_id"] == "batch-1:cell-1|236"
        assert updated["evidence_coverage"]["state"] == "incomplete"
        assert updated["verdict"]["result"] == "pending"

        listed = _get_json(
            base_url,
            "/api/v2/cases?profile_name=battery-soh-diagnostics&limit=1",
        )
        assert [item["id"] for item in listed["cases"]] == [case["id"]]

        workspace = _get_json(
            base_url,
            f"/api/v2/cases/{case['id']}/evidence-workspace?cycle_id=438&evidence_tab=icdv&context_binding_id={updated['context_binding']['binding_id']}",
        )
        assert workspace["context"]["cycle_id"] == "438"
        assert workspace["context"]["evidence_tab"] == "icdv"
        assert workspace["status_axes"] == {
            "execution": "completed",
            "evidence": "incomplete",
            "verdict": "pending",
            "review": "not_started",
        }

        events = _get_json(base_url, f"/api/v2/cases/{case['id']}/events")
        assert [event["event_type"] for event in events["events"]] == [
            "case.created", "evidence.pinned"
        ]

        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(
                base_url,
                f"/api/v2/cases/{case['id']}/evidence-selection",
                {
                    "expected_version": updated["version"],
                    "expected_context_binding_id": updated["context_binding"]["binding_id"],
                    "coverage": {"items": []},
                    "verdict": {"result": "pass"},
                },
            )
        assert error.value.code == 400
    finally:
        server.shutdown()


def test_api_rejects_client_fabricated_training_and_inference_lineage(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        case = _post_json(
            base_url,
            "/api/v2/cases",
            {
                "profile_name": "pipeline-profile",
                "idempotency_key": "pipeline:api-case",
                "source_class": "measured",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            _post_json(
                base_url,
                f"/api/v2/cases/{case['id']}/identity",
                {
                    "expected_version": case["version"],
                    "expected_context_binding_id": case["context_binding"]["binding_id"],
                    "execution": {
                        "state": "completed",
                        "training_job_id": "train-01",
                        "run_id": "train-01",
                    },
                    "candidates": [{
                        "candidate_id": "artifact:ridge-voltage",
                        "model": "ridge",
                        "featureset": "voltage",
                        "role": "champion",
                        "model_artifact_id": "model@sha256:abc123",
                        "run_id": "train-01",
                    }],
                },
            )
        assert error.value.code in {400, 404}
        unchanged = _get_json(base_url, f"/api/v2/cases/{case['id']}")
        assert unchanged["execution"]["training_job_id"] == ""
        assert unchanged["candidates"] == []
        events = _get_json(base_url, f"/api/v2/cases/{case['id']}/events")
        assert [event["event_type"] for event in events["events"]] == ["case.created"]
    finally:
        server.shutdown()


def test_api_exposes_phase2_review_workspace_and_blocks_premature_signing(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        case = _post_json(
            base_url,
            "/api/v2/cases",
            {
                "profile_name": "phase2-profile",
                "idempotency_key": "phase2:api-workspace",
                "source_class": "measured",
                "execution": {"state": "completed", "run_id": "run-phase2"},
            },
        )
        issue = _post_json(
            base_url,
            "/api/v2/review-actions/issues",
            {
                "case_id": case["id"],
                "expected_case_version": case["version"],
                "title": "缺少可复测的精确 Cycle",
                "severity": "major",
                "actor": "api-reviewer",
            },
        )
        regression_set = _post_json(
            base_url,
            "/api/v2/regression-sets",
            {
                "issue_id": issue["id"],
                "expected_case_version": case["version"],
                "actor": "api-reviewer",
            },
        )

        workspace = _get_json(
            base_url,
            f"/api/v2/cases/{case['id']}/review-workspace"
            f"?context_binding_id={case['context_binding']['binding_id']}",
        )
        assert workspace["issues"][0]["id"] == issue["id"]
        assert workspace["regression_sets"][0]["id"] == regression_set["id"]
        assert workspace["signing_readiness"]["can_sign"] is False
        assert "required evidence is incomplete" in workspace["signing_readiness"]["blockers"]

        pack = _post_json(
            base_url,
            f"/api/v2/cases/{case['id']}/evidence-packs",
            {
                "expected_case_version": case["version"],
                "actor": "api-reviewer",
            },
        )
        assert pack["state"] == "draft"

        status, _, body = _request_json(
            base_url,
            f"/api/v2/evidence-packs/{pack['id']}/sign",
            method="POST",
            payload={
                "expected_case_version": case["version"],
                "signer": "api-reviewer",
                "reason": "premature",
            },
        )
        assert status == 400
        assert "signing is blocked" in body["error"]
    finally:
        server.shutdown()


def test_api_phase21_retest_is_accepted_and_auto_fills_controlled_result(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    case = _prepare_phase21_api_case(service, tmp_path)
    server, base_url = _serve(service)
    try:
        issue = _post_json(base_url, "/api/v2/review-actions/issues", {
            "case_id": case["id"],
            "expected_case_version": case["version"],
            "title": "cycle 438 requires controlled retest",
        })
        regression_set = _post_json(base_url, "/api/v2/regression-sets", {
            "issue_id": issue["id"],
            "expected_case_version": case["version"],
        })
        _post_json(
            base_url,
            f"/api/v2/regression-sets/{regression_set['id']}/members",
            {"expected_case_version": case["version"]},
        )

        status, _, retest = _request_json(
            base_url,
            f"/api/v2/cases/{case['id']}/retests",
            method="POST",
            payload={
                "expected_case_version": case["version"],
                "issue_id": issue["id"],
                "regression_set_id": regression_set["id"],
            },
        )

        assert status == 202
        assert retest["controlled_run"]["status"] == "queued"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            workspace = _get_json(
                base_url,
                f"/api/v2/cases/{case['id']}/review-workspace"
                f"?context_binding_id={case['context_binding']['binding_id']}",
            )
            finished = next(item for item in workspace["retests"] if item["id"] == retest["id"])
            if finished["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        assert finished["status"] == "completed"
        assert finished["run_id"].startswith("RTR-")
        assert finished["result"]["runner"] == "controlled_model_reinference/v1"
        assert finished["result"]["execution_mode"] == "model_reinference"
        assert finished["result"]["prediction_recomputed"] is True
        assert finished["result"]["metrics"]["mae"] == pytest.approx(0.14)
        assert finished["result"]["result_digest"].startswith("sha256:")
    finally:
        server.shutdown()


def test_api_adds_a_bounded_neighboring_cycle_batch_to_a_regression_set(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    case = _prepare_phase21_api_case(service, tmp_path)
    server, base_url = _serve(service)
    try:
        issue = _post_json(base_url, "/api/v2/review-actions/issues", {
            "case_id": case["id"],
            "expected_case_version": case["version"],
            "title": "batch neighboring cycles",
        })
        regression_set = _post_json(base_url, "/api/v2/regression-sets", {
            "issue_id": issue["id"],
            "expected_case_version": case["version"],
        })
        batch = _post_json(
            base_url,
            f"/api/v2/regression-sets/{regression_set['id']}/members",
            {"expected_case_version": case["version"], "neighbor_radius": 2},
        )

        assert batch["neighbor_radius"] == 2
        assert batch["added_count"] == 5
        assert [item["cycle_id"] for item in batch["members"]] == [436, 437, 438, 439, 440]
        workspace = _get_json(
            base_url,
            f"/api/v2/cases/{case['id']}/review-workspace"
            f"?context_binding_id={case['context_binding']['binding_id']}",
        )
        assert len(workspace["regression_sets"][0]["members"]) == 5
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


def test_running_api_hot_refreshes_profiles_jobs_and_cases_from_another_service(tmp_path):
    paths = {
        "data_dir": tmp_path / "data",
        "work_dir": tmp_path / "work",
        "runs_dir": tmp_path / "runs",
        "frontend_data_dir": tmp_path / "frontend_data",
    }
    service = EvaluationService(**paths)
    server, base_url = _serve(service)
    try:
        producer = EvaluationService(**paths)
        producer.publish_frontend_profile(
            "hot-profile",
            {
                "label": "运行中新增 Profile",
                "kind": "production_hot_refresh",
                "datasets": ["nasa"],
                "training_job_id": "hot-train-01",
                "inference_job_id": "hot-infer-01",
            },
        )
        producer._save_job({
            "id": "hot-train-01",
            "kind": "training",
            "status": "succeeded",
            "created_at": "2026-08-15T00:00:00+00:00",
            "request": {"datasets": ["nasa"]},
        })
        producer._save_inference_job({
            "id": "hot-infer-01",
            "kind": "inference",
            "status": "succeeded",
            "created_at": "2026-08-15T00:01:00+00:00",
            "request": {"datasets": ["nasa"]},
        })
        case = producer.resolve_evaluation_case({
            "profile_name": "hot-profile",
            "idempotency_key": "hot-refresh:case-01",
            "source_class": "measured",
        })["case"]

        profiles = _get_json(base_url, "/api/profiles")
        assert any(item["name"] == "hot-profile" for item in profiles["profiles"])
        assert _get_json(base_url, "/api/jobs/hot-train-01")["status"] == "succeeded"
        assert _get_json(base_url, "/api/inference-jobs/hot-infer-01")["status"] == "succeeded"
        cases = _get_json(base_url, "/api/v2/cases?profile_name=hot-profile")
        assert [item["id"] for item in cases["cases"]] == [case["id"]]
        assert cases["cases"][0]["lifecycle_state"] == "draft"

        archived = producer.transition_evaluation_case(case["id"], {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
            "lifecycle_state": "archived",
            "actor": "test",
        })
        assert archived["lifecycle_state"] == "archived"
        assert _get_json(base_url, "/api/v2/cases?profile_name=hot-profile")["cases"] == []
        history = _get_json(
            base_url,
            "/api/v2/cases?profile_name=hot-profile&include_archived=1",
        )
        assert [item["id"] for item in history["cases"]] == [case["id"]]
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

        case = _post_json(
            base_url,
            "/api/v2/cases",
            {
                "profile_name": "uploaded-model-inference",
                "idempotency_key": "uploaded-model-inference:case",
                "source_class": "measured",
                "candidates": [{
                    "candidate_id": "artifact:api_model",
                    "model": "api_model",
                    "featureset": "full",
                    "model_artifact_id": uploaded["resource"]["artifact_id"],
                    "role": "candidate",
                }],
            },
        )
        job = _post_json(
            base_url,
            "/api/inference-jobs",
            {
                "datasets": ["api_demo"],
                "models": ["api_model"],
                "case_id": case["id"],
                "expected_context_binding_id": case["context_binding"]["binding_id"],
            },
        )
        deadline = time.time() + 20
        status = job
        while time.time() < deadline and status["status"] not in {"succeeded", "failed"}:
            time.sleep(0.2)
            status = _get_json(base_url, f"/api/inference-jobs/{job['id']}")
        assert status["status"] == "succeeded"
        artifacts = _get_json(base_url, f"/api/inference-jobs/{job['id']}/artifacts")
        assert artifacts["job_id"] == job["id"]
        assert artifacts["kind"] == "inference"
        assert artifacts["artifacts"]["inference_run_config"]["job_id"] == job["id"]
        assert artifacts["artifacts"]["inference_benchmark"]["records"][0]["dataset"] == "api_demo"
    finally:
        server.shutdown()


def test_api_starts_and_reports_quickstart_example(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        example = _post_json(base_url, "/api/examples/quickstart", {})
        assert example["status"] in {"queued", "running", "succeeded"}

        deadline = time.time() + 20
        status = example
        while time.time() < deadline and status["status"] not in {"succeeded", "failed"}:
            time.sleep(0.2)
            status = _get_json(base_url, f"/api/examples/{example['id']}")

        assert status["status"] == "succeeded"
        assert status["training_job_id"]
        assert status["inference_job_id"]
        listed = _get_json(base_url, "/api/examples")
        assert any(item["id"] == status["id"] for item in listed["examples"])
    finally:
        server.shutdown()
