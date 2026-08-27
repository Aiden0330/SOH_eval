from __future__ import annotations

import argparse
import base64
import json
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .evaluation_service import EvaluationService
from .paths import DEFAULT_DATA_DIR
from .evaluation_cases import (
    EvaluationCaseConflictError,
    EvaluationCaseNotFoundError,
    EvaluationCaseValidationError,
)
from .model_store import ModelStoreError
from .trusted_plugins import TrustedPluginError
from .review_workflow import (
    ReviewWorkflowConflictError,
    ReviewWorkflowError,
    ReviewWorkflowNotFoundError,
)


DEFAULT_ALLOWED_ORIGINS = {
    # Documented quick-start: python -m http.server 8000 --directory frontend/app
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    # Repo-parent static serve used by the acceptance workflow docs
    "http://127.0.0.1:8765",
    "http://localhost:8765",
}

DANGEROUS_POST_PATHS = {
    "/api/upload/dataset",
    "/api/upload/algorithm",
    "/api/upload/featureset",
    "/api/upload/model",
    "/api/jobs",
    "/api/inference-jobs",
    "/api/search-jobs",
    "/api/examples/quickstart",
    "/api/import/battery-dashboard",
    "/api/import/battery-soh-plugins",
    "/api/sync/native-dashboard",
    "/api/package/portable",
    "/api/package/standalone",
}


def create_handler(
    service: EvaluationService,
    *,
    allowed_origins: set[str] | None = None,
    access_token: str | None = None,
    allow_dangerous_operations: bool = False,
):
    origins = set(allowed_origins or DEFAULT_ALLOWED_ORIGINS)
    token = access_token or secrets.token_urlsafe(32)

    class SohbenchApiHandler(BaseHTTPRequestHandler):
        server_version = "sohbench-api/0.1"

        def do_OPTIONS(self):  # noqa: N802
            if not self._origin_allowed():
                self._send_json({"error": "origin not allowed"}, status=HTTPStatus.FORBIDDEN)
                return
            self._send_json({"ok": True})

        def do_GET(self):  # noqa: N802
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)
                if not self._origin_allowed():
                    self._send_json({"error": "origin not allowed"}, status=HTTPStatus.FORBIDDEN)
                    return
                if path == "/api/health":
                    self._send_json({"ok": True, "mode": "trusted_local"})
                elif path == "/api/session":
                    self._send_json(
                        {
                            "access_token": token,
                            "legacy_execution_enabled": allow_dangerous_operations,
                        }
                    )
                elif path == "/api/catalog":
                    self._send_json(
                        service.catalog(
                            model_limit=_positive_int(query.get("model_limit", [None])[0]),
                            job_limit=_positive_int(query.get("job_limit", [None])[0]),
                            compact_jobs=_truthy(query.get("compact_jobs", ["0"])[0]),
                        )
                    )
                elif path == "/api/models":
                    self._send_json({"models": service.model_store.catalog()})
                elif path == "/api/profiles":
                    self._send_json(service.frontend_profile_index())
                elif path == "/api/examples":
                    self._send_json({"examples": service.list_examples()})
                elif path.startswith("/api/examples/"):
                    example_id = path.rsplit("/", 1)[-1]
                    self._send_json(service.get_example(example_id))
                elif path == "/api/battery-curves":
                    self._send_json(
                        service.battery_curve_dataset(
                            profile_name=query.get("profile", ["battery-soh-diagnostics"])[0],
                            dataset=query.get("dataset", [""])[0],
                            limit=_positive_int(query.get("limit", [None])[0]),
                        )
                    )
                elif path == "/api/package/status":
                    self._send_json(
                        {
                            "ok": True,
                            "report": service.package_status(
                                native_dashboard=_query_value(query, "native_dashboard"),
                                portable_output=_query_value(query, "portable_output"),
                                standalone_output=_query_value(query, "standalone_output"),
                            ),
                        }
                    )
                elif path == "/api/jobs":
                    self._send_json({"jobs": service.list_jobs()})
                elif path == "/api/inference-jobs":
                    self._send_json({"jobs": service.list_inference_jobs()})
                elif path == "/api/v2/cases":
                    self._send_json(
                        {
                            "cases": service.list_evaluation_cases(
                                profile_name=_query_value(query, "profile_name"),
                                lifecycle_state=_query_value(query, "lifecycle_state"),
                                include_archived=_truthy(
                                    query.get("include_archived", ["0"])[0]
                                ),
                                limit=_positive_int(query.get("limit", [None])[0]),
                            )
                        }
                    )
                elif path == "/api/v2/case-migration-errors":
                    self._send_json({"errors": service.evaluation_case_migration_errors()})
                elif path.startswith("/api/v2/cases/") and path.endswith("/review-workspace"):
                    case_id = _case_id_from_path(path, "/review-workspace")
                    context_binding_id = _query_value(query, "context_binding_id")
                    service.evaluation_cases.assert_context_binding(case_id, context_binding_id)
                    self._send_json(service.evaluation_case_review_workspace(
                        case_id,
                        expected_context_binding_id=context_binding_id,
                    ))
                elif path.startswith("/api/v2/cases/") and path.endswith("/evidence-workspace"):
                    case_id = _case_id_from_path(path, "/evidence-workspace")
                    overrides = {
                        key: _query_value(query, key)
                        for key in (
                            "comparison_id", "candidate_id", "dataset", "dataset_snapshot_id",
                            "split_assignment_id", "slice_id", "asset_id", "sample_id", "cycle_id",
                            "evidence_tab",
                        )
                    }
                    self._send_json(service.evaluation_case_evidence_workspace(
                        case_id,
                        overrides,
                        expected_context_binding_id=_query_value(query, "context_binding_id"),
                    ))
                elif path.startswith("/api/v2/cases/") and path.endswith("/prediction-record"):
                    case_id = _case_id_from_path(path, "/prediction-record")
                    self._send_json(
                        service.evaluation_case_prediction_record(
                            case_id,
                            sample_id=_query_value(query, "sample_id"),
                            expected_context_binding_id=_query_value(query, "context_binding_id"),
                        )
                    )
                elif path.startswith("/api/v2/cases/") and path.endswith("/trajectory"):
                    case_id = _case_id_from_path(path, "/trajectory")
                    self._send_json(
                        service.evaluation_case_prediction_trajectory(
                            case_id,
                            inference_job_id=_query_value(query, "inference_job_id"),
                            dataset=_query_value(query, "dataset"),
                            asset_id=_query_value(query, "asset_id"),
                            compact=_truthy(_query_value(query, "compact")),
                            candidate_id=_query_value(query, "candidate_id"),
                            model_artifact_id=_query_value(query, "model_artifact_id"),
                            expected_context_binding_id=_query_value(query, "context_binding_id"),
                        )
                    )
                elif path.startswith("/api/v2/prediction-records/") and path.endswith("/curves"):
                    case_id = _query_value(query, "case_id")
                    if not case_id:
                        raise EvaluationCaseValidationError("case_id is required")
                    record_id = unquote(_entity_id_from_path(path, "/curves"))
                    signal_text = _query_value(query, "signals") or ""
                    signals = [item.strip() for item in signal_text.split(",") if item.strip()]
                    self._send_json(
                        service.evaluation_prediction_record_curves(
                            case_id,
                            record_id,
                            signals=signals or None,
                            resolution=_query_value(query, "resolution") or "viewport",
                            max_points=_positive_int(query.get("max_points", [None])[0]),
                            expected_context_binding_id=_query_value(query, "context_binding_id"),
                        )
                    )
                elif path.startswith("/api/v2/cases/") and path.endswith("/events"):
                    case_id = _case_id_from_path(path, "/events")
                    self._send_json(
                        {
                            "events": service.evaluation_case_events(
                                case_id,
                                after_sequence=_positive_int(query.get("after_sequence", [0])[0]) or 0,
                            )
                        }
                    )
                elif path.startswith("/api/v2/cases/"):
                    case_id = _case_id_from_path(path)
                    self._send_json(service.get_evaluation_case(case_id))
                elif path.startswith("/api/inference-jobs/") and path.endswith("/artifacts"):
                    job_id = _entity_id_from_path(path, "/artifacts")
                    self._send_json(service.get_inference_job_artifacts(job_id))
                elif path.startswith("/api/jobs/") and path.endswith("/artifacts"):
                    job_id = _entity_id_from_path(path, "/artifacts")
                    self._send_json(service.get_job_artifacts(job_id))
                elif path.startswith("/api/inference-jobs/"):
                    job_id = path.rsplit("/", 1)[-1]
                    self._send_json(service.get_inference_job(job_id))
                elif path.startswith("/api/jobs/"):
                    job_id = path.rsplit("/", 1)[-1]
                    self._send_json(service.get_job(job_id))
                else:
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except EvaluationCaseNotFoundError:
                self._send_json({"error": "case not found"}, status=HTTPStatus.NOT_FOUND)
            except EvaluationCaseConflictError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
            except EvaluationCaseValidationError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except KeyError:
                self._send_json({"error": "job not found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):  # noqa: N802
            try:
                path = urlparse(self.path).path
                payload = self._read_json()
                if not self._origin_allowed():
                    self._send_json({"error": "origin not allowed"}, status=HTTPStatus.FORBIDDEN)
                    return
                if not self._token_allowed():
                    self._send_json({"error": "invalid local access token"}, status=HTTPStatus.UNAUTHORIZED)
                    return
                if path in DANGEROUS_POST_PATHS and not allow_dangerous_operations:
                    self._send_json(
                        {
                            "error": "legacy execution endpoint is disabled; start with --enable-legacy-execution for trusted local use"
                        },
                        status=HTTPStatus.FORBIDDEN,
                    )
                    return
                if path == "/api/upload/dataset":
                    resource = service.store.upload_dataset(
                        payload.get("name"),
                        payload.get("filename", ""),
                        _decode_content(payload),
                        replace=bool(payload.get("replace", False)),
                    )
                    self._send_json({"ok": True, "resource": resource})
                elif path == "/api/upload/algorithm":
                    resource = service.store.upload_estimator_plugin(
                        payload.get("name"),
                        payload.get("filename", ""),
                        _decode_content(payload),
                        replace=bool(payload.get("replace", False)),
                    )
                    self._send_json({"ok": True, "resource": resource})
                elif path == "/api/upload/featureset":
                    resource = service.store.upload_featureset_plugin(
                        payload.get("name"),
                        payload.get("filename", ""),
                        _decode_content(payload),
                        replace=bool(payload.get("replace", False)),
                    )
                    self._send_json({"ok": True, "resource": resource})
                elif path == "/api/upload/model":
                    resource = service.model_store.upload_model(
                        payload.get("name"),
                        payload.get("modelFilename", ""),
                        _decode_named_content(payload, "modelContentBase64"),
                        payload.get("manifestFilename", ""),
                        _decode_named_content(payload, "manifestContentBase64"),
                        replace=bool(payload.get("replace", False)),
                    )
                    self._send_json({"ok": True, "resource": resource})
                elif path == "/api/training/preflight":
                    self._send_json(service.training_preflight(payload))
                elif path == "/api/jobs":
                    _require_case_job_binding(payload)
                    self._send_json(service.create_job(payload, start=True), status=HTTPStatus.ACCEPTED)
                elif path == "/api/inference-jobs":
                    _require_case_job_binding(payload)
                    self._send_json(service.create_inference_job(payload, start=True), status=HTTPStatus.ACCEPTED)
                elif path == "/api/v2/cases":
                    self._send_json(service.create_evaluation_case(payload), status=HTTPStatus.CREATED)
                elif path == "/api/v2/cases/resolve":
                    result = service.resolve_evaluation_case(payload)
                    self._send_json(result, status=HTTPStatus.CREATED if result["created"] else HTTPStatus.OK)
                elif path.startswith("/api/v2/cases/") and path.endswith("/identity"):
                    case_id = _case_id_from_path(path, "/identity")
                    self._send_json(service.update_evaluation_case_identity(case_id, payload))
                elif path.startswith("/api/v2/cases/") and path.endswith("/saved-combinations/activate"):
                    case_id = _case_id_from_path(path, "/saved-combinations/activate")
                    self._send_json(
                        service.activate_evaluation_case_saved_combination(case_id, payload)
                    )
                elif path.startswith("/api/v2/cases/") and path.endswith("/saved-combinations"):
                    case_id = _case_id_from_path(path, "/saved-combinations")
                    self._send_json(
                        service.update_evaluation_case_saved_combinations(case_id, payload)
                    )
                elif path.startswith("/api/v2/cases/") and path.endswith("/evidence-selection"):
                    case_id = _case_id_from_path(path, "/evidence-selection")
                    self._send_json(service.update_evaluation_case_evidence(case_id, payload))
                elif path.startswith("/api/v2/cases/") and path.endswith("/lifecycle"):
                    case_id = _case_id_from_path(path, "/lifecycle")
                    self._send_json(service.transition_evaluation_case(case_id, payload))
                elif path.startswith("/api/v2/cases/") and path.endswith("/review"):
                    case_id = _case_id_from_path(path, "/review")
                    self._send_json(service.update_evaluation_case_review(case_id, payload))
                elif path == "/api/v2/review-actions/issues":
                    self._send_json(service.create_review_issue(payload), status=HTTPStatus.CREATED)
                elif path.startswith("/api/v2/review-actions/issues/") and path.endswith("/status"):
                    issue_id = _entity_id_from_path(path, "/status")
                    self._send_json(service.update_review_issue_status(issue_id, payload))
                elif path == "/api/v2/regression-sets":
                    self._send_json(service.create_regression_set(payload), status=HTTPStatus.CREATED)
                elif path.startswith("/api/v2/regression-sets/") and path.endswith("/members"):
                    regression_set_id = _entity_id_from_path(path, "/members")
                    self._send_json(
                        service.add_regression_member(regression_set_id, payload),
                        status=HTTPStatus.CREATED,
                    )
                elif path.startswith("/api/v2/cases/") and path.endswith("/retests"):
                    case_id = _case_id_from_path(path, "/retests")
                    self._send_json(service.create_case_retest(case_id, payload), status=HTTPStatus.ACCEPTED)
                elif path.startswith("/api/v2/cases/") and path.endswith("/reconcile"):
                    case_id = _case_id_from_path(path, "/reconcile")
                    self._send_json(service.reconcile_case_kpis(case_id, payload), status=HTTPStatus.CREATED)
                elif path.startswith("/api/v2/cases/") and path.endswith("/evidence-packs"):
                    case_id = _case_id_from_path(path, "/evidence-packs")
                    self._send_json(
                        service.create_case_evidence_pack(case_id, payload),
                        status=HTTPStatus.CREATED,
                    )
                elif path.startswith("/api/v2/evidence-packs/") and path.endswith("/sign"):
                    pack_id = _entity_id_from_path(path, "/sign")
                    self._send_json(service.sign_evidence_pack(pack_id, payload))
                elif path == "/api/examples/quickstart":
                    self._send_json(service.create_quickstart_example(start=True), status=HTTPStatus.ACCEPTED)
                elif path == "/api/import/battery-dashboard":
                    profile = service.import_battery_dashboard_profile(
                        source_dir=payload.get("source_dir"),
                        profile_name=payload.get("profile_name", "battery-soh-diagnostics"),
                        include_curves=bool(payload.get("include_curves", False)),
                    )
                    self._send_json({"ok": True, "profile": profile})
                elif path == "/api/import/battery-soh-plugins":
                    plugins = service.import_battery_soh_plugins(
                        battery_soh_root=payload.get("battery_soh_root", r"D:\battery_SOH"),
                        replace=bool(payload.get("replace", False)),
                    )
                    self._send_json({"ok": True, "plugins": plugins})
                elif path == "/api/sync/native-dashboard":
                    manifest = service.sync_native_dashboard(
                        source=payload.get("source"),
                        target=payload.get("target"),
                        include_curves=bool(payload.get("include_curves", True)),
                    )
                    self._send_json({"ok": True, "action": "sync_native_dashboard", "manifest": manifest})
                elif path == "/api/package/portable":
                    manifest = service.package_portable_frontend(
                        app_dir=payload.get("app_dir"),
                        output=payload.get("output"),
                    )
                    self._send_json({"ok": True, "action": "package_portable", "manifest": manifest})
                elif path == "/api/package/standalone":
                    manifest = service.package_standalone_frontend(
                        app_dir=payload.get("app_dir"),
                        output=payload.get("output"),
                    )
                    self._send_json({"ok": True, "action": "package_standalone", "manifest": manifest})
                else:
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except EvaluationCaseNotFoundError:
                self._send_json({"error": "case not found"}, status=HTTPStatus.NOT_FOUND)
            except ReviewWorkflowNotFoundError:
                self._send_json({"error": "workflow entity not found"}, status=HTTPStatus.NOT_FOUND)
            except EvaluationCaseConflictError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
            except ReviewWorkflowConflictError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
            except (
                EvaluationCaseValidationError,
                ReviewWorkflowError,
                TrustedPluginError,
                ModelStoreError,
                ValueError,
                KeyError,
            ) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, fmt, *args):
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            return json.loads(body.decode("utf-8"))

        def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            origin = self.headers.get("Origin", "")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type,X-SOHBench-Token")
            self.end_headers()
            self.wfile.write(data)

        def _origin_allowed(self) -> bool:
            origin = self.headers.get("Origin", "")
            return not origin or origin in origins

        def _token_allowed(self) -> bool:
            origin = self.headers.get("Origin", "")
            return not origin or secrets.compare_digest(
                self.headers.get("X-SOHBench-Token", ""), token
            )

    return SohbenchApiHandler


def _decode_content(payload: dict) -> bytes:
    encoded = payload.get("contentBase64")
    if not encoded:
        raise ValueError("contentBase64 is required")
    return base64.b64decode(encoded)


def _decode_named_content(payload: dict, key: str) -> bytes:
    encoded = payload.get(key)
    if not encoded:
        raise ValueError(f"{key} is required")
    return base64.b64decode(encoded)


def _positive_int(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, parsed)


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _query_value(query: dict, key: str) -> str | None:
    values = query.get(key) or []
    value = values[0] if values else None
    return value if value not in (None, "") else None


def _require_case_job_binding(payload: dict) -> None:
    if not str(payload.get("case_id") or "").strip():
        raise EvaluationCaseValidationError("case_id is required for workflow jobs")
    if not str(payload.get("expected_context_binding_id") or "").strip():
        raise EvaluationCaseValidationError(
            "expected_context_binding_id is required for workflow jobs"
        )


def _case_id_from_path(path: str, suffix: str = "") -> str:
    base = path[:-len(suffix)] if suffix and path.endswith(suffix) else path
    case_id = base.rstrip("/").rsplit("/", 1)[-1]
    if not case_id or case_id == "cases":
        raise EvaluationCaseValidationError("case id is required")
    return case_id


def _entity_id_from_path(path: str, suffix: str = "") -> str:
    base = path[:-len(suffix)] if suffix and path.endswith(suffix) else path
    entity_id = base.rstrip("/").rsplit("/", 1)[-1]
    if not entity_id:
        raise ValueError("entity id is required")
    return entity_id


def serve(
    *,
    host: str,
    port: int,
    data_dir: str | Path,
    work_dir: str | Path,
    runs_dir: str | Path,
    frontend_data_dir: str | Path,
    allowed_origins: set[str] | None = None,
    allow_dangerous_operations: bool = False,
) -> None:
    service = EvaluationService(
        data_dir=data_dir,
        work_dir=work_dir,
        runs_dir=runs_dir,
        frontend_data_dir=frontend_data_dir,
        allow_unsafe_plugins=allow_dangerous_operations,
    )
    httpd = ThreadingHTTPServer(
        (host, port),
        create_handler(
            service,
            allowed_origins=allowed_origins,
            allow_dangerous_operations=allow_dangerous_operations,
        ),
    )
    print(f"sohbench API listening on http://{host}:{port}", flush=True)
    print(f"data_dir={Path(data_dir)}", flush=True)
    httpd.serve_forever()


def parse_args(argv=None):
    parser = argparse.ArgumentParser("sohbench-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--work-dir", default="work/trusted_plugins")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--frontend-data-dir", default="frontend/app/data")
    parser.add_argument("--ui-origin", action="append", default=[])
    parser.add_argument("--enable-legacy-execution", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    serve(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        runs_dir=args.runs_dir,
        frontend_data_dir=args.frontend_data_dir,
        allowed_origins=set(args.ui_origin) if args.ui_origin else None,
        allow_dangerous_operations=args.enable_legacy_execution,
    )


if __name__ == "__main__":
    main()
