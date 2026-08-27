"""FastAPI surface over :class:`sohbench.evaluation_service.EvaluationService`.

Drop-in replacement for the stdlib server in ``sohbench.api_server`` (B3 of
the redesign program).  The V1 frontend must run unmodified, so this module
reproduces the legacy wire contract exactly:

- error bodies are always ``{"error": "<message>"}``, never ``{"detail": ...}``
- the exception -> status mapping differs between GET and POST, as it did in
  the legacy ``do_GET``/``do_POST`` handlers
- CORS headers, the ``X-SOHBench-Token`` POST gate, and the
  ``--enable-legacy-execution`` gate behave identically
- handlers return raw service payloads through ``JSONResponse`` (no
  response-model filtering); the pydantic models in ``.schemas`` document the
  contract and drive OpenAPI/type generation only

Run with ``python -m sohbench.api`` (same flags as the legacy server).
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..api_server import (
    DANGEROUS_POST_PATHS,
    DEFAULT_ALLOWED_ORIGINS,
    _positive_int,
    _truthy,
    parse_args,
)
from ..evaluation_cases import (
    EvaluationCaseConflictError,
    EvaluationCaseNotFoundError,
    EvaluationCaseValidationError,
)
from ..evaluation_service import EvaluationService
from ..model_store import ModelStoreError
from ..review_workflow import (
    ReviewWorkflowConflictError,
    ReviewWorkflowError,
    ReviewWorkflowNotFoundError,
)
from ..trusted_plugins import TrustedPluginError
from . import schemas

LEGACY_EXECUTION_DISABLED_ERROR = (
    "legacy execution endpoint is disabled; "
    "start with --enable-legacy-execution for trusted local use"
)


def _opt(value: str | None) -> str | None:
    """Legacy ``_query_value`` semantics: empty string counts as absent."""
    return value if value not in (None, "") else None


def _json(payload, status: int = 200) -> JSONResponse:
    try:
        return JSONResponse(payload, status_code=status)
    except ValueError as exc:  # allow_nan=False parity with the legacy server
        return JSONResponse({"error": str(exc)}, status_code=500)


def _map_exception(exc: Exception, method: str) -> tuple[int, dict]:
    """Reproduce the legacy do_GET / do_POST exception -> response mapping."""
    if isinstance(exc, EvaluationCaseNotFoundError):
        return 404, {"error": "case not found"}
    if method == "POST":
        if isinstance(exc, ReviewWorkflowNotFoundError):
            return 404, {"error": "workflow entity not found"}
        if isinstance(exc, (EvaluationCaseConflictError, ReviewWorkflowConflictError)):
            return 409, {"error": str(exc)}
        if isinstance(
            exc,
            (
                EvaluationCaseValidationError,
                ReviewWorkflowError,
                TrustedPluginError,
                ModelStoreError,
                ValueError,
                KeyError,
            ),
        ):
            return 400, {"error": str(exc)}
        return 500, {"error": str(exc)}
    if isinstance(exc, EvaluationCaseConflictError):
        return 409, {"error": str(exc)}
    if isinstance(exc, EvaluationCaseValidationError):
        return 400, {"error": str(exc)}
    if isinstance(exc, KeyError):
        return 404, {"error": "job not found"}
    return 500, {"error": str(exc)}


def _require_case_job_binding(payload: dict) -> None:
    if not str(payload.get("case_id") or "").strip():
        raise EvaluationCaseValidationError("case_id is required for workflow jobs")
    if not str(payload.get("expected_context_binding_id") or "").strip():
        raise EvaluationCaseValidationError(
            "expected_context_binding_id is required for workflow jobs"
        )


def _dump(payload) -> dict:
    """Recover the raw request payload from an optional pydantic model."""
    if payload is None:
        return {}
    return payload.model_dump(exclude_unset=True)


def create_app(
    service: EvaluationService | None,
    *,
    allowed_origins: set[str] | None = None,
    access_token: str | None = None,
    allow_dangerous_operations: bool = False,
) -> FastAPI:
    origins = set(allowed_origins or DEFAULT_ALLOWED_ORIGINS)
    token = access_token or secrets.token_urlsafe(32)

    app = FastAPI(
        title="sohbench API",
        version="0.1.0",
        description="Battery SOH benchmark evaluation service",
    )

    def _apply_cors(response, origin: str) -> None:
        if origin in origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,X-SOHBench-Token"

    @app.middleware("http")
    async def _gates(request: Request, call_next):
        origin = request.headers.get("origin", "")

        def reply(payload: dict, status: int = 200) -> JSONResponse:
            response = _json(payload, status)
            _apply_cors(response, origin)
            return response

        if origin and origin not in origins:
            return reply({"error": "origin not allowed"}, 403)
        if request.method == "OPTIONS":
            return reply({"ok": True})
        if request.method == "POST":
            supplied = request.headers.get("x-sohbench-token", "")
            if origin and not secrets.compare_digest(supplied, token):
                return reply({"error": "invalid local access token"}, 401)
            if request.url.path in DANGEROUS_POST_PATHS and not allow_dangerous_operations:
                return reply({"error": LEGACY_EXECUTION_DISABLED_ERROR}, 403)
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - legacy mapped every exception
            status, payload = _map_exception(exc, request.method)
            return reply(payload, status)
        _apply_cors(response, origin)
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = first.get("msg", "invalid request body")
        detail = f"{location}: {message}" if location else message
        return _json({"error": detail}, 400)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return _json({"error": "not found"}, 404)
        return _json({"error": str(exc.detail)}, exc.status_code)

    err = {"model": schemas.ErrorResponse}

    # ------------------------------------------------------------------
    # System / catalog
    # ------------------------------------------------------------------

    @app.get("/api/health", responses={200: {"model": schemas.HealthResponse}})
    def health():
        return _json({"ok": True, "mode": "trusted_local"})

    @app.get("/api/session", responses={200: {"model": schemas.SessionResponse}})
    def session():
        return _json(
            {
                "access_token": token,
                "legacy_execution_enabled": allow_dangerous_operations,
            }
        )

    @app.get("/api/catalog", responses={200: {"model": schemas.CatalogResponse}, 500: err})
    def catalog(
        model_limit: str | None = None,
        job_limit: str | None = None,
        compact_jobs: str | None = "0",
    ):
        return _json(
            service.catalog(
                model_limit=_positive_int(model_limit),
                job_limit=_positive_int(job_limit),
                compact_jobs=_truthy(compact_jobs),
            )
        )

    @app.get("/api/models", responses={200: {"model": schemas.ModelsResponse}, 500: err})
    def models():
        return _json({"models": service.model_store.catalog()})

    @app.get("/api/profiles", responses={200: {"model": schemas.ProfileIndexResponse}, 500: err})
    def profiles():
        return _json(service.frontend_profile_index())

    @app.get("/api/examples", responses={200: {"model": schemas.ExamplesResponse}, 500: err})
    def examples():
        return _json({"examples": service.list_examples()})

    @app.get(
        "/api/examples/{example_id}",
        responses={200: {"model": schemas.ExampleRecord}, 404: err},
    )
    def example(example_id: str):
        return _json(service.get_example(example_id))

    @app.get(
        "/api/battery-curves",
        responses={200: {"model": schemas.BatteryCurvesResponse}, 400: err, 500: err},
    )
    def battery_curves(
        profile: str = "battery-soh-diagnostics",
        dataset: str = "",
        limit: str | None = None,
    ):
        return _json(
            service.battery_curve_dataset(
                profile_name=profile,
                dataset=dataset,
                limit=_positive_int(limit),
            )
        )

    @app.get(
        "/api/package/status",
        responses={200: {"model": schemas.PackageStatusResponse}, 500: err},
    )
    def package_status(
        native_dashboard: str | None = None,
        portable_output: str | None = None,
        standalone_output: str | None = None,
    ):
        return _json(
            {
                "ok": True,
                "report": service.package_status(
                    native_dashboard=_opt(native_dashboard),
                    portable_output=_opt(portable_output),
                    standalone_output=_opt(standalone_output),
                ),
            }
        )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    @app.get("/api/jobs", responses={200: {"model": schemas.JobsResponse}, 500: err})
    def jobs():
        return _json({"jobs": service.list_jobs()})

    @app.get("/api/inference-jobs", responses={200: {"model": schemas.JobsResponse}, 500: err})
    def inference_jobs():
        return _json({"jobs": service.list_inference_jobs()})

    @app.get(
        "/api/inference-jobs/{job_id}/artifacts",
        responses={200: {"model": schemas.JobArtifactsResponse}, 404: err},
    )
    def inference_job_artifacts(job_id: str):
        return _json(service.get_inference_job_artifacts(job_id))

    @app.get(
        "/api/jobs/{job_id}/artifacts",
        responses={200: {"model": schemas.JobArtifactsResponse}, 404: err},
    )
    def job_artifacts(job_id: str):
        return _json(service.get_job_artifacts(job_id))

    @app.get(
        "/api/inference-jobs/{job_id}",
        responses={200: {"model": schemas.JobRecord}, 404: err},
    )
    def inference_job(job_id: str):
        return _json(service.get_inference_job(job_id))

    @app.get("/api/jobs/{job_id}", responses={200: {"model": schemas.JobRecord}, 404: err})
    def job(job_id: str):
        return _json(service.get_job(job_id))

    @app.post(
        "/api/training/preflight",
        responses={200: {"model": schemas.TrainingPreflightResponse}, 400: err},
    )
    def training_preflight(payload: schemas.TrainingPreflightRequest | None = None):
        return _json(service.training_preflight(_dump(payload)))

    @app.post(
        "/api/jobs",
        status_code=202,
        responses={202: {"model": schemas.JobRecord}, 400: err, 403: err},
    )
    def create_job(payload: schemas.TrainingJobRequest | None = None):
        data = _dump(payload)
        _require_case_job_binding(data)
        return _json(service.create_job(data, start=True), 202)

    @app.post(
        "/api/inference-jobs",
        status_code=202,
        responses={202: {"model": schemas.JobRecord}, 400: err, 403: err},
    )
    def create_inference_job(payload: schemas.InferenceJobRequest | None = None):
        data = _dump(payload)
        _require_case_job_binding(data)
        return _json(service.create_inference_job(data, start=True), 202)

    # ------------------------------------------------------------------
    # Agent search jobs
    # ------------------------------------------------------------------

    @app.get("/api/search-jobs", responses={200: {"model": schemas.SearchJobsResponse}, 500: err})
    def search_jobs():
        return _json({"jobs": service.list_search_jobs()})

    @app.get(
        "/api/search-jobs/{job_id}",
        responses={200: {"model": schemas.SearchJobRecord}, 404: err},
    )
    def search_job(job_id: str):
        return _json(service.get_search_job(job_id))

    @app.post(
        "/api/search-jobs",
        status_code=202,
        responses={202: {"model": schemas.SearchJobRecord}, 400: err, 403: err},
    )
    def create_search_job(payload: schemas.SearchJobRequest | None = None):
        return _json(service.create_search_job(_dump(payload), start=True), 202)

    # ------------------------------------------------------------------
    # Uploads / examples / import / packaging
    # ------------------------------------------------------------------

    def _decode_base64(payload, field: str) -> bytes:
        import base64

        encoded = getattr(payload, field, None) if payload is not None else None
        if not encoded:
            raise ValueError(f"{field} is required")
        return base64.b64decode(encoded)

    @app.post(
        "/api/upload/dataset",
        responses={200: {"model": schemas.UploadResponse}, 400: err, 403: err},
    )
    def upload_dataset(payload: schemas.UploadPluginRequest | None = None):
        body = payload or schemas.UploadPluginRequest()
        resource = service.store.upload_dataset(
            body.name,
            body.filename,
            _decode_base64(body, "contentBase64"),
            replace=bool(body.replace),
        )
        return _json({"ok": True, "resource": resource})

    @app.post(
        "/api/upload/algorithm",
        responses={200: {"model": schemas.UploadResponse}, 400: err, 403: err},
    )
    def upload_algorithm(payload: schemas.UploadPluginRequest | None = None):
        body = payload or schemas.UploadPluginRequest()
        resource = service.store.upload_estimator_plugin(
            body.name,
            body.filename,
            _decode_base64(body, "contentBase64"),
            replace=bool(body.replace),
        )
        return _json({"ok": True, "resource": resource})

    @app.post(
        "/api/upload/featureset",
        responses={200: {"model": schemas.UploadResponse}, 400: err, 403: err},
    )
    def upload_featureset(payload: schemas.UploadPluginRequest | None = None):
        body = payload or schemas.UploadPluginRequest()
        resource = service.store.upload_featureset_plugin(
            body.name,
            body.filename,
            _decode_base64(body, "contentBase64"),
            replace=bool(body.replace),
        )
        return _json({"ok": True, "resource": resource})

    @app.post(
        "/api/upload/model",
        responses={200: {"model": schemas.UploadResponse}, 400: err, 403: err},
    )
    def upload_model(payload: schemas.UploadModelRequest | None = None):
        body = payload or schemas.UploadModelRequest()
        resource = service.model_store.upload_model(
            body.name,
            body.modelFilename,
            _decode_base64(body, "modelContentBase64"),
            body.manifestFilename,
            _decode_base64(body, "manifestContentBase64"),
            replace=bool(body.replace),
        )
        return _json({"ok": True, "resource": resource})

    @app.post(
        "/api/examples/quickstart",
        status_code=202,
        responses={202: {"model": schemas.ExampleRecord}, 403: err},
    )
    def quickstart_example():
        return _json(service.create_quickstart_example(start=True), 202)

    @app.post(
        "/api/import/battery-dashboard",
        responses={200: {"model": schemas.ImportBatteryDashboardResponse}, 400: err, 403: err},
    )
    def import_battery_dashboard(
        payload: schemas.ImportBatteryDashboardRequest | None = None,
    ):
        body = payload or schemas.ImportBatteryDashboardRequest()
        profile = service.import_battery_dashboard_profile(
            source_dir=body.source_dir,
            profile_name=body.profile_name,
            include_curves=bool(body.include_curves),
        )
        return _json({"ok": True, "profile": profile})

    @app.post(
        "/api/import/battery-soh-plugins",
        responses={200: {"model": schemas.ImportBatterySohPluginsResponse}, 400: err, 403: err},
    )
    def import_battery_soh_plugins(
        payload: schemas.ImportBatterySohPluginsRequest | None = None,
    ):
        body = payload or schemas.ImportBatterySohPluginsRequest()
        plugins = service.import_battery_soh_plugins(
            battery_soh_root=body.battery_soh_root,
            replace=bool(body.replace),
        )
        return _json({"ok": True, "plugins": plugins})

    @app.post(
        "/api/sync/native-dashboard",
        responses={200: {"model": schemas.PackageActionResponse}, 400: err, 403: err},
    )
    def sync_native_dashboard(payload: schemas.SyncNativeDashboardRequest | None = None):
        body = payload or schemas.SyncNativeDashboardRequest()
        manifest = service.sync_native_dashboard(
            source=body.source,
            target=body.target,
            include_curves=bool(body.include_curves),
        )
        return _json({"ok": True, "action": "sync_native_dashboard", "manifest": manifest})

    @app.post(
        "/api/package/portable",
        responses={200: {"model": schemas.PackageActionResponse}, 400: err, 403: err},
    )
    def package_portable(payload: schemas.PackageBuildRequest | None = None):
        body = payload or schemas.PackageBuildRequest()
        manifest = service.package_portable_frontend(app_dir=body.app_dir, output=body.output)
        return _json({"ok": True, "action": "package_portable", "manifest": manifest})

    @app.post(
        "/api/package/standalone",
        responses={200: {"model": schemas.PackageActionResponse}, 400: err, 403: err},
    )
    def package_standalone(payload: schemas.PackageBuildRequest | None = None):
        body = payload or schemas.PackageBuildRequest()
        manifest = service.package_standalone_frontend(app_dir=body.app_dir, output=body.output)
        return _json({"ok": True, "action": "package_standalone", "manifest": manifest})

    # ------------------------------------------------------------------
    # Evaluation cases (v2)
    # ------------------------------------------------------------------

    @app.get("/api/v2/cases", responses={200: {"model": schemas.CasesResponse}, 500: err})
    def list_cases(
        profile_name: str | None = None,
        lifecycle_state: str | None = None,
        include_archived: str | None = "0",
        limit: str | None = None,
    ):
        return _json(
            {
                "cases": service.list_evaluation_cases(
                    profile_name=_opt(profile_name),
                    lifecycle_state=_opt(lifecycle_state),
                    include_archived=_truthy(include_archived),
                    limit=_positive_int(limit),
                )
            }
        )

    @app.get(
        "/api/v2/case-migration-errors",
        responses={200: {"model": schemas.CaseMigrationErrorsResponse}, 500: err},
    )
    def case_migration_errors():
        return _json({"errors": service.evaluation_case_migration_errors()})

    @app.get(
        "/api/v2/cases/{case_id}/review-workspace",
        responses={200: {"model": schemas.ReviewWorkspaceResponse}, 404: err, 409: err},
    )
    def case_review_workspace(case_id: str, context_binding_id: str | None = None):
        binding_id = _opt(context_binding_id)
        service.evaluation_cases.assert_context_binding(case_id, binding_id)
        return _json(
            service.evaluation_case_review_workspace(
                case_id,
                expected_context_binding_id=binding_id,
            )
        )

    @app.get(
        "/api/v2/cases/{case_id}/evidence-workspace",
        responses={200: {"model": schemas.CaseRecord}, 404: err, 409: err},
    )
    def case_evidence_workspace(
        case_id: str,
        comparison_id: str | None = None,
        candidate_id: str | None = None,
        dataset: str | None = None,
        dataset_snapshot_id: str | None = None,
        split_assignment_id: str | None = None,
        slice_id: str | None = None,
        asset_id: str | None = None,
        sample_id: str | None = None,
        cycle_id: str | None = None,
        evidence_tab: str | None = None,
        context_binding_id: str | None = None,
    ):
        overrides = {
            "comparison_id": _opt(comparison_id),
            "candidate_id": _opt(candidate_id),
            "dataset": _opt(dataset),
            "dataset_snapshot_id": _opt(dataset_snapshot_id),
            "split_assignment_id": _opt(split_assignment_id),
            "slice_id": _opt(slice_id),
            "asset_id": _opt(asset_id),
            "sample_id": _opt(sample_id),
            "cycle_id": _opt(cycle_id),
            "evidence_tab": _opt(evidence_tab),
        }
        return _json(
            service.evaluation_case_evidence_workspace(
                case_id,
                overrides,
                expected_context_binding_id=_opt(context_binding_id),
            )
        )

    @app.get(
        "/api/v2/cases/{case_id}/prediction-record",
        responses={200: {"model": schemas.PredictionRecordCurvesResponse}, 404: err, 409: err},
    )
    def case_prediction_record(
        case_id: str,
        sample_id: str | None = None,
        context_binding_id: str | None = None,
    ):
        return _json(
            service.evaluation_case_prediction_record(
                case_id,
                sample_id=_opt(sample_id),
                expected_context_binding_id=_opt(context_binding_id),
            )
        )

    @app.get(
        "/api/v2/cases/{case_id}/trajectory",
        responses={200: {"model": schemas.TrajectoryResponse}, 400: err, 404: err, 409: err},
    )
    def case_trajectory(
        case_id: str,
        inference_job_id: str | None = None,
        dataset: str | None = None,
        asset_id: str | None = None,
        compact: str | None = None,
        candidate_id: str | None = None,
        model_artifact_id: str | None = None,
        context_binding_id: str | None = None,
    ):
        return _json(
            service.evaluation_case_prediction_trajectory(
                case_id,
                inference_job_id=_opt(inference_job_id),
                dataset=_opt(dataset),
                asset_id=_opt(asset_id),
                compact=_truthy(_opt(compact)),
                candidate_id=_opt(candidate_id),
                model_artifact_id=_opt(model_artifact_id),
                expected_context_binding_id=_opt(context_binding_id),
            )
        )

    @app.get(
        "/api/v2/prediction-records/{record_id}/curves",
        responses={200: {"model": schemas.PredictionRecordCurvesResponse}, 400: err, 404: err},
    )
    def prediction_record_curves(
        record_id: str,
        case_id: str | None = None,
        signals: str | None = None,
        resolution: str | None = None,
        max_points: str | None = None,
        context_binding_id: str | None = None,
    ):
        bound_case_id = _opt(case_id)
        if not bound_case_id:
            raise EvaluationCaseValidationError("case_id is required")
        signal_list = [item.strip() for item in (signals or "").split(",") if item.strip()]
        return _json(
            service.evaluation_prediction_record_curves(
                bound_case_id,
                record_id,
                signals=signal_list or None,
                resolution=_opt(resolution) or "viewport",
                max_points=_positive_int(max_points),
                expected_context_binding_id=_opt(context_binding_id),
            )
        )

    @app.get(
        "/api/v2/cases/{case_id}/events",
        responses={200: {"model": schemas.CaseEventsResponse}, 404: err},
    )
    def case_events(case_id: str, after_sequence: str | None = None):
        return _json(
            {
                "events": service.evaluation_case_events(
                    case_id,
                    after_sequence=_positive_int(after_sequence) or 0,
                )
            }
        )

    @app.get(
        "/api/v2/cases/{case_id}",
        responses={200: {"model": schemas.CaseRecord}, 404: err},
    )
    def get_case(case_id: str):
        return _json(service.get_evaluation_case(case_id))

    @app.post(
        "/api/v2/cases",
        status_code=201,
        responses={201: {"model": schemas.CaseRecord}, 400: err, 409: err},
    )
    def create_case(payload: schemas.CreateCaseRequest | None = None):
        return _json(service.create_evaluation_case(_dump(payload)), 201)

    @app.post(
        "/api/v2/cases/resolve",
        responses={
            200: {"model": schemas.ResolveCaseResponse},
            201: {"model": schemas.ResolveCaseResponse},
            400: err,
        },
    )
    def resolve_case(payload: schemas.CreateCaseRequest | None = None):
        result = service.resolve_evaluation_case(_dump(payload))
        return _json(result, 201 if result["created"] else 200)

    @app.post(
        "/api/v2/cases/{case_id}/identity",
        responses={200: {"model": schemas.CaseRecord}, 400: err, 404: err, 409: err},
    )
    def update_case_identity(case_id: str, payload: schemas.CaseIdentityRequest | None = None):
        return _json(service.update_evaluation_case_identity(case_id, _dump(payload)))

    @app.post(
        "/api/v2/cases/{case_id}/saved-combinations/activate",
        responses={200: {"model": schemas.CaseRecord}, 400: err, 404: err, 409: err},
    )
    def activate_saved_combination(
        case_id: str, payload: schemas.ActivateSavedCombinationRequest | None = None
    ):
        return _json(service.activate_evaluation_case_saved_combination(case_id, _dump(payload)))

    @app.post(
        "/api/v2/cases/{case_id}/saved-combinations",
        responses={200: {"model": schemas.CaseRecord}, 400: err, 404: err, 409: err},
    )
    def update_saved_combinations(
        case_id: str, payload: schemas.SavedCombinationsRequest | None = None
    ):
        return _json(service.update_evaluation_case_saved_combinations(case_id, _dump(payload)))

    @app.post(
        "/api/v2/cases/{case_id}/evidence-selection",
        responses={200: {"model": schemas.CaseRecord}, 400: err, 404: err, 409: err},
    )
    def update_case_evidence(
        case_id: str, payload: schemas.EvidenceSelectionRequest | None = None
    ):
        return _json(service.update_evaluation_case_evidence(case_id, _dump(payload)))

    @app.post(
        "/api/v2/cases/{case_id}/lifecycle",
        responses={200: {"model": schemas.CaseRecord}, 400: err, 404: err, 409: err},
    )
    def transition_case(case_id: str, payload: schemas.LifecycleRequest | None = None):
        return _json(service.transition_evaluation_case(case_id, _dump(payload)))

    @app.post(
        "/api/v2/cases/{case_id}/review",
        responses={200: {"model": schemas.CaseRecord}, 400: err, 404: err, 409: err},
    )
    def update_case_review(case_id: str, payload: schemas.CaseReviewRequest | None = None):
        return _json(service.update_evaluation_case_review(case_id, _dump(payload)))

    # ------------------------------------------------------------------
    # Review workflow
    # ------------------------------------------------------------------

    @app.post(
        "/api/v2/review-actions/issues",
        status_code=201,
        responses={201: {"model": schemas.ReviewWorkspaceResponse}, 400: err, 409: err},
    )
    def create_review_issue(payload: schemas.CreateIssueRequest | None = None):
        return _json(service.create_review_issue(_dump(payload)), 201)

    @app.post(
        "/api/v2/review-actions/issues/{issue_id}/status",
        responses={200: {"model": schemas.ReviewWorkspaceResponse}, 400: err, 404: err, 409: err},
    )
    def update_review_issue_status(
        issue_id: str, payload: schemas.IssueStatusRequest | None = None
    ):
        return _json(service.update_review_issue_status(issue_id, _dump(payload)))

    @app.post(
        "/api/v2/regression-sets",
        status_code=201,
        responses={201: {"model": schemas.ReviewWorkspaceResponse}, 400: err, 409: err},
    )
    def create_regression_set(payload: schemas.CreateRegressionSetRequest | None = None):
        return _json(service.create_regression_set(_dump(payload)), 201)

    @app.post(
        "/api/v2/regression-sets/{regression_set_id}/members",
        status_code=201,
        responses={201: {"model": schemas.ReviewWorkspaceResponse}, 400: err, 404: err, 409: err},
    )
    def add_regression_member(
        regression_set_id: str, payload: schemas.AddRegressionMemberRequest | None = None
    ):
        return _json(service.add_regression_member(regression_set_id, _dump(payload)), 201)

    @app.post(
        "/api/v2/cases/{case_id}/retests",
        status_code=202,
        responses={202: {"model": schemas.ReviewWorkspaceResponse}, 400: err, 404: err, 409: err},
    )
    def create_case_retest(case_id: str, payload: schemas.CreateRetestRequest | None = None):
        return _json(service.create_case_retest(case_id, _dump(payload)), 202)

    @app.post(
        "/api/v2/cases/{case_id}/reconcile",
        status_code=201,
        responses={201: {"model": schemas.ReviewWorkspaceResponse}, 400: err, 404: err, 409: err},
    )
    def reconcile_case(case_id: str, payload: schemas.ReconcileRequest | None = None):
        return _json(service.reconcile_case_kpis(case_id, _dump(payload)), 201)

    @app.post(
        "/api/v2/cases/{case_id}/evidence-packs",
        status_code=201,
        responses={201: {"model": schemas.ReviewWorkspaceResponse}, 400: err, 404: err, 409: err},
    )
    def create_evidence_pack(
        case_id: str, payload: schemas.ReviewActionRequest | None = None
    ):
        return _json(service.create_case_evidence_pack(case_id, _dump(payload)), 201)

    @app.post(
        "/api/v2/evidence-packs/{pack_id}/sign",
        responses={200: {"model": schemas.ReviewWorkspaceResponse}, 400: err, 404: err, 409: err},
    )
    def sign_evidence_pack(pack_id: str, payload: schemas.SignEvidencePackRequest | None = None):
        return _json(service.sign_evidence_pack(pack_id, _dump(payload)))

    _strip_validation_noise(app)
    return app


def _strip_validation_noise(app: FastAPI) -> None:
    """Drop FastAPI's auto-documented 422 responses from the OpenAPI schema.

    Validation failures are converted to legacy-style ``400 {"error": ...}``
    responses at runtime, so the 422/HTTPValidationError entries would
    document behavior this server never exhibits.
    """
    build = app.openapi

    def openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = build()
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation.get("responses", {}).pop("422", None)
        components = schema.get("components", {}).get("schemas", {})
        components.pop("HTTPValidationError", None)
        components.pop("ValidationError", None)
        return schema

    app.openapi = openapi


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
    import uvicorn

    service = EvaluationService(
        data_dir=data_dir,
        work_dir=work_dir,
        runs_dir=runs_dir,
        frontend_data_dir=frontend_data_dir,
        allow_unsafe_plugins=allow_dangerous_operations,
    )
    app = create_app(
        service,
        allowed_origins=allowed_origins,
        allow_dangerous_operations=allow_dangerous_operations,
    )
    print(f"sohbench API listening on http://{host}:{port}", flush=True)
    print(f"data_dir={Path(data_dir)}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")


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
