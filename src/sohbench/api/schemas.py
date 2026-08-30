"""Pydantic contracts for the sohbench HTTP API.

These models document the request/response envelopes served by
``sohbench.api.app`` and drive the OpenAPI schema that
``python -m sohbench.api.typegen`` turns into frontend types.

Compatibility contract (B3): the V1 frontend must run unmodified, so every
model allows extra fields (nothing is filtered on the wire) and request fields
stay optional — the EvaluationService keeps its role as the validating
authority and its error messages reach the client unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ApiModel(BaseModel):
    """Base for all API contracts: pass extra fields through untouched."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


# ---------------------------------------------------------------------------
# Generic envelopes
# ---------------------------------------------------------------------------


class ErrorResponse(ApiModel):
    error: str


class OkResponse(ApiModel):
    ok: bool


class HealthResponse(ApiModel):
    ok: bool
    mode: str


class SessionResponse(ApiModel):
    access_token: str
    legacy_execution_enabled: bool


# ---------------------------------------------------------------------------
# Catalog / resources
# ---------------------------------------------------------------------------


class DatasetInfo(ApiModel):
    name: str | None = None
    rows: int | None = None
    record_count: int | None = None
    cycle_count: int | None = None
    battery_count: int | None = None
    chemistry: str | None = None
    max_capacity_ah: float | None = None
    capacity_source: str | None = None
    held_out_ready: bool | None = None
    columns: list[str] | None = None
    uploaded: bool | None = None


class EstimatorInfo(ApiModel):
    name: str | None = None
    pairing_mode: str | None = None
    feature_mode: str | None = None
    method_category: str | None = None
    method_family: str | None = None
    backend: str | None = None
    combination_policy: str | None = None
    execution_contract: str | None = None
    compatible_featuresets: list[str] | None = None
    preferred_featureset: str | None = None
    data_granularity: str | None = None
    required_signals: list[str] | None = None
    physics_constraints: dict[str, Any] | None = None
    description: str | None = None
    uploaded: bool | None = None


class FeaturesetInfo(ApiModel):
    name: str | None = None
    columns: list[str] | None = None
    has_transform: bool | None = None
    family: str | None = None
    uploaded: bool | None = None


class ModelRecord(ApiModel):
    name: str | None = None
    artifact_id: str | None = None
    estimator: str | None = None
    featureset: str | None = None
    created_by_run: str | None = None


class JobRecord(ApiModel):
    id: str
    kind: str | None = None
    status: Literal["queued", "running", "succeeded", "failed"] | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    traceback: str | None = None
    run_dir: str | None = None
    raw_run_dir: str | None = None
    run_id: str | None = None
    preflight_id: str | None = None
    saved_models: list[dict[str, Any]] | None = None
    model_warnings: list[Any] | None = None
    result_summary: dict[str, Any] | None = None
    request: dict[str, Any] | None = None


class CatalogResponse(ApiModel):
    datasets: list[DatasetInfo] = []
    estimators: list[EstimatorInfo] = []
    featuresets: list[FeaturesetInfo] = []
    data_manifest: dict[str, Any] | None = None
    job_count: int = 0
    inference_job_count: int = 0
    model_count: int = 0
    jobs_limited: bool = False
    inference_jobs_limited: bool = False
    models_limited: bool = False
    jobs: list[JobRecord] = []
    inference_jobs: list[JobRecord] = []
    models: list[ModelRecord] = []


class ModelsResponse(ApiModel):
    models: list[ModelRecord]


class JobsResponse(ApiModel):
    jobs: list[JobRecord]


class JobArtifactsResponse(ApiModel):
    job_id: str
    kind: str | None = None
    status: str | None = None
    run_dir: str | None = None
    artifacts: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Profiles / examples / curves / packaging
# ---------------------------------------------------------------------------


class ProfileEntry(ApiModel):
    name: str | None = None
    label: str | None = None
    kind: str | None = None
    datasets: list[str] | None = None
    training_job_id: str | None = None
    inference_job_id: str | None = None


class ProfileIndexResponse(ApiModel):
    schema_version: int | str | None = None
    profiles: list[ProfileEntry] = []


class ExampleStep(ApiModel):
    id: str | None = None
    label: str | None = None
    status: str | None = None


class ExampleRecord(ApiModel):
    id: str
    kind: str | None = None
    status: str | None = None
    current_step: str | None = None
    created_at: str | None = None
    error: str | None = None
    profile_name: str | None = None
    steps: list[ExampleStep] | None = None


class ExamplesResponse(ApiModel):
    examples: list[ExampleRecord]


class BatteryCurvesResponse(ApiModel):
    ok: bool | None = None
    profile: str | None = None
    dataset: str | None = None
    source: str | None = None
    bytes: int | None = None
    total_curves: int | None = None
    returned_curves: int | None = None
    truncated: bool | None = None
    curves: dict[str, Any] = {}


class PackageStatusReport(ApiModel):
    generated_at: str | None = None
    summary: dict[str, Any] | None = None
    checks: list[dict[str, Any]] | None = None
    outputs: dict[str, Any] | None = None


class PackageStatusResponse(ApiModel):
    ok: bool
    report: PackageStatusReport


class PackageActionResponse(ApiModel):
    ok: bool
    action: str | None = None
    manifest: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Uploads / imports
# ---------------------------------------------------------------------------


class UploadPluginRequest(ApiModel):
    name: str | None = None
    filename: str = ""
    contentBase64: str | None = None
    replace: bool = False


class UploadModelRequest(ApiModel):
    name: str | None = None
    modelFilename: str = ""
    modelContentBase64: str | None = None
    manifestFilename: str = ""
    manifestContentBase64: str | None = None
    replace: bool = False


class UploadResponse(ApiModel):
    ok: bool
    resource: dict[str, Any]


class ImportBatteryDashboardRequest(ApiModel):
    source_dir: str | None = None
    profile_name: str = "battery-soh-diagnostics"
    include_curves: bool = False


class ImportBatteryDashboardResponse(ApiModel):
    ok: bool
    profile: dict[str, Any]


class ImportBatterySohPluginsRequest(ApiModel):
    battery_soh_root: str = r"D:\battery_SOH"
    replace: bool = False


class ImportBatterySohPluginsResponse(ApiModel):
    ok: bool
    plugins: dict[str, Any]


class SyncNativeDashboardRequest(ApiModel):
    source: str | None = None
    target: str | None = None
    include_curves: bool = True


class PackageBuildRequest(ApiModel):
    app_dir: str | None = None
    output: str | None = None


# ---------------------------------------------------------------------------
# Training / inference jobs
# ---------------------------------------------------------------------------


class TrainingPreflightRequest(ApiModel):
    datasets: list[str] | None = None
    estimators: list[str] | None = None
    featuresets: list[str] | None = None
    protocols: list[str] | None = None
    seeds: list[int] | None = None


class PreflightCombination(ApiModel):
    status: str | None = None
    reasons: list[str] | None = None
    warnings: list[str] | None = None


class TrainingPreflightResponse(ApiModel):
    ready: bool | None = None
    preflight_id: str | None = None
    combinations: list[PreflightCombination] = []
    request: dict[str, Any] | None = None


class TrainingJobRequest(TrainingPreflightRequest):
    preflight_id: str | None = None
    case_id: str | None = None
    expected_context_binding_id: str | None = None


class SearchJobRequest(ApiModel):
    """Protocol-gated agent search: inner pool is searchable, the outer
    dataset is held out for a single-shot generalization claim."""

    mode: Literal["training", "adaptation"] | None = None
    inner_datasets: list[str] | None = None
    outer_dataset: str | None = None
    estimators: list[str] | None = None
    featuresets: list[str] | None = None
    policy: Literal["grid", "llm"] | None = None
    window_targets: list[float] | None = None
    # adaptation mode: frozen model x deployment-legal operator on a target
    target_dataset: str | None = None
    models: list[str] | None = None
    operators: list[str] | None = None
    calibration_budget: int | None = None
    top_k: int | None = None
    seeds: list[int] | None = None


class SearchJobRecord(ApiModel):
    id: str
    kind: str | None = None
    status: Literal["queued", "running", "succeeded", "failed"] | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    request: dict[str, Any] | None = None
    progress: list[str] | None = None
    report: dict[str, Any] | None = None
    result_summary: dict[str, Any] | None = None


class SearchJobsResponse(ApiModel):
    jobs: list[SearchJobRecord]


class LlmConfigRequest(ApiModel):
    """AI-agent connection settings. The api_key is write-only: it is stored
    server-side (never echoed back) and an empty value keeps the stored one."""

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    clear_api_key: bool | None = None


class LlmConfigResponse(ApiModel):
    config: dict[str, Any]
    status: dict[str, Any]


class InferenceJobRequest(ApiModel):
    models: list[str] | None = None
    datasets: list[str] | None = None
    evaluation_policy: str | None = None
    launch_strategy: str | None = None
    selected_models: list[str] | None = None
    submitted_models: list[str] | None = None
    skipped_models: list[Any] | None = None
    selected_datasets: list[str] | None = None
    submitted_datasets: list[str] | None = None
    skipped_datasets: list[Any] | None = None
    profile_name: str | None = None
    # Deployment-legal adaptation applied inside the audited run:
    # {"operator": "none|coral|recalibration|coral+recalibration",
    #  "calibration_budget": int}
    adaptation: dict[str, Any] | None = None
    case_id: str | None = None
    expected_context_binding_id: str | None = None


# ---------------------------------------------------------------------------
# Evaluation cases (v2)
# ---------------------------------------------------------------------------


class CaseCandidate(ApiModel):
    candidate_id: str | None = None
    model: str | None = None
    featureset: str | None = None
    role: str | None = None
    model_artifact_id: str | None = None
    run_id: str | None = None


class CaseExecution(ApiModel):
    state: str | None = None
    training_job_id: str | None = None
    inference_job_id: str | None = None
    run_id: str | None = None
    failure_reason: str | None = None


class CaseRecord(ApiModel):
    id: str
    version: int | None = None
    profile_name: str | None = None
    idempotency_key: str | None = None
    lifecycle_state: str | None = None
    title: str | None = None
    source_class: str | None = None
    context_binding: dict[str, Any] | None = None
    execution: CaseExecution | None = None
    candidates: list[CaseCandidate] | None = None
    evidence_selection: dict[str, Any] | None = None
    saved_combinations: list[dict[str, Any]] | None = None
    active_saved_combination_id: str | None = None
    verdict: dict[str, Any] | None = None
    review: dict[str, Any] | None = None


class CasesResponse(ApiModel):
    cases: list[CaseRecord]


class CaseMigrationErrorsResponse(ApiModel):
    errors: list[dict[str, Any]]


class CaseEventsResponse(ApiModel):
    events: list[dict[str, Any]]


class CreateCaseRequest(ApiModel):
    profile_name: str | None = None
    idempotency_key: str | None = None
    title: str | None = None
    source_class: str | None = None
    execution: dict[str, Any] | None = None
    candidates: list[CaseCandidate] | None = None


class ResolveCaseResponse(ApiModel):
    case: CaseRecord
    created: bool


class CaseMutationRequest(ApiModel):
    """Shared shape of the optimistic-concurrency case mutations."""

    expected_version: int | None = None
    expected_context_binding_id: str | None = None
    actor: str | None = None
    reason: str | None = None


class CaseIdentityRequest(CaseMutationRequest):
    title: str | None = None
    source_class: str | None = None
    execution: dict[str, Any] | None = None
    candidates: list[CaseCandidate] | None = None


class SavedCombinationsRequest(CaseMutationRequest):
    action: str | None = None
    source_stage: str | None = None
    job_id: str | None = None
    criterion_set_id: str | None = None
    model_names: list[str] | None = None
    datasets: list[str] | None = None


class ActivateSavedCombinationRequest(CaseMutationRequest):
    saved_combination_id: str | None = None


class EvidenceSelectionRequest(CaseMutationRequest):
    criterion_set_id: str | None = None
    criterion_thresholds: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None


class LifecycleRequest(CaseMutationRequest):
    lifecycle_state: str | None = None


class CaseReviewRequest(CaseMutationRequest):
    state: str | None = None
    reviewer: str | None = None
    note: str | None = None


# ---------------------------------------------------------------------------
# Review workflow (issues / regression sets / retests / evidence packs)
# ---------------------------------------------------------------------------


class ReviewActionRequest(ApiModel):
    expected_case_version: int | None = None
    actor: str | None = None


class CreateIssueRequest(ReviewActionRequest):
    case_id: str | None = None
    title: str | None = None
    severity: str | None = None
    description: str | None = None


class IssueStatusRequest(ReviewActionRequest):
    status: str | None = None
    reason: str | None = None


class CreateRegressionSetRequest(ReviewActionRequest):
    issue_id: str | None = None
    name: str | None = None


class AddRegressionMemberRequest(ReviewActionRequest):
    neighbor_radius: int | None = None


class CreateRetestRequest(ReviewActionRequest):
    issue_id: str | None = None
    regression_set_id: str | None = None


class ReconcileRequest(ReviewActionRequest):
    tolerance: float | None = None


class SignEvidencePackRequest(ReviewActionRequest):
    signer: str | None = None
    reason: str | None = None


class ReviewWorkspaceResponse(ApiModel):
    case_id: str | None = None
    case_version: int | None = None
    issues: list[dict[str, Any]] = []
    regression_sets: list[dict[str, Any]] = []
    retests: list[dict[str, Any]] = []
    reconciliations: list[dict[str, Any]] = []
    evidence_packs: list[dict[str, Any]] = []
    signing_readiness: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    context_binding: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Evidence payloads
# ---------------------------------------------------------------------------


class TrajectoryPoint(ApiModel):
    cell_id: str | None = None
    asset_id: str | None = None
    cycle_index: int | None = None
    cycle_id: int | str | None = None
    true_soh: float | None = None
    pred_soh: float | None = None
    lower: float | None = None
    upper: float | None = None
    absolute_error: float | None = None
    raw_curve_available: bool | None = None
    prediction_record_id: str | None = None
    sample_id: str | None = None
    metric_contributions: dict[str, Any] | None = None


class TrajectoryResponse(ApiModel):
    case_id: str | None = None
    inference_job_id: str | None = None
    dataset: str | None = None
    candidate_id: str | None = None
    model_artifact_id: str | None = None
    context_binding_id: str | None = None
    points: list[TrajectoryPoint] = []


class PredictionRecordCurvesResponse(ApiModel):
    prediction_record: dict[str, Any] | None = None
    sampling: dict[str, Any] | None = None
    signals: dict[str, Any] | None = None
    feature_window: dict[str, Any] | None = None
    feature_curves: list[dict[str, Any]] | None = None
    missing_signals: list[str] | None = None
    curve_segment_ref: dict[str, Any] | None = None
