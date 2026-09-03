from __future__ import annotations

import json
import hashlib
import math
import os
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import registry as R
from .battery_soh_plugins import register_battery_soh_plugins
from .contracts import feature_columns
from .data import DatasetLoader
from .evaluation_cases import EvaluationCaseConflictError, EvaluationCaseStore
from .case_evidence import (
    EvidenceDerivationService,
    curve_signal_names,
    derive_verdict,
    write_prediction_trajectory_index,
)
from .battery_dashboard_import import publish_battery_dashboard_profile
from .frontend_publish import (
    load_frontend_profile_index,
    publish_frontend_profile,
    sync_frontend_artifacts,
)
from .feature_safety import (
    FeatureSafetyError,
    LABEL_SAFETY_CONTRACT,
    audit_feature_frame,
    is_label_source_feature,
)
from .inference import run_inference_benchmark_with_evidence
from .method_contracts import incompatible_pairing_reason, method_contract
from .model_store import ModelStore
from .protocols import LeaveOneCellOut, LeaveOneChemistryOut, LeaveOneDatasetOut
from .report import write_artifacts, write_inference_artifacts
from .runner import BenchmarkRunner
from .review_workflow import ReviewWorkflowStore
from .tasks import SOHEstimationTask
from sohkernel.transforms import (  # the rule moved to the kernel (Task 5); agent_search imports the alias
    inference_transform_featuresets as _inference_transform_featuresets,
)
from .trusted_plugins import TrustedPluginStore


PROTOCOLS = {
    "loco": LeaveOneCellOut,
    "lodo": LeaveOneDatasetOut,
    "lcho": LeaveOneChemistryOut,
}
TRAINING_PREFLIGHT_CONTRACT = "training-preflight/1.0"
INFERENCE_JOB_SCHEMA_VERSION = "1.1.0"


def ensure_builtin_resources() -> None:
    import sohbench.estimators.baselines as baselines
    import sohbench.estimators.byo as byo
    import sohbench.estimators.mechanism as mechanism
    import sohbench.estimators.neural as neural
    import sohbench.estimators.physics as physics
    import sohbench.featuresets as featuresets

    for cls in (
        baselines.RidgeEstimator,
        baselines.GBDTEstimator,
        baselines.SVREstimator,
        baselines.GPREstimator,
        mechanism.ElasticNetEstimator,
        mechanism.ExtraTreesEstimator,
        mechanism.PLSEstimator,
        mechanism.StressResidualEstimator,
        physics.PhysicsResidualEstimator,
        neural.MLPEstimator,
        byo.ByoSelectKBestRidge,
    ):
        R._ESTIMATORS.setdefault(cls.name, cls)

    R._FEATURESETS.setdefault("full", featuresets._full_fs)
    for fs in (
        featuresets.FeatureSet(
            name="voltage",
            candidates=["voltage_mean", "voltage_std", "voltage_min", "voltage_max"],
        ),
        featuresets.FeatureSet(
            name="ic_dva",
            candidates=["ic_peak_ah_per_v", "ic_peak_voltage_v"],
        ),
        featuresets.FeatureSet(
            name="partial_window",
            candidates=["q_win_ah", "t_win_s"],
        ),
        featuresets.FeatureSet(
            name="capacity_time",
            # charge_capacity_max is the label source; banned as a model input
            candidates=["time_s_max", "cycle_index_feature", "cycle_fraction"],
        ),
        featuresets.FeatureSet(
            name="partial_window_norm",
            candidates=["q_win_frac", "t_win_frac"],
            description=(
                "无量纲部分窗口健康因子：窗口容量/额定容量、窗口时长/充电时长 —— "
                "跨电芯规格与倍率可迁移（L1 可迁移 HI）。"
            ),
            transform=featuresets._partial_window_norm_transform,
        ),
        featuresets.FeatureSet(
            name="electrochemical_diagnostics",
            candidates=[
                "ic_peak_ah_per_v", "ic_peak_voltage_v", "q_win_ah", "t_win_s",
                "voltage_span_v",
            ],
            family="mechanism_derived",
            mechanism_level="diagnostic_proxy",
            description="IC/DVA 峰、局部充电窗口与电压/能量代理，适合曲线诊断和退化模式比较。",
            required_signals=["cycle_voltage", "cycle_current", "cycle_time"],
            transform=featuresets._mechanism_proxy_transform,
        ),
        featuresets.FeatureSet(
            name="operating_stress",
            candidates=[
                "current_rms_proxy_a", "current_variability_ratio", "temperature_mean",
                "temperature_over_25c", "arrhenius_stress_factor", "voltage_span_v",
            ],
            family="mechanism_derived",
            mechanism_level="stress_proxy",
            description="电流波动、温度与 Arrhenius 应力代理，用于跨工况泛化与灰盒退化模型。",
            required_signals=["cycle_current", "cycle_temperature", "cycle_voltage"],
            transform=featuresets._mechanism_proxy_transform,
        ),
        featuresets.FeatureSet(
            name="degradation_kinetics",
            candidates=[
                "cycle_sqrt", "cycle_log1p", "time_s_max", "arrhenius_stress_factor",
            ],
            family="mechanism_derived",
            mechanism_level="semi_empirical",
            description="平方根/对数老化坐标、容量吞吐与温度应力代理，面向半经验退化先验。",
            required_signals=["cycle_index", "cycle_capacity", "cycle_temperature"],
            transform=featuresets._mechanism_proxy_transform,
        ),
    ):
        R._FEATURESETS.setdefault(fs.name, fs)

    try:
        import torch  # noqa: F401
    except ImportError:
        return
    import sohbench.estimators.deep  # noqa: F401
    import sohbench.estimators.pinn  # noqa: F401


def _effective_training_featuresets(estimator_names: list[str], requested: list[str]) -> list[str]:
    """Include a joint method's preferred package when no selected feature is compatible."""
    effective = list(dict.fromkeys(map(str, requested)))
    for estimator_name in estimator_names:
        contract = method_contract(R.get_estimator(estimator_name))
        for feature_name in contract.feature_iterations(requested):
            if feature_name != "(byo)" and feature_name not in effective:
                effective.append(feature_name)
    return effective


class EvaluationService:
    def __init__(
        self,
        *,
        data_dir: str | Path,
        work_dir: str | Path = Path("work/trusted_plugins"),
        runs_dir: str | Path = Path("runs"),
        frontend_data_dir: str | Path = Path("frontend/app/data"),
        allow_unsafe_plugins: bool = True,
    ):
        ensure_builtin_resources()
        self.data_dir = Path(data_dir)
        self.work_dir = Path(work_dir)
        self.runs_dir = Path(runs_dir)
        self.frontend_data_dir = Path(frontend_data_dir)
        self.jobs_dir = self.work_dir / "jobs"
        self.inference_jobs_dir = self.work_dir / "inference_jobs"
        self.examples_dir = self.work_dir / "examples"
        self.retest_runs_dir = self.work_dir / "retest_runs"
        self.search_jobs_dir = self.work_dir / "search_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.inference_jobs_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.retest_runs_dir.mkdir(parents=True, exist_ok=True)
        self.search_jobs_dir.mkdir(parents=True, exist_ok=True)
        self._example_lock = threading.RLock()
        # Single-flight guard for the quickstart run itself; the example
        # record lock above is only ever held for a read/write.
        self._quickstart_run_lock = threading.Lock()
        self._job_lock = threading.RLock()
        self._inference_job_lock = threading.RLock()
        self._retest_run_lock = threading.RLock()
        self._search_job_lock = threading.RLock()
        self.store = TrustedPluginStore(
            self.work_dir,
            self.data_dir,
            load_plugins=allow_unsafe_plugins,
        )
        self.model_store = ModelStore(self.work_dir / "models")
        self.evidence_deriver = EvidenceDerivationService(
            self.frontend_data_dir,
            self.data_dir,
            inference_jobs_dir=self.inference_jobs_dir,
            runs_dir=self.runs_dir,
        )
        self.evaluation_cases = EvaluationCaseStore(
            self.work_dir / "cases",
            evidence_deriver=self.evidence_deriver.derive,
            require_context_binding=True,
        )
        self.review_workflow = ReviewWorkflowStore(
            self.work_dir / "evaluation_cases.sqlite3",
            self.work_dir / "artifacts",
            get_case=self.evaluation_cases.get,
            list_case_events=self.evaluation_cases.list_events,
            frontend_data_dir=self.frontend_data_dir,
            has_executable_model_artifact=self._has_executable_model_artifact,
        )
        self._dataset_snapshot_cache: dict[str, tuple[int, int, str]] = {}
        self._reconcile_interrupted_jobs()

    def _reconcile_interrupted_jobs(self) -> None:
        """Runners are daemon threads inside this process: any record still
        ``queued``/``running`` when a new service starts belonged to a process
        that is gone, and would otherwise stay "running" forever (the UI keeps
        polling it and the bound case copies the status)."""
        stamp = _now()
        for directory, saver in (
            (self.jobs_dir, self._save_job),
            (self.inference_jobs_dir, self._save_inference_job),
            (self.search_jobs_dir, self._save_search_job),
            (self.examples_dir, self._save_example),
        ):
            for path in sorted(directory.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(record, dict) or record.get("id") is None:
                    continue
                if record.get("status") not in {"queued", "running"}:
                    continue
                record["status"] = "failed"
                record["error"] = "interrupted: the API process that ran this job exited before it finished"
                record.setdefault("finished_at", stamp)
                try:
                    saver(record)
                except OSError:
                    continue

    def _has_executable_model_artifact(self, artifact_id: str) -> bool:
        try:
            self.model_store.load_artifact(artifact_id)
        except Exception:
            return False
        return True

    def catalog(
        self,
        *,
        model_limit: int | None = None,
        job_limit: int | None = None,
        compact_jobs: bool = False,
    ) -> dict:
        catalog = self.store.catalog()
        jobs = self.list_jobs()
        inference_jobs = self.list_inference_jobs()
        models = self.model_store.catalog()
        catalog["job_count"] = len(jobs)
        catalog["inference_job_count"] = len(inference_jobs)
        catalog["model_count"] = len(models)
        catalog["jobs_limited"] = job_limit is not None and len(jobs) > job_limit
        catalog["inference_jobs_limited"] = job_limit is not None and len(inference_jobs) > job_limit
        catalog["models_limited"] = model_limit is not None and len(models) > model_limit
        if job_limit is not None:
            jobs = jobs[: max(0, int(job_limit))]
            inference_jobs = inference_jobs[: max(0, int(job_limit))]
        if compact_jobs:
            jobs = [_compact_job(job) for job in jobs]
            inference_jobs = [_compact_job(job) for job in inference_jobs]
        if model_limit is not None:
            models = models[: max(0, int(model_limit))]
        catalog["jobs"] = jobs
        catalog["inference_jobs"] = inference_jobs
        catalog["models"] = models
        return catalog

    def training_preflight(self, payload: dict) -> dict:
        """Validate an exact data x feature x estimator x protocol matrix.

        The result is derived from the loaded, QC-filtered tables and is bound
        to content digests for every selected processed dataset.  It is the
        server-side authority used by both the UI and job admission.
        """
        started = time.perf_counter()
        request = self._normalize_request(payload)
        loader = DatasetLoader(self.data_dir)
        datasets = loader.resolve_datasets(request["datasets"])
        request = {**request, "datasets": datasets}
        snapshots = [self._dataset_snapshot(name) for name in datasets]

        table, meta, qc = loader.load(datasets)
        effective_featuresets = _effective_training_featuresets(
            request["estimators"], request["featuresets"]
        )
        feature_transform_errors: dict[str, str] = {}
        for feature_name in effective_featuresets:
            try:
                table = self.store.apply_feature_transforms(table, meta, [feature_name])
            except Exception as exc:
                feature_transform_errors[feature_name] = str(exc)
        dataset_profiles = _training_dataset_profiles(table, datasets)
        feature_profiles = {
            name: _training_feature_profile(
                table,
                datasets,
                name,
                transform_error=feature_transform_errors.get(name, ""),
            )
            for name in effective_featuresets
        }
        protocol_profiles = {
            name: _training_protocol_profile(table, meta, name)
            for name in request["protocols"]
        }

        combinations: list[dict] = []
        for estimator_name in request["estimators"]:
            estimator = R.get_estimator(estimator_name)
            contract = method_contract(estimator)
            feature_names = ["(byo)"] if contract.feature_mode == "byo" else list(dict.fromkeys([
                *request["featuresets"],
                *contract.feature_iterations(request["featuresets"]),
            ]))
            for feature_name in feature_names:
                feature = None if feature_name == "(byo)" else feature_profiles[feature_name]
                for protocol_name in request["protocols"]:
                    protocol = protocol_profiles[protocol_name]
                    incompatible = bool(
                        feature is not None
                        and not contract.accepts_featureset(feature_name)
                    )
                    if incompatible:
                        combinations.append(
                            {
                                "estimator": estimator_name,
                                "featureset": feature_name,
                                "protocol": protocol_name,
                                **contract.catalog_metadata(),
                                "selected_columns": [],
                                "status": "not_applicable",
                                "reasons": [],
                                "warnings": [incompatible_pairing_reason(contract, feature_name)],
                                "seed_count": len(request["seeds"]),
                                "run_instances": 0,
                            }
                        )
                        continue
                    reasons = list(protocol["errors"])
                    warnings = list(protocol["warnings"])
                    selected_columns: list[str] = []
                    if feature is not None:
                        selected_columns = list(feature["selected_columns"])
                        reasons.extend(feature["errors"])
                        warnings.extend(feature["warnings"])
                    if contract.feature_mode not in {"unified", "byo"}:
                        reasons.append(f"unsupported estimator feature_mode: {contract.feature_mode}")
                    status = "blocked" if reasons else ("warning" if warnings else "ready")
                    combinations.append(
                        {
                            "estimator": estimator_name,
                            "featureset": feature_name,
                            "protocol": protocol_name,
                            **contract.catalog_metadata(),
                            "selected_columns": selected_columns,
                            "status": status,
                            "reasons": reasons,
                            "warnings": warnings,
                            "seed_count": len(request["seeds"]),
                            "run_instances": len(request["seeds"]),
                        }
                    )

        blocked = [item for item in combinations if item["status"] == "blocked"]
        warned = [item for item in combinations if item["status"] == "warning"]
        not_applicable = [item for item in combinations if item["status"] == "not_applicable"]
        runnable = [item for item in combinations if item["status"] in {"ready", "warning"}]
        identity_payload = {
            "contract": TRAINING_PREFLIGHT_CONTRACT,
            "request": request,
            "dataset_snapshots": [
                {"dataset": item["dataset"], "snapshot_id": item["snapshot_id"]}
                for item in snapshots
            ],
            "combinations": [
                {
                    "estimator": item["estimator"],
                    "featureset": item["featureset"],
                    "protocol": item["protocol"],
                    "status": item["status"],
                    "selected_columns": item["selected_columns"],
                }
                for item in combinations
            ],
        }
        preflight_id = f"preflight@sha256:{_canonical_digest(identity_payload)}"
        return {
            "contract": TRAINING_PREFLIGHT_CONTRACT,
            "preflight_id": preflight_id,
            "ready": bool(runnable) and not blocked,
            "request": request,
            "dataset_snapshots": snapshots,
            "datasets": dataset_profiles,
            "featuresets": list(feature_profiles.values()),
            "protocols": list(protocol_profiles.values()),
            "combinations": combinations,
            "summary": {
                "rows": int(len(table)),
                "datasets": len(datasets),
                "base_combinations": len(combinations),
                "applicable_combinations": len(runnable) + len(blocked),
                "run_instances": sum(item["run_instances"] for item in combinations),
                "runnable_combinations": len(runnable),
                "warning_combinations": len(warned),
                "blocked_combinations": len(blocked),
                "not_applicable_combinations": len(not_applicable),
                "qc_pass_rate": float(qc.pass_rate),
            },
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "checked_at": _now(),
        }

    def _dataset_snapshot(self, name: str) -> dict:
        path = self.data_dir / "processed" / f"{name}_features.pkl"
        stat = path.stat()
        key = str(path.resolve())
        cached = self._dataset_snapshot_cache.get(key)
        signature = (int(stat.st_size), int(stat.st_mtime_ns))
        if cached is None or cached[:2] != signature:
            digest = _sha256_path(path)
            self._dataset_snapshot_cache[key] = (*signature, digest)
        else:
            digest = cached[2]
        return {
            "dataset": name,
            "snapshot_id": f"dataset@sha256:{digest}",
            "path": str(path),
            "bytes": int(stat.st_size),
        }

    def create_evaluation_case(self, payload: dict) -> dict:
        return self.evaluation_cases.create(payload)

    def resolve_evaluation_case(self, payload: dict) -> dict:
        case, created = self.evaluation_cases.resolve(payload)
        return {"case": case, "created": created}

    def get_evaluation_case(self, case_id: str) -> dict:
        return self.evaluation_cases.get(case_id)

    def update_evaluation_case_identity(self, case_id: str, payload: dict) -> dict:
        payload = dict(payload or {})
        current_case = self.evaluation_cases.get(case_id)
        execution = dict(payload.get("execution") or {})
        training_job_id = str(execution.get("training_job_id") or "").strip()
        inference_job_id = str(execution.get("inference_job_id") or "").strip()
        authoritative_candidates: list[dict] | None = None
        authoritative_job = None
        replace_candidate_pool = True
        if inference_job_id:
            authoritative_job = self.get_inference_job(inference_job_id)
            self._assert_job_case(authoritative_job, case_id)
            self._assert_job_can_update_case(
                authoritative_job,
                current_case,
                stage="inference",
            )
            if authoritative_job.get("status") == "succeeded":
                # An inference run validates a subset of the case's model
                # pool; it must not shrink the training-derived pool, so
                # its candidates MERGE instead of replacing.
                replace_candidate_pool = False
                authoritative_candidates = _unique_candidate_rows(
                    item["candidate"]
                    for item in self._inference_saved_combinations(
                        inference_job_id,
                        str((self.evaluation_cases.get(case_id).get("verdict") or {}).get("criterion_set_version") or "engineering"),
                    )
                )
        elif training_job_id:
            authoritative_job = self.get_job(training_job_id)
            self._assert_job_case(authoritative_job, case_id)
            self._assert_job_can_update_case(
                authoritative_job,
                current_case,
                stage="training",
            )
            if authoritative_job.get("status") == "succeeded":
                authoritative_candidates = _unique_candidate_rows(
                    item["candidate"]
                    for item in self._training_saved_combinations(
                        training_job_id,
                        str((self.evaluation_cases.get(case_id).get("verdict") or {}).get("criterion_set_version") or "engineering"),
                    )
                )
        if authoritative_job is not None:
            execution["state"] = _case_execution_state(authoritative_job.get("status"))
            execution["run_id"] = str(
                authoritative_job.get("run_id")
                or authoritative_job.get("id")
                or ""
            )
            execution["failure_reason"] = (
                str(authoritative_job.get("error") or "")
                if authoritative_job.get("status") == "failed" else ""
            )
            payload["execution"] = execution
        if authoritative_candidates is not None:
            payload["candidates"] = authoritative_candidates
            payload["replace_candidates"] = replace_candidate_pool
        elif "candidates" in payload:
            payload.pop("candidates", None)
            payload.pop("replace_candidates", None)
        return self.evaluation_cases.update_identity(case_id, payload)

    def update_evaluation_case_saved_combinations(self, case_id: str, payload: dict) -> dict:
        """Save or remove completed combinations using server-owned artifacts only."""
        payload = dict(payload or {})
        current = self.evaluation_cases.get(case_id)
        action = str(payload.get("action") or "save")
        expected_version = payload.get("expected_version")
        expected_context_binding_id = payload.get("expected_context_binding_id")
        if action == "save":
            source_stage = str(payload.get("source_stage") or "")
            job_id = str(
                payload.get("job_id")
                or payload.get(f"{source_stage}_job_id")
                or ""
            ).strip()
            if source_stage not in {"training", "inference"}:
                raise ValueError("source_stage must be training or inference")
            if not job_id:
                raise ValueError("job_id is required")
            source_job = (
                self.get_job(job_id)
                if source_stage == "training"
                else self.get_inference_job(job_id)
            )
            self._assert_job_case(source_job, case_id)
            if source_job.get("status") != "succeeded":
                raise ValueError("only a succeeded case-bound job can be saved")
            criterion_set_id = str(
                payload.get("criterion_set_id")
                or (current.get("verdict") or {}).get("criterion_set_version")
                or "engineering"
            )
            if source_stage == "training":
                items = self._training_saved_combinations(job_id, criterion_set_id)
            else:
                items = self._inference_saved_combinations(job_id, criterion_set_id)
            requested_ids = set(_string_list(payload.get("candidate_ids") or []))
            requested_models = set(_string_list(payload.get("model_names") or []))
            requested_datasets = set(_string_list(payload.get("datasets") or []))
            if requested_ids:
                items = [
                    item for item in items
                    if item["candidate"]["candidate_id"] in requested_ids
                ]
            if requested_models:
                catalog_by_artifact = {
                    str(item.get("artifact_id") or ""): str(item.get("name") or "")
                    for item in self.model_store.catalog()
                }
                items = [
                    item for item in items
                    if catalog_by_artifact.get(item["candidate"]["model_artifact_id"])
                    in requested_models
                ]
            if requested_datasets:
                items = [
                    item for item in items
                    if item.get("target_dataset") in requested_datasets
                ]
            if not items:
                raise ValueError("the completed job has no matching combinations to save")
            return self.evaluation_cases.update_saved_combinations(
                case_id,
                {
                    "action": "save",
                    "items": items,
                    "expected_version": expected_version,
                    "expected_context_binding_id": expected_context_binding_id,
                    "actor": payload.get("actor") or "local-ui",
                    "reason": payload.get("reason")
                    or f"saved completed {source_stage} combinations from {job_id}",
                },
            )
        if action == "remove":
            saved_ids = _string_list(payload.get("saved_combination_ids") or [])
            if not saved_ids:
                source_stage = str(payload.get("source_stage") or "")
                job_id = str(
                    payload.get("job_id")
                    or payload.get(f"{source_stage}_job_id")
                    or ""
                ).strip()
                saved_ids = [
                    item["saved_combination_id"]
                    for item in current.get("saved_combinations") or []
                    if item.get("source_stage") == source_stage
                    and job_id in {
                        str(item.get("training_job_id") or ""),
                        str(item.get("inference_job_id") or ""),
                    }
                ]
            if not saved_ids:
                raise ValueError("no matching saved combinations were found")
            return self.evaluation_cases.update_saved_combinations(
                case_id,
                {
                    "action": "remove",
                    "saved_combination_ids": saved_ids,
                    "expected_version": expected_version,
                    "expected_context_binding_id": expected_context_binding_id,
                    "actor": payload.get("actor") or "local-ui",
                    "reason": payload.get("reason") or "removed saved combinations",
                },
            )
        raise ValueError(f"unsupported saved combination action: {action}")

    def activate_evaluation_case_saved_combination(self, case_id: str, payload: dict) -> dict:
        return self.evaluation_cases.activate_saved_combination(case_id, payload)

    def list_evaluation_cases(
        self,
        *,
        profile_name: str | None = None,
        lifecycle_state: str | None = None,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> list[dict]:
        return self.evaluation_cases.list(
            profile_name=profile_name,
            lifecycle_state=lifecycle_state,
            include_archived=include_archived,
            limit=limit,
        )

    def frontend_profile_index(self) -> dict:
        return load_frontend_profile_index(self.frontend_data_dir)

    def update_evaluation_case_evidence(self, case_id: str, payload: dict) -> dict:
        return self.evaluation_cases.update_evidence(case_id, payload)

    def transition_evaluation_case(self, case_id: str, payload: dict) -> dict:
        return self.evaluation_cases.transition_lifecycle(case_id, payload)

    def update_evaluation_case_review(self, case_id: str, payload: dict) -> dict:
        return self.evaluation_cases.update_review(case_id, payload)

    def evaluation_case_events(self, case_id: str, *, after_sequence: int = 0) -> list[dict]:
        return self.evaluation_cases.list_events(case_id, after_sequence=after_sequence)

    def evaluation_case_migration_errors(self) -> list[dict]:
        return self.evaluation_cases.migration_errors()

    def evaluation_case_evidence_workspace(
        self,
        case_id: str,
        overrides: dict | None = None,
        *,
        expected_context_binding_id: str | None = None,
    ) -> dict:
        self.evaluation_cases.assert_context_binding(case_id, expected_context_binding_id)
        return self.evaluation_cases.evidence_workspace(case_id, overrides)

    def evaluation_case_prediction_record(
        self,
        case_id: str,
        *,
        sample_id: str | None = None,
        expected_context_binding_id: str | None = None,
    ) -> dict:
        return self.evidence_deriver.prediction_detail(
            self.evaluation_cases.assert_context_binding(case_id, expected_context_binding_id),
            sample_id=sample_id,
        )

    def evaluation_prediction_record_curves(
        self,
        case_id: str,
        record_id: str,
        *,
        signals: list[str] | None = None,
        resolution: str = "viewport",
        max_points: int | None = None,
        expected_context_binding_id: str | None = None,
    ) -> dict:
        return self.evidence_deriver.prediction_curves(
            self.evaluation_cases.assert_context_binding(case_id, expected_context_binding_id),
            record_id=record_id,
            signals=signals,
            resolution=resolution,
            max_points=max_points,
        )

    def evaluation_case_prediction_trajectory(
        self,
        case_id: str,
        *,
        inference_job_id: str | None = None,
        dataset: str | None = None,
        asset_id: str | None = None,
        compact: bool = False,
        candidate_id: str | None = None,
        model_artifact_id: str | None = None,
        expected_context_binding_id: str | None = None,
    ) -> dict:
        return self.evidence_deriver.prediction_trajectory(
            self.evaluation_cases.assert_context_binding(case_id, expected_context_binding_id),
            inference_job_id=inference_job_id,
            dataset=dataset,
            asset_id=asset_id,
            compact=compact,
            candidate_id=candidate_id,
            model_artifact_id=model_artifact_id,
        )

    def evaluation_case_review_workspace(
        self,
        case_id: str,
        *,
        expected_context_binding_id: str | None = None,
    ) -> dict:
        # The review workspace is a semantic read just like curves and
        # predictions.  Bind API callers to the exact case generation so a
        # delayed response cannot repaint a newer candidate/dataset context.
        if expected_context_binding_id is not None:
            bound_case = self.evaluation_cases.assert_context_binding(
                case_id,
                expected_context_binding_id,
            )
        else:
            bound_case = self.evaluation_cases.get(case_id)
        workspace = self.review_workflow.workspace(case_id)
        if expected_context_binding_id is not None:
            # Detect a semantic switch that races the workspace projection.
            bound_case = self.evaluation_cases.assert_context_binding(
                case_id,
                expected_context_binding_id,
            )
        workspace["context_binding"] = bound_case["context_binding"]
        return workspace

    def create_review_issue(self, payload: dict) -> dict:
        return self.review_workflow.create_issue(payload)

    def update_review_issue_status(self, issue_id: str, payload: dict) -> dict:
        return self.review_workflow.update_issue_status(issue_id, payload)

    def create_regression_set(self, payload: dict) -> dict:
        return self.review_workflow.create_regression_set(payload)

    def add_regression_member(self, regression_set_id: str, payload: dict) -> dict:
        return self.review_workflow.add_current_evidence_member(regression_set_id, payload)

    def create_case_retest(self, case_id: str, payload: dict, *, start: bool = True) -> dict:
        retest = self.review_workflow.create_retest(case_id, payload)
        run_id = "RTR-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8].upper()
        queued = {
            "id": run_id,
            "kind": "controlled_retest",
            "execution_mode": retest.get("execution_mode") or "evidence_replay",
            "retest_id": retest["id"],
            "case_id": case_id,
            "status": "queued",
            "created_at": _now(),
        }
        self._save_retest_run(queued)
        if start:
            thread = threading.Thread(
                target=self.run_controlled_retest_inline,
                args=(retest["id"], run_id),
                daemon=True,
            )
            thread.start()
        return {**retest, "controlled_run": queued}

    def run_controlled_retest_inline(self, retest_id: str, run_id: str | None = None) -> dict:
        run_id = run_id or (
            "RTR-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8].upper()
        )
        context = None
        execution_mode = "evidence_replay"
        dispatch = {
            "runner": "controlled_retest/v2",
            "execution_mode": execution_mode,
        }
        run = {
            "id": run_id,
            "kind": "controlled_retest",
            "retest_id": retest_id,
            "case_id": "",
            "status": "running",
            "created_at": _now(),
            "started_at": _now(),
        }
        try:
            context = self.review_workflow.retest_execution_context(retest_id)
            execution_mode = self._controlled_retest_execution_mode(context)
            run.update(
                {
                    "case_id": context["retest"]["case_id"],
                    "execution_mode": execution_mode,
                }
            )
            dispatch = {
                "runner": "controlled_model_reinference/v1" if execution_mode == "model_reinference" else "controlled_evidence_replay/v1",
                "execution_mode": execution_mode,
                "baseline_case_version": context["retest"]["baseline_case_version"],
                "protocol_ref": context["retest"]["requested_protocol_ref"],
                "regression_set_id": context["regression_set"]["id"],
                "member_count": len(context["regression_set"]["members"]),
            }
            self.review_workflow.start_retest_from_runner(
                retest_id,
                run_id=run_id,
                dispatch=dispatch,
            )
            self._save_retest_run({**run, "dispatch": dispatch})
            result = (
                self._evaluate_model_reinference(context, run_id=run_id)
                if execution_mode == "model_reinference"
                else self._evaluate_controlled_evidence_replay(context, run_id=run_id)
            )
            run.update({"status": "completed", "finished_at": _now(), "result": result})
            self._save_retest_run(run)
            workflow_status = "completed" if result.get("prediction_recomputed") else "failed"
            workflow_result = result if workflow_status == "completed" else {
                **result,
                "error": "immutable model artifact unavailable; evidence replay completed but cannot close the retest gate",
                "replay_completed": True,
            }
            self.review_workflow.complete_retest_from_runner(
                retest_id,
                run_id=run_id,
                status=workflow_status,
                result=workflow_result,
            )
            if workflow_status == "failed":
                run.update({"status": "failed", "result": workflow_result})
                self._save_retest_run(run)
        except Exception as exc:
            failure = {
                "runner": dispatch["runner"],
                "execution_mode": execution_mode,
                "error": str(exc),
            }
            run.update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "result": failure,
                    "traceback": traceback.format_exc(limit=12),
                }
            )
            self._save_retest_run(run)
            self.review_workflow.complete_retest_from_runner(
                retest_id,
                run_id=run_id,
                status="failed",
                result=failure,
            )
        return run

    def get_controlled_retest_run(self, run_id: str) -> dict:
        path = self.retest_runs_dir / f"{run_id}.json"
        if not path.is_file():
            raise KeyError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _controlled_retest_execution_mode(self, context: dict) -> str:
        artifact_id = str((context["issue"].get("context") or {}).get("model_artifact_id") or "")
        return "model_reinference" if self._has_executable_model_artifact(artifact_id) else "evidence_replay"

    def _evaluate_model_reinference(self, context: dict, *, run_id: str) -> dict:
        retest = context["retest"]
        issue_context = context["issue"].get("context") or {}
        regression_set = context["regression_set"]
        members = regression_set.get("members") or []
        if not members:
            raise ValueError("controlled retest requires at least one pinned regression member")
        if retest["requested_protocol_ref"] != retest["baseline_protocol_ref"]:
            raise ValueError("controlled model runner currently requires the original protocol")
        artifact_id = str(issue_context.get("model_artifact_id") or "")
        loaded = self.model_store.load_artifact(artifact_id)
        datasets = sorted({str(member.get("dataset") or "") for member in members})
        if "" in datasets:
            raise ValueError("regression member dataset is missing")
        loader = DatasetLoader(self.data_dir)
        self._verify_dataset_snapshot(issue_context, datasets)
        table, meta, _qc = loader.load(datasets)
        featureset = str(loaded.manifest.get("featureset") or "")
        table = self.store.apply_feature_transforms(table, meta, [featureset] if featureset else [])
        rows = []
        for member in members:
            mask = (
                (table["dataset"].astype(str) == str(member.get("dataset") or ""))
                & (table["cell_id"].astype(str) == str(member.get("asset_id") or ""))
                & (table["cycle_index"].astype(str) == str(member.get("cycle_id")))
            )
            matches = table.loc[mask]
            if len(matches) != 1:
                raise ValueError(f"pinned dataset row is not unique for {member['sample_id']}")
            row_meta = meta.loc[matches.index] if meta is not None and len(meta) == len(table) else meta
            prediction = float(loaded.predict(matches, row_meta)[0])
            interval = loaded.predict_interval(matches, row_meta)
            lower = float(interval[0][0]) if interval is not None else None
            upper = float(interval[1][0]) if interval is not None else None
            true_soh = _finite_number(matches.iloc[0]["soh"], "soh")
            rows.append(
                self._controlled_retest_record(
                    member,
                    true_soh=true_soh,
                    pred_soh=prediction,
                    lower=lower,
                    upper=upper,
                )
            )
        return self._controlled_retest_result(
            context,
            run_id=run_id,
            execution_mode="model_reinference",
            runner="controlled_model_reinference/v1",
            records=rows,
            source_artifacts=[
                {
                    "kind": "model",
                    "artifact_id": artifact_id,
                    "model_name": loaded.name,
                },
                *[
                    {
                        "kind": "dataset",
                        "dataset": dataset,
                        "snapshot_ref": issue_context.get("dataset_snapshot_id") or "",
                    }
                    for dataset in datasets
                ],
            ],
            prediction_recomputed=True,
        )

    def _verify_dataset_snapshot(self, issue_context: dict, datasets: list[str]) -> None:
        snapshot_ref = str(issue_context.get("dataset_snapshot_id") or "")
        if not snapshot_ref.startswith("dataset-snapshot@sha256:") or len(datasets) != 1:
            return
        path = self.data_dir / "processed" / f"{datasets[0]}_features.pkl"
        if not path.is_file():
            raise ValueError(f"pinned dataset snapshot is unavailable for {datasets[0]}")
        expected = snapshot_ref.split(":", 1)[1]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("dataset snapshot digest changed after the issue was created")

    def _evaluate_controlled_evidence_replay(self, context: dict, *, run_id: str) -> dict:
        retest = context["retest"]
        issue_context = context["issue"].get("context") or {}
        regression_set = context["regression_set"]
        members = regression_set.get("members") or []
        if not members:
            raise ValueError("controlled retest requires at least one pinned regression member")
        if retest["requested_protocol_ref"] != retest["baseline_protocol_ref"]:
            raise ValueError("controlled regression runner currently requires the original protocol")

        profile_name = _safe_profile_name(context["case"].get("profile_name") or "")
        expected_dataset = str(issue_context.get("dataset") or "")
        records = []
        source_artifacts: dict[str, dict] = {}
        for member in members:
            dataset = str(member.get("dataset") or "")
            if expected_dataset and dataset != expected_dataset:
                raise ValueError("regression member dataset differs from the pinned issue snapshot")
            scenario_path = _controlled_scenario_path(
                self.frontend_data_dir / "profiles" / profile_name,
                dataset,
            )
            if scenario_path is None:
                raise ValueError(f"frozen prediction scenario is unavailable for dataset {dataset}")
            source_digest = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
            snapshot_ref = str(issue_context.get("prediction_snapshot_id") or "")
            if snapshot_ref.startswith("prediction-snapshot@sha256:"):
                expected_digest = snapshot_ref.split(":", 1)[1]
                if source_digest != expected_digest:
                    raise ValueError("prediction snapshot digest changed after the issue was created")
            scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
            point = next(
                (
                    item for item in scenario.get("points") or []
                    if str(item.get("cell_id") or "") == str(member.get("asset_id") or "")
                    and str(item.get("cycle_index")) == str(member.get("cycle_id"))
                ),
                None,
            )
            if point is None:
                raise ValueError(f"prediction record is unavailable for {member['sample_id']}")
            true_soh = _finite_number(point.get("true_soh"), "true_soh")
            pred_soh = _finite_number(point.get("pred_soh"), "pred_soh")
            lower = _optional_finite_number(point.get("lower"), "lower")
            upper = _optional_finite_number(point.get("upper"), "upper")
            records.append(
                self._controlled_retest_record(
                    member,
                    true_soh=true_soh,
                    pred_soh=pred_soh,
                    lower=lower,
                    upper=upper,
                )
            )
            source_artifacts[dataset] = {
                "kind": "prediction_snapshot",
                "dataset": dataset,
                "digest": f"sha256:{source_digest}",
                "snapshot_ref": snapshot_ref,
            }

        return self._controlled_retest_result(
            context,
            run_id=run_id,
            execution_mode="evidence_replay",
            runner="controlled_evidence_replay/v1",
            records=records,
            source_artifacts=list(source_artifacts.values()),
            prediction_recomputed=False,
        )

    @staticmethod
    def _controlled_retest_record(
        member: dict,
        *,
        true_soh: float,
        pred_soh: float,
        lower: float | None,
        upper: float | None,
    ) -> dict:
        covered = None if lower is None or upper is None else lower <= true_soh <= upper
        return {
            "sample_id": member["sample_id"],
            "dataset": member["dataset"],
            "asset_id": member["asset_id"],
            "cycle_id": member["cycle_id"],
            "true_soh": true_soh,
            "pred_soh": pred_soh,
            "lower": lower,
            "upper": upper,
            "absolute_error": abs(pred_soh - true_soh),
            "interval_covered": covered,
        }

    def _controlled_retest_baseline_records(self, context: dict) -> list[dict]:
        issue_context = context["issue"].get("context") or {}
        members = context["regression_set"].get("members") or []
        profile_name = _safe_profile_name(context["case"].get("profile_name") or "")
        profile_dir = self.frontend_data_dir / "profiles" / profile_name
        prediction_payload = _read_json_object(profile_dir / "prediction_records.json")
        prediction_records = prediction_payload.get("records") or []
        contribution_payload = _read_json_object(profile_dir / "metric_contributions.json")
        contribution_records = contribution_payload.get("records") or []
        contributions_by_sample = {
            str(item.get("sample_id") or item.get("prediction_record_id") or ""): item
            for item in contribution_records
        }

        baseline_records = []
        for member in members:
            record = next(
                (
                    item for item in prediction_records
                    if _same_prediction_member(item, member)
                ),
                None,
            )
            if prediction_records:
                if record is None:
                    raise ValueError(f"baseline prediction record is unavailable for {member['sample_id']}")
                _verify_prediction_record_identity(record, issue_context)
            else:
                scenario_path = _controlled_scenario_path(
                    profile_dir,
                    str(member.get("dataset") or ""),
                )
                if scenario_path is None:
                    raise ValueError(f"baseline prediction snapshot is unavailable for {member['sample_id']}")
                scenario = _read_json_object(scenario_path)
                record = next(
                    (
                        item for item in scenario.get("points") or []
                        if _same_prediction_member(item, member)
                    ),
                    None,
                )
                if record is None:
                    raise ValueError(f"baseline prediction record is unavailable for {member['sample_id']}")

            baseline = self._controlled_retest_record(
                member,
                true_soh=_finite_number(record.get("true_soh"), "baseline true_soh"),
                pred_soh=_finite_number(record.get("pred_soh"), "baseline pred_soh"),
                lower=_optional_finite_number(record.get("lower"), "baseline lower"),
                upper=_optional_finite_number(record.get("upper"), "baseline upper"),
            )
            contribution = record.get("metric_contributions") or contributions_by_sample.get(
                str(record.get("sample_id") or record.get("prediction_record_id") or member["sample_id"]),
                {},
            )
            if contribution:
                baseline["metric_contributions"] = contribution
            baseline_records.append(baseline)
        return baseline_records

    @staticmethod
    def _controlled_retest_metrics(records: list[dict]) -> dict:
        errors = [record["absolute_error"] for record in records]
        interval_records = [record for record in records if record["interval_covered"] is not None]
        widths = [
            record["upper"] - record["lower"]
            for record in interval_records
            if record["upper"] is not None and record["lower"] is not None
        ]
        return {
            "mae": sum(errors) / len(errors),
            "max_absolute_error": max(errors),
            "picp": (
                sum(1 for record in interval_records if record["interval_covered"]) / len(interval_records)
                if interval_records else None
            ),
            "mean_interval_width": sum(widths) / len(widths) if widths else None,
            "evaluated_count": len(records),
            "interval_count": len(interval_records),
        }

    @staticmethod
    def _controlled_retest_record_deltas(
        baseline_records: list[dict],
        records: list[dict],
    ) -> list[dict]:
        baseline_by_sample = {record["sample_id"]: record for record in baseline_records}
        denominator = len(records)
        deltas = []
        for record in records:
            baseline = baseline_by_sample.get(record["sample_id"])
            if baseline is None:
                raise ValueError(f"baseline comparator is unavailable for {record['sample_id']}")
            absolute_error_delta = record["absolute_error"] - baseline["absolute_error"]
            if absolute_error_delta < -1e-12:
                change = "improved"
            elif absolute_error_delta > 1e-12:
                change = "regressed"
            else:
                change = "unchanged"
            baseline_case_contribution = (baseline.get("metric_contributions") or {}).get(
                "mae_contribution"
            )
            baseline_covered = baseline.get("interval_covered")
            retest_covered = record.get("interval_covered")
            coverage_change = (
                "unavailable" if baseline_covered is None or retest_covered is None
                else "unchanged" if baseline_covered == retest_covered
                else "gained" if retest_covered
                else "lost"
            )
            deltas.append(
                {
                    "sample_id": record["sample_id"],
                    "dataset": record["dataset"],
                    "asset_id": record["asset_id"],
                    "cycle_id": record["cycle_id"],
                    "baseline_prediction": baseline["pred_soh"],
                    "retest_prediction": record["pred_soh"],
                    "prediction_delta": record["pred_soh"] - baseline["pred_soh"],
                    "baseline_absolute_error": baseline["absolute_error"],
                    "retest_absolute_error": record["absolute_error"],
                    "absolute_error_delta": absolute_error_delta,
                    "baseline_case_mae_contribution": baseline_case_contribution,
                    "baseline_subset_mae_contribution": baseline["absolute_error"] / denominator,
                    "retest_subset_mae_contribution": record["absolute_error"] / denominator,
                    "subset_mae_contribution_delta": absolute_error_delta / denominator,
                    "baseline_interval_covered": baseline_covered,
                    "retest_interval_covered": retest_covered,
                    "coverage_change": coverage_change,
                    "change": change,
                }
            )
        return deltas

    def _controlled_retest_result(
        self,
        context: dict,
        *,
        run_id: str,
        execution_mode: str,
        runner: str,
        records: list[dict],
        source_artifacts: list[dict],
        prediction_recomputed: bool,
    ) -> dict:
        retest = context["retest"]
        issue_context = context["issue"].get("context") or {}
        metrics = self._controlled_retest_metrics(records)
        baseline_records = self._controlled_retest_baseline_records(context)
        baseline = self._controlled_retest_metrics(baseline_records)
        record_deltas = self._controlled_retest_record_deltas(baseline_records, records)
        thresholds = dict(issue_context.get("verdict_thresholds") or {})
        checks = []
        if isinstance(thresholds.get("mae_max"), (int, float)):
            checks.append(
                {
                    "criterion": "mae_max",
                    "observed": metrics["mae"],
                    "threshold": thresholds["mae_max"],
                    "passed": metrics["mae"] <= thresholds["mae_max"],
                }
            )
        if isinstance(thresholds.get("picp_min"), (int, float)) and metrics["picp"] is not None:
            checks.append(
                {
                    "criterion": "picp_min",
                    "observed": metrics["picp"],
                    "threshold": thresholds["picp_min"],
                    "passed": metrics["picp"] >= thresholds["picp_min"],
                }
            )
        deltas = {
            key: metrics[key] - baseline[key]
            for key in ("mae", "picp")
            if isinstance(metrics.get(key), (int, float)) and isinstance(baseline.get(key), (int, float))
        }
        failed_criteria = [item["criterion"] for item in checks if not item["passed"]]
        threshold_exceeding_records = []
        for record, record_delta in zip(records, record_deltas):
            failed = []
            if isinstance(thresholds.get("mae_max"), (int, float)) and record["absolute_error"] > thresholds["mae_max"]:
                failed.append("mae_max")
            if record.get("interval_covered") is False:
                failed.append("picp_min")
            if failed:
                threshold_exceeding_records.append(
                    {
                        "sample_id": record["sample_id"],
                        "dataset": record["dataset"],
                        "asset_id": record["asset_id"],
                        "cycle_id": record["cycle_id"],
                        "absolute_error": record["absolute_error"],
                        "absolute_error_delta": record_delta["absolute_error_delta"],
                        "failed_criteria": failed,
                    }
                )
        worst_records = sorted(
            record_deltas,
            key=lambda item: (item["retest_absolute_error"], item["absolute_error_delta"]),
            reverse=True,
        )[:5]
        regression_summary = {
            "member_count": len(record_deltas),
            "improved_count": sum(item["change"] == "improved" for item in record_deltas),
            "unchanged_count": sum(item["change"] == "unchanged" for item in record_deltas),
            "regressed_count": sum(item["change"] == "regressed" for item in record_deltas),
            "worst_records": worst_records,
        }
        result = {
            "schema_version": "1.1.0",
            "comparison_contract": "controlled-retest-comparison/1.0",
            "runner": runner,
            "execution_mode": execution_mode,
            "prediction_recomputed": prediction_recomputed,
            "run_id": run_id,
            "case_id": retest["case_id"],
            "baseline_case_version": retest["baseline_case_version"],
            "issue_id": retest["issue_id"],
            "regression_set_id": retest["regression_set_id"],
            "protocol_ref": retest["requested_protocol_ref"],
            "protocol_unchanged": True,
            "candidate_id": issue_context.get("candidate_id") or "",
            "model_artifact_id": issue_context.get("model_artifact_id") or "",
            "criterion_set_version": issue_context.get("criterion_set_version") or "",
            "metrics": metrics,
            "baseline_metrics": baseline,
            "case_baseline_metrics": dict(issue_context.get("verdict_observed") or {}),
            "metric_deltas": deltas,
            "record_deltas": record_deltas,
            "regression_summary": regression_summary,
            "failure_localization": {
                "datasets": sorted({record["dataset"] for record in records}),
                "assets": sorted({record["asset_id"] for record in records}),
                "protocol_ref": retest["requested_protocol_ref"],
                "split_assignment_id": issue_context.get("split_assignment_id") or "",
                "candidate_id": issue_context.get("candidate_id") or "",
                "model_artifact_id": issue_context.get("model_artifact_id") or "",
                "failed_criteria": failed_criteria,
                "threshold_exceeding_records": threshold_exceeding_records,
                "worst_records": worst_records,
            },
            "checks": checks,
            "verdict": "pass" if checks and all(item["passed"] for item in checks) else (
                "fail" if checks else "not_evaluated"
            ),
            "records": records,
            "source_artifacts": source_artifacts,
            "finished_at": _now(),
        }
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        result["result_digest"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        return result

    def reconcile_case_kpis(self, case_id: str, payload: dict) -> dict:
        return self.review_workflow.reconcile(case_id, payload)

    def create_case_evidence_pack(self, case_id: str, payload: dict) -> dict:
        return self.review_workflow.create_evidence_pack(case_id, payload)

    def sign_evidence_pack(self, pack_id: str, payload: dict) -> dict:
        return self.review_workflow.sign_evidence_pack(pack_id, payload)

    def create_job(self, payload: dict, *, start: bool = True) -> dict:
        preflight = self.training_preflight(payload)
        provided_preflight_id = str(payload.get("preflight_id") or "").strip()
        if provided_preflight_id and provided_preflight_id != preflight["preflight_id"]:
            raise ValueError("training preflight is stale; refresh the combination check")
        if not preflight["ready"]:
            blocked = [
                "; ".join(item.get("reasons") or [])
                for item in preflight["combinations"]
                if item.get("status") == "blocked"
            ]
            reason = next((item for item in blocked if item), "no runnable combinations")
            raise ValueError(f"training preflight blocked: {reason}")
        request = dict(preflight["request"])
        case_context = self._validated_job_case_context(payload)
        if case_context:
            request.update(case_context)
        job_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "kind": "training",
            "status": "queued",
            "created_at": _now(),
            "label_safety_contract": LABEL_SAFETY_CONTRACT,
            "request": request,
            "preflight": preflight,
        }
        self._save_job(job)
        if start:
            thread = threading.Thread(target=self.run_job_inline, args=(job_id,), daemon=True)
            thread.start()
        return job

    def create_inference_job(self, payload: dict, *, start: bool = True) -> dict:
        request = self._normalize_inference_request(payload)
        job_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "kind": "inference",
            "status": "queued",
            "created_at": _now(),
            "request": request,
        }
        self._save_inference_job(job)
        if start:
            thread = threading.Thread(target=self.run_inference_job_inline, args=(job_id,), daemon=True)
            thread.start()
        return job

    def create_quickstart_example(self, *, start: bool = True) -> dict:
        if start and self._quickstart_run_lock.locked():
            raise EvaluationCaseConflictError("a quickstart example is already running")
        example_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        example = {
            "id": example_id,
            "kind": "quickstart_e2e",
            "status": "queued",
            "current_step": "dataset",
            "created_at": _now(),
            "dataset_name": "soh_quickstart_demo",
            "estimator": "ridge",
            "featureset": "voltage",
            "protocol": "loco",
            "steps": _quickstart_steps(),
        }
        self._save_example(example)
        if start:
            thread = threading.Thread(
                target=self.run_quickstart_example_inline,
                args=(example_id,),
                daemon=True,
            )
            thread.start()
        return example

    def run_quickstart_example_inline(self, example_id: str) -> dict:
        # Holding the example *record* lock for the whole run made every
        # other quickstart POST (which saves its record under the same
        # lock) hang until this run finished.
        with self._quickstart_run_lock:
            example = self.get_example(example_id)
            example.update({"status": "running", "started_at": _now()})
            self._save_example(example)
            try:
                self._set_example_step(example, "dataset", "running")
                dataset = self.store.upload_dataset(
                    example["dataset_name"],
                    "soh_quickstart_demo.csv",
                    _quickstart_dataset_csv(),
                    replace=True,
                )
                example["dataset"] = dataset
                self._set_example_step(example, "dataset", "succeeded")

                self._set_example_step(example, "training", "running")
                training_job = self.create_job(
                    {
                        "datasets": [example["dataset_name"]],
                        "estimators": [example["estimator"]],
                        "featuresets": [example["featureset"]],
                        "protocols": [example["protocol"]],
                        "seeds": [0],
                    },
                    start=False,
                )
                example["training_job_id"] = training_job["id"]
                self._save_example(example)
                training_job = self.run_job_inline(training_job["id"], publish=False)
                if training_job["status"] != "succeeded":
                    raise RuntimeError(training_job.get("error") or "quickstart training failed")
                self._set_example_step(example, "training", "succeeded")

                self._set_example_step(example, "model", "running")
                saved_models = list(training_job.get("saved_models") or [])
                if not saved_models:
                    warnings = "; ".join(training_job.get("model_warnings") or [])
                    raise RuntimeError(warnings or "quickstart training did not persist a model")
                example["model_name"] = saved_models[0]
                self._set_example_step(example, "model", "succeeded")

                self._set_example_step(example, "inference", "running")
                inference_job = self.create_inference_job(
                    {
                        "datasets": [example["dataset_name"]],
                        "models": [example["model_name"]],
                        "launch_strategy": "quickstart_e2e",
                        "selected_datasets": [example["dataset_name"]],
                        "submitted_datasets": [example["dataset_name"]],
                        "skipped_datasets": [],
                    },
                    start=False,
                )
                example["inference_job_id"] = inference_job["id"]
                self._save_example(example)
                inference_job = self.run_inference_job_inline(inference_job["id"], publish=False)
                if inference_job["status"] != "succeeded":
                    raise RuntimeError(inference_job.get("error") or "quickstart inference failed")
                example["result"] = _quickstart_result(Path(inference_job["run_dir"]))
                profile_name = "quickstart-e2e"
                profile = self.publish_frontend_profile(
                    profile_name,
                    {
                        "label": "内置端到端示例",
                        "kind": "quickstart_e2e",
                        "datasets": [example["dataset_name"]],
                        "training_job_id": example["training_job_id"],
                        "inference_job_id": example["inference_job_id"],
                    },
                    source_dirs=[training_job["run_dir"], inference_job["run_dir"]],
                )
                example["profile_name"] = profile_name
                example["profile_path"] = profile["path"]
                self._set_example_step(example, "inference", "succeeded")
                example.update(
                    {
                        "status": "succeeded",
                        "current_step": "completed",
                        "finished_at": _now(),
                    }
                )
            except Exception as exc:
                current_step = str(example.get("current_step") or "dataset")
                self._set_example_step(example, current_step, "failed", error=str(exc))
                example.update(
                    {
                        "status": "failed",
                        "finished_at": _now(),
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=12),
                    }
                )
            self._save_example(example)
            return example

    def get_example(self, example_id: str) -> dict:
        path = self.examples_dir / f"{_safe_entity_id(example_id)}.json"
        if not path.exists():
            raise KeyError(example_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_examples(self) -> list[dict]:
        examples = []
        for path in sorted(self.examples_dir.glob("*.json"), reverse=True):
            try:
                examples.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return sorted(examples, key=_job_sort_key, reverse=True)

    def _set_example_step(
        self,
        example: dict,
        step_id: str,
        status: str,
        *,
        error: str = "",
    ) -> None:
        for step in example.get("steps", []):
            if step.get("id") != step_id:
                continue
            step["status"] = status
            if status == "running":
                step["started_at"] = _now()
                example["current_step"] = step_id
            elif status in {"succeeded", "failed"}:
                step["finished_at"] = _now()
            if error:
                step["error"] = error
            break
        self._save_example(example)

    def run_job_inline(self, job_id: str, *, publish: bool = True) -> dict:
        job = self.get_job(job_id)
        job.update({"status": "running", "started_at": _now()})
        self._save_job(job)
        try:
            request = job["request"]
            current_preflight = self.training_preflight(request)
            admitted_preflight_id = str((job.get("preflight") or {}).get("preflight_id") or "")
            if admitted_preflight_id and current_preflight["preflight_id"] != admitted_preflight_id:
                raise RuntimeError("dataset or combination contract changed after job admission")
            if not current_preflight["ready"]:
                raise RuntimeError("training preflight no longer passes")
            loader = DatasetLoader(self.data_dir)
            datasets = loader.resolve_datasets(request["datasets"])
            table, meta, qc = loader.load(datasets)
            effective_featuresets = _effective_training_featuresets(
                request["estimators"], request["featuresets"]
            )
            table = self.store.apply_feature_transforms(table, meta, effective_featuresets)
            estimators = [R.get_estimator(name) for name in request["estimators"]]
            # Label-safety gate on the actual training table: passthrough
            # featuresets and the BYO track bypass transform-level audits, so
            # audit their concrete columns here before any model is fitted.
            label_safety_findings: list[str] = []
            for fs_name in dict.fromkeys(str(name) for name in request["featuresets"]):
                try:
                    spec = R.get_featureset(fs_name)
                except KeyError:
                    continue
                candidates = [
                    col for col in (getattr(spec, "candidates", None) or [])
                    if col in table.columns
                ]
                if not candidates:
                    continue
                audit = audit_feature_frame(table, candidates)
                if audit["status"] != "passed":
                    label_safety_findings.extend(
                        f"{fs_name}: {item['reason']}" for item in audit["violations"]
                    )
            if any(getattr(est, "feature_mode", "") == "byo" for est in estimators):
                # Audit the columns the BYO track can actually observe: the
                # runner strips structurally banned label-source columns
                # before any BYO estimator sees the table.
                byo_columns = [
                    col for col in table.columns
                    if col not in ("dataset", "cell_id", "cycle_index", "soh")
                    and not is_label_source_feature(col)
                ]
                audit = audit_feature_frame(table, byo_columns)
                if audit["status"] != "passed":
                    label_safety_findings.extend(
                        f"(byo): {item['reason']}" for item in audit["violations"]
                    )
            if label_safety_findings:
                raise FeatureSafetyError(
                    "label-safety audit blocked training: "
                    + " ; ".join(sorted(set(label_safety_findings)))
                )
            protocols = {name: PROTOCOLS[name]() for name in request["protocols"]}
            saved_models = []
            model_warnings = []

            def save_fitted_model(**kwargs) -> None:
                record = kwargs["record"]
                fitted = kwargs["estimator"]
                train_table = kwargs["train_table"]
                fs_name = str(record.get("featureset", "full"))
                model_name = "_".join(
                    [
                        _model_name_token(record["estimator"]),
                        _model_name_token(fs_name),
                        _model_name_token(record["protocol"]),
                        _model_name_token(record["held_out_key"]),
                        f"seed{int(record['seed'])}",
                    ]
                )
                manifest = {
                    "model_name": model_name,
                    "model_kind": "pickle",
                    "estimator": str(record["estimator"]),
                    "featureset": fs_name,
                    # The transforms that produced the training table; needed
                    # to rebuild derived feature columns at inference-only
                    # (BYO models have no reconstructable featureset of their
                    # own — this list is their feature recipe).
                    "training_featuresets": list(effective_featuresets),
                    "feature_columns": feature_columns(train_table),
                    "input_columns": list(map(str, train_table.columns)),
                    "target": "soh",
                    "trained_on": sorted(map(str, train_table["dataset"].dropna().unique())),
                    "held_out_key": str(record["held_out_key"]),
                    "protocol": str(record["protocol"]),
                    "seed": int(record["seed"]),
                    "created_by_run": job_id,
                    "predict_contract": "predict(X, meta=None)",
                    "label_safety": audit_feature_frame(
                        train_table,
                        feature_columns(train_table),
                    ),
                }
                try:
                    saved_models.append(
                        self.model_store.save_training_model(
                            model_name,
                            fitted,
                            manifest,
                            replace=True,
                        )
                    )
                except Exception as exc:
                    model_warnings.append(f"{model_name}: {exc}")

            results = BenchmarkRunner(seeds=request["seeds"]).run(
                SOHEstimationTask(),
                table=table,
                meta=meta,
                protocols=protocols,
                estimators=estimators,
                featuresets=request["featuresets"],
                on_model_fitted=save_fitted_model,
            )
            failures = _training_failures(results)
            if failures:
                first = failures[0]
                raise RuntimeError(
                    f"training produced {len(failures)} failed fold(s); "
                    f"{first['estimator']} / {first['featureset']} / {first['protocol']} / "
                    f"{first['held_out_key']}: {first['error']}"
                )
            results["dataset"] = ",".join(datasets)
            model_artifacts = [
                {
                    key: item.get(key)
                    for key in (
                        "name", "artifact_id", "estimator", "featureset", "protocol",
                        "held_out_key", "seed", "contract_version", "manifest_sha256",
                        "model_sha256",
                    )
                }
                for item in saved_models
            ]
            raw_run_dir = Path(
                write_artifacts(
                    results,
                    qc_report=qc,
                    task=SOHEstimationTask(),
                    config={
                        **request,
                        "datasets": datasets,
                        "job_id": job_id,
                        "run_id": job_id,
                        "preflight_id": current_preflight["preflight_id"],
                        "preflight_contract": current_preflight["contract"],
                        "label_safety_contract": LABEL_SAFETY_CONTRACT,
                        "dataset_snapshots": current_preflight["dataset_snapshots"],
                        "trusted_local_plugins": True,
                        "saved_models": [item["name"] for item in saved_models],
                        "model_artifacts": model_artifacts,
                        "model_warnings": model_warnings,
                    },
                    out_dir=self.runs_dir,
                )
            )
            (raw_run_dir / "model_catalog.json").write_text(
                json.dumps(
                    {"models": self.model_store.catalog()},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            published_run_dir = self._publish_frontend_artifacts(raw_run_dir) if publish else raw_run_dir
            job.update(
                {
                    "status": "succeeded",
                    "finished_at": _now(),
                    "raw_run_dir": str(raw_run_dir),
                    "run_dir": str(published_run_dir),
                    "run_id": job_id,
                    "preflight_verified_at": current_preflight["checked_at"],
                    "preflight_id": current_preflight["preflight_id"],
                    "label_safety_contract": LABEL_SAFETY_CONTRACT,
                    "saved_models": [item["name"] for item in saved_models],
                    "model_artifacts": model_artifacts,
                    "model_warnings": model_warnings,
                    "result_summary": _training_result_summary(
                        results,
                        datasets=datasets,
                        model_artifacts=model_artifacts,
                    ),
                }
            )
        except Exception as exc:
            job.update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=12),
                }
            )
        self._save_job(job)
        return job

    def run_inference_job_inline(self, job_id: str, *, publish: bool = True) -> dict:
        job = self.get_inference_job(job_id)
        job.update({"status": "running", "started_at": _now()})
        self._save_inference_job(job)
        try:
            request = job["request"]
            loader = DatasetLoader(self.data_dir)
            datasets = loader.resolve_datasets(request["datasets"])
            table, meta, _qc = loader.load(datasets)
            model_bindings = list(request.get("model_bindings") or [])
            if len(model_bindings) != len(request["models"]):
                raise RuntimeError("inference job has no complete immutable model binding")
            inferred_featuresets = _inference_transform_featuresets(model_bindings)
            table = self.store.apply_feature_transforms(table, meta, inferred_featuresets)
            dataset_snapshot_ids = {
                name: f"dataset-snapshot@sha256:{_sha256_path(self.data_dir / 'processed' / f'{name}_features.pkl')}"
                for name in datasets
            }
            records, evidence = run_inference_benchmark_with_evidence(
                self.model_store,
                table,
                meta,
                request["models"],
                job_id=job_id,
                dataset_snapshot_ids=dataset_snapshot_ids,
                evaluation_policy=request.get("evaluation_policy", "research_any"),
                model_bindings=model_bindings,
                adapter=self._inference_adapter(request),
            )
            curve_artifacts, curve_refs = self._inference_curve_evidence(
                evidence["prediction_records"],
                datasets,
            )
            evidence["curve_artifacts"] = curve_artifacts
            evidence["curve_segment_refs"] = curve_refs
            model_manifests = []
            for item in model_bindings:
                try:
                    model_manifests.append(
                        self.model_store.load_artifact(item["artifact_id"]).manifest
                    )
                except Exception:
                    continue
            protocols = sorted({
                str(manifest.get("protocol") or "")
                for manifest in model_manifests
                if manifest.get("protocol")
            })
            seeds = sorted({
                int(manifest["seed"])
                for manifest in model_manifests
                if manifest.get("seed") is not None
            })
            raw_run_dir = Path(
                write_inference_artifacts(
                    records,
                    {
                        **request,
                        "datasets": datasets,
                        "job_id": job_id,
                        "trusted_local_pickle_models": True,
                        "applied_featureset_transforms": inferred_featuresets,
                        "dataset_snapshots": [
                            {"dataset": name, "snapshot_id": dataset_snapshot_ids[name]}
                            for name in datasets
                        ],
                        "protocols": protocols,
                        "seeds": seeds,
                    },
                    self.runs_dir,
                    evidence=evidence,
                )
            )
            write_prediction_trajectory_index(raw_run_dir, evidence["prediction_records"])
            config_path = raw_run_dir / "inference_run_config.json"
            if config_path.exists():
                run_config = json.loads(config_path.read_text(encoding="utf-8"))
                run_config.update({"job_id": job_id, "run_dir": str(raw_run_dir), "raw_run_dir": str(raw_run_dir)})
                config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
            (raw_run_dir / "model_catalog.json").write_text(
                json.dumps(
                    {"models": model_bindings},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            published_run_dir = self._publish_frontend_artifacts(raw_run_dir) if publish else raw_run_dir
            job.update(
                {
                    "status": "succeeded",
                    "finished_at": _now(),
                    "raw_run_dir": str(raw_run_dir),
                    "run_dir": str(published_run_dir),
                    "run_id": job_id,
                    "evidence_summary": {
                        "prediction_records": len(evidence["prediction_records"]),
                        "curve_segment_refs": len(curve_refs),
                        "curve_datasets": [item["dataset"] for item in curve_artifacts],
                    },
                }
            )
        except Exception as exc:
            job.update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=12),
                }
            )
        self._save_inference_job(job)
        return job

    def get_job(self, job_id: str) -> dict:
        path = self.jobs_dir / f"{_safe_entity_id(job_id)}.json"
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def get_job_artifacts(self, job_id: str) -> dict:
        return self._read_job_artifacts(
            self.get_job(job_id),
            kind="training",
            names=("leaderboard.json", "dataset_leaderboard.json", "qc_report.json", "run_config.json"),
        )

    def list_jobs(self) -> list[dict]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return sorted(jobs, key=_job_sort_key, reverse=True)

    def get_inference_job(self, job_id: str) -> dict:
        path = self.inference_jobs_dir / f"{_safe_entity_id(job_id)}.json"
        if not path.exists():
            raise KeyError(job_id)
        with self._inference_job_lock:
            raw = json.loads(path.read_text(encoding="utf-8"))
            migrated = self._migrate_inference_job_document(raw)
            if migrated != raw:
                self._save_inference_job(migrated)
            return migrated

    def get_inference_job_artifacts(self, job_id: str) -> dict:
        return self._read_job_artifacts(
            self.get_inference_job(job_id),
            kind="inference",
            names=(
                "inference_benchmark.json",
                "inference_run_config.json",
                "inference_evidence_manifest.json",
            ),
        )

    def _training_saved_combinations(self, job_id: str, criterion_set_id: str) -> list[dict]:
        job = self.get_job(job_id)
        if job.get("status") != "succeeded":
            raise ValueError("only completed training jobs can be saved")
        payload = self.get_job_artifacts(job_id).get("artifacts") or {}
        leaderboard = payload.get("leaderboard") or []
        if not isinstance(leaderboard, list) or not leaderboard:
            raise ValueError("completed training job has no authoritative leaderboard")
        artifacts = [
            item for item in (job.get("model_artifacts") or [])
            if item.get("artifact_id")
            and not self.model_store.is_artifact_invalidated(str(item.get("artifact_id") or ""))
        ]
        by_group: dict[tuple[str, str, str], list[dict]] = {}
        for artifact in artifacts:
            key = (
                str(artifact.get("estimator") or ""),
                str(artifact.get("featureset") or "full"),
                str(artifact.get("protocol") or ""),
            )
            by_group.setdefault(key, []).append(artifact)
        summary = job.get("result_summary") or {}
        training_datasets = _string_list(
            (job.get("request") or {}).get("datasets")
            or summary.get("datasets")
            or []
        )
        snapshot_rows = (job.get("preflight") or {}).get("dataset_snapshots") or []
        snapshot_by_dataset = {
            str(item.get("dataset") or ""): str(item.get("snapshot_id") or "")
            for item in snapshot_rows
        }
        output = []
        for row in leaderboard:
            estimator = str(row.get("estimator") or "")
            featureset = str(row.get("featureset") or "full")
            protocol = str(row.get("protocol") or "")
            group = by_group.get((estimator, featureset, protocol), [])
            if not group:
                continue
            preferred_held_out = (
                str(summary.get("best_held_out_key") or "")
                if estimator == str(summary.get("best_estimator") or "")
                and featureset == str(summary.get("best_featureset") or "")
                and protocol == str(summary.get("best_protocol") or protocol)
                else ""
            )
            champion_artifact = next(
                (
                    item for item in group
                    if preferred_held_out
                    and str(item.get("held_out_key") or "") == preferred_held_out
                ),
                group[0],
            )
            # LODO folds are not interchangeable: each fold is trained on a
            # different dataset composition and is its own deployable model,
            # so every fold becomes a selectable combination.  Cell-level
            # protocols keep the single champion (77 fungible folds must not
            # flood the case).
            selected_artifacts = group if protocol == "lodo" else [champion_artifact]
            metrics = _flatten_saved_metrics(row.get("metrics") or {})
            verdict = derive_verdict(metrics, criterion_set_id)
            combo_is_best = (
                estimator == str(summary.get("best_estimator") or "")
                and featureset == str(summary.get("best_featureset") or "")
                and protocol == str(summary.get("best_protocol") or protocol)
            )
            for artifact in selected_artifacts:
                artifact_id = str(artifact.get("artifact_id") or "")
                held_out_key = str(artifact.get("held_out_key") or "")
                fold_note = (
                    f" · 留出 {held_out_key}" if protocol == "lodo" and held_out_key else ""
                )
                candidate = {
                    "candidate_id": _artifact_candidate_id(artifact_id),
                    "model": estimator,
                    "featureset": featureset,
                    "role": "champion" if (
                        combo_is_best
                        and artifact_id == str(champion_artifact.get("artifact_id") or "")
                    ) else "candidate",
                    "model_artifact_id": artifact_id,
                    "run_id": str(job.get("run_id") or job_id),
                }
                identity = {
                    "source_stage": "training",
                    "job_id": job_id,
                    "artifact_id": artifact_id,
                    "protocol": protocol,
                }
                output.append({
                    "saved_combination_id": f"combo:{_canonical_digest(identity)[:40]}",
                    "source_stage": "training",
                    "validation_kind": "training_internal",
                    "label": f"{estimator} × {featureset} · 训练内评估{fold_note}",
                    "candidate": candidate,
                    "training_job_id": job_id,
                    "inference_job_id": "",
                    "run_id": str(job.get("run_id") or job_id),
                    "training_datasets": training_datasets,
                    "target_dataset": "",
                    "dataset_snapshot_id": (
                        snapshot_by_dataset.get(training_datasets[0], "")
                        if len(training_datasets) == 1 else ""
                    ),
                    "protocol": protocol,
                    "criterion_set_version": criterion_set_id,
                    "metrics": metrics,
                    "verdict_result": verdict["result"],
                })
        return output

    def _inference_saved_combinations(self, job_id: str, criterion_set_id: str) -> list[dict]:
        job = self.get_inference_job(job_id)
        if job.get("status") != "succeeded":
            raise ValueError("only completed inference jobs can be saved")
        payload = self.get_inference_job_artifacts(job_id).get("artifacts") or {}
        benchmark = payload.get("inference_benchmark") or {}
        records = benchmark.get("records") or []
        run_config = payload.get("inference_run_config") or {}
        if not isinstance(records, list) or not records:
            raise ValueError("completed inference job has no authoritative benchmark records")
        models_by_name = {
            str(item.get("name") or ""): item
            for item in (job.get("request") or {}).get("model_bindings") or []
        }
        snapshot_by_dataset = {
            str(item.get("dataset") or ""): str(item.get("snapshot_id") or "")
            for item in (run_config.get("dataset_snapshots") or [])
        }
        output = []
        for record in records:
            if record.get("status") != "succeeded":
                continue
            model_name = str(record.get("model") or "")
            model = models_by_name.get(model_name) or {}
            artifact_id = str(
                record.get("model_artifact_id") or model.get("artifact_id") or ""
            )
            if not artifact_id:
                continue
            if self.model_store.is_artifact_invalidated(artifact_id):
                continue
            estimator = str(record.get("estimator") or model.get("estimator") or model_name)
            featureset = str(record.get("featureset") or model.get("featureset") or "full")
            dataset = str(record.get("dataset") or "")
            training_datasets = _string_list(model.get("trained_on") or [])
            metrics = _flatten_saved_metrics(record)
            verdict = derive_verdict(metrics, criterion_set_id)
            candidate = {
                "candidate_id": str(record.get("candidate_id") or _artifact_candidate_id(artifact_id)),
                "model": estimator,
                "featureset": featureset,
                "role": "candidate",
                "model_artifact_id": artifact_id,
                "run_id": str(model.get("created_by_run") or ""),
            }
            identity = {
                "source_stage": "inference",
                "job_id": job_id,
                "artifact_id": artifact_id,
                "dataset": dataset,
            }
            validation_kind = "replay" if dataset in training_datasets else "cross_dataset"
            output.append({
                "saved_combination_id": f"combo:{_canonical_digest(identity)[:40]}",
                "source_stage": "inference",
                "validation_kind": validation_kind,
                "label": f"{estimator} × {featureset} · {dataset}",
                "candidate": candidate,
                "training_job_id": str(model.get("created_by_run") or ""),
                "inference_job_id": job_id,
                "run_id": str(job.get("run_id") or job_id),
                "training_datasets": training_datasets,
                "target_dataset": dataset,
                "dataset_snapshot_id": snapshot_by_dataset.get(dataset, ""),
                "protocol": str(model.get("protocol") or ""),
                "criterion_set_version": criterion_set_id,
                "metrics": metrics,
                "verdict_result": verdict["result"],
            })
        return output

    def _read_job_artifacts(self, job: dict, *, kind: str, names: tuple[str, ...]) -> dict:
        raw_run_dir = str(job.get("raw_run_dir") or job.get("run_dir") or "").strip()
        if not raw_run_dir:
            raise ValueError(f"{kind} job has no completed run artifacts")
        run_dir = Path(raw_run_dir).resolve()
        runs_root = self.runs_dir.resolve()
        if not run_dir.is_relative_to(runs_root) or not run_dir.is_dir():
            # Job records written on another machine store that host's absolute
            # paths (e.g. a Windows drive path <drive>:\SOH_eval\runs\...); relocate by run-dir name under
            # the local runs root before giving up.
            leaf = raw_run_dir.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            relocated = (runs_root / leaf).resolve() if leaf else runs_root
            if not leaf or not relocated.is_relative_to(runs_root) or relocated == runs_root or not relocated.is_dir():
                raise ValueError(f"{kind} job run directory is unavailable")
            run_dir = relocated
        artifacts = {}
        for name in names:
            path = run_dir / name
            if path.is_file():
                artifacts[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
        return {
            "job_id": str(job.get("id") or ""),
            "kind": kind,
            "status": str(job.get("status") or ""),
            "run_dir": str(run_dir),
            "artifacts": artifacts,
        }

    def list_inference_jobs(self) -> list[dict]:
        jobs = []
        for path in sorted(self.inference_jobs_dir.glob("*.json"), reverse=True):
            try:
                jobs.append(self.get_inference_job(path.stem))
            except (json.JSONDecodeError, ValueError):
                continue
        return sorted(jobs, key=_job_sort_key, reverse=True)

    # ------------------------------------------------------------------
    # Agent search jobs (protocol-gated algorithm x featureset selection)
    # ------------------------------------------------------------------

    def create_search_job(self, payload: dict, *, start: bool = True) -> dict:
        mode = str(payload.get("mode") or "training").strip().lower()
        if mode not in {"training", "adaptation"}:
            raise ValueError(f"unknown search mode: {mode}")
        if mode == "adaptation":
            from .adaptation import ADAPTATION_OPERATORS, CALIBRATION_SPLITS, INTERVAL_MODES

            target = str(payload.get("target_dataset") or "").strip()
            models = _string_list(payload.get("models") or [])
            operators = _string_list(payload.get("operators") or []) or list(ADAPTATION_OPERATORS)
            if not target:
                raise ValueError("target_dataset is required for an adaptation search")
            if not models:
                raise ValueError("models is required for an adaptation search")
            unknown_operators = [op for op in operators if op not in ADAPTATION_OPERATORS]
            if unknown_operators:
                raise ValueError(f"unknown adaptation operators: {unknown_operators}")
            known_models = {item["name"] for item in self.model_store.catalog()}
            missing = [name for name in models if name not in known_models]
            if missing:
                raise ValueError(f"unknown models: {missing[:5]}")
            calibration_split = str(payload.get("calibration_split") or "earliest")
            if calibration_split not in CALIBRATION_SPLITS:
                raise ValueError(f"unknown calibration split: {calibration_split}")
            interval_mode = str(payload.get("interval") or "model")
            if interval_mode not in INTERVAL_MODES:
                raise ValueError(f"unknown interval mode: {interval_mode}")
            request_extra = {
                "mode": "adaptation",
                "target_dataset": target,
                "models": models,
                "operators": operators,
                "calibration_budget": max(0, int(payload.get("calibration_budget") or 32)),
                "calibration_split": calibration_split,
                "interval": interval_mode,
            }
            inner, outer, estimators, featuresets = [], "", [], []
        else:
            inner = _string_list(payload.get("inner_datasets") or [])
            outer = str(payload.get("outer_dataset") or "").strip()
            estimators = _string_list(payload.get("estimators") or [])
            featuresets = _string_list(payload.get("featuresets") or [])
            if not inner:
                raise ValueError("inner_datasets is required for an agent search")
            if not outer:
                raise ValueError("outer_dataset is required for an agent search")
            if outer in inner:
                raise ValueError("outer dataset must not appear in the inner search pool")
            if not estimators:
                raise ValueError("estimators is required for an agent search")
            if not featuresets:
                raise ValueError("featuresets is required for an agent search")
            for name in estimators:
                R.get_estimator(name)
            for name in featuresets:
                R.get_featureset(name)
            request_extra = {"mode": "training"}
        policy = str(payload.get("policy") or "grid").strip().lower()
        if policy not in {"grid", "llm"}:
            raise ValueError(f"unknown search policy: {policy}")
        if policy == "llm":
            from .llm_policy import llm_transport_status, load_llm_config

            status = llm_transport_status(load_llm_config(self.work_dir))
            if not status["available"]:
                raise ValueError(f"AI 策略不可用 — {status['detail']}")
        window_targets = [
            float(target) for target in (payload.get("window_targets") or [])
        ]
        for target in window_targets:
            if not 0.1 <= target <= 0.9:
                raise ValueError(f"window target out of range: {target}")
        top_k = int(payload.get("top_k") or 3)
        seeds = [int(seed) for seed in (payload.get("seeds") or [0])]
        job_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "kind": "agent_search",
            "status": "queued",
            "created_at": _now(),
            "request": {
                "inner_datasets": inner,
                "outer_dataset": outer,
                "estimators": estimators,
                "featuresets": featuresets,
                "window_targets": window_targets,
                "policy": policy,
                "top_k": max(1, top_k),
                "seeds": seeds,
                "protocol": "loco",
                **request_extra,
            },
            "progress": [],
        }
        self._save_search_job(job)
        if start:
            thread = threading.Thread(
                target=self.run_search_job_inline, args=(job_id,), daemon=True
            )
            thread.start()
        return job

    def run_search_job_inline(self, job_id: str) -> dict:
        from .agent_search import AgentSearch, GridPolicy

        job = self.get_search_job(job_id)
        job.update({"status": "running", "started_at": _now()})
        self._save_search_job(job)

        def log(message: str) -> None:
            job["progress"].append(f"{_now()} {message}".strip())
            self._save_search_job(job)

        try:
            request = job["request"]
            use_llm = str(request.get("policy") or "grid") == "llm"
            if str(request.get("mode") or "training") == "adaptation":
                from .agent_search import AdaptationGridPolicy, AdaptationSearch

                search = AdaptationSearch(
                    self,
                    target_dataset=request["target_dataset"],
                    calibration_budget=int(request.get("calibration_budget") or 32),
                    calibration_split=str(request.get("calibration_split") or "earliest"),
                    interval=str(request.get("interval") or "model"),
                    top_k=int(request.get("top_k") or 3),
                )
                if use_llm:
                    from .llm_policy import AdaptationLLMPolicy, load_llm_config

                    requested = set(request["models"])
                    entries = [
                        item for item in self.model_store.catalog()
                        if item["name"] in requested
                    ]
                    policy = AdaptationLLMPolicy(
                        models=entries,
                        target_dataset=request["target_dataset"],
                        calibration_budget=int(request.get("calibration_budget") or 32),
                        config=load_llm_config(self.work_dir),
                        allowed_operators=list(request.get("operators") or []),
                    )
                else:
                    policy = AdaptationGridPolicy(request["models"], request["operators"])
            else:
                search = AgentSearch(
                    self,
                    inner_datasets=request["inner_datasets"],
                    outer_dataset=request["outer_dataset"],
                    seeds=request.get("seeds") or [0],
                    top_k=int(request.get("top_k") or 3),
                )
                if use_llm:
                    from .llm_policy import LLMPolicy, load_llm_config

                    policy = LLMPolicy(
                        estimators=request["estimators"],
                        featuresets=request["featuresets"],
                        inner_datasets=request["inner_datasets"],
                        window_targets_allowed=True,
                        config=load_llm_config(self.work_dir),
                    )
                else:
                    policy = GridPolicy(
                        request["estimators"],
                        request["featuresets"],
                        request.get("window_targets") or None,
                    )
            report = search.run(policy, log=log)
            job.update(
                {
                    "status": "succeeded",
                    "finished_at": _now(),
                    "report": report,
                    "result_summary": {
                        "candidates_scored": len(report.get("ranking") or []),
                        "candidates_excluded": len(report.get("excluded") or []),
                        "best_candidate": (report.get("ranking") or [{}])[0].get("candidate", ""),
                        "best_outer_mae": (
                            (report.get("outer_claims") or [{}])[0].get("outer_mae")
                            if report.get("outer_claims")
                            else (report.get("ranking") or [{}])[0].get("target_mae")
                        ),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - job records carry the failure
            job.update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        self._save_search_job(job)
        return job

    def get_search_job(self, job_id: str) -> dict:
        path = self.search_jobs_dir / f"{_safe_entity_id(job_id)}.json"
        if not path.is_file():
            raise KeyError(job_id)
        with self._search_job_lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def list_search_jobs(self) -> list[dict]:
        jobs = []
        for path in sorted(self.search_jobs_dir.glob("*.json"), reverse=True):
            try:
                jobs.append(self.get_search_job(path.stem))
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        return sorted(jobs, key=_job_sort_key, reverse=True)

    def _save_search_job(self, job: dict) -> None:
        with self._search_job_lock:
            _write_json_atomic(self.search_jobs_dir / f"{job['id']}.json", job)

    def _inference_adapter(self, request: dict):
        spec = request.get("adaptation")
        if not spec:
            return None
        from .adaptation import InferenceAdapter

        return InferenceAdapter(
            self,
            str(spec.get("operator") or "none"),
            int(spec.get("calibration_budget") or 32),
            calibration_split=str(spec.get("calibration_split") or "earliest"),
            interval=str(spec.get("interval") or "model"),
        )

    def get_llm_config(self) -> dict:
        from .llm_policy import llm_transport_status, load_llm_config, masked_llm_config

        config = load_llm_config(self.work_dir)
        return {
            "config": masked_llm_config(config),
            "status": llm_transport_status(config),
        }

    def set_llm_config(self, payload: dict) -> dict:
        from .llm_policy import llm_transport_status, load_llm_config, save_llm_config

        masked = save_llm_config(self.work_dir, payload or {})
        return {
            "config": masked,
            "status": llm_transport_status(load_llm_config(self.work_dir)),
        }

    def _require_known_datasets(self, datasets: list[str]) -> None:
        """Unknown names used to surface as FileNotFoundError deep inside the
        pipeline (a 500 that leaked the server's data path); reject them at
        request normalization with a 400-mapped ValueError instead."""
        from .data import ALL_DATASETS

        missing = [
            name for name in datasets
            if str(name).lower() not in ALL_DATASETS
            and not (self.data_dir / "processed" / f"{name}_features.pkl").is_file()
        ]
        if missing:
            raise ValueError(f"unknown datasets: {missing}")

    def _normalize_request(self, payload: dict) -> dict:
        datasets = _string_list(payload.get("datasets") or ["all"])
        self._require_known_datasets(datasets)
        estimators = _string_list(payload.get("estimators") or ["ridge"])
        featuresets = _string_list(payload.get("featuresets") or ["full"])
        protocols = _string_list(payload.get("protocols") or ["loco"])
        seeds = [int(seed) for seed in (payload.get("seeds") or [0])]
        unknown_protocols = [name for name in protocols if name not in PROTOCOLS]
        if unknown_protocols:
            raise ValueError(f"unknown protocols: {unknown_protocols}")
        for name in estimators:
            R.get_estimator(name)
        for name in featuresets:
            R.get_featureset(name)
        return {
            "datasets": datasets,
            "estimators": estimators,
            "featuresets": featuresets,
            "protocols": protocols,
            "seeds": seeds,
        }

    def _normalize_inference_request(self, payload: dict) -> dict:
        datasets = _string_list(payload.get("datasets") or ["all"])
        self._require_known_datasets(datasets)
        models = _string_list(payload.get("models") or [])
        if not models:
            raise ValueError("at least one model is required")
        catalog = {
            str(item.get("name") or ""): item
            for item in self.model_store.catalog()
        }
        known = set(catalog)
        missing = [name for name in models if name not in known]
        if missing:
            raise ValueError(f"unknown models: {missing}")
        invalid = [
            name for name in models
            if not bool(catalog[name].get("usable", True))
        ]
        if invalid:
            reasons = {
                name: str(
                    ((catalog[name].get("validation") or {}).get("invalidation") or {}).get("reason")
                    or "label-safety policy"
                )
                for name in invalid
            }
            raise ValueError(f"models are invalidated and cannot be used for inference: {reasons}")
        model_bindings = [
            _immutable_model_binding(catalog[name])
            for name in models
        ]
        evaluation_policy = str(payload.get("evaluation_policy") or "research_any")
        if evaluation_policy not in {"held_out_same_chemistry", "research_any"}:
            raise ValueError(f"unknown inference evaluation policy: {evaluation_policy}")
        adaptation_request = None
        if payload.get("adaptation"):
            from .adaptation import ADAPTATION_OPERATORS, CALIBRATION_SPLITS, INTERVAL_MODES, OPERATOR_CHANNELS

            spec = payload["adaptation"] if isinstance(payload["adaptation"], dict) else {}
            operator = str(spec.get("operator") or "").strip()
            if operator not in ADAPTATION_OPERATORS:
                raise ValueError(f"unknown adaptation operator: {operator!r}")
            calibration_split = str(spec.get("calibration_split") or "earliest")
            if calibration_split not in CALIBRATION_SPLITS:
                raise ValueError(f"unknown calibration split: {calibration_split}")
            interval_mode = str(spec.get("interval") or "model")
            if interval_mode not in INTERVAL_MODES:
                raise ValueError(f"unknown interval mode: {interval_mode}")
            channel = OPERATOR_CHANNELS[operator]
            if interval_mode == "conformal":
                channel += "; interval: split-conformal quantile from the same K calibration residuals"
            adaptation_request = {
                "operator": operator,
                "calibration_budget": max(0, int(spec.get("calibration_budget") or 32)),
                "calibration_split": calibration_split,
                "interval": interval_mode,
                "channel": channel,
            }
        request = {
            "datasets": datasets,
            "models": models,
            "model_bindings": model_bindings,
            "model_artifact_ids": [item["artifact_id"] for item in model_bindings],
            "launch_strategy": str(payload.get("launch_strategy") or "unspecified"),
            "evaluation_policy": evaluation_policy,
            "selected_models": _string_list(payload.get("selected_models") or models),
            "submitted_models": _string_list(payload.get("submitted_models") or models),
            "skipped_models": _string_list(payload.get("skipped_models") or []),
            "selected_datasets": _string_list(payload.get("selected_datasets") or datasets),
            "submitted_datasets": _string_list(payload.get("submitted_datasets") or datasets),
            "skipped_datasets": _string_list(payload.get("skipped_datasets") or []),
        }
        if adaptation_request:
            request["adaptation"] = adaptation_request
        case_id = str(payload.get("case_id") or "").strip()
        profile_name = str(payload.get("profile_name") or "").strip()
        if case_id:
            binding_id = str(payload.get("expected_context_binding_id") or "").strip()
            case = self.evaluation_cases.assert_context_binding(case_id, binding_id)
            case_profile = str(case.get("profile_name") or "")
            if profile_name and profile_name != case_profile:
                raise ValueError("inference profile does not match the evaluation case")
            request["case_id"] = case_id
            request["profile_name"] = case_profile
            request["case_context_binding_id"] = binding_id
            request["case_context_generation"] = int(
                (case.get("context_binding") or {}).get("generation") or 0
            )
            bound_artifacts = {
                str(item.get("model_artifact_id") or "")
                for item in case.get("candidates") or []
                if item.get("model_artifact_id")
            }
            unbound = [
                item["name"] for item in model_bindings
                if item["artifact_id"] not in bound_artifacts
            ]
            if unbound:
                raise EvaluationCaseConflictError(
                    f"inference models are not bound to the current case context: {unbound}"
                )
        elif profile_name:
            request["profile_name"] = _safe_profile_name(profile_name)
        return request

    def _migrate_inference_job_document(self, job: dict) -> dict:
        """Upgrade legacy inference jobs to immutable artifact bindings.

        Completed legacy jobs are backfilled from the catalog snapshot written
        into their own run directory. The live model catalog is deliberately
        never used for this migration because a logical model name may have
        been replaced after the job ran.
        """
        migrated = dict(job or {})
        request = dict(migrated.get("request") or {})
        bindings = [
            _immutable_model_binding(item)
            for item in request.get("model_bindings") or []
            if item.get("name") and item.get("artifact_id")
        ]
        if not bindings:
            bindings = self._legacy_inference_model_bindings(migrated)
        request["model_bindings"] = bindings
        request["model_artifact_ids"] = [item["artifact_id"] for item in bindings]
        migrated["request"] = request
        migrated["schema_version"] = INFERENCE_JOB_SCHEMA_VERSION
        migrated["model_binding_status"] = (
            "bound" if len(bindings) == len(_string_list(request.get("models") or []))
            else "unavailable"
        )
        return migrated

    def _legacy_inference_model_bindings(self, job: dict) -> list[dict]:
        model_names = _string_list((job.get("request") or {}).get("models") or [])
        if not model_names:
            return []
        raw_run_dir = str(job.get("raw_run_dir") or job.get("run_dir") or "").strip()
        if not raw_run_dir:
            return []
        run_dir = Path(raw_run_dir).resolve()
        if not run_dir.is_relative_to(self.runs_dir.resolve()):
            return []
        catalog_path = run_dir / "model_catalog.json"
        if not catalog_path.is_file():
            return []
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        by_name = {
            str(item.get("name") or ""): item
            for item in payload.get("models") or []
        }
        return [
            _immutable_model_binding(by_name[name])
            for name in model_names
            if name in by_name and by_name[name].get("artifact_id")
        ]

    def _validated_job_case_context(self, payload: dict) -> dict:
        case_id = str(payload.get("case_id") or "").strip()
        if not case_id:
            return {}
        binding_id = str(payload.get("expected_context_binding_id") or "").strip()
        case = self.evaluation_cases.assert_context_binding(case_id, binding_id)
        return {
            "case_id": case_id,
            "profile_name": str(case.get("profile_name") or ""),
            "case_context_binding_id": binding_id,
            "case_context_generation": int(
                (case.get("context_binding") or {}).get("generation") or 0
            ),
        }

    @staticmethod
    def _assert_job_case(job: dict, case_id: str) -> None:
        bound_case_id = str((job.get("request") or {}).get("case_id") or "")
        if not bound_case_id:
            raise EvaluationCaseConflictError(
                "job has no immutable evaluation-case binding and cannot update a case"
            )
        if bound_case_id != case_id:
            raise EvaluationCaseConflictError(
                f"job belongs to evaluation case {bound_case_id}, not {case_id}"
            )

    @staticmethod
    def _assert_job_can_update_case(job: dict, case: dict, *, stage: str) -> None:
        """Reject delayed jobs from an older semantic generation.

        A job may bind itself once when it was launched from the case's current
        context, and may then continue updating only the exact execution slot it
        owns. A same-case job from an older generation cannot overwrite a newer
        training or inference selection.
        """
        request = job.get("request") or {}
        launched_from = str(request.get("case_context_binding_id") or "")
        current_binding = str((case.get("context_binding") or {}).get("binding_id") or "")
        execution = case.get("execution") or {}
        slot = "training_job_id" if stage == "training" else "inference_job_id"
        active_job_id = str(execution.get(slot) or "")
        job_id = str(job.get("id") or "")
        if launched_from == current_binding or (active_job_id and active_job_id == job_id):
            return
        raise EvaluationCaseConflictError(
            f"stale {stage} job {job_id} cannot update the current case generation"
        )

    def _featuresets_for_models(self, model_names: list[str]) -> list[str]:
        selected = set(model_names)
        featuresets: list[str] = []
        for item in self.model_store.catalog():
            if item.get("name") not in selected:
                continue
            name = str(item.get("featureset") or "").strip()
            if name and name not in featuresets:
                featuresets.append(name)
        return featuresets

    def _inference_curve_evidence(
        self,
        prediction_records: list[dict],
        datasets: list[str],
    ) -> tuple[list[dict], list[dict]]:
        """Join canonical predictions to measured curve sources without copying them."""
        dashboard_data = self.data_dir / "curves"
        if not dashboard_data.is_dir():
            legacy_dashboard_data = self.data_dir.parent / "dashboard" / "data"
            dashboard_data = legacy_dashboard_data if legacy_dashboard_data.is_dir() else dashboard_data
        catalogs: dict[str, tuple[Path, str, dict]] = {}
        curve_artifacts: list[dict] = []
        for dataset in datasets:
            token = _dashboard_dataset_token(dataset)
            source = dashboard_data / f"{token}_curves.json"
            if not source.is_file():
                continue
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            source_uri = f"curve@sha256:{_sha256_path(source)}"
            catalogs[dataset] = (source, source_uri, payload)
            curve_artifacts.append(
                {
                    "dataset": dataset,
                    "filename": source.name,
                    "source": str(source),
                    "source_uri": source_uri,
                    "bytes": source.stat().st_size,
                    "total_curves": len(payload),
                    "copied": False,
                }
            )

        curve_refs: list[dict] = []
        for record in prediction_records:
            dataset = str(record.get("dataset") or "")
            catalog = catalogs.get(dataset)
            if catalog is None:
                continue
            _source, source_uri, payload = catalog
            curve_key = f"{record.get('cell_id')}|{record.get('cycle_index')}"
            curve = payload.get(curve_key)
            if not isinstance(curve, dict):
                continue
            signals = curve_signal_names(curve)
            curve_ref_id = f"curve:{dataset}:{record.get('cell_id')}:{record.get('cycle_index')}"
            record["curve_segment_ref_id"] = curve_ref_id
            curve_refs.append(
                {
                    "schema_version": "1.0.0",
                    "curve_segment_ref_id": curve_ref_id,
                    "prediction_record_id": record.get("prediction_record_id") or "",
                    "sample_id": record.get("sample_id") or "",
                    "candidate_id": record.get("candidate_id") or "",
                    "dataset": dataset,
                    "asset_id": str(record.get("cell_id") or ""),
                    "cycle_id": record.get("cycle_index"),
                    "curve_key": curve_key,
                    "source_uri": source_uri,
                    "signals": signals,
                    "availability": "available",
                }
            )
        return curve_artifacts, curve_refs

    def _publish_frontend_artifacts(self, raw_run_dir: Path) -> Path:
        sync_frontend_artifacts(raw_run_dir, self.frontend_data_dir)
        return raw_run_dir

    def publish_frontend_profile(
        self,
        profile_name: str,
        manifest: dict,
        *,
        source_dirs: list[str | Path] | tuple[str | Path, ...] = (),
    ) -> dict:
        return publish_frontend_profile(
            self.frontend_data_dir,
            profile_name,
            manifest,
            source_dirs=source_dirs,
        )

    def import_battery_dashboard_profile(
        self,
        *,
        source_dir: str | Path | None = None,
        profile_name: str = "battery-soh-diagnostics",
        include_curves: bool = False,
    ) -> dict:
        return publish_battery_dashboard_profile(
            self.frontend_data_dir,
            Path(source_dir) if source_dir is not None else self.data_dir / "curves",
            include_curves=include_curves,
            profile_name=profile_name,
        )

    def battery_curve_dataset(
        self,
        *,
        profile_name: str = "battery-soh-diagnostics",
        dataset: str,
        limit: int | None = None,
    ) -> dict:
        profile = _safe_profile_name(profile_name)
        dataset_name = str(dataset or "").strip()
        if not dataset_name:
            raise ValueError("dataset is required")
        manifest_path = self.frontend_data_dir / "profiles" / profile / "manifest.json"
        if not manifest_path.exists():
            raise KeyError(f"profile not found: {profile}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.get("curve_artifacts") if isinstance(manifest.get("curve_artifacts"), list) else []
        artifact = next((item for item in artifacts if str(item.get("dataset") or "") == dataset_name), None)
        if artifact is None:
            raise KeyError(f"curve dataset not found: {dataset_name}")
        source = Path(str(artifact.get("source") or ""))
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"curve source not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("curve source must be a JSON object")
        curve_items = list(payload.items())
        max_items = None if limit is None else max(0, int(limit))
        selected_items = curve_items if max_items is None else curve_items[:max_items]
        return {
            "ok": True,
            "profile": profile,
            "dataset": dataset_name,
            "source": str(source),
            "bytes": source.stat().st_size,
            "total_curves": len(curve_items),
            "returned_curves": len(selected_items),
            "truncated": len(selected_items) < len(curve_items),
            "curves": {key: value for key, value in selected_items},
        }

    def import_battery_soh_plugins(
        self,
        *,
        battery_soh_root: str | Path | None = None,
        replace: bool = False,
    ) -> dict:
        # None / "" -> BATTERY_SOH_ROOT or the sibling checkout (see sohbench.paths).
        return register_battery_soh_plugins(
            self.store,
            battery_root=battery_soh_root,
            replace=replace,
        )

    def sync_native_dashboard(
        self,
        *,
        source: str | Path | None = None,
        target: str | Path | None = None,
        include_curves: bool = True,
    ) -> dict:
        from scripts.sync_battery_dashboard_native import (
            DEFAULT_SOURCE,
            DEFAULT_TARGET,
            sync_native_dashboard,
        )

        return sync_native_dashboard(
            Path(source) if source else DEFAULT_SOURCE,
            Path(target) if target else DEFAULT_TARGET,
            include_curves=include_curves,
        )

    def package_portable_frontend(
        self,
        *,
        app_dir: str | Path | None = None,
        output: str | Path | None = None,
    ) -> dict:
        from scripts.package_portable_frontend import (
            DEFAULT_APP_DIR,
            DEFAULT_OUTPUT,
            package_portable_frontend,
        )

        return package_portable_frontend(
            Path(app_dir) if app_dir else DEFAULT_APP_DIR,
            Path(output) if output else DEFAULT_OUTPUT,
        )

    def package_standalone_frontend(
        self,
        *,
        app_dir: str | Path | None = None,
        output: str | Path | None = None,
    ) -> dict:
        from scripts.build_standalone_frontend import APP_DIR, DEFAULT_OUTPUT, build_standalone_html

        source_app = Path(app_dir) if app_dir else APP_DIR
        output_path = Path(output) if output else DEFAULT_OUTPUT
        html, artifact_count = build_standalone_html(source_app)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8", newline="\n")
        embedded_payload = _standalone_data_payload(html)
        return {
            "standalone": True,
            "app_dir": str(source_app),
            "output": str(output_path),
            "artifact_count": artifact_count,
            "total_bytes": output_path.stat().st_size,
            "excluded_detection_assets": not _payload_contains_detection_asset(embedded_payload),
        }

    def package_status(
        self,
        *,
        native_dashboard: str | Path | None = None,
        portable_output: str | Path | None = None,
        standalone_output: str | Path | None = None,
    ) -> dict:
        from scripts.build_standalone_frontend import DEFAULT_OUTPUT as DEFAULT_STANDALONE_OUTPUT
        from scripts.package_portable_frontend import DEFAULT_OUTPUT as DEFAULT_PORTABLE_OUTPUT
        from scripts.sync_battery_dashboard_native import DEFAULT_TARGET as DEFAULT_NATIVE_DASHBOARD

        native = _package_native_dashboard_info(Path(native_dashboard) if native_dashboard else DEFAULT_NATIVE_DASHBOARD)
        portable = _package_portable_info(Path(portable_output) if portable_output else DEFAULT_PORTABLE_OUTPUT)
        standalone = _package_standalone_info(Path(standalone_output) if standalone_output else DEFAULT_STANDALONE_OUTPUT)

        checks = [
            _package_check(
                "native_dashboard",
                "SOH 看板同步",
                native["present"] and native["scenario_count"] > 0,
                f"{native['scenario_count']} 个场景 / {native['file_count']} 个文件",
            ),
            _package_check(
                "native_curves",
                "SOH 曲线资产",
                native["include_soh_curves"],
                f"{native['curve_file_count']} 个曲线文件",
            ),
            _package_check(
                "portable_package",
                "目录包",
                portable["present"] and portable["file_count"] > 0 and portable["has_manifest"],
                f"{portable['file_count']} 个文件 / {_format_bytes(portable['total_bytes'])}",
            ),
            _package_check(
                "standalone_html",
                "单 HTML",
                standalone["present"] and standalone["standalone"] and standalone["artifact_count"] > 0,
                f"{standalone['artifact_count']} 个内嵌 JSON / {_format_bytes(standalone['total_bytes'])}",
            ),
            _package_check(
                "detection_excluded",
                "异常检测资产排除",
                not native["detection_assets"]
                and not portable["detection_assets"]
                and not standalone["contains_detection_asset"],
                "未发现 ess_detection 资产",
            ),
        ]
        passed = sum(1 for check in checks if check["passed"])
        return {
            "generated_at": _now(),
            "summary": {
                "ready": passed == len(checks),
                "passed": passed,
                "failed": len(checks) - passed,
                "total": len(checks),
            },
            "checks": checks,
            "outputs": {
                "native_dashboard": native,
                "portable": portable,
                "standalone": standalone,
            },
        }

    def _save_job(self, job: dict) -> None:
        with self._job_lock:
            _write_json_atomic(self.jobs_dir / f"{job['id']}.json", job)

    def _save_inference_job(self, job: dict) -> None:
        with self._inference_job_lock:
            _write_json_atomic(self.inference_jobs_dir / f"{job['id']}.json", job)

    def _save_example(self, example: dict) -> None:
        with self._example_lock:
            _write_json_atomic(self.examples_dir / f"{example['id']}.json", example)

    def _save_retest_run(self, run: dict) -> None:
        with self._retest_run_lock:
            _write_json_atomic(self.retest_runs_dir / f"{run['id']}.json", run)


def _immutable_model_binding(item: dict) -> dict:
    """Keep only immutable model identity and lineage fields in a job request."""
    return {
        "name": str(item.get("name") or ""),
        "artifact_id": str(item.get("artifact_id") or ""),
        "estimator": str(item.get("estimator") or ""),
        "featureset": str(item.get("featureset") or "full"),
        "training_featuresets": _string_list(item.get("training_featuresets") or []),
        "feature_columns": _string_list(item.get("feature_columns") or []),
        "created_by_run": str(item.get("created_by_run") or ""),
        "trained_on": _string_list(item.get("trained_on") or []),
        "protocol": str(item.get("protocol") or ""),
        "held_out_key": str(item.get("held_out_key") or ""),
        "seed": item.get("seed"),
        "manifest_sha256": str(item.get("manifest_sha256") or ""),
        "model_sha256": str(item.get("model_sha256") or ""),
        "label_safety": dict((item.get("validation") or {}).get("invalidation") or {}),
        "usable": bool(item.get("usable", True)),
    }


def _training_failures(results) -> list[dict]:
    if "error" not in results.columns:
        return []
    failed = results[
        results["error"].notna() & results["error"].astype(str).str.strip().ne("")
    ]
    if failed.empty:
        return []
    columns = [
        column for column in (
            "estimator", "featureset", "protocol", "held_out_key", "seed", "error"
        )
        if column in failed.columns
    ]
    return [
        {key: (int(value) if key == "seed" else str(value)) for key, value in row.items()}
        for row in failed[columns].drop_duplicates().to_dict(orient="records")
    ]


def _training_result_summary(results, *, datasets: list[str], model_artifacts: list[dict]) -> dict:
    key_columns = [
        column for column in (
            "estimator", "featureset", "protocol", "held_out_key", "seed"
        )
        if column in results.columns
    ]
    evaluation_count = int(results[key_columns].drop_duplicates().shape[0]) if key_columns else 0
    mae_rows = results[results["metric"] == "mae"] if "metric" in results.columns else results.iloc[0:0]
    finite_mae = []
    for row in mae_rows.to_dict(orient="records"):
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite_mae.append((value, row))
    best = min(finite_mae, key=lambda item: item[0], default=None)
    best_row = best[1] if best else {}
    aggregate_mae = (
        sum(item[0] for item in finite_mae) / len(finite_mae)
        if finite_mae else None
    )
    return {
        "source_class": "measured",
        "datasets": list(datasets),
        "metric_records": int(len(results)),
        "evaluations": evaluation_count,
        "successful_mae_records": len(finite_mae),
        "failed_folds": 0,
        "best_mae": aggregate_mae,
        "aggregate_mae": aggregate_mae,
        "best_fold_mae": best[0] if best else None,
        "best_estimator": best_row.get("estimator"),
        "best_featureset": best_row.get("featureset"),
        "best_protocol": best_row.get("protocol"),
        "best_held_out_key": best_row.get("held_out_key"),
        "model_artifact_count": len(model_artifacts),
        "model_artifact_ids": [item.get("artifact_id") for item in model_artifacts if item.get("artifact_id")],
    }


def _training_dataset_profiles(table, datasets: list[str]) -> list[dict]:
    profiles = []
    for name in datasets:
        frame = table[table["dataset"].astype(str) == str(name)]
        usable = [
            column for column in feature_columns(frame)
            if column in frame.columns and bool(frame[column].notna().any())
        ]
        profiles.append(
            {
                "dataset": name,
                "rows": int(len(frame)),
                "cells": int(frame["cell_id"].astype(str).nunique()),
                "usable_feature_columns": usable,
            }
        )
    return profiles


def _training_feature_profile(
    table,
    datasets: list[str],
    name: str,
    *,
    transform_error: str = "",
) -> dict:
    feature_set = R.get_featureset(name)
    selected = list(feature_set.columns(table))
    expected = list(getattr(feature_set, "candidates", []) or selected)
    dataset_coverage = []
    errors: list[str] = []
    warnings: list[str] = []
    if transform_error:
        errors.append(transform_error)
    if not selected:
        errors.append(f"feature set {name!r} selects zero usable columns")

    label_safety = audit_feature_frame(table, selected)
    errors.extend(item["reason"] for item in label_safety["violations"])
    warnings.extend(item["reason"] for item in label_safety.get("warnings", []))

    for dataset in datasets:
        frame = table[table["dataset"].astype(str) == str(dataset)]
        usable = [
            column for column in selected
            if column in frame.columns and bool(frame[column].notna().any())
        ]
        missing = [column for column in expected if column not in usable]
        status = "ready"
        if not usable:
            status = "blocked"
            errors.append(f"{dataset} has no usable columns for feature set {name}")
        elif missing:
            status = "warning"
            warnings.append(f"{dataset} is missing {len(missing)} expected feature columns")
        dataset_coverage.append(
            {
                "dataset": dataset,
                "status": status,
                "usable_columns": usable,
                "missing_columns": missing,
            }
        )
    return {
        "name": name,
        "selected_columns": selected,
        "expected_columns": expected,
        "dataset_coverage": dataset_coverage,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "label_safety": label_safety,
    }


def _training_protocol_profile(table, meta, name: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict = {}
    dataset_count = int(table["dataset"].astype(str).nunique())
    cell_count = int(
        table[["dataset", "cell_id"]].astype(str).drop_duplicates().shape[0]
    )
    if name == "loco":
        details = {"cells": cell_count}
        if cell_count < 2:
            errors.append("LOCO requires at least two cells")
    elif name == "lodo":
        details = {"datasets": dataset_count}
        if dataset_count < 2:
            errors.append("LODO requires at least two datasets")
    elif name == "lcho":
        chemistry = meta.get("chemistry") if hasattr(meta, "get") else None
        values = [] if chemistry is None else [str(value) for value in chemistry.dropna().unique()]
        unknown = [value for value in values if not value or value.lower() == "unknown"]
        known = sorted({value for value in values if value and value.lower() != "unknown"})
        details = {"chemistries": known, "unknown_metadata": unknown}
        if unknown:
            errors.append("LCHO requires known chemistry metadata for every selected cell")
        if len(known) < 2:
            errors.append("LCHO requires at least two known chemistries")
    else:
        errors.append(f"unknown protocol: {name}")
    if len(table) < 4:
        warnings.append("selected scope contains fewer than four QC-passed rows")
    return {
        "name": name,
        "status": "blocked" if errors else ("warning" if warnings else "ready"),
        "details": details,
        "errors": errors,
        "warnings": warnings,
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _artifact_candidate_id(artifact_id: str) -> str:
    # Must stay byte-identical with inference.py's _candidate_id (token
    # truncated to 48) — the evidence resolver joins candidates from both
    # derivations on this id.
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(artifact_id or "model")).strip("-.")[:48]
    return f"artifact:{token or 'model'}"


def _flatten_saved_metrics(raw: dict) -> dict:
    aliases = {
        "monotonicity_violation_rate": "monotonicity",
        "value_range_violation": "value_range",
    }
    output = {}
    for name, value in (raw or {}).items():
        if isinstance(value, dict):
            value = value.get("mean")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            output[aliases.get(str(name), str(name))] = number
    return output


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 8) -> None:
    """Handle the short Windows sharing window created by concurrent JSON readers."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.01 * (attempt + 1))


def _compact_job(job: dict) -> dict:
    keep = {
        "id",
        "kind",
        "status",
        "created_at",
        "started_at",
        "finished_at",
        "run_dir",
        "raw_run_dir",
        "request",
        "error",
    }
    return {key: job[key] for key in keep if key in job}


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _safe_profile_name(value: str) -> str:
    name = str(value or "battery-soh-diagnostics").strip()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", name):
        raise ValueError(f"invalid profile name: {name}")
    return name


def _controlled_scenario_path(profile_dir: Path, dataset: str) -> Path | None:
    safe_dataset = str(dataset or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", safe_dataset):
        raise ValueError(f"invalid controlled retest dataset: {safe_dataset!r}")
    candidates = [
        profile_dir / f"battery_dashboard_{safe_dataset}.json",
        profile_dir / f"battery_dashboard_{'calce' if safe_dataset == 'calce_cs2' else safe_dataset}.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _read_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"controlled retest artifact must be a JSON object: {path.name}")
    return payload


def _same_prediction_member(record: dict, member: dict) -> bool:
    record_dataset = str(record.get("dataset") or "")
    if record_dataset and record_dataset != str(member.get("dataset") or ""):
        return False
    record_asset = str(record.get("cell_id") or record.get("asset_id") or "")
    if record_asset != str(member.get("asset_id") or ""):
        return False
    try:
        record_cycle = _finite_number(
            record.get("cycle_index", record.get("cycle_id")),
            "baseline cycle_index",
        )
        member_cycle = _finite_number(member.get("cycle_id"), "member cycle_id")
    except ValueError:
        return False
    return record_cycle == member_cycle


def _verify_prediction_record_identity(record: dict, issue_context: dict) -> None:
    for field in (
        "candidate_id",
        "model_artifact_id",
        "dataset_snapshot_id",
        "split_assignment_id",
    ):
        expected = str(issue_context.get(field) or "")
        if expected and str(record.get(field) or "") != expected:
            raise ValueError(f"baseline prediction record {field} differs from the pinned issue")


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"controlled retest {field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"controlled retest {field} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"controlled retest {field} must be finite")
    return number


def _optional_finite_number(value, field: str) -> float | None:
    return None if value is None else _finite_number(value, field)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")



_ENTITY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")


def _safe_entity_id(value) -> str:
    """Ids resolve to ``<dir>/<id>.json``; anything that is not a plain token
    (path separators, ``..``, backslashes on Windows) is treated as unknown."""
    text = str(value or "")
    if not _ENTITY_ID_RE.fullmatch(text) or ".." in text:
        raise KeyError(text)
    return text

def _quickstart_steps() -> list[dict]:
    return [
        {"id": "dataset", "label": "注册示例数据", "status": "pending"},
        {"id": "training", "label": "执行 LOCO 训练", "status": "pending"},
        {"id": "model", "label": "持久化 Pickle 模型", "status": "pending"},
        {"id": "inference", "label": "执行 inference-only", "status": "pending"},
    ]


def _quickstart_dataset_csv() -> bytes:
    rows = [
        "dataset,cell_id,cycle_index,soh,voltage_mean,voltage_std,voltage_min,voltage_max"
    ]
    for cell_index in range(3):
        cell_id = f"demo_cell_{cell_index + 1:02d}"
        for cycle in range(12):
            cell_shift = cell_index * 0.0015
            soh = 1.0 - 0.0105 * cycle - cell_shift
            voltage_mean = 3.62 - 0.014 * cycle - cell_shift
            voltage_std = 0.045 + 0.0012 * cycle + cell_shift
            voltage_min = 3.18 - 0.010 * cycle - cell_shift
            voltage_max = 4.18 - 0.004 * cycle - cell_shift
            rows.append(
                ",".join(
                    [
                        "soh_quickstart_demo",
                        cell_id,
                        str(cycle),
                        f"{soh:.6f}",
                        f"{voltage_mean:.6f}",
                        f"{voltage_std:.6f}",
                        f"{voltage_min:.6f}",
                        f"{voltage_max:.6f}",
                    ]
                )
            )
    return ("\n".join(rows) + "\n").encode("utf-8")


def _quickstart_result(run_dir: Path) -> dict:
    artifact = json.loads((run_dir / "inference_benchmark.json").read_text(encoding="utf-8"))
    summary = artifact.get("summary") or {}
    succeeded = [row for row in artifact.get("records") or [] if row.get("status") == "succeeded"]
    first = succeeded[0] if succeeded else {}
    return {
        "n_records": int(summary.get("n_records") or 0),
        "n_succeeded": int(summary.get("n_succeeded") or 0),
        "dataset": first.get("dataset"),
        "model": first.get("model"),
        "mae": first.get("mae"),
        "rmse": first.get("rmse"),
        "predict_ms": first.get("predict_ms"),
        "ms_per_1000_samples": first.get("ms_per_1000_samples"),
    }


def _package_check(check_id: str, label: str, passed: bool, detail: str) -> dict:
    return {
        "id": check_id,
        "label": label,
        "passed": bool(passed),
        "detail": detail,
        "severity": "required",
    }


def _read_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _directory_file_stats(path: Path) -> tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return file_count, total_bytes


def _format_bytes(value: int | float) -> str:
    size = float(value or 0)
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{int(size)} B"


def _package_native_dashboard_info(path: Path) -> dict:
    path = Path(path)
    data_dir = path / "data"
    manifest = _read_json_if_exists(path / "SYNC_MANIFEST.json")
    source_manifest = _read_json_if_exists(data_dir / "manifest.json")
    scenarios = source_manifest.get("scenarios") or []
    curve_files = [
        item
        for item in data_dir.glob("*_curves.json")
        if item.is_file() and not item.name.startswith("ess_detection")
    ]
    detection_assets = [item.relative_to(path).as_posix() for item in path.rglob("*ess_detection*.json") if item.is_file()]
    file_count, total_bytes = _directory_file_stats(path)
    scenario_count = int(manifest.get("scenario_count") or len(scenarios))
    return {
        "path": str(path),
        "present": (path / "index.html").is_file() and data_dir.is_dir(),
        "has_manifest": bool(manifest),
        "scenario_count": scenario_count,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "include_soh_curves": bool(manifest.get("include_soh_curves") or curve_files),
        "curve_file_count": len(curve_files),
        "excluded_detection_assets": not detection_assets,
        "detection_assets": detection_assets,
    }


def _package_portable_info(path: Path) -> dict:
    path = Path(path)
    manifest = _read_json_if_exists(path / "MANIFEST.json")
    file_count, total_bytes = _directory_file_stats(path)
    detection_assets = [item.relative_to(path).as_posix() for item in path.rglob("*ess_detection*.json") if item.is_file()]
    return {
        "path": str(path),
        "present": path.is_dir(),
        "has_manifest": bool(manifest),
        "file_count": int(manifest.get("file_count") or file_count),
        "total_bytes": int(manifest.get("total_bytes") or total_bytes),
        "include_soh_curves": bool(manifest.get("include_soh_curves")),
        "excluded_detection_assets": bool(manifest.get("excluded_detection_assets")) and not detection_assets,
        "detection_assets": detection_assets,
    }


def _package_standalone_info(path: Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    payload = _standalone_data_payload(text)
    total_bytes = path.stat().st_size if path.is_file() else 0
    return {
        "path": str(path),
        "present": path.is_file(),
        "standalone": "window.__SOH_STANDALONE__ = true" in text,
        "artifact_count": len(payload),
        "total_bytes": total_bytes,
        "contains_detection_asset": _payload_contains_detection_asset(payload),
        "has_package_operations": "package-operations-card" in text,
        "has_package_report": "package-report-card" in text,
    }


def _standalone_data_payload(text: str) -> dict:
    if not text:
        return {}
    match = re.search(
        r"const\s+SOH_STANDALONE_DATA\s*=\s*(\{.*?\});\s*window\.__SOH_STANDALONE__",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_contains_detection_asset(value) -> bool:
    if isinstance(value, dict):
        return any(
            "ess_detection" in str(key) or _payload_contains_detection_asset(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_payload_contains_detection_asset(item) for item in value)
    return isinstance(value, str) and "ess_detection" in value


def _case_execution_state(status: object) -> str:
    value = str(status or "idle")
    if value == "succeeded":
        return "completed"
    return value if value in {"idle", "queued", "running", "completed", "failed"} else "idle"


def _unique_candidate_rows(values) -> list[dict]:
    rows = []
    seen = set()
    for value in values:
        candidate_id = str((value or {}).get("candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        rows.append(dict(value))
    return rows


def _job_sort_key(job: dict) -> tuple[str, str]:
    timestamp = job.get("finished_at") or job.get("started_at") or job.get("created_at") or ""
    return str(timestamp), str(job.get("id") or "")


def _model_name_token(value) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return token or "model"


def _dashboard_dataset_token(dataset: str) -> str:
    return {
        "calce_cs2": "calce",
        "rwth_isea_public": "rwth",
    }.get(str(dataset or ""), str(dataset or ""))


def _sync_frontend_files(run_dir: Path, frontend_data_dir: Path) -> None:
    sync_frontend_artifacts(run_dir, frontend_data_dir)
