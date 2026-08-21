from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .phase22_golden import Phase22GoldenError, build_and_sign_phase22_golden
from .paths import DEFAULT_CURVE_DIR


class Phase24PipelineError(ValueError):
    pass


@dataclass(frozen=True)
class Phase24PipelineConfig:
    profile_name: str = ""
    dataset: str = "nasa"
    asset_id: str = "B0018"
    estimator: str = "ridge"
    featureset: str = "full"
    protocol: str = "loco"
    seed: int = 0
    curve_source: str | Path = DEFAULT_CURVE_DIR / "nasa_curves.json"
    criterion_set_id: str = "exploratory"
    reviewer: str = "phase24-pipeline-reviewer"
    version: str = "phase24-e2e-20260814f"
    release_overrides: Mapping[str, object] | None = None


def run_phase24_pipeline(service, config: Phase24PipelineConfig) -> dict:
    """Run one measured case from preflight through a signed EvidencePack."""
    profile_name = config.profile_name or time.strftime("phase24-e2e-%Y%m%d-%H%M%S")
    training_request = {
        "datasets": [config.dataset],
        "estimators": [config.estimator],
        "featuresets": [config.featureset],
        "protocols": [config.protocol],
        "seeds": [int(config.seed)],
    }
    preflight = service.training_preflight(training_request)
    if not preflight.get("ready"):
        reasons = [
            reason
            for item in preflight.get("combinations") or []
            for reason in item.get("reasons") or []
        ]
        raise Phase24PipelineError(
            "training preflight is blocked: " + ("; ".join(reasons) or "unknown reason")
        )

    admitted = service.create_job(
        {**training_request, "preflight_id": preflight["preflight_id"]},
        start=False,
    )
    training = service.run_job_inline(admitted["id"], publish=False)
    _require_job(training, "training")
    model = _select_held_out_model(
        service,
        training,
        dataset=config.dataset,
        asset_id=config.asset_id,
        estimator=config.estimator,
        featureset=config.featureset,
        protocol=config.protocol,
        seed=config.seed,
    )

    inference_request = {
        "datasets": [config.dataset],
        "models": [model["name"]],
        "launch_strategy": "compatible_only",
        "selected_datasets": [config.dataset],
        "submitted_datasets": [config.dataset],
        "skipped_datasets": [],
    }
    admitted_inference = service.create_inference_job(inference_request, start=False)
    inference = service.run_inference_job_inline(admitted_inference["id"], publish=False)
    _require_job(inference, "inference")
    inference_records = _successful_inference_records(
        Path(inference["run_dir"]) / "inference_benchmark.json",
        dataset=config.dataset,
        model_name=model["name"],
    )

    try:
        release = {
            "phase": "2.4",
            "profile_label": "阶段 2.4 · 真实案例流水线",
            "profile_kind": "phase24_case_pipeline",
            "case_title": f"阶段 2.4 · {config.dataset.upper()}/{config.asset_id} 全链路验收",
            "idempotency_prefix": "phase24",
            "scenario_title": f"{config.dataset.upper()}/{config.asset_id} 实测 {config.protocol.upper()} 案例轨迹",
            "experiment": "phase24-real-case-pipeline",
            "training_job_id": training["id"],
            "inference_job_id": inference["id"],
            "execution_run_id": inference["id"],
            "url_version": config.version,
            "evidence_reason": "phase 2.4 auto-selected representative evidence from the completed inference scope",
            "review_note": "已核对真实训练预检、折次产物分组、自动推理继承、代表 Cycle、原始曲线、真实复测、KPI 对账和案例连续性。",
            "signature_reason": "阶段 2.4 真实案例流水线验收发布；该签署为工作台审计签署，不代表法规认证。",
        }
        release.update(dict(config.release_overrides or {}))
        case_summary = build_and_sign_phase22_golden(
            service,
            profile_name=profile_name,
            dataset=config.dataset,
            asset_id=config.asset_id,
            model_name=model["name"],
            curve_source=config.curve_source,
            criterion_set_id=config.criterion_set_id,
            reviewer=config.reviewer,
            release=release,
        )
    except Phase22GoldenError as exc:
        raise Phase24PipelineError(str(exc)) from exc

    return {
        **case_summary,
        "pipeline": {
            "preflight_id": preflight["preflight_id"],
            "preflight_ready": True,
            "dataset_rows": int((preflight.get("summary") or {}).get("rows") or 0),
            "training_job_id": training["id"],
            "training_status": training["status"],
            "training_model_count": len(training.get("saved_models") or []),
            "selected_model": model["name"],
            "selected_model_artifact_id": model.get("artifact_id") or "",
            "inference_job_id": inference["id"],
            "inference_status": inference["status"],
            "inference_records": len(inference_records),
        },
    }


def _select_held_out_model(
    service,
    training: dict,
    *,
    dataset: str,
    asset_id: str,
    estimator: str,
    featureset: str,
    protocol: str,
    seed: int,
) -> dict:
    expected_held_out = f"{dataset}/{asset_id}"
    artifacts = training.get("model_artifacts") or []
    match = next(
        (
            item for item in artifacts
            if str(item.get("held_out_key") or "") == expected_held_out
            and str(item.get("estimator") or "") == estimator
            and str(item.get("featureset") or "") == featureset
            and str(item.get("protocol") or "") == protocol
            and int(item.get("seed") or 0) == int(seed)
        ),
        None,
    )
    if match is None:
        raise Phase24PipelineError(
            f"training job {training.get('id')} produced no model for {expected_held_out}"
        )
    catalog_match = next(
        (
            item for item in service.model_store.catalog()
            if item.get("name") == match.get("name")
            and item.get("artifact_id") == match.get("artifact_id")
        ),
        None,
    )
    if catalog_match is None:
        raise Phase24PipelineError("selected model artifact is not the immutable training output")
    return catalog_match


def _successful_inference_records(path: Path, *, dataset: str, model_name: str) -> list[dict]:
    if not path.is_file():
        raise Phase24PipelineError(f"inference benchmark is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [
        item for item in payload.get("records") or []
        if str(item.get("dataset") or "") == dataset
        and str(item.get("model") or "") == model_name
    ]
    if not records:
        raise Phase24PipelineError("inference produced no record for the selected model and dataset")
    failed = [item for item in records if item.get("status") != "succeeded"]
    if failed:
        raise Phase24PipelineError(f"inference record failed: {failed[0]}")
    return records


def _require_job(job: dict, label: str) -> None:
    if job.get("status") != "succeeded":
        raise Phase24PipelineError(
            f"{label} job failed: {job.get('error') or job.get('traceback') or 'unknown error'}"
        )
    warnings = job.get("model_warnings") or []
    if warnings:
        raise Phase24PipelineError(f"{label} job produced model warnings: {warnings}")
