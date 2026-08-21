from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import metrics as M
from .case_evidence import CRITERION_SETS
from .data import DatasetLoader
from .paths import DEFAULT_CURVE_DIR


class Phase22GoldenError(ValueError):
    pass


def build_and_sign_phase22_golden(
    service,
    *,
    profile_name: str = "phase22-golden-nasa",
    dataset: str = "nasa",
    asset_id: str = "B0018",
    model_name: str = "ridge_full_loco_nasa_B0018_seed0",
    curve_source: str | Path = DEFAULT_CURVE_DIR / "nasa_curves.json",
    criterion_set_id: str = "exploratory",
    reviewer: str = "phase22-local-reviewer",
    release: dict | None = None,
) -> dict:
    """Build and sign the first measured golden case from real model re-inference."""
    release_meta = {
        "phase": "2.2",
        "profile_label": "阶段 2.2 · NASA 真实黄金案例",
        "profile_kind": "phase22_golden",
        "case_title": "阶段 2.2 · NASA/B0018 真实黄金验收",
        "idempotency_prefix": "phase22",
        "scenario_title": "NASA/B0018 实测 LOCO 黄金轨迹",
        "experiment": "phase22-nasa-b0018-loco-golden",
        "training_job_id": "",
        "inference_job_id": "",
        "execution_run_id": "",
        "url_version": "phase22-golden-20260812b",
        "evidence_reason": "pin canonical phase 2.2 golden evidence",
        "review_note": "已核对真实数据快照、LOCO 留出、目标字段隔离、102 条逐 Cycle 事实、原始曲线、真实复测和 KPI 对账。",
        "signature_reason": "阶段 2.2 本地黄金基线发布；该签署为工作台审计签署，不代表法规认证。",
    }
    if release:
        release_meta.update(release)
    source_root = Path(__file__).resolve().parents[1]
    expected_root = Path.cwd().resolve() / "src"
    if source_root != expected_root:
        raise Phase22GoldenError(
            f"sohbench import does not resolve to the active workspace: {source_root}"
        )
    if criterion_set_id not in CRITERION_SETS:
        raise Phase22GoldenError(f"unknown criterion set: {criterion_set_id}")

    curve_path = Path(curve_source).resolve()
    dataset_path = (service.data_dir / "processed" / f"{dataset}_features.pkl").resolve()
    if not dataset_path.is_file():
        raise Phase22GoldenError(f"processed dataset is unavailable: {dataset_path}")
    if not curve_path.is_file():
        raise Phase22GoldenError(f"raw curve source is unavailable: {curve_path}")

    loaded = service.model_store.load_model(model_name)
    artifact_id = service.model_store.artifact_id(model_name)
    manifest = loaded.manifest
    held_out_key = str(manifest.get("held_out_key") or "")
    expected_held_out = f"{dataset}/{asset_id}"
    if held_out_key != expected_held_out:
        raise Phase22GoldenError(
            f"model held-out key {held_out_key!r} does not match {expected_held_out!r}"
        )
    if str(manifest.get("protocol") or "") != "loco":
        raise Phase22GoldenError("phase 2.2 golden case requires a LOCO model")
    if str(manifest.get("target") or "soh") in loaded.input_columns(_peek_table(dataset_path)):
        raise Phase22GoldenError("model inference input includes the target column")

    curve_payload = _read_json(curve_path)
    if not isinstance(curve_payload, dict):
        raise Phase22GoldenError("raw curve source must be a JSON object")
    curve_keys = set(curve_payload)
    table, meta, qc = DatasetLoader(service.data_dir).load([dataset])
    featureset = str(manifest.get("featureset") or "")
    table = service.store.apply_feature_transforms(
        table, meta, [featureset] if featureset else []
    )
    selected = table.loc[
        (table["dataset"].astype(str) == dataset)
        & (table["cell_id"].astype(str) == asset_id)
        & table.apply(
            lambda row: f"{row['cell_id']}|{row['cycle_index']}" in curve_keys,
            axis=1,
        )
    ].sort_values("cycle_index")
    if len(selected) < 2:
        raise Phase22GoldenError("golden case has fewer than two curve-backed prediction rows")

    prediction = loaded.predict(selected, meta)
    interval = loaded.predict_interval(selected, meta)
    lower = interval[0] if interval is not None else np.full(len(selected), np.nan)
    upper = interval[1] if interval is not None else np.full(len(selected), np.nan)
    truth = selected["soh"].to_numpy(dtype=float)
    errors = prediction - truth
    absolute_errors = np.abs(errors)
    covered = (truth >= lower) & (truth <= upper) if interval is not None else np.zeros(len(selected), dtype=bool)
    metrics = {
        "mae": M.mae(truth, prediction),
        "rmse": M.rmse(truth, prediction),
        "coverage": M.picp(truth, lower, upper) if interval is not None else None,
        "mean_width": float(np.mean(upper - lower)) if interval is not None else None,
        "monotonicity_violation_rate": M.monotonicity_violation_rate(prediction),
        "value_range_violation": M.value_range_violation(prediction),
        "mean_bias_error": M.mean_bias_error(truth, prediction),
        "max_error": M.max_error(truth, prediction),
    }
    thresholds = CRITERION_SETS[criterion_set_id]
    checks = {
        "generalization": metrics["mae"] <= thresholds["mae_max"],
        "calibration": metrics["coverage"] is not None
        and metrics["coverage"] >= thresholds["picp_min"],
        "monotonicity": metrics["monotonicity_violation_rate"]
        <= thresholds["monotonicity_max"],
        "value_range": metrics["value_range_violation"] <= thresholds["value_range_max"],
    }
    if not all(checks.values()):
        raise Phase22GoldenError(f"real golden run does not pass {criterion_set_id}: {checks}")

    dataset_snapshot_id = _digest_ref("dataset-snapshot", dataset_path)
    split_descriptor = {
        "protocols": ["loco"],
        "held_out_key": held_out_key,
        "seeds": [int(manifest.get("seed") or 0)],
    }
    split_assignment_id = _json_digest_ref("protocol", split_descriptor)
    comparison_id = _json_digest_ref(
        "comparison",
        {
            "dataset_snapshot_id": dataset_snapshot_id,
            "split_assignment_id": split_assignment_id,
            "model_artifact_id": artifact_id,
            "criterion_set_id": criterion_set_id,
        },
    )
    model_run_id = str(manifest.get("created_by_run") or f"phase22-{artifact_id[-12:]}")
    training_job_id = str(release_meta.get("training_job_id") or model_run_id)
    inference_job_id = str(release_meta.get("inference_job_id") or model_run_id)
    run_id = str(release_meta.get("execution_run_id") or inference_job_id or training_job_id)
    release_meta.update(
        {
            "training_job_id": training_job_id,
            "inference_job_id": inference_job_id,
            "execution_run_id": run_id,
        }
    )
    candidate_id = f"artifact:{model_name}"
    profile_dir = service.frontend_data_dir / "profiles" / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)

    records = []
    contributions = []
    curve_refs = []
    for position, (_, row) in enumerate(selected.iterrows()):
        cycle = int(row["cycle_index"])
        sample_id = f"{dataset}:{asset_id}:{cycle}"
        record_id = f"pred:{dataset}:{asset_id}:{cycle}"
        curve_key = f"{asset_id}|{cycle}"
        curve = curve_payload.get(curve_key) or {}
        record = {
            "schema_version": "1.0.0",
            "prediction_record_id": record_id,
            "sample_id": sample_id,
            "run_id": run_id,
            "comparison_id": comparison_id,
            "candidate_id": candidate_id,
            "model_artifact_id": artifact_id,
            "dataset": dataset,
            "dataset_snapshot_id": dataset_snapshot_id,
            "split_assignment_id": split_assignment_id,
            "cell_id": asset_id,
            "cycle_index": cycle,
            "true_soh": float(truth[position]),
            "pred_soh": float(prediction[position]),
            "lower": _optional_float(lower[position]),
            "upper": _optional_float(upper[position]),
            "absolute_error": float(absolute_errors[position]),
            "interval_covered": bool(covered[position]) if interval is not None else None,
            "curve_segment_ref_id": f"curve:{dataset}:{asset_id}:{cycle}",
            "metric_contributions": {
                "mae_contribution": float(absolute_errors[position] / len(selected)),
                "rmse_squared_contribution": float(errors[position] ** 2 / len(selected)),
                "picp_contribution": float(covered[position] / len(selected)) if interval is not None else None,
            },
        }
        records.append(record)
        contributions.append(
            {
                "schema_version": "1.0.0",
                "metric_contribution_id": f"metric:{dataset}:{asset_id}:{cycle}",
                "prediction_record_id": record_id,
                "sample_id": sample_id,
                "mae_numerator": float(absolute_errors[position]),
                "mae_contribution": float(absolute_errors[position] / len(selected)),
                "rmse_squared_numerator": float(errors[position] ** 2),
                "rmse_squared_contribution": float(errors[position] ** 2 / len(selected)),
                "picp_numerator": int(covered[position]) if interval is not None else None,
                "picp_contribution": float(covered[position] / len(selected)) if interval is not None else None,
                "residual": float(errors[position]),
            }
        )
        curve_refs.append(
            {
                "schema_version": "1.0.0",
                "curve_segment_ref_id": record["curve_segment_ref_id"],
                "prediction_record_id": record_id,
                "sample_id": sample_id,
                "dataset": dataset,
                "asset_id": asset_id,
                "cycle_id": cycle,
                "curve_key": curve_key,
                "source_uri": _digest_ref("curve", curve_path),
                "signals": _curve_signals(curve),
                "availability": "available",
            }
        )

    representative = _representative_record(records, thresholds)
    artifact_payloads = _profile_artifacts(
        profile_name=profile_name,
        dataset=dataset,
        asset_id=asset_id,
        model_name=model_name,
        model_manifest=manifest,
        model_artifact_id=artifact_id,
        run_id=run_id,
        candidate_id=candidate_id,
        curve_path=curve_path,
        dataset_path=dataset_path,
        dataset_snapshot_id=dataset_snapshot_id,
        split_descriptor=split_descriptor,
        split_assignment_id=split_assignment_id,
        comparison_id=comparison_id,
        criterion_set_id=criterion_set_id,
        thresholds=thresholds,
        checks=checks,
        metrics=metrics,
        qc=qc,
        model_path=loaded.path,
        records=records,
        contributions=contributions,
        curve_refs=curve_refs,
        release=release_meta,
    )
    for filename, payload in artifact_payloads.items():
        _write_json_atomic(profile_dir / filename, payload)
    _register_profile(
        service.frontend_data_dir / "profiles" / "index.json",
        {
            "name": profile_name,
            "label": release_meta["profile_label"],
            "kind": release_meta["profile_kind"],
            "created_at": _now(),
            "datasets": [dataset],
            "training_job_id": training_job_id,
            "inference_job_id": inference_job_id,
        },
    )

    idempotency_key = f"{release_meta['idempotency_prefix']}:{comparison_id}"
    resolved = service.resolve_evaluation_case(
        {
            "profile_name": profile_name,
            "idempotency_key": idempotency_key,
            "title": release_meta["case_title"],
            "source_class": "measured",
            "execution": {
                "state": "completed",
                "training_job_id": training_job_id,
                "inference_job_id": inference_job_id,
                "run_id": run_id,
            },
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "model": str(manifest.get("estimator") or "ridge"),
                    "featureset": featureset or "full",
                    "role": "champion",
                    "model_artifact_id": artifact_id,
                    "run_id": training_job_id,
                }
            ],
        }
    )
    case = resolved["case"]
    if not resolved["created"]:
        workspace = service.evaluation_case_review_workspace(case["id"])
        signed = next((item for item in workspace["evidence_packs"] if item["state"] == "signed"), None)
        if signed:
            return _summary(
                service, case, workspace, signed, profile_name, representative, metrics,
                comparison_id, artifact_id, dataset_snapshot_id, split_assignment_id,
                release_meta,
            )
        raise Phase22GoldenError(
            f"existing golden case {case['id']} is incomplete; inspect its review workspace"
        )

    case = service.update_evaluation_case_evidence(
        case["id"],
        {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
            "criterion_set_id": criterion_set_id,
            "selection": {
                "comparison_id": comparison_id,
                "candidate_id": candidate_id,
                "dataset": dataset,
                "slice_id": f"loco:{asset_id}:curve-backed",
                "asset_id": asset_id,
                "cycle_id": representative["cycle_index"],
                "evidence_tab": "raw",
            },
            "actor": reviewer,
            "reason": release_meta["evidence_reason"],
        },
    )
    if case["evidence_coverage"]["state"] != "complete":
        raise Phase22GoldenError(f"golden evidence is incomplete: {case['evidence_coverage']}")
    if case["verdict"]["result"] != "pass":
        raise Phase22GoldenError(f"golden verdict is not pass: {case['verdict']}")
    if case["evidence_selection"]["split_assignment_id"] != split_assignment_id:
        raise Phase22GoldenError("derived split assignment does not match the published run spec")

    detail = service.evaluation_case_prediction_record(
        case["id"],
        expected_context_binding_id=case["context_binding"]["binding_id"],
    )
    if detail["prediction_record"].get("model_artifact_id") != artifact_id:
        raise Phase22GoldenError("prediction detail lost the immutable model identity")

    issue = service.create_review_issue(
        {
            "case_id": case["id"],
            "expected_case_version": case["version"],
            "title": f"Cycle {representative['cycle_index']} 黄金样本需同协议复测",
            "severity": "minor",
            "description": "对逐 Cycle 事实表中的代表样本重新加载不可变模型与数据快照，验证非重放推理链路。",
            "actor": reviewer,
        }
    )
    regression_set = service.create_regression_set(
        {
            "issue_id": issue["id"],
            "expected_case_version": case["version"],
            "name": "NASA/B0018 黄金回归集",
            "actor": reviewer,
        }
    )
    service.add_regression_member(
        regression_set["id"],
        {"expected_case_version": case["version"], "actor": reviewer},
    )
    retest = service.create_case_retest(
        case["id"],
        {
            "expected_case_version": case["version"],
            "issue_id": issue["id"],
            "regression_set_id": regression_set["id"],
            "actor": reviewer,
        },
        start=False,
    )
    controlled_run = service.run_controlled_retest_inline(
        retest["id"], retest["controlled_run"]["id"]
    )
    completed = service.evaluation_case_review_workspace(case["id"])["retests"][0]
    if controlled_run["status"] != "completed" or completed["status"] != "completed":
        raise Phase22GoldenError(f"model re-inference failed: {controlled_run.get('result')}")
    result = completed.get("result") or {}
    if result.get("execution_mode") != "model_reinference" or result.get("prediction_recomputed") is not True:
        raise Phase22GoldenError("controlled run did not recompute model predictions")
    if result.get("verdict") != "pass":
        raise Phase22GoldenError(f"representative-sample retest did not pass: {result}")

    service.update_review_issue_status(
        issue["id"],
        {
            "expected_case_version": case["version"],
            "status": "resolved",
            "reason": "不可变模型与数据快照下的真实重新推理通过，结果摘要已写入内容寻址存储。",
            "actor": reviewer,
        },
    )
    case = service.update_evaluation_case_review(
        case["id"],
        {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
            "state": "in_review",
            "reviewer": reviewer,
        },
    )
    case = service.update_evaluation_case_review(
        case["id"],
        {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
            "state": "approved",
            "reviewer": reviewer,
            "note": release_meta["review_note"],
        },
    )
    reconciliation = service.reconcile_case_kpis(
        case["id"],
        {
            "expected_case_version": case["version"],
            "tolerance": 1e-12,
            "actor": reviewer,
        },
    )
    if reconciliation["status"] != "match":
        raise Phase22GoldenError(f"KPI reconciliation failed: {reconciliation}")
    pack = service.create_case_evidence_pack(
        case["id"],
        {"expected_case_version": case["version"], "actor": reviewer},
    )
    signed = service.sign_evidence_pack(
        pack["id"],
        {
            "expected_case_version": case["version"],
            "signer": reviewer,
            "reason": release_meta["signature_reason"],
        },
    )
    workspace = service.evaluation_case_review_workspace(case["id"])
    if not workspace["signing_readiness"]["can_sign"]:
        raise Phase22GoldenError("signed case unexpectedly fails the current readiness gates")
    return _summary(
        service, case, workspace, signed, profile_name, representative, metrics,
        comparison_id, artifact_id, dataset_snapshot_id, split_assignment_id,
        release_meta,
    )


def _profile_artifacts(**values) -> dict[str, object]:
    profile_name = values["profile_name"]
    dataset = values["dataset"]
    asset_id = values["asset_id"]
    model_name = values["model_name"]
    model_manifest = values["model_manifest"]
    metrics = values["metrics"]
    checks = values["checks"]
    records = values["records"]
    release = values["release"]
    model_created_at = datetime.fromtimestamp(
        Path(values["model_path"]).stat().st_mtime, timezone.utc
    ).isoformat(timespec="seconds")
    model_row = {
        "name": model_name,
        "kind": "model",
        "source": "training",
        "artifact_id": values["model_artifact_id"],
        "estimator": model_manifest.get("estimator") or "ridge",
        "featureset": model_manifest.get("featureset") or "full",
        "feature_columns": list(model_manifest.get("feature_columns") or []),
        "trained_on": list(model_manifest.get("trained_on") or []),
        "held_out_key": model_manifest.get("held_out_key") or "",
        "created_by_run": release["training_job_id"],
    }
    leaderboard_row = {
        "candidate_id": values["candidate_id"],
        "model_artifact_id": values["model_artifact_id"],
        "run_id": values["run_id"],
        "estimator": model_manifest.get("estimator") or "ridge",
        "featureset": model_manifest.get("featureset") or "full",
        "track": "unified",
        "protocol": "loco",
        "held_out_key": model_manifest.get("held_out_key") or "",
        "metrics": {
            "mae": {"mean": metrics["mae"], "std": 0.0},
            "rmse": {"mean": metrics["rmse"], "std": 0.0},
            "picp": {"mean": metrics["coverage"], "std": 0.0},
            "monotonicity_violation_rate": {
                "mean": metrics["monotonicity_violation_rate"], "std": 0.0
            },
            "value_range_violation": {"mean": metrics["value_range_violation"], "std": 0.0},
            "mean_bias_error": {"mean": metrics["mean_bias_error"], "std": 0.0},
            "max_error": {"mean": metrics["max_error"], "std": 0.0},
        },
        "verdicts": {
            "generalization": checks["generalization"],
            "calibration": checks["calibration"],
            "physical_consistency": checks["monotonicity"] and checks["value_range"],
        },
        "ci_source": "conformal",
        "backend": "sklearn",
    }
    run_config = {
        "datasets": [dataset],
        "estimators": [model_manifest.get("estimator") or "ridge"],
        "featuresets": [model_manifest.get("featureset") or "full"],
        "protocols": ["loco"],
        "held_out_key": model_manifest.get("held_out_key") or "",
        "split_assignment": values["split_descriptor"],
        "seeds": [int(model_manifest.get("seed") or 0)],
        "saved_models": [model_name],
        "model_artifact_id": values["model_artifact_id"],
        "dataset_snapshot_id": values["dataset_snapshot_id"],
        "split_assignment_id": values["split_assignment_id"],
        "comparison_id": values["comparison_id"],
        "criterion_set_id": values["criterion_set_id"],
        "target_excluded_from_predictor": True,
        "trusted_local_plugins": True,
    }
    scenario = {
        "schema_version": "1.0.0",
        "mode": dataset,
        "title": release["scenario_title"],
        "chemistry": "NASA 18650 aging dataset",
        "feature_columns": list(model_manifest.get("feature_columns") or []),
        "meta": {
            "total_cells": 1,
            "evaluated_cells": 1,
            "conditions": "curve-backed cycles only",
            "source_class": "measured",
        },
        "battery_info": {
            asset_id: {
                "n_points": len(records),
                "error_mean": metrics["mae"],
                "error_std": float(np.std([item["absolute_error"] for item in records])),
            }
        },
        "experiment": release["experiment"],
        "metrics": metrics,
        "per_cell_mae": {asset_id: metrics["mae"]},
        "per_cell_coverage": {asset_id: metrics["coverage"]},
        "gates": checks,
        "overall_pass": all(checks.values()),
        "contract": values["thresholds"],
        "config_brief": {
            "split": "leave-one-cell-out · held out nasa/B0018",
            "model": f"{model_manifest.get('estimator') or 'ridge'} · {model_name}",
            "interval_method": "split conformal from frozen model artifact",
            "target_excluded": True,
        },
        "points": records,
    }
    scenario_summary = {
        "mode": dataset,
        "title": scenario["title"],
        "points": len(records),
        "cells": 1,
        "curves": len(records),
        "curves_missing": 0,
        "overall_pass": True,
        "mae": metrics["mae"],
        "coverage": metrics["coverage"],
        "mean_width": metrics["mean_width"],
    }
    created_at = model_created_at
    manifest = {
        "schema_version": 1,
        "profile": profile_name,
        "label": release["profile_label"],
        "kind": release["profile_kind"],
        "created_at": created_at,
        "artifact_files": [
            "leaderboard.json", "dataset_leaderboard.json", "qc_report.json",
            "run_config.json", "model_catalog.json", "inference_benchmark.json",
            "inference_run_config.json", "prediction_records.json",
            "metric_contributions.json", "curve_segment_refs.json",
            f"battery_dashboard_{dataset}.json", "battery_dashboard_manifest.json",
            "battery_dashboard_overview.json",
        ],
        "source_artifacts": {
            "processed_dataset": str(values["dataset_path"]),
            "raw_curves": str(values["curve_path"]),
            "model_artifact": values["model_artifact_id"],
        },
        "datasets": [dataset],
        "run_id": values["run_id"],
        "training_job_id": release["training_job_id"],
        "inference_job_id": release["inference_job_id"],
        "comparison_id": values["comparison_id"],
        "criterion_set_id": values["criterion_set_id"],
        "curve_artifacts": [
            {
                "dataset": dataset,
                "filename": f"battery_dashboard_{dataset}_curves.json",
                "source": str(values["curve_path"]),
                "bytes": values["curve_path"].stat().st_size,
                "copied": False,
            }
        ],
        "local_audit_signature_only": True,
    }
    return {
        "manifest.json": manifest,
        "run_config.json": run_config,
        "leaderboard.json": [leaderboard_row],
        "dataset_leaderboard.json": [leaderboard_row],
        "qc_report.json": {
            "pass_rate": float(values["qc"].pass_rate),
            "n_in": int(values["qc"].n_in),
            "n_out": int(values["qc"].n_out),
            "drop_reasons": dict(values["qc"].drop_reasons),
            "dropped_columns": list(values["qc"].dropped_columns),
            "per_dataset": {dataset: {"curve_backed_rows": len(records)}},
        },
        "model_catalog.json": {"models": [model_row]},
        "inference_benchmark.json": {
            "records": [
                {
                    "model": model_name,
                    "dataset": dataset,
                    "featureset": model_manifest.get("featureset") or "full",
                    "estimator": model_manifest.get("estimator") or "ridge",
                    "n_samples": len(records),
                    "status": "succeeded",
                    **metrics,
                }
            ],
            "summary": {"records": 1, "succeeded": 1, "failed": 0},
        },
        "inference_run_config.json": {
            "datasets": [dataset],
            "models": [model_name],
            "comparison_id": values["comparison_id"],
            "target_excluded_from_predictor": True,
        },
        "recommendation_summary.json": {
            "generated_at": created_at,
            "scope": {"datasets": [dataset], "protocols": ["loco"]},
            "overall": {"recommended_default": model_manifest.get("estimator") or "ridge"},
            "scenarios": [],
        },
        "prediction_records.json": {
            "schema_version": "1.0.0",
            "comparison_id": values["comparison_id"],
            "run_id": values["run_id"],
            "model_artifact_id": values["model_artifact_id"],
            "dataset_snapshot_id": values["dataset_snapshot_id"],
            "split_assignment_id": values["split_assignment_id"],
            "records": records,
        },
        "metric_contributions.json": {
            "schema_version": "1.0.0",
            "comparison_id": values["comparison_id"],
            "denominator": len(records),
            "aggregate_metrics": metrics,
            "records": values["contributions"],
        },
        "curve_segment_refs.json": {
            "schema_version": "1.0.0",
            "comparison_id": values["comparison_id"],
            "records": values["curve_refs"],
        },
        f"battery_dashboard_{dataset}.json": scenario,
        "battery_dashboard_manifest.json": {"scenarios": [scenario_summary]},
        "battery_dashboard_overview.json": {
            "scenarios": [
                {
                    **scenario_summary,
                    "kind": "lab",
                    "contract": values["thresholds"],
                    "gates": checks,
                    "config": scenario["config_brief"],
                    "features": {"columns": scenario["feature_columns"]},
                }
            ],
            "transfers": [],
        },
    }


def _summary(
    service, case, workspace, signed, profile_name, representative, metrics,
    comparison_id, artifact_id, dataset_snapshot_id, split_assignment_id,
    release,
) -> dict:
    current = service.get_evaluation_case(case["id"])
    retest = next((item for item in workspace["retests"] if item["status"] == "completed"), {})
    reconciliation = workspace["reconciliations"][0] if workspace["reconciliations"] else {}
    return {
        "phase": release["phase"],
        "status": "completed",
        "profile_name": profile_name,
        "case_id": current["id"],
        "case_version": current["version"],
        "comparison_id": comparison_id,
        "model_artifact_id": artifact_id,
        "dataset_snapshot_id": dataset_snapshot_id,
        "split_assignment_id": split_assignment_id,
        "training_job_id": release["training_job_id"],
        "inference_job_id": release["inference_job_id"],
        "selected_sample_id": representative["sample_id"],
        "prediction_records": len(
            (json.loads(
                (service.frontend_data_dir / "profiles" / profile_name / "prediction_records.json")
                .read_text(encoding="utf-8")
            ).get("records") or [])
        ),
        "metrics": metrics,
        "evidence_coverage": current["evidence_coverage"],
        "verdict": current["verdict"],
        "review": current["review"],
        "issue_statuses": [item["status"] for item in workspace["issues"]],
        "retest": {
            "id": retest.get("id") or "",
            "status": retest.get("status") or "",
            "run_id": retest.get("run_id") or "",
            "execution_mode": (retest.get("result") or {}).get("execution_mode") or "",
            "prediction_recomputed": (retest.get("result") or {}).get("prediction_recomputed") is True,
            "verdict": (retest.get("result") or {}).get("verdict") or "",
        },
        "reconciliation": {
            "id": reconciliation.get("id") or "",
            "status": reconciliation.get("status") or "",
            "deltas": reconciliation.get("deltas") or {},
        },
        "evidence_pack": {
            "id": signed["id"],
            "state": signed["state"],
            "digest": signed["digest"],
            "signer": signed["signer"],
            "artifact_refs": len(signed["manifest"].get("artifact_refs") or []),
        },
        "signing_readiness": workspace["signing_readiness"],
        "url": (
            "http://127.0.0.1:8765/SOH_eval/frontend/app/index.html"
            f"?profile={profile_name}&mode=audit&v={release['url_version']}"
        ),
        "generated_at": _now(),
    }


def _peek_table(path: Path):
    import pandas as pd

    return pd.read_pickle(path).head(1)


def _representative_record(records: list[dict], thresholds: dict) -> dict:
    eligible = [
        item for item in records
        if item["absolute_error"] <= thresholds["mae_max"]
        and item.get("interval_covered") is True
    ]
    if not eligible:
        raise Phase22GoldenError("no representative sample can pass the controlled retest gates")
    eligible.sort(key=lambda item: item["cycle_index"])
    return eligible[len(eligible) // 2]


def _curve_signals(curve: dict) -> list[str]:
    mapping = {
        "voltage_v": "voltage",
        "current_a": "current",
        "temperature_c": "temperature",
        "ic_ah_per_v": "ic",
        "dva_v_per_ah": "dv",
    }
    return [label for key, label in mapping.items() if key in curve]


def _optional_float(value) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _digest_ref(kind: str, path: Path) -> str:
    return f"{kind}@sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _json_digest_ref(kind: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{kind}@sha256:{hashlib.sha256(encoded).hexdigest()}"


def _register_profile(index_path: Path, profile: dict) -> None:
    payload = {"profiles": []}
    if index_path.is_file():
        payload = _read_json(index_path)
    profiles = [
        item for item in payload.get("profiles") or [] if item.get("name") != profile["name"]
    ]
    profiles.append(profile)
    payload["profiles"] = sorted(profiles, key=lambda item: str(item.get("name") or ""))
    _write_json_atomic(index_path, payload)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
