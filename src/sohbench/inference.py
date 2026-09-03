from __future__ import annotations

import hashlib
import json
import re
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .metadata import DATASET_CHEMISTRY
from sohkernel.records import (  # record shape, ids and metrics moved to the kernel (Task 8)
    EOL_SOH_THRESHOLD as _EOL_SOH_THRESHOLD,
    candidate_id as _candidate_id,
    extended_metrics as _extended_metrics,
    failed_record as _failed_record,
    id_token as _id_token,
    inference_metrics as _metrics,
    json_number as _json_number,
    per_cell_monotonicity_violation as _per_cell_monotonicity_violation,
    prediction_evidence_rows as _prediction_evidence_rows,
    split_assignment_id as _split_assignment_id,
)


def run_inference_benchmark(
    model_store,
    table: pd.DataFrame,
    meta: pd.DataFrame | None,
    model_names: list[str],
) -> list[dict]:
    records, _evidence = _run_inference(
        model_store,
        table,
        meta,
        model_names,
        job_id="",
        dataset_snapshot_ids={},
        capture_evidence=False,
    )
    return records


def run_inference_benchmark_with_evidence(
    model_store,
    table: pd.DataFrame,
    meta: pd.DataFrame | None,
    model_names: list[str],
    *,
    job_id: str,
    dataset_snapshot_ids: dict[str, str] | None = None,
    evaluation_policy: str = "research_any",
    model_bindings: list[dict] | None = None,
    adapter=None,
) -> tuple[list[dict], dict]:
    """Run point inference and retain canonical per-cycle evidence facts.

    The legacy benchmark API intentionally continues to return aggregate rows.
    This companion API performs the same predictions once and additionally
    returns PredictionRecord and metric-contribution payloads for case-scoped
    evidence views.  ``adapter`` (sohbench.adaptation.InferenceAdapter)
    optionally applies a deployment-legal adaptation operator; its
    calibration rows are excluded from all metrics and flagged in evidence.
    """
    return _run_inference(
        model_store,
        table,
        meta,
        model_names,
        job_id=str(job_id or ""),
        dataset_snapshot_ids=dict(dataset_snapshot_ids or {}),
        capture_evidence=True,
        evaluation_policy=evaluation_policy,
        model_bindings=list(model_bindings or []),
        adapter=adapter,
    )


def _run_inference(
    model_store,
    table: pd.DataFrame,
    meta: pd.DataFrame | None,
    model_names: list[str],
    *,
    job_id: str,
    dataset_snapshot_ids: dict[str, str],
    capture_evidence: bool,
    evaluation_policy: str = "research_any",
    model_bindings: list[dict] | None = None,
    adapter=None,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    prediction_records: list[dict] = []
    metric_contributions: list[dict] = []
    bindings_by_name = {
        str(item.get("name") or ""): dict(item)
        for item in (model_bindings or [])
        if item.get("name")
    }
    for model_name in model_names:
        binding = bindings_by_name.get(str(model_name)) or {}
        bound_artifact_id = str(binding.get("artifact_id") or "")
        try:
            loaded = (
                model_store.load_artifact(bound_artifact_id)
                if bound_artifact_id
                else model_store.load_model(model_name)
            )
        except Exception as exc:
            records.append(_failed_record(model_name, "all", 0, 0.0, str(exc)))
            continue

        if bound_artifact_id:
            artifact_id = bound_artifact_id
        else:
            try:
                artifact_id = model_store.artifact_id(model_name)
            except Exception:
                artifact_id = ""
        candidate_id = _candidate_id(artifact_id or model_name)
        split_assignment_id = _split_assignment_id(loaded.manifest)

        for dataset, group in table.groupby("dataset", sort=True):
            dataset_name = str(dataset)
            scope = _evaluation_scope(
                loaded.manifest,
                dataset_name,
                group,
                policy=evaluation_policy,
            )
            scoped_group = scope["group"]
            base = {
                "model": model_name,
                "dataset": dataset_name,
                "featureset": loaded.manifest.get("featureset"),
                "estimator": loaded.manifest.get("estimator"),
                "candidate_id": candidate_id,
                "model_artifact_id": artifact_id,
                "n_samples": int(len(scoped_group)),
                "load_ms": float(loaded.load_ms),
                "evaluation_scope": scope["scope"],
                "evaluation_reason": scope["reason"],
                "source_chemistries": scope["source_chemistries"],
                "target_chemistry": scope["target_chemistry"],
            }
            if not scope["eligible"]:
                records.append({
                    **base,
                    "mae": None,
                    "rmse": None,
                    "mean_bias_error": None,
                    "monotonicity_violation_rate": None,
                    "value_range_violation": None,
                    "predict_ms": None,
                    "ms_per_1000_samples": None,
                    "samples_per_second": None,
                    "status": "skipped",
                    "error": scope["reason"],
                })
                continue
            try:
                calibration_mask = None
                adaptation_info = None
                if adapter is not None:
                    scoped_group, calibration_mask, adaptation_info = adapter.prepare(
                        loaded, binding, scoped_group
                    )
                group_meta = _group_meta(meta, scoped_group)
                start = time.perf_counter()
                pred = loaded.predict(scoped_group, group_meta)
                predict_ms = (time.perf_counter() - start) * 1000.0
                recal = None
                if adapter is not None and adapter.uses_recalibration:
                    pred, recal = adapter.recalibrate(scoped_group, pred, calibration_mask)
                    adaptation_info = {**adaptation_info, "recalibration": recal}
                interval = _prediction_interval(loaded, scoped_group, group_meta)
                if adapter is not None and recal is not None:
                    interval = adapter.recalibrate_interval(interval, recal)
                eval_mask = (
                    ~calibration_mask
                    if calibration_mask is not None and calibration_mask.any()
                    else None
                )
                records.append(
                    {
                        **base,
                        **_metrics(scoped_group, pred, eval_mask=eval_mask, interval=interval),
                        "predict_ms": float(predict_ms),
                        "ms_per_1000_samples": float(predict_ms / max(len(scoped_group), 1) * 1000.0),
                        "samples_per_second": float(len(scoped_group) / max(predict_ms / 1000.0, 1e-12)),
                        "status": "succeeded",
                        "error": "",
                        **({"adaptation": adaptation_info} if adaptation_info else {}),
                    }
                )
                if capture_evidence:
                    evidence_rows, contribution_rows = _prediction_evidence_rows(
                        scoped_group,
                        pred,
                        interval,
                        job_id=job_id,
                        model_name=model_name,
                        candidate_id=candidate_id,
                        model_artifact_id=artifact_id,
                        featureset=str(loaded.manifest.get("featureset") or "full"),
                        feature_columns=loaded.feature_columns(),
                        dataset=dataset_name,
                        dataset_snapshot_id=dataset_snapshot_ids.get(dataset_name, ""),
                        split_assignment_id=split_assignment_id,
                        evaluation_scope=scope["scope"],
                        calibration_mask=calibration_mask,
                    )
                    prediction_records.extend(evidence_rows)
                    metric_contributions.extend(contribution_rows)
            except Exception as exc:
                records.append(
                    {
                        **base,
                        "mae": None,
                        "rmse": None,
                        "mean_bias_error": None,
                        "monotonicity_violation_rate": None,
                        "value_range_violation": None,
                        "predict_ms": None,
                        "ms_per_1000_samples": None,
                        "samples_per_second": None,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    return records, {
        "prediction_records": prediction_records,
        "metric_contributions": metric_contributions,
    }


def _evaluation_scope(manifest: dict, dataset: str, group: pd.DataFrame, *, policy: str) -> dict:
    """Resolve the non-overlapping test slice used by inference validation."""
    trained_on = [str(value) for value in (manifest.get("trained_on") or []) if str(value)]
    held_out_key = str(manifest.get("held_out_key") or "")
    protocol = str(manifest.get("protocol") or "").lower()
    target_chemistry = str(DATASET_CHEMISTRY.get(dataset) or "unknown")
    source_chemistries = sorted({
        str(DATASET_CHEMISTRY.get(name) or "unknown")
        for name in trained_on
        if str(DATASET_CHEMISTRY.get(name) or "unknown").lower() != "unknown"
    })
    base = {
        "group": group,
        "eligible": True,
        "scope": "research_any",
        "reason": "研究模式：允许训练回放或跨化学体系验证",
        "source_chemistries": source_chemistries,
        "target_chemistry": target_chemistry,
    }
    if policy != "held_out_same_chemistry":
        return base

    if protocol == "loco" and held_out_key.startswith(f"{dataset}/"):
        held_out_cell = held_out_key.split("/", 1)[1]
        held_out_group = group[group["cell_id"].astype(str) == held_out_cell]
        return {
            **base,
            "group": held_out_group,
            "eligible": not held_out_group.empty,
            "scope": "held_out_test",
            "reason": f"仅使用训练未见电池 {held_out_key}",
        }
    if protocol == "lodo" and held_out_key == dataset:
        return {**base, "scope": "held_out_test", "reason": f"使用训练完整留出的数据集 {dataset}"}
    if dataset in trained_on:
        return {
            **base,
            "group": group.iloc[0:0],
            "eligible": False,
            "scope": "training_overlap",
            "reason": "目标数据参与过训练且没有匹配的留出切分",
        }
    if not trained_on:
        return {
            **base,
            "group": group.iloc[0:0],
            "eligible": False,
            "scope": "provenance_unknown",
            "reason": "模型未声明训练数据范围",
        }
    if target_chemistry.lower() == "unknown":
        return {
            **base,
            "group": group.iloc[0:0],
            "eligible": False,
            "scope": "chemistry_unknown",
            "reason": "目标数据集化学体系待确认",
        }
    if target_chemistry not in source_chemistries:
        return {
            **base,
            "group": group.iloc[0:0],
            "eligible": False,
            "scope": "cross_chemistry",
            "reason": f"迁移默认限制为同化学体系：{source_chemistries or ['unknown']} -> {target_chemistry}",
        }
    return {
        **base,
        "scope": "same_chemistry_transfer",
        "reason": f"同化学体系迁移：{target_chemistry}",
    }


def _group_meta(meta: pd.DataFrame | None, group: pd.DataFrame) -> pd.DataFrame | None:
    """Per-cell metadata for the cells present in ``group``.

    ``meta`` holds one row per cell while ``group`` holds cycle rows, so the
    old ``meta.loc[group.index]`` positional lookup either raised (→ None,
    starving plugin models that used meta at training time) or silently
    returned unrelated cells."""
    if meta is None:
        return None
    if "cell_id" in meta.columns and "cell_id" in group.columns:
        cells = set(group["cell_id"].astype(str))
        scoped = meta[meta["cell_id"].astype(str).isin(cells)]
        if "dataset" in meta.columns and "dataset" in group.columns:
            datasets = set(group["dataset"].astype(str))
            scoped = scoped[scoped["dataset"].astype(str).isin(datasets)]
        return scoped.reset_index(drop=True)
    return meta if len(meta) == len(group) else meta


def _prediction_interval(loaded, group, group_meta):
    try:
        interval = loaded.predict_interval(group, group_meta)
    except Exception:
        # Point prediction remains valid when an optional interval adapter is
        # unavailable for an imported model package.
        return None
    if interval is None:
        return None
    lower, upper = (np.asarray(values, dtype=float) for values in interval)
    if lower.shape[0] != len(group) or upper.shape[0] != len(group):
        return None
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return None
    return lower, upper
