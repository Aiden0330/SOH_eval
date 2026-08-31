from __future__ import annotations

import hashlib
import json
import re
import time

import numpy as np
import pandas as pd

from . import metrics as M
from .metadata import DATASET_CHEMISTRY


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
    if meta is None:
        return None
    try:
        return meta.loc[group.index]
    except (KeyError, TypeError):
        return meta if len(meta) == len(group) else None


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


def _prediction_evidence_rows(
    group: pd.DataFrame,
    pred: np.ndarray,
    interval,
    *,
    job_id: str,
    model_name: str,
    candidate_id: str,
    model_artifact_id: str,
    featureset: str,
    feature_columns: list[str],
    dataset: str,
    dataset_snapshot_id: str,
    split_assignment_id: str,
    evaluation_scope: str,
    calibration_mask=None,
) -> tuple[list[dict], list[dict]]:
    count = len(group)
    if calibration_mask is not None:
        # Contributions normalize over the evaluated subset only; calibration
        # rows keep their predictions but carry no metric weight.
        count = int(count - int(np.asarray(calibration_mask).sum()))
    lower = interval[0] if interval is not None else None
    upper = interval[1] if interval is not None else None
    rows: list[dict] = []
    contributions: list[dict] = []
    for position, (_index, source) in enumerate(group.iterrows()):
        is_calibration = bool(calibration_mask[position]) if calibration_mask is not None else False
        cell_id = str(source.get("cell_id") or "")
        cycle_index = _json_number(source.get("cycle_index"))
        true_soh = _json_number(source.get("soh"))
        pred_soh = float(pred[position])
        lower_value = float(lower[position]) if lower is not None else None
        upper_value = float(upper[position]) if upper is not None else None
        absolute_error = abs(pred_soh - true_soh) if true_soh is not None else None
        interval_covered = (
            bool(lower_value <= true_soh <= upper_value)
            if true_soh is not None and lower_value is not None and upper_value is not None
            else None
        )
        sample_id = f"{dataset}:{cell_id}:{cycle_index}"
        digest_payload = f"{job_id}|{model_name}|{sample_id}".encode("utf-8")
        prediction_record_id = f"pred:{_id_token(job_id or 'inference', 32)}:{hashlib.sha256(digest_payload).hexdigest()[:24]}"
        contribution = {
            "mae_contribution": (
                absolute_error / max(count, 1)
                if absolute_error is not None and not is_calibration else None
            ),
            "rmse_squared_contribution": (
                absolute_error * absolute_error / max(count, 1)
                if absolute_error is not None and not is_calibration else None
            ),
            "picp_contribution": (
                (1.0 if interval_covered else 0.0) / max(count, 1)
                if interval_covered is not None and not is_calibration else None
            ),
        }
        feature_values = {
            column: value
            for column in feature_columns
            if column in source.index and (value := _json_number(source.get(column))) is not None
        }
        record = {
            "schema_version": "1.0.0",
            "prediction_record_id": prediction_record_id,
            "sample_id": sample_id,
            "run_id": job_id,
            "candidate_id": candidate_id,
            "model_name": model_name,
            "model_artifact_id": model_artifact_id,
            "featureset": featureset,
            "dataset": dataset,
            "dataset_snapshot_id": dataset_snapshot_id,
            "split_assignment_id": split_assignment_id,
            "evaluation_scope": evaluation_scope,
            "cell_id": cell_id,
            "cycle_index": cycle_index,
            "true_soh": true_soh,
            "pred_soh": pred_soh,
            "lower": lower_value,
            "upper": upper_value,
            "absolute_error": absolute_error,
            "interval_covered": interval_covered,
            "curve_segment_ref_id": "",
            "feature_values": feature_values,
            "metric_contributions": contribution,
            **({"calibration_sample": True} if is_calibration else {}),
        }
        rows.append(record)
        contributions.append(
            {
                "schema_version": "1.0.0",
                "prediction_record_id": prediction_record_id,
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                **contribution,
            }
        )
    return rows, contributions


def _candidate_id(identity: str) -> str:
    return f"artifact:{_id_token(identity, 48)}"


def _id_token(value: object, limit: int) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "unknown"))
    token = token.strip("-.")[:limit]
    return token or "unknown"


def _split_assignment_id(manifest: dict) -> str:
    descriptor = {
        "protocol": str(manifest.get("protocol") or ""),
        "held_out_key": str(manifest.get("held_out_key") or ""),
        "seed": manifest.get("seed"),
    }
    if not any(value not in (None, "") for value in descriptor.values()):
        return ""
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"protocol@sha256:{hashlib.sha256(encoded).hexdigest()}"


def _json_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _metrics(group: pd.DataFrame, pred: np.ndarray, eval_mask=None, interval=None) -> dict:
    """Aggregate metrics; ``eval_mask`` excludes adaptation-calibration rows."""
    if eval_mask is not None:
        eval_mask = np.asarray(eval_mask, dtype=bool)
        group = group.iloc[eval_mask]
        pred = pred[eval_mask]
        if interval is not None:
            interval = (interval[0][eval_mask], interval[1][eval_mask])
    out = {
        "monotonicity_violation_rate": M.monotonicity_violation_rate(
            pred[np.argsort(group["cycle_index"].to_numpy())]
        ),
        "value_range_violation": M.value_range_violation(pred),
    }
    if "soh" in group.columns:
        y = group["soh"].to_numpy(dtype=float)
        out.update(
            {
                "mae": M.mae(y, pred),
                "rmse": M.rmse(y, pred),
                "mean_bias_error": M.mean_bias_error(y, pred),
            }
        )
        out.update(_extended_metrics(group, y, pred, interval))
    else:
        out.update({"mae": None, "rmse": None, "mean_bias_error": None})
    return out


_EOL_SOH_THRESHOLD = 0.80


def _extended_metrics(group: pd.DataFrame, y: np.ndarray, pred: np.ndarray, interval) -> dict:
    """Pooled MAE hides where the error lives; these answer the questions it
    can't: which SOH band fails, which cells collapse, whether EOL timing is
    usable, and whether the model's own intervals are honest."""
    finite = np.isfinite(y) & np.isfinite(pred)
    error = pred - y
    bands = {
        "soh_ge_90": y >= 0.90,
        "soh_80_90": (y >= 0.80) & (y < 0.90),
        "soh_lt_80": y < 0.80,
    }
    band_metrics = {}
    for name, in_band in bands.items():
        rows = finite & in_band
        n = int(rows.sum())
        band_metrics[name] = {
            "mae": float(np.mean(np.abs(error[rows]))) if n else None,
            "bias": float(np.mean(error[rows])) if n else None,
            "n": n,
        }
    out: dict = {"band_metrics": band_metrics}

    cells = group["cell_id"].astype(str).to_numpy() if "cell_id" in group.columns else None
    cycles = group["cycle_index"].to_numpy(dtype=float) if "cycle_index" in group.columns else None
    if cells is not None and finite.any():
        cell_frame = pd.DataFrame({
            "cell": cells[finite],
            "abs_error": np.abs(error[finite]),
        })
        per_cell = cell_frame.groupby("cell")["abs_error"].mean()
        worst_cell = per_cell.idxmax()
        out.update({
            "cell_mae_worst": float(per_cell.max()),
            "cell_mae_worst_cell": str(worst_cell),
            "cell_mae_p90": float(per_cell.quantile(0.9)),
            "cells_evaluated": int(len(per_cell)),
        })

    if cells is not None and cycles is not None:
        cycle_errors: list[float] = []
        cells_with_true_eol = 0
        undetected = 0
        eol_frame = pd.DataFrame({
            "cell": cells[finite],
            "cycle": cycles[finite],
            "true": y[finite],
            "pred": pred[finite],
        })
        for _cell, rows in eol_frame.groupby("cell", sort=False):
            rows = rows.sort_values("cycle")
            true_hits = rows.loc[rows["true"] < _EOL_SOH_THRESHOLD, "cycle"]
            if true_hits.empty:
                continue
            cells_with_true_eol += 1
            pred_hits = rows.loc[rows["pred"] < _EOL_SOH_THRESHOLD, "cycle"]
            if pred_hits.empty:
                undetected += 1
                continue
            cycle_errors.append(abs(float(pred_hits.iloc[0]) - float(true_hits.iloc[0])))
        out["eol"] = {
            "threshold": _EOL_SOH_THRESHOLD,
            "cells_with_true_eol": cells_with_true_eol,
            "cells_predicted": len(cycle_errors),
            "undetected_cells": undetected,
            "mean_abs_cycle_error": float(np.mean(cycle_errors)) if cycle_errors else None,
            "median_abs_cycle_error": float(np.median(cycle_errors)) if cycle_errors else None,
        }

    if interval is not None:
        lower = np.asarray(interval[0], dtype=float)
        upper = np.asarray(interval[1], dtype=float)
        rows = finite & np.isfinite(lower) & np.isfinite(upper)
        if rows.any():
            out["picp"] = float(np.mean((lower[rows] <= y[rows]) & (y[rows] <= upper[rows])))
            out["mean_interval_width"] = float(np.mean(upper[rows] - lower[rows]))
    return out


def _failed_record(model_name: str, dataset: str, n_samples: int, load_ms: float, error: str) -> dict:
    return {
        "model": model_name,
        "dataset": dataset,
        "featureset": None,
        "estimator": None,
        "n_samples": n_samples,
        "load_ms": load_ms,
        "mae": None,
        "rmse": None,
        "mean_bias_error": None,
        "monotonicity_violation_rate": None,
        "value_range_violation": None,
        "predict_ms": None,
        "ms_per_1000_samples": None,
        "samples_per_second": None,
        "status": "failed",
        "error": error,
    }
