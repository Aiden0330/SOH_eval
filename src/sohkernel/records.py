"""Prediction records and inference metrics (Spec 01 §4, §8; extraction plan Task 8).

The per-cycle ``PredictionRecord`` is the evidence unit every downstream view (cases,
evidence pages, claims) is built from; its shape, the identity derivations that join it to
artifacts and splits, and the aggregate / extended inference metrics live in the kernel so
the workbench and the platform produce byte-identical records.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from . import metrics as M

PREDICTION_RECORD_SCHEMA_VERSION = "1.0.0"
PREDICTION_RECORD_FIELDS = (
    "schema_version",
    "prediction_record_id",
    "sample_id",
    "run_id",
    "candidate_id",
    "model_name",
    "model_artifact_id",
    "featureset",
    "dataset",
    "dataset_snapshot_id",
    "split_assignment_id",
    "evaluation_scope",
    "cell_id",
    "cycle_index",
    "true_soh",
    "pred_soh",
    "lower",
    "upper",
    "absolute_error",
    "interval_covered",
    "curve_segment_ref_id",
    "feature_values",
    "metric_contributions",
)
METRIC_CONTRIBUTION_FIELDS = ("mae_contribution", "rmse_squared_contribution", "picp_contribution")


def validate_prediction_record(record: dict) -> dict:
    """Return ``record`` if it carries every PredictionRecord field with a sane type; else raise."""
    missing = [field for field in PREDICTION_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"prediction record is missing fields: {missing}")
    if record["schema_version"] != PREDICTION_RECORD_SCHEMA_VERSION:
        raise ValueError(f"unsupported prediction record schema_version {record['schema_version']!r}")
    if not str(record["prediction_record_id"]).startswith("pred:"):
        raise ValueError("prediction_record_id must start with 'pred:'")
    if not isinstance(record["feature_values"], dict) or not isinstance(record["metric_contributions"], dict):
        raise ValueError("feature_values and metric_contributions must be objects")
    unknown = [k for k in record["metric_contributions"] if k not in METRIC_CONTRIBUTION_FIELDS]
    if unknown:
        raise ValueError(f"unknown metric contributions: {unknown}")
    return record


def prediction_evidence_rows(
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
        cycle_index = json_number(source.get("cycle_index"))
        true_soh = json_number(source.get("soh"))
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
        prediction_record_id = f"pred:{id_token(job_id or 'inference', 32)}:{hashlib.sha256(digest_payload).hexdigest()[:24]}"
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
            if column in source.index and (value := json_number(source.get(column))) is not None
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


def candidate_id(identity: str) -> str:
    return f"artifact:{id_token(identity, 48)}"


def id_token(value: object, limit: int) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "unknown"))
    token = token.strip("-.")[:limit]
    return token or "unknown"


def split_assignment_id(manifest: dict) -> str:
    descriptor = {
        "protocol": str(manifest.get("protocol") or ""),
        "held_out_key": str(manifest.get("held_out_key") or ""),
        "seed": manifest.get("seed"),
    }
    if not any(value not in (None, "") for value in descriptor.values()):
        return ""
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"protocol@sha256:{hashlib.sha256(encoded).hexdigest()}"


def json_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def inference_metrics(group: pd.DataFrame, pred: np.ndarray, eval_mask=None, interval=None) -> dict:
    """Aggregate metrics; ``eval_mask`` excludes adaptation-calibration rows."""
    if eval_mask is not None:
        eval_mask = np.asarray(eval_mask, dtype=bool)
        group = group.iloc[eval_mask]
        pred = pred[eval_mask]
        if interval is not None:
            interval = (interval[0][eval_mask], interval[1][eval_mask])
    out = {
        "monotonicity_violation_rate": per_cell_monotonicity_violation(group, pred),
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
        out.update(extended_metrics(group, y, pred, interval))
    else:
        out.update({"mae": None, "rmse": None, "mean_bias_error": None})
    return out


EOL_SOH_THRESHOLD = 0.80


def per_cell_monotonicity_violation(group: pd.DataFrame, pred: np.ndarray) -> float:
    """Sample-weighted mean of per-cell violation rates.  A global cycle sort
    interleaves every cell at each cycle index, so cross-cell level jumps
    would dominate the physical (within-cell) signal."""
    if "cell_id" not in group.columns:
        return M.monotonicity_violation_rate(
            pred[np.argsort(group["cycle_index"].to_numpy())]
        )
    return M.per_cell_monotonicity_violation_rate(
        group["cell_id"].to_numpy(), group["cycle_index"].to_numpy(), pred
    )


def extended_metrics(group: pd.DataFrame, y: np.ndarray, pred: np.ndarray, interval) -> dict:
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
            true_hits = rows.loc[rows["true"] < EOL_SOH_THRESHOLD, "cycle"]
            if true_hits.empty:
                continue
            cells_with_true_eol += 1
            pred_hits = rows.loc[rows["pred"] < EOL_SOH_THRESHOLD, "cycle"]
            if pred_hits.empty:
                undetected += 1
                continue
            cycle_errors.append(abs(float(pred_hits.iloc[0]) - float(true_hits.iloc[0])))
        out["eol"] = {
            "threshold": EOL_SOH_THRESHOLD,
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


def failed_record(model_name: str, dataset: str, n_samples: int, load_ms: float, error: str) -> dict:
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
