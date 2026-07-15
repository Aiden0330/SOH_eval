from __future__ import annotations
import json, math, time
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from .acceptance import evaluate_verdicts
from .fabric.outcomes import (
    RunOutcome,
    is_finite_real_scalar,
    iter_execution_units,
    successful_result_rows,
)
from .tasks import SOHEstimationTask


def _json_safe(o):
    """Recursively replace non-finite floats (NaN/inf) with None so the output is
    valid JSON (browsers/JSON.parse reject bare NaN). NaN appears for failed folds
    or estimators that are N/A on a feature set (e.g. phys_resid without throughput)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def _dump(obj, path):
    path.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False, default=str))


def _successful_units_have_complete_metrics(
    evidence: pd.DataFrame, required_metrics: set[str],
) -> bool:
    for unit in iter_execution_units(evidence):
        finite_mask = unit["value"].map(is_finite_real_scalar).to_numpy(dtype=bool)
        finite_unit = unit.loc[finite_mask]
        if not required_metrics.issubset(set(finite_unit["metric"].dropna())):
            return False
    return True


def _leaderboard_entries(results: pd.DataFrame, task, group_cols: list[str]) -> list[dict]:
    entries = []
    has_track = "track" in results.columns

    for group_key, g in results.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        keys = dict(zip(group_cols, group_key))

        est = keys["estimator"]
        fs_name = keys.get("featureset", "full")
        proto = keys["protocol"]
        track = keys.get("track")
        if track is None:
            if has_track:
                track = g["track"].iloc[0]
            else:
                track = results["feature_mode"].iloc[0] if "feature_mode" in results.columns else "unified"

        execution = RunOutcome.from_results(g)
        evidence = successful_result_rows(g)
        finite_mask = evidence["value"].map(is_finite_real_scalar).to_numpy(dtype=bool)
        finite_evidence = evidence.loc[finite_mask].copy()
        finite_evidence["value"] = [float(value) for value in finite_evidence["value"]]
        agg = finite_evidence.groupby("metric")["value"].agg(["mean", "std"])
        metric_names = g["metric"].dropna().drop_duplicates()
        metrics = {}
        for metric in metric_names:
            if metric in agg.index:
                row = agg.loc[metric]
                metrics[metric] = {
                    "mean": float(row["mean"]),
                    "std": float(row["std"] if math.isfinite(row["std"]) else 0.0),
                }
            else:
                metrics[metric] = {"mean": float("nan"), "std": 0.0}
        flat = {
            metric: values["mean"]
            for metric, values in metrics.items()
            if math.isfinite(values["mean"])
        }
        coverage = execution.succeeded / execution.attempted if execution.attempted else 0.0
        required_acceptance_metrics = {
            "mae", "picp", "monotonicity_violation_rate", "value_range_violation",
        }
        verdicts = (
            {v.name: v.passed for v in evaluate_verdicts(flat, task)}
            if (
                required_acceptance_metrics.issubset(flat)
                and _successful_units_have_complete_metrics(
                    evidence, required_acceptance_metrics,
                )
            )
            else {}
        )
        ci_source = g["ci_source"].iloc[0] if "ci_source" in g.columns else "conformal"

        entry = {
            "estimator": est,
            "featureset": fs_name,
            "track": track,
            "protocol": proto,
            "metrics": metrics,
            "verdicts": verdicts,
            "coverage": coverage,
            "execution": asdict(execution),
            "ci_source": ci_source,
        }
        if "held_out_key" in keys:
            entry["held_out_key"] = str(keys["held_out_key"])
        if "backend" in g.columns:
            entry["backend"] = g["backend"].iloc[0]
        entries.append(entry)

    return entries


def write_artifacts(results: pd.DataFrame, qc_report, task, config, out_dir):
    out = Path(out_dir) / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out / "results.parquet")
    results.to_csv(out / "results.csv", index=False)

    # Group by featureset when the column is present; fall back for backward compat.
    has_featureset = "featureset" in results.columns
    has_track = "track" in results.columns

    if has_featureset:
        group_cols = ["estimator", "featureset", "protocol"]
    else:
        group_cols = ["estimator", "protocol"]
    if has_track:
        group_cols.append("track")

    leaderboard = _leaderboard_entries(results, task, group_cols)

    dataset_group_cols = group_cols + ["held_out_key"] if "held_out_key" in results.columns else []
    dataset_leaderboard = (
        _leaderboard_entries(results, task, dataset_group_cols)
        if dataset_group_cols else []
    )

    _dump(leaderboard, out / "leaderboard.json")
    _dump(dataset_leaderboard, out / "dataset_leaderboard.json")
    if qc_report is not None:
        _dump(qc_report.__dict__, out / "qc_report.json")
    _dump(config, out / "run_config.json")
    return str(out)


def write_inference_artifacts(records: list[dict], config: dict, out_dir):
    out = Path(out_dir) / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    frame.to_parquet(out / "inference_results.parquet")
    frame.to_csv(out / "inference_results.csv", index=False)
    _dump(
        {
            "records": records,
            "summary": _inference_summary(records),
        },
        out / "inference_benchmark.json",
    )
    _dump(config, out / "inference_run_config.json")
    return str(out)


def _inference_summary(records: list[dict]) -> dict:
    succeeded = [row for row in records if row.get("status") == "succeeded"]
    fastest = min(
        succeeded,
        key=lambda row: row.get("ms_per_1000_samples") or float("inf"),
        default=None,
    )
    best = min(
        [row for row in succeeded if row.get("mae") is not None],
        key=lambda row: row.get("mae"),
        default=None,
    )
    return {
        "n_records": len(records),
        "n_succeeded": len(succeeded),
        "fastest_model": fastest.get("model") if fastest else None,
        "fastest_ms_per_1000_samples": fastest.get("ms_per_1000_samples") if fastest else None,
        "best_transfer_model": best.get("model") if best else None,
        "best_transfer_mae": best.get("mae") if best else None,
    }
