from __future__ import annotations
import json, math, time
from pathlib import Path
import pandas as pd
from .acceptance import evaluate_verdicts
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


UNIT_COLUMNS = ("estimator", "featureset", "protocol", "held_out_key", "fold", "seed")


def _unit_outcomes(group: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Split a leaderboard group into its evidence rows and the unit accounting.

    An execution unit is one (estimator, featureset, protocol, held-out key, fold, seed).
    Units the runner marked ``failed`` (or, for legacy frames, units carrying an error
    message) never enter the aggregates; the entry states how many were attempted,
    succeeded and failed (``outcome``: complete | partial | failed).
    """
    unit_cols = [c for c in UNIT_COLUMNS if c in group.columns]
    if "status" in group.columns:
        failed_rows = group["status"].astype(str).eq("failed")
    elif "error" in group.columns:
        failed_rows = group["error"].notna() & group["error"].astype(str).str.strip().ne("")
    else:
        failed_rows = pd.Series(False, index=group.index)
    units = group.assign(_failed=failed_rows)
    units = units.drop_duplicates(unit_cols) if unit_cols else units.iloc[:1]
    attempted = int(len(units))
    failed = int(units["_failed"].sum())
    succeeded = attempted - failed
    outcome = "complete" if attempted and not failed else ("partial" if succeeded else "failed")
    evidence = group.loc[~failed_rows] if succeeded else group
    return evidence, {"attempted": attempted, "succeeded": succeeded, "failed": failed, "outcome": outcome}


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

        evidence, units = _unit_outcomes(g)
        agg = evidence.groupby("metric")["value"].agg(["mean", "std"])
        metrics = {m: {"mean": float(r["mean"]),
                        "std": float(r["std"] if r["std"] == r["std"] else 0.0)}
                   for m, r in agg.iterrows()}
        flat = {m: v["mean"] for m, v in metrics.items()}
        verdicts = {v.name: v.passed for v in evaluate_verdicts(flat, task)}
        ci_source = g["ci_source"].iloc[0] if "ci_source" in g.columns else "conformal"

        entry = {
            "estimator": est,
            "featureset": fs_name,
            "track": track,
            "protocol": proto,
            "metrics": metrics,
            "verdicts": verdicts,
            "ci_source": ci_source,
            "units": units,
        }
        if "held_out_key" in keys:
            entry["held_out_key"] = str(keys["held_out_key"])
        if "backend" in g.columns:
            entry["backend"] = g["backend"].iloc[0]
        entries.append(entry)

    return entries


def _attach_training_provenance(entries: list[dict], config: dict) -> list[dict]:
    artifacts = list(config.get("model_artifacts") or [])
    snapshots = list(config.get("dataset_snapshots") or [])
    snapshot_ids = [
        item.get("snapshot_id") for item in snapshots if item.get("snapshot_id")
    ]
    for entry in entries:
        matching = [
            item for item in artifacts
            if item.get("estimator") == entry.get("estimator")
            and (item.get("featureset") or "full") == (entry.get("featureset") or "full")
            and item.get("protocol") == entry.get("protocol")
            and (
                not entry.get("held_out_key")
                or str(item.get("held_out_key")) == str(entry.get("held_out_key"))
            )
        ]
        artifact_ids = sorted(
            {str(item.get("artifact_id")) for item in matching if item.get("artifact_id")}
        )
        entry["run_id"] = config.get("run_id") or config.get("job_id")
        entry["preflight_id"] = config.get("preflight_id")
        entry["source_datasets"] = list(config.get("datasets") or [])
        entry["dataset_snapshot_ids"] = snapshot_ids
        entry["model_artifact_ids"] = artifact_ids
        entry["model_artifact_count"] = len(artifact_ids)
        if len(artifact_ids) == 1:
            entry["model_artifact_id"] = artifact_ids[0]
    return entries


def _claim_run_dir(out_dir) -> Path:
    """Claim a fresh run directory; suffix on same-second collisions so
    two jobs can never silently merge their artifacts into one directory."""
    base = Path(out_dir)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / stamp
    suffix = 1
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base / f"{stamp}-{suffix}"
            suffix += 1


def write_artifacts(results: pd.DataFrame, qc_report, task, config, out_dir):
    out = _claim_run_dir(out_dir)
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

    leaderboard = _attach_training_provenance(
        _leaderboard_entries(results, task, group_cols),
        config,
    )

    dataset_group_cols = group_cols + ["held_out_key"] if "held_out_key" in results.columns else []
    dataset_leaderboard = (
        _attach_training_provenance(
            _leaderboard_entries(results, task, dataset_group_cols),
            config,
        )
        if dataset_group_cols else []
    )

    _dump(leaderboard, out / "leaderboard.json")
    _dump(dataset_leaderboard, out / "dataset_leaderboard.json")
    if qc_report is not None:
        _dump(qc_report.__dict__, out / "qc_report.json")
    _dump(config, out / "run_config.json")
    return str(out)


def write_inference_artifacts(
    records: list[dict],
    config: dict,
    out_dir,
    *,
    evidence: dict | None = None,
):
    out = _claim_run_dir(out_dir)
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
    if evidence is not None:
        prediction_records = list(evidence.get("prediction_records") or [])
        metric_contributions = list(evidence.get("metric_contributions") or [])
        curve_segment_refs = list(evidence.get("curve_segment_refs") or [])
        _dump(
            {
                "schema_version": "1.0.0",
                "job_id": str(config.get("job_id") or ""),
                "case_id": str(config.get("case_id") or ""),
                "profile_name": str(config.get("profile_name") or ""),
                "records": prediction_records,
            },
            out / "prediction_records.json",
        )
        _dump(
            {
                "schema_version": "1.0.0",
                "job_id": str(config.get("job_id") or ""),
                "records": metric_contributions,
            },
            out / "metric_contributions.json",
        )
        _dump(
            {
                "schema_version": "1.0.0",
                "job_id": str(config.get("job_id") or ""),
                "records": curve_segment_refs,
            },
            out / "curve_segment_refs.json",
        )
        _dump(
            {
                "schema_version": "1.0.0",
                "case_id": str(config.get("case_id") or ""),
                "profile_name": str(config.get("profile_name") or ""),
                "inference_job_id": str(config.get("job_id") or ""),
                "datasets": list(config.get("datasets") or []),
                "candidate_ids": sorted({
                    str(item.get("candidate_id") or "")
                    for item in prediction_records
                    if item.get("candidate_id")
                }),
                "prediction_record_count": len(prediction_records),
                "curve_segment_ref_count": len(curve_segment_refs),
                "curve_artifacts": list(evidence.get("curve_artifacts") or []),
            },
            out / "inference_evidence_manifest.json",
        )
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
