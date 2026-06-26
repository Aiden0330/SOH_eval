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

def write_artifacts(results: pd.DataFrame, qc_report, task, config, out_dir):
    out = Path(out_dir) / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out / "results.parquet")
    results.to_csv(out / "results.csv", index=False)

    leaderboard = []
    feature_mode = results["feature_mode"].iloc[0] if "feature_mode" in results else "unified"

    # Group by featureset when the column is present; fall back for backward compat.
    has_featureset = "featureset" in results.columns
    if has_featureset:
        group_cols = ["estimator", "featureset", "protocol"]
    else:
        group_cols = ["estimator", "protocol"]

    for group_key, g in results.groupby(group_cols):
        if has_featureset:
            est, fs_name, proto = group_key
        else:
            est, proto = group_key
            fs_name = "full"

        agg = g.groupby("metric")["value"].agg(["mean","std"])
        metrics = {m: {"mean": float(r["mean"]), "std": float(r["std"] if r["std"]==r["std"] else 0.0)}
                   for m, r in agg.iterrows()}
        flat = {m: v["mean"] for m, v in metrics.items()}
        verdicts = {v.name: v.passed for v in evaluate_verdicts(flat, task)}
        ci_source = g["ci_source"].iloc[0] if "ci_source" in g.columns else "conformal"
        leaderboard.append({"estimator": est, "featureset": fs_name,
                            "track": feature_mode, "protocol": proto,
                            "metrics": metrics, "verdicts": verdicts, "ci_source": ci_source})
    _dump(leaderboard, out / "leaderboard.json")
    if qc_report is not None:
        _dump(qc_report.__dict__, out / "qc_report.json")
    _dump(config, out / "run_config.json")
    return str(out)
