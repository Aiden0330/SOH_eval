from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
from .acceptance import evaluate_verdicts
from .tasks import SOHEstimationTask

def write_artifacts(results: pd.DataFrame, qc_report, task, config, out_dir):
    out = Path(out_dir) / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out / "results.parquet")
    results.to_csv(out / "results.csv", index=False)

    leaderboard = []
    group_cols = ["estimator","protocol"]
    feature_mode = results["feature_mode"].iloc[0] if "feature_mode" in results else "unified"
    for (est, proto), g in results.groupby(group_cols):
        agg = g.groupby("metric")["value"].agg(["mean","std"])
        metrics = {m: {"mean": float(r["mean"]), "std": float(r["std"] if r["std"]==r["std"] else 0.0)}
                   for m, r in agg.iterrows()}
        flat = {m: v["mean"] for m, v in metrics.items()}
        verdicts = {v.name: v.passed for v in evaluate_verdicts(flat, task)}
        leaderboard.append({"estimator": est, "track": feature_mode, "protocol": proto,
                            "metrics": metrics, "verdicts": verdicts, "ci_source": "conformal"})
    (out / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    if qc_report is not None:
        (out / "qc_report.json").write_text(json.dumps(qc_report.__dict__, indent=2, default=str))
    (out / "run_config.json").write_text(json.dumps(config, indent=2, default=str))
    return str(out)
