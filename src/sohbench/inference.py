from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import metrics as M


def run_inference_benchmark(
    model_store,
    table: pd.DataFrame,
    meta: pd.DataFrame | None,
    model_names: list[str],
) -> list[dict]:
    records: list[dict] = []
    for model_name in model_names:
        try:
            loaded = model_store.load_model(model_name)
        except Exception as exc:
            records.append(_failed_record(model_name, "all", 0, 0.0, str(exc)))
            continue

        for dataset, group in table.groupby("dataset", sort=True):
            base = {
                "model": model_name,
                "dataset": str(dataset),
                "featureset": loaded.manifest.get("featureset"),
                "estimator": loaded.manifest.get("estimator"),
                "n_samples": int(len(group)),
                "load_ms": float(loaded.load_ms),
            }
            try:
                start = time.perf_counter()
                pred = loaded.predict(group, meta)
                predict_ms = (time.perf_counter() - start) * 1000.0
                records.append(
                    {
                        **base,
                        **_metrics(group, pred),
                        "predict_ms": float(predict_ms),
                        "ms_per_1000_samples": float(predict_ms / max(len(group), 1) * 1000.0),
                        "samples_per_second": float(len(group) / max(predict_ms / 1000.0, 1e-12)),
                        "status": "succeeded",
                        "error": "",
                    }
                )
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
    return records


def _metrics(group: pd.DataFrame, pred: np.ndarray) -> dict:
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
    else:
        out.update({"mae": None, "rmse": None, "mean_bias_error": None})
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
