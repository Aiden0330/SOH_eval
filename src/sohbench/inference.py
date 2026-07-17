from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import metrics as M
from .contracts import (
    validate_finite_metrics,
    validate_finite_vector,
    validate_interval_bounds,
)
from .fabric import scoped_meta_for_frame


def run_inference_benchmark(
    model_store,
    table: pd.DataFrame,
    meta: pd.DataFrame | None,
    model_names: list[str],
    *,
    loaded_models: dict[str, object] | None = None,
) -> list[dict]:
    records: list[dict] = []
    dataset_groups = [
        (str(dataset), group)
        for dataset, group in table.groupby("dataset", sort=True)
    ]
    for model_name in model_names:
        try:
            loaded = (loaded_models or {}).get(model_name)
            if loaded is None:
                loaded = model_store.load_model(model_name)
        except Exception as exc:
            records.extend(
                _failed_record(
                    model_name,
                    dataset,
                    int(len(group)),
                    0.0,
                    str(exc),
                )
                for dataset, group in dataset_groups
            )
            continue

        for dataset, group in dataset_groups:
            base = {
                "model": model_name,
                "dataset": dataset,
                "featureset": loaded.manifest.get("featureset"),
                "estimator": loaded.manifest.get("estimator"),
                "n_samples": int(len(group)),
                "load_ms": float(loaded.load_ms),
            }
            try:
                group_meta = scoped_meta_for_frame(meta, group)
                start = time.perf_counter()
                pred = validate_finite_vector(
                    loaded.predict(group, group_meta),
                    name="predictions",
                    n_samples=len(group),
                )
                predict_ms = (time.perf_counter() - start) * 1000.0
                _validate_optional_intervals(loaded, group, group_meta)
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
    out = _trajectory_metrics(group, pred)
    if "soh" in group.columns:
        y = pd.to_numeric(group["soh"], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if np.isfinite(y).all():
            out.update(
                {
                    "mae": M.mae(y, pred),
                    "rmse": M.rmse(y, pred),
                    "mean_bias_error": M.mean_bias_error(y, pred),
                }
            )
    return validate_finite_metrics(out)


def _validate_optional_intervals(loaded, group, meta) -> None:
    predict_interval = getattr(loaded, "predict_interval", None)
    if not callable(predict_interval):
        return
    intervals = predict_interval(group, meta)
    if intervals is None:
        return
    try:
        low, high = intervals
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "predict_interval() must return interval_low and interval_high arrays"
        ) from exc
    low = validate_finite_vector(low, name="interval_low", n_samples=len(group))
    high = validate_finite_vector(
        high, name="interval_high", n_samples=len(group)
    )
    validate_interval_bounds(low, high)


def _trajectory_metrics(group: pd.DataFrame, pred: np.ndarray) -> dict:
    frame = group[["dataset", "cell_id", "cycle_index"]].copy()
    frame["prediction"] = pred
    identity = ["dataset", "cell_id", "cycle_index"]
    if frame.duplicated(identity).any():
        raise ValueError("duplicate cycle identity")

    per_asset = []
    for _, asset in frame.groupby(
        ["dataset", "cell_id"], sort=True, dropna=False
    ):
        if len(asset) < 2:
            continue
        ordered = asset.sort_values("cycle_index", kind="stable")[
            "prediction"
        ].to_numpy()
        per_asset.append(
            {
                "monotonicity_violation_rate": M.monotonicity_violation_rate(
                    ordered
                ),
                "value_range_violation": M.value_range_violation(ordered),
            }
        )
    if not per_asset:
        raise ValueError(
            "trajectory metrics require an asset with at least two cycles"
        )
    result = {
        metric: float(np.mean([values[metric] for values in per_asset]))
        for metric in (
            "monotonicity_violation_rate",
            "value_range_violation",
        )
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("trajectory metrics must be finite")
    return result


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
