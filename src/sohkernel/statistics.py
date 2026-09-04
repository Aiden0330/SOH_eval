"""Claim statistics (Spec 04 §4; extraction plan Task 7h): asset-level bootstrap intervals.

The unit of resampling is the asset (a cell).  Protocols whose folds *are* assets — ``loco``
and ``forward`` — expose one metric value per asset (averaged over seeds) directly in the
kernel's results frame, so their leaderboard entries carry bootstrap intervals; folds that pool
many assets (``lodo``, ``lcho``) do not, and the entry says so instead of guessing.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

ASSET_LEVEL_PROTOCOLS = frozenset({"loco", "forward"})
DEFAULT_CONFIDENCE = 0.90
DEFAULT_RESAMPLES = 1000
CLAIM_METRICS = ("mae", "rmse", "picp", "monotonicity_violation_rate", "eol_region_error")


def bootstrap_mean_ci(values: Iterable[float], *, confidence: float = DEFAULT_CONFIDENCE,
                      n_resamples: int = DEFAULT_RESAMPLES, seed: int = 0) -> dict:
    """Percentile bootstrap interval of the mean of finite unit values (deterministic given ``seed``)."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("bootstrap needs at least one finite value")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    mean = float(array.mean())
    if array.size == 1:
        lower = upper = mean
    else:
        rng = np.random.default_rng(seed)
        draws = array[rng.integers(0, array.size, size=(int(n_resamples), array.size))].mean(axis=1)
        alpha = (1.0 - float(confidence)) / 2.0
        lower, upper = float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))
    return {"mean": mean, "lower": lower, "upper": upper, "n": int(array.size),
            "confidence": float(confidence), "resamples": int(n_resamples), "seed": int(seed)}


def asset_metric_values(results: pd.DataFrame, metric: str) -> pd.Series | None:
    """One value per asset (mean over seeds of the fold metric) when every fold is an asset."""
    if results.empty or "held_out_key" not in results.columns:
        return None
    protocols = set(map(str, results["protocol"].unique())) if "protocol" in results.columns else set()
    if not protocols or not protocols <= ASSET_LEVEL_PROTOCOLS:
        return None
    rows = results[results["metric"].astype(str) == metric]
    if "status" in rows.columns:
        rows = rows[rows["status"].astype(str) != "failed"]
    values = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows.assign(value=values)[np.isfinite(values)]
    if rows.empty:
        return None
    return rows.groupby(rows["held_out_key"].astype(str))["value"].mean()


def asset_confidence_intervals(results: pd.DataFrame, *, metrics: Iterable[str] = CLAIM_METRICS,
                               confidence: float = DEFAULT_CONFIDENCE, n_resamples: int = DEFAULT_RESAMPLES,
                               seed: int = 0) -> dict:
    """Per-metric asset-level bootstrap intervals, or an explicit ``unavailable`` record."""
    protocols = sorted(map(str, results["protocol"].unique())) if "protocol" in results.columns else []
    if not protocols or not set(protocols) <= ASSET_LEVEL_PROTOCOLS:
        return {"status": "unavailable", "unit": "asset",
                "reason": f"folds of {protocols or ['?']} pool many assets; asset-level intervals need loco/forward folds"}
    intervals: dict[str, dict] = {}
    for metric in metrics:
        per_asset = asset_metric_values(results, metric)
        if per_asset is None or len(per_asset) < 2:
            continue
        intervals[metric] = bootstrap_mean_ci(per_asset.to_numpy(), confidence=confidence, n_resamples=n_resamples, seed=seed)
    if not intervals:
        return {"status": "unavailable", "unit": "asset", "reason": "fewer than two assets with a finite value"}
    return {"status": "available", "unit": "asset", "confidence": float(confidence), "resamples": int(n_resamples),
            "assets": int(max(item["n"] for item in intervals.values())), "intervals": intervals}
