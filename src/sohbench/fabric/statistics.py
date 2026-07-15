from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .outcomes import is_finite_real_scalar


_ASSET_COLUMNS = ("dataset", "cell_id")
_METRIC_COLUMNS = (*_ASSET_COLUMNS, "y_true", "y_pred")
_SAMPLE_IDENTITY_COLUMNS = (
    "run_id",
    "candidate_id",
    "featureset",
    "protocol",
    "fold",
    "seed",
    "dataset",
    "cell_id",
    "cycle_index",
    "sample_id",
)
_SAMPLE_MARKERS = frozenset(("cycle_index", "sample_id"))
_SUPPORTED_METRICS = frozenset(("mae", "rmse"))


def _require_frame(frame: object, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing {name} columns: {', '.join(missing)}")


def _finite_numeric_values(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not all(is_finite_real_scalar(value) for value in array):
        raise ValueError(f"{name} must contain only finite numeric values")
    numeric = np.asarray(array, dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    return numeric


def asset_metric_table(predictions: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """Return one unweighted MAE or RMSE value per dataset/cell asset."""
    if metric not in _SUPPORTED_METRICS:
        raise ValueError("metric must be one of: mae, rmse")
    frame = _require_frame(predictions, "predictions")
    _require_columns(frame, _METRIC_COLUMNS, "prediction")
    if frame.empty:
        raise ValueError("predictions must be non-empty")
    for column in _ASSET_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"asset column {column} must not contain null values")

    identity_columns = [
        column for column in _SAMPLE_IDENTITY_COLUMNS if column in frame.columns
    ]
    if _SAMPLE_MARKERS.intersection(identity_columns):
        if frame[identity_columns].isna().any(axis=None):
            raise ValueError("sample identity columns must not contain null values")
        if frame.duplicated(identity_columns).any():
            raise ValueError("duplicate prediction sample identity")

    y_true = _finite_numeric_values(frame["y_true"].to_numpy(), "y_true")
    y_pred = _finite_numeric_values(frame["y_pred"].to_numpy(), "y_pred")
    scored = frame.loc[:, list(_ASSET_COLUMNS)].copy()
    scored["y_true"] = y_true
    scored["y_pred"] = y_pred

    records: list[dict] = []
    with np.errstate(over="ignore", invalid="ignore"):
        for (dataset, cell_id), group in scored.groupby(
            list(_ASSET_COLUMNS), sort=True, dropna=False
        ):
            error = group["y_pred"].to_numpy() - group["y_true"].to_numpy()
            if metric == "mae":
                components = np.sort(np.abs(error))
                value = components.mean()
            else:
                components = np.sort(np.square(error))
                value = np.sqrt(components.mean())
            if not is_finite_real_scalar(value):
                raise ValueError(f"{metric} produced a non-finite asset result")
            records.append(
                {"dataset": dataset, "cell_id": cell_id, "value": float(value)}
            )

    result = pd.DataFrame.from_records(records, columns=[*_ASSET_COLUMNS, "value"])
    if result.empty or not np.isfinite(result["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"{metric} asset results must be non-empty and finite")
    return result.sort_values(list(_ASSET_COLUMNS), kind="stable").reset_index(drop=True)


def bootstrap_mean_ci(
    values: object,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap the unweighted mean of finite, one-dimensional unit values."""
    numeric = _finite_numeric_values(values, "bootstrap values")
    if not is_finite_real_scalar(confidence):
        raise ValueError("confidence must be a finite number between zero and one")
    selected_confidence = float(confidence)
    if not 0.0 < selected_confidence < 1.0:
        raise ValueError("confidence must be greater than zero and less than one")
    if (
        isinstance(n_resamples, (bool, np.bool_))
        or not isinstance(n_resamples, (int, np.integer))
        or int(n_resamples) <= 0
    ):
        raise ValueError("n_resamples must be a positive integer")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a nonnegative integer")

    with np.errstate(over="ignore", invalid="ignore"):
        draws = np.random.default_rng(int(seed)).choice(
            numeric,
            size=(int(n_resamples), numeric.size),
            replace=True,
        ).mean(axis=1)
        mean = numeric.mean()
        alpha = (1.0 - selected_confidence) / 2.0
        low = np.quantile(draws, alpha)
        high = np.quantile(draws, 1.0 - alpha)
    summary = np.asarray([mean, low, high], dtype=float)
    if not np.isfinite(draws).all() or not np.isfinite(summary).all():
        raise ValueError("bootstrap produced a non-finite result")
    return {
        "mean": float(mean),
        "low": float(low),
        "high": float(high),
        "confidence": selected_confidence,
        "n_assets": int(numeric.size),
    }


def paired_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_columns: Sequence[str],
    value_column: str,
) -> dict:
    """Bootstrap paired unit deltas, defined as ``left - right`` for each key."""
    left_frame = _require_frame(left, "left")
    right_frame = _require_frame(right, "right")
    if isinstance(key_columns, (str, bytes)) or not isinstance(key_columns, Sequence):
        raise TypeError("key_columns must be a non-empty sequence of column names")
    keys = list(key_columns)
    if not keys or any(not isinstance(column, str) or not column for column in keys):
        raise ValueError("key_columns must be a non-empty sequence of column names")
    if len(keys) != len(set(keys)):
        raise ValueError("pairing key columns must be unique")
    if not isinstance(value_column, str) or not value_column:
        raise ValueError("value_column must be a non-empty column name")
    if value_column in keys:
        raise ValueError("value_column must not also be a pairing key")

    required = [*keys, value_column]
    _require_columns(left_frame, required, "left pairing")
    _require_columns(right_frame, required, "right pairing")
    for name, frame in (("left", left_frame), ("right", right_frame)):
        if frame[keys].isna().any(axis=None):
            raise ValueError(f"{name} pairing keys must not contain null values")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} pairing keys must be unique")
        _finite_numeric_values(frame[value_column].to_numpy(), f"{name} {value_column}")

    paired = left_frame.loc[:, required].merge(
        right_frame.loc[:, required],
        on=keys,
        how="inner",
        suffixes=("_left", "_right"),
        sort=False,
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("paired inputs have zero overlapping pairs")
    try:
        paired = paired.sort_values(keys, kind="stable")
    except TypeError as exc:
        raise ValueError("pairing keys must have deterministic sortable values") from exc

    with np.errstate(over="ignore", invalid="ignore"):
        deltas = (
            paired[f"{value_column}_left"].to_numpy(dtype=float)
            - paired[f"{value_column}_right"].to_numpy(dtype=float)
        )
    if not np.isfinite(deltas).all():
        raise ValueError("paired deltas must be finite")
    interval = bootstrap_mean_ci(deltas)
    n_pairs = interval.pop("n_assets")
    return {
        "direction": "left-minus-right",
        **interval,
        "n_pairs": int(n_pairs),
    }
