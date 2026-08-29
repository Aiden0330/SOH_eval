"""Inference-phase adaptation operators (deployment-legal, channel-declared).

At inference the model artifact is frozen — algorithm and features cannot
change without invalidating the evidence chain.  What MAY change is how the
frozen model meets a new dataset, using only information that would exist
at deployment:

- ``none``     — baseline: the frozen model as-is.
- ``coral``    — CORrelation ALignment: a closed-form linear map that
  re-colors the *target's* feature distribution to match the *source*
  (training) distribution before predict.  Channel: unlabeled target
  feature statistics + the model's own training data statistics.
- ``recalibration`` — two-parameter output correction (slope/intercept)
  fitted on a small labeled calibration budget from the target (the "few
  RPT points" scenario).  Channel: K labeled target samples, which are
  EXCLUDED from evaluation.
- ``coral+recalibration`` — both, in that order.

Every operator declares its information channel in ``OPERATOR_CHANNELS``;
the adaptation search reports it, and evaluation always runs on the
non-calibration subset so operators with and without label budgets stay
comparable.

Design doc: docs/superpowers/specs/2026-08-26-agent-guided-search-design.md §1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADAPTATION_OPERATORS = ("none", "coral", "recalibration", "coral+recalibration")

OPERATOR_CHANNELS = {
    "none": "frozen model only",
    "coral": "unlabeled target feature statistics + source training statistics",
    "recalibration": "K labeled target calibration samples (excluded from evaluation)",
    "coral+recalibration": (
        "unlabeled target feature statistics + source training statistics "
        "+ K labeled target calibration samples (excluded from evaluation)"
    ),
}

_COV_RIDGE = 1e-6


def _feature_matrix(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    frame = table[[c for c in columns if c in table.columns]]
    return frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _matrix_sqrt(matrix: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, _COV_RIDGE, None)
    powered = values ** (-0.5 if inverse else 0.5)
    return (vectors * powered) @ vectors.T


def fit_coral(
    source: np.ndarray, target: np.ndarray, columns: list[str]
) -> dict:
    """Closed-form CORAL: x' = (x - mu_t) @ Ct^-1/2 @ Cs^1/2 + mu_s.

    Rows with any non-finite value are ignored for the statistics; the
    returned dict is a serializable description of the linear map (goes into
    the report as provenance).
    """
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if len(source) < 8 or len(target) < 8:
        raise ValueError("CORAL needs at least 8 finite rows on each side")
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    cov_s = np.cov(source, rowvar=False) + _COV_RIDGE * np.eye(source.shape[1])
    cov_t = np.cov(target, rowvar=False) + _COV_RIDGE * np.eye(target.shape[1])
    transform = _matrix_sqrt(cov_t, inverse=True) @ _matrix_sqrt(cov_s)
    return {
        "columns": list(columns),
        "mu_source": mu_s.tolist(),
        "mu_target": mu_t.tolist(),
        "matrix": transform.tolist(),
    }


def apply_coral(table: pd.DataFrame, coral: dict) -> pd.DataFrame:
    columns = [c for c in coral["columns"] if c in table.columns]
    if columns != list(coral["columns"]):
        raise ValueError("target table is missing CORAL feature columns")
    out = table.copy()
    x = _feature_matrix(out, columns)
    mu_t = np.asarray(coral["mu_target"], dtype=float)
    mu_s = np.asarray(coral["mu_source"], dtype=float)
    matrix = np.asarray(coral["matrix"], dtype=float)
    aligned = (x - mu_t) @ matrix + mu_s
    # Rows with missing values pass through unaligned (imputation happens
    # inside the model pipeline, as at training time).
    finite = np.isfinite(x).all(axis=1)
    for j, column in enumerate(columns):
        values = x[:, j].copy()
        values[finite] = aligned[finite, j]
        out[column] = values
    return out


def calibration_split(
    table: pd.DataFrame, budget: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic budget: the EARLIEST cycles of each cell, round-robin.

    Mirrors the deployment reality (early reference tests are what you get
    first) and keeps the evaluation subset identical across operators.
    """
    if budget <= 0:
        return np.zeros(len(table), dtype=bool), np.ones(len(table), dtype=bool)
    frame = table.reset_index(drop=True)
    ordered = frame.sort_values(["cell_id", "cycle_index"], kind="stable")
    rank_in_cell = ordered.groupby("cell_id", sort=False).cumcount()
    priority = ordered.assign(_rank=rank_in_cell).sort_values(
        ["_rank", "cell_id", "cycle_index"], kind="stable"
    )
    calibration = np.zeros(len(frame), dtype=bool)
    calibration[priority.index[: min(budget, len(frame))]] = True
    return calibration, ~calibration


def fit_recalibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 4:
        raise ValueError("recalibration needs at least 4 finite calibration samples")
    if float(np.ptp(y_pred[mask])) <= 1e-9:
        slope, intercept = 1.0, float(np.mean(y_true[mask] - y_pred[mask]))
    else:
        slope, intercept = np.polyfit(y_pred[mask], y_true[mask], 1)
    return {"slope": float(slope), "intercept": float(intercept), "samples": int(mask.sum())}


def apply_recalibration(y_pred: np.ndarray, recal: dict) -> np.ndarray:
    return y_pred * float(recal["slope"]) + float(recal["intercept"])
