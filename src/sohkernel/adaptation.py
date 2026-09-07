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
- ``residual`` — ridge on the target's feature values predicting the frozen
  model's residual, fitted on the same K-sample budget.  Feature-level, so
  it can restore a degradation trend that output recalibration cannot.
- ``monotone`` — per-cell isotonic projection onto non-increasing SOH.
  Label-free physical constraint; composes as ``recalibration+monotone``
  and ``residual+monotone`` (applied after the labeled correction).

Calibration rows are drawn per ``calibration_split``: ``earliest`` (first
cycles per cell) or ``spread`` (cycles spread over each cell's life — the
periodic-RPT-checkup scenario).

Every operator declares its information channel in ``OPERATOR_CHANNELS``;
the adaptation search reports it, and evaluation always runs on the
non-calibration subset so operators with and without label budgets stay
comparable — the calibration rows are held out for EVERY operator (also
label-free ones, which simply never read their labels), both in the search
and inside the audited inference job.

Conformal intervals are calibrated on leave-one-out residuals of the
labeled correction (closed form for the affine / ridge steps), not on the
in-sample residuals of a fit that just saw those rows — in-sample residuals
shrink with the number of correction parameters and under-cover.

Design doc: docs/superpowers/specs/2026-08-26-agent-guided-search-design.md §1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ADAPTATION_OPERATORS = (
    "none",
    "coral",
    "recalibration",
    "coral+recalibration",
    "residual",
    "monotone",
    "recalibration+monotone",
    "residual+monotone",
    "residual_wide",
    "residual_wide+monotone",
)

OPERATOR_CHANNELS = {
    "none": "frozen model only",
    "coral": "unlabeled target feature statistics + source training statistics",
    "recalibration": "K labeled target calibration samples (excluded from evaluation)",
    "coral+recalibration": (
        "unlabeled target feature statistics + source training statistics "
        "+ K labeled target calibration samples (excluded from evaluation)"
    ),
    "residual": (
        "K labeled target calibration samples + target feature values "
        "(ridge residual correction, calibration rows excluded from evaluation)"
    ),
    "monotone": (
        "none — output post-processing only: per-cell non-increasing SOH "
        "enforced (physical constraint, no target information)"
    ),
    "recalibration+monotone": (
        "K labeled target calibration samples (excluded from evaluation) "
        "+ per-cell monotone projection (physical constraint, label-free)"
    ),
    "residual+monotone": (
        "K labeled target calibration samples + target feature values "
        "(ridge residual correction, calibration rows excluded from evaluation) "
        "+ per-cell monotone projection (physical constraint, label-free)"
    ),
    "residual_wide": (
        "K labeled target calibration samples + ALL label-safe target feature "
        "columns (ridge residual correction over the extended feature set, "
        "calibration rows excluded from evaluation)"
    ),
    "residual_wide+monotone": (
        "K labeled target calibration samples + ALL label-safe target feature "
        "columns (ridge residual correction over the extended feature set, "
        "calibration rows excluded from evaluation) "
        "+ per-cell monotone projection (physical constraint, label-free)"
    ),
}

CALIBRATION_SPLITS = ("earliest", "spread")
# What each split simulates (Spec 01 §7): ``earliest`` is the deployment case (the first K
# reference cycles are all a fleet ever gets) and is the only split that can support a
# forward claim; ``spread`` is a periodic-checkup backtest and is always labelled as such.
CALIBRATION_SCENARIOS = {
    "earliest": "forward_deployment",
    "spread": "periodic_checkup_backtest",
}

INTERVAL_MODES = ("model", "conformal")

# What the calibration budget counts (Spec 12 §5, K13).  ``rows``: K rows chosen by the split
# strategy, round-robin across units.  ``capacity_tests``: K field capacity tests per unit — the
# labelled segments whose label comes from one of the unit's first K reference tests are calibration
# (a deployment that has run K tests), the segments labelled by later tests are evaluation.  A test
# budget is forward by construction and pairs only with the ``earliest`` strategy.
CALIBRATION_UNITS = ("rows", "capacity_tests")

_COV_RIDGE = 1e-6
# Fewer evaluable rows than this after the calibration split makes the
# reported metric meaningless (the search applies the same floor).
_MIN_EVALUATION_ROWS = 8


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


def _spread_order(n: int) -> np.ndarray:
    """Deterministic pick order covering [0, n) evenly as the budget grows:
    both endpoints first, then binary midpoints (0, n-1, mid, quarter, ...)."""
    if n <= 0:
        return np.zeros(0, dtype=int)
    if n <= 2:
        return np.arange(n, dtype=int)
    fractions = [0.0, 1.0]
    depth = 1
    while len(fractions) < n:
        step = 1.0 / (2 ** depth)
        fractions.extend(step * k for k in range(1, 2 ** depth, 2))
        depth += 1
    order: list[int] = []
    seen: set[int] = set()
    for fraction in fractions:
        index = int(round(fraction * (n - 1)))
        if index not in seen:
            seen.add(index)
            order.append(index)
        if len(order) == n:
            break
    return np.asarray(order, dtype=int)


def calibration_split(
    table: pd.DataFrame, budget: int, strategy: str = "earliest", *, unit: str = "rows"
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic budget split, round-robin across cells.

    ``earliest`` takes the first cycles of each cell (early reference tests
    are what a deployment gets first).  ``spread`` takes cycles spread over
    each cell's observed life (endpoints first, then binary midpoints) —
    the periodic-RPT-checkup scenario, which lets output/residual
    corrections see the degradation range instead of only the fresh plateau.
    ``unit="capacity_tests"`` counts the budget in field capacity tests per
    unit instead of rows (:func:`capacity_test_split`).
    """
    if strategy not in CALIBRATION_SPLITS:
        raise ValueError(f"unknown calibration split: {strategy}")
    if unit not in CALIBRATION_UNITS:
        raise ValueError(f"unknown calibration unit: {unit}")
    if unit == "capacity_tests":
        return capacity_test_split(table, budget, strategy)
    if budget <= 0:
        return np.zeros(len(table), dtype=bool), np.ones(len(table), dtype=bool)
    frame = table.reset_index(drop=True)
    ordered = frame.sort_values(["cell_id", "cycle_index"], kind="stable")
    if strategy == "earliest":
        rank_in_cell = ordered.groupby("cell_id", sort=False).cumcount()
    else:
        rank_in_cell = pd.Series(0, index=ordered.index, dtype=int)
        for _cell, indices in ordered.groupby("cell_id", sort=False).indices.items():
            positions = np.asarray(indices, dtype=int)
            ranks = np.empty(len(positions), dtype=int)
            ranks[_spread_order(len(positions))] = np.arange(len(positions))
            rank_in_cell.iloc[positions] = ranks
    priority = ordered.assign(_rank=rank_in_cell).sort_values(
        ["_rank", "cell_id", "cycle_index"], kind="stable"
    )
    calibration = np.zeros(len(frame), dtype=bool)
    calibration[priority.index[: min(budget, len(frame))]] = True
    return calibration, ~calibration


def _first_k_label_times(table: pd.DataFrame, tests: int) -> tuple[np.ndarray, dict[str, int]]:
    """Per row, whether its label comes from one of its unit's first K reference tests (distinct
    ``label_time_s``, ascending); per unit, the number of distinct label times."""
    cells = table["cell_id"].astype(str).to_numpy()
    label_time = pd.to_numeric(table["label_time_s"], errors="coerce").to_numpy(dtype=float)
    within = np.zeros(len(table), dtype=bool)
    counts: dict[str, int] = {}
    for cell in pd.unique(cells):
        rows = np.flatnonzero(cells == cell)
        times = np.unique(label_time[rows])
        times = times[np.isfinite(times)]
        counts[str(cell)] = int(len(times))
        if len(times) and tests > 0:
            within[rows] = np.isin(label_time[rows], times[:tests])
    return within, counts


def capacity_test_split(
    table: pd.DataFrame, tests: int, strategy: str = "earliest"
) -> tuple[np.ndarray, np.ndarray]:
    """Spec 12 §5 (K13): the calibration budget counted in field capacity tests per unit.

    The calibration rows of a unit are its labelled segments whose label comes from one of the
    unit's first K reference tests (the K smallest distinct ``label_time_s``); with forward-only
    labels these are the segments between the first test and the (K+1)-th, so every evaluation row
    of the unit starts after every calibration row — a forward cut by construction, which is why
    only the ``earliest`` strategy applies.  Segments before the first test (unlabelled) and segments
    labelled by later tests are the evaluation side.  A unit with fewer than K tests puts all its
    labelled rows in calibration and contributes no labelled evaluation row;
    :func:`capacity_test_summary` reports those units.
    """
    if strategy != "earliest":
        raise ValueError("calibration unit 'capacity_tests' is a forward cut: only the 'earliest' strategy applies")
    required = ("cell_id", "label_time_s")
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise ValueError(
            "calibration unit 'capacity_tests' needs a segment table with "
            f"{list(required)} (Spec 12 §3 labels); missing {missing}"
        )
    if tests <= 0:
        return np.zeros(len(table), dtype=bool), np.ones(len(table), dtype=bool)
    frame = table.reset_index(drop=True)
    calibration, _counts = _first_k_label_times(frame, int(tests))
    return calibration, ~calibration


def capacity_test_summary(table: pd.DataFrame, calibration_mask, tests: int) -> dict:
    """What a capacity-test budget did: units, units with fewer than K tests, and the labelled row
    counts on both sides — for the job record next to ``calibration_samples``."""
    frame = table.reset_index(drop=True)
    cal = np.asarray(calibration_mask, dtype=bool)
    _within, counts = _first_k_label_times(frame, int(tests))
    labeled = np.isfinite(pd.to_numeric(frame["soh"], errors="coerce").to_numpy(dtype=float)) if "soh" in frame.columns \
        else np.zeros(len(frame), dtype=bool)
    return {
        "calibration_unit": "capacity_tests",
        "calibration_tests": int(tests),
        "units": len(counts),
        "units_with_fewer_tests": int(sum(1 for n in counts.values() if n < int(tests))),
        "calibration_labeled": int((cal & labeled).sum()),
        "evaluation_labeled": int((~cal & labeled).sum()),
        "evaluation_unlabeled": int((~cal & ~labeled).sum()),
    }


# A frozen model that predicts a near-constant on the calibration set gives
# the affine fit almost no prediction variance to lean on, so the fitted
# slope explodes and the "correction" destroys the output scale.  Outside
# these bounds the honest correction is intercept-only.
_RECAL_SLOPE_BOUNDS = (0.1, 10.0)
# The slope is only identifiable when the calibration rows span real
# degradation AND prediction co-varies with truth.  On a fresh-plateau
# calibration set (truth ≈ 1.0 ± 0.01, the "earliest" split) the OLS slope
# is pure noise — it can come out negative and flip a correct downward
# trajectory into a rising one — so those fits are refused.
_RECAL_MIN_TRUTH_RANGE = 0.05  # SOH units
_RECAL_MIN_ABS_CORRELATION = 0.3


def fit_recalibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n < 4:
        raise ValueError("recalibration needs at least 4 finite calibration samples")
    yt, yp = y_true[mask], y_pred[mask]
    intercept_only = (1.0, float(np.mean(yt - yp)))
    truth_range = float(np.ptp(yt))
    correlation = None
    slope, intercept = intercept_only
    reason = None
    if float(np.ptp(yp)) <= 1e-9:
        reason = "constant_predictions"
    elif truth_range < _RECAL_MIN_TRUTH_RANGE:
        reason = "flat_calibration_truth"
    else:
        correlation = float(np.corrcoef(yp, yt)[0, 1])
        if not np.isfinite(correlation) or abs(correlation) < _RECAL_MIN_ABS_CORRELATION:
            reason = "weak_correlation"
        else:
            slope, intercept = np.polyfit(yp, yt, 1)
            low, high = _RECAL_SLOPE_BOUNDS
            if (
                not np.isfinite(slope)
                or not np.isfinite(intercept)
                or not (low <= abs(slope) <= high)
            ):
                slope, intercept = intercept_only
                reason = "slope_out_of_bounds"
    result = {
        "slope": float(slope),
        "intercept": float(intercept),
        "samples": n,
        "truth_range": truth_range,
        "correlation": correlation,
    }
    if reason:
        result["fallback"] = "intercept_only"
        result["fallback_reason"] = reason
    return result


def apply_recalibration(y_pred: np.ndarray, recal: dict) -> np.ndarray:
    return y_pred * float(recal["slope"]) + float(recal["intercept"])


def fit_residual(
    features: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    columns: list[str],
    *,
    alpha: float = 1.0,
) -> dict:
    """Ridge on standardized target features predicting the frozen model's
    residual (y - pred).  Unlike affine output recalibration this re-weights
    features, so it can restore a degradation trend the frozen model missed.
    """
    features = np.asarray(features, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Missing feature values are imputed with the calibration medians instead
    # of dropping the row: a partially-NaN wide column used to shrink the
    # calibration set (or empty it when a transform column was all-NaN on the
    # target) and gave those rows an intercept-only correction at apply time,
    # creating within-cell discontinuities.
    impute = _column_medians(features, np.isfinite(y_true) & np.isfinite(y_pred))
    features = _impute_columns(features, impute)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(features).all(axis=1)
    n = int(mask.sum())
    if n < 4:
        raise ValueError("residual correction needs at least 4 finite calibration samples")
    residual = y_true[mask] - y_pred[mask]
    model = {
        "columns": list(columns),
        "alpha": float(alpha),
        "samples": n,
        "intercept": float(residual.mean()),
        "impute": impute.tolist(),
    }
    if n < max(8, len(columns) + 2):
        model.update({"mean": None, "scale": None, "coef": None, "fallback": "intercept_only"})
        return model
    x = features[mask]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    z = (x - mean) / scale
    centered = residual - residual.mean()
    gram = z.T @ z + alpha * np.eye(z.shape[1])
    coef = np.linalg.solve(gram, z.T @ centered)
    model.update({"mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist()})
    return model


def fit_conformal_interval(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1
) -> dict:
    """Split-conformal band: the finite-sample-corrected (1-alpha) quantile
    of |y - pred| on the calibration samples.  Reuses the already-declared
    K-label channel; coverage is marginal and assumes the calibration rows
    are exchangeable with evaluation rows (the 'spread' split approximates
    this — measured PICP remains the arbiter).  When a labeled correction
    was fitted on the same rows, pass its leave-one-out predictions
    (``recalibration_loo_predictions`` / ``residual_loo_predictions``) so the
    residuals are not the shrunken in-sample ones."""
    residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    residuals = np.sort(residuals[np.isfinite(residuals)])
    n = len(residuals)
    if n < 10:
        raise ValueError("conformal interval needs at least 10 finite calibration samples")
    rank = min(n - 1, int(np.ceil((n + 1) * (1.0 - alpha))) - 1)
    return {"quantile": float(residuals[rank]), "samples": n, "alpha": float(alpha)}


def apply_conformal_interval(y_pred: np.ndarray, conformal: dict) -> tuple[np.ndarray, np.ndarray]:
    q = float(conformal["quantile"])
    pred = np.asarray(y_pred, dtype=float)
    return pred - q, pred + q


def wide_residual_columns(table: pd.DataFrame, model_columns: list[str]) -> list[str]:
    """Extended residual inputs: every numeric, label-safe column of the
    prepared target table (structural label-source ban enforced — a column
    like charge_capacity_max would hand the correction the answer)."""
    from .feature_safety import is_label_source_feature

    usable = []
    for column in table.columns:
        if column == "soh" or is_label_source_feature(column):
            continue
        if not pd.api.types.is_numeric_dtype(table[column]):
            continue
        # A transform column the target cannot derive is materialized as
        # all-NaN; it carries no information and would only be imputed to a
        # constant, so leave it out of the residual inputs entirely.
        if not np.isfinite(pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)).any():
            continue
        usable.append(column)
    usable_set = set(usable)
    head = [c for c in model_columns if c in usable_set]
    tail = sorted(usable_set - set(head))
    return head + tail


def apply_monotone(table: pd.DataFrame, y_pred: np.ndarray) -> tuple[np.ndarray, dict]:
    """Project each cell's prediction sequence onto the non-increasing cone
    (isotonic regression over cycle order).  SOH cannot recover, so this is a
    label-free physical constraint, not a data channel."""
    from sklearn.isotonic import IsotonicRegression

    smoothed = np.asarray(y_pred, dtype=float).copy()
    cells = table["cell_id"].astype(str).to_numpy()
    cycles = pd.to_numeric(table["cycle_index"], errors="coerce").to_numpy(dtype=float)
    n_cells = 0
    for cell in pd.unique(cells):
        rows = np.flatnonzero(cells == cell)
        order = rows[np.argsort(cycles[rows], kind="stable")]
        values = smoothed[order]
        finite = np.isfinite(values)
        if int(finite.sum()) < 2:
            continue
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        values[finite] = iso.fit_transform(np.flatnonzero(finite), values[finite])
        smoothed[order] = values
        n_cells += 1
    delta = smoothed - np.asarray(y_pred, dtype=float)
    finite_delta = delta[np.isfinite(delta)]
    info = {
        "cells": n_cells,
        "adjusted_fraction": float(np.mean(np.abs(finite_delta) > 1e-12)) if finite_delta.size else 0.0,
        "mean_abs_adjustment": float(np.mean(np.abs(finite_delta))) if finite_delta.size else 0.0,
    }
    return smoothed, info


def _column_medians(features: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] == 0:
        return np.zeros(features.shape[1] if features.ndim == 2 else 0)
    subset = features[rows] if rows.any() else features
    with np.errstate(all="ignore"):
        medians = np.nanmedian(subset, axis=0) if len(subset) else np.full(features.shape[1], np.nan)
    return np.where(np.isfinite(medians), medians, 0.0)


def _impute_columns(features: np.ndarray, impute) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if impute is None or features.ndim != 2:
        return features
    fill = np.asarray(impute, dtype=float)
    if fill.shape[0] != features.shape[1]:
        return features
    missing = ~np.isfinite(features)
    if not missing.any():
        return features
    filled = features.copy()
    filled[missing] = np.broadcast_to(fill, features.shape)[missing]
    return filled


def apply_residual(features: np.ndarray, y_pred: np.ndarray, model: dict) -> np.ndarray:
    features = _impute_columns(np.asarray(features, dtype=float), model.get("impute"))
    correction = np.full(len(y_pred), float(model["intercept"]))
    if model.get("coef") is not None:
        mean = np.asarray(model["mean"], dtype=float)
        scale = np.asarray(model["scale"], dtype=float)
        coef = np.asarray(model["coef"], dtype=float)
        finite = np.isfinite(features).all(axis=1)
        z = (features[finite] - mean) / scale
        correction[finite] = correction[finite] + z @ coef
    return y_pred + correction


def _loo_predictions(y_true: np.ndarray, fitted: np.ndarray, hat_diag: np.ndarray) -> np.ndarray:
    """Leave-one-out predictions of a linear smoother from its in-sample fit:
    e_i^(-i) = e_i / (1 - h_ii), so pred_i^(-i) = y_i - e_i^(-i)."""
    residual = y_true - fitted
    scale = 1.0 - np.clip(hat_diag, 0.0, 1.0 - 1e-6)
    return y_true - residual / scale


def recalibration_loo_predictions(y_true: np.ndarray, y_pred: np.ndarray, recal: dict) -> np.ndarray:
    """Corrected predictions on the calibration rows as if each row had been
    left out of the affine fit (exact for OLS; intercept-only: h_ii = 1/n)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fitted = apply_recalibration(y_pred, recal)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    out = fitted.copy()
    if n < 2:
        return out
    if recal.get("fallback") == "intercept_only":
        hat = np.full(n, 1.0 / n)
    else:
        x = y_pred[mask]
        centered = x - x.mean()
        sum_sq = float(centered @ centered)
        hat = 1.0 / n + (centered ** 2 / sum_sq if sum_sq > 0 else 0.0)
    out[mask] = _loo_predictions(y_true[mask], fitted[mask], hat)
    return out


def residual_loo_predictions(
    features: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, model: dict
) -> np.ndarray:
    """Residual-corrected predictions on the calibration rows with each row
    left out of the ridge fit (hat matrix 11'/n + Z (Z'Z + αI)^-1 Z' with the
    standardization treated as fixed — the usual ridge LOO shortcut)."""
    features = _impute_columns(np.asarray(features, dtype=float), model.get("impute"))
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    corrected = apply_residual(features, y_pred, model)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(features).all(axis=1)
    n = int(mask.sum())
    out = corrected.copy()
    if n < 2:
        return out
    if model.get("coef") is None:
        hat = np.full(n, 1.0 / n)
    else:
        mean = np.asarray(model["mean"], dtype=float)
        scale = np.asarray(model["scale"], dtype=float)
        z = (features[mask] - mean) / scale
        gram = z.T @ z + float(model["alpha"]) * np.eye(z.shape[1])
        solved = np.linalg.solve(gram, z.T)  # (p, n)
        hat = 1.0 / n + np.einsum("ij,ji->i", z, solved)
    out[mask] = _loo_predictions(y_true[mask], corrected[mask], hat)
    return out


# ---------------------------------------------------------------------------
# The operator seam (Spec 12 §9, K13): fit on the calibration rows, apply anywhere
# ---------------------------------------------------------------------------

INTERVAL_SOURCES = ("model", "model_affine", "model_shifted", "calibration_conformal")


def _strip_step(operator: str, step: str) -> str:
    parts = [p for p in operator.split("+") if p != step]
    return "+".join(parts) if parts else "none"


@dataclass
class FittedOperator:
    """A deployment-legal operator fitted on one unit's calibration rows (the Spec 12 §9 seam).

    Everything :func:`apply_operator` needs to reproduce the fit — on the calibration rows, on the
    unit's evaluation rows, or on new query segments at serving: the CORAL map, the labelled
    correction, whether the monotone projection runs, the conformal band, and the audit trail
    (``applied`` / ``refused``).  :meth:`record` is the JSON-safe summary for a job record or an
    estimate response.  ``interval_source`` says where the served band comes from: the frozen
    model's own band (``model``), that band mapped through the affine recalibration
    (``model_affine``), shifted by the realised per-row correction (``model_shifted``), or the
    split-conformal band of the calibration residuals (``calibration_conformal``).
    """

    operator: str
    interval: str
    columns: list[str]
    correction_columns: list[str]
    coral: dict | None = None
    correction: dict | None = None
    monotone: bool = False
    monotone_info: dict | None = None
    conformal: dict | None = None
    calibration_rows: int = 0
    applied: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    @property
    def channel(self) -> str:
        channel = OPERATOR_CHANNELS[self.operator]
        if self.interval == "conformal":
            channel += "; interval: split-conformal quantile from the same K calibration residuals"
        return channel

    @property
    def verdict(self) -> str:
        if self.operator == "none" and self.interval != "conformal":
            return "not_applicable"
        if self.applied and not self.refused:
            return "applied"
        return "partial" if self.applied else "refused"

    @property
    def interval_source(self) -> str:
        if self.conformal is not None:
            return "calibration_conformal"
        if self.correction is not None and "residual" not in self.operator and not self.monotone:
            return "model_affine"
        if self.correction is not None or self.monotone:
            return "model_shifted"
        return "model"

    def record(self) -> dict:
        def trim(value: dict) -> dict:
            return {k: v for k, v in value.items() if not isinstance(v, (list, np.ndarray)) or len(v) <= 12}

        fitted: dict = {}
        if self.coral is not None:
            fitted["coral"] = {"columns": list(self.coral["columns"])}
        if self.correction is not None:
            fitted["residual" if "residual" in self.operator else "recalibration"] = trim(self.correction)
        if self.monotone_info is not None:
            fitted["monotone"] = dict(self.monotone_info)
        if self.conformal is not None:
            fitted["conformal"] = dict(self.conformal)
        return {
            "operator": self.operator, "channel": self.channel, "interval": self.interval,
            "calibration_rows": int(self.calibration_rows), "applied": list(self.applied),
            "refused": list(self.refused), "verdict": self.verdict, "interval_source": self.interval_source,
            "fitted": fitted,
        }


def _guard(fitted: FittedOperator, strict: bool, message: str) -> None:
    if strict:
        raise ValueError(message)
    fitted.refused.append(message)


def fit_operator(
    operator: str,
    target: pd.DataFrame,
    calibration_mask,
    predict,
    *,
    columns: list[str],
    interval: str = "model",
    source_features: np.ndarray | None = None,
    correction_columns: list[str] | None = None,
    strict: bool = True,
    skip_coral: bool = False,
) -> FittedOperator:
    """Fit ``operator`` around a frozen model on one target table — the Spec 12 §9 seam shared by the
    evaluation ladder (:class:`InferenceAdapter`) and serving.

    ``target`` holds every row of the unit: the calibration rows (``calibration_mask``) and the rows
    the fit will be applied to (evaluation rows, or query segments at serving); ``cell_id`` and
    ``cycle_index`` order the monotone projection, ``soh`` is read on the calibration rows only.
    ``predict(rows) -> ndarray`` is the frozen model's prediction on rows carrying its feature
    ``columns`` — the kernel never holds the model, the caller binds estimator and meta.
    ``source_features`` is the training feature sample CORAL aligns to (its unlabeled channel);
    ``correction_columns`` are the residual step's inputs (default: the model columns;
    :func:`wide_residual_columns` for ``residual_wide``).  ``skip_coral`` tells the seam that the
    caller has already aligned ``target`` (the adapter's ``prepare`` step).

    ``strict=True`` raises the kernel guards (the ladder's rule: an operator that cannot be fitted is
    not a rung); ``strict=False`` records them in ``refused`` and runs the remaining steps — serving
    reports the refusal and never works around it (Spec 12 §9).
    """
    if operator not in ADAPTATION_OPERATORS:
        raise ValueError(f"unknown adaptation operator: {operator}")
    if interval not in INTERVAL_MODES:
        raise ValueError(f"unknown interval mode: {interval}")
    frame = target.reset_index(drop=True)
    cal = np.asarray(calibration_mask, dtype=bool).reshape(-1)
    if len(cal) != len(frame):
        raise ValueError("calibration_mask must have one entry per target row")
    columns = [c for c in columns if c in frame.columns]
    fitted = FittedOperator(
        operator=operator, interval=interval, columns=list(columns),
        correction_columns=list(correction_columns) if correction_columns is not None else list(columns),
        calibration_rows=int(cal.sum()),
    )
    adapted = frame
    if "coral" in operator and not skip_coral:
        try:
            if source_features is None:
                raise ValueError("CORAL needs the training feature sample (source_features) to align to")
            fitted.coral = fit_coral(np.asarray(source_features, dtype=float), _feature_matrix(frame, columns), columns)
            adapted = apply_coral(frame, fitted.coral)
            fitted.applied.append("coral")
        except ValueError as exc:
            _guard(fitted, strict, f"coral: {exc}")
    pred = np.asarray(predict(adapted), dtype=float).reshape(-1)
    if len(pred) != len(frame):
        raise ValueError("predict must return one value per target row")
    truth = (
        pd.to_numeric(frame["soh"], errors="coerce").to_numpy(dtype=float)
        if "soh" in frame.columns else np.full(len(frame), np.nan)
    )
    corrected = pred.copy()
    loo = None
    needs_labels = "residual" in operator or "recalibration" in operator
    if needs_labels and fitted.calibration_rows == 0:
        _guard(fitted, strict, f"{operator}: needs calibration rows with labels; none were passed")
    elif "residual" in operator:
        try:
            features = _feature_matrix(adapted, fitted.correction_columns)
            model = fit_residual(features[cal], corrected[cal], truth[cal], fitted.correction_columns)
            loo = residual_loo_predictions(features[cal], corrected[cal], truth[cal], model)
            corrected = apply_residual(features, corrected, model)
            fitted.correction = model
            fitted.applied.append("residual")
        except ValueError as exc:
            _guard(fitted, strict, f"residual: {exc}")
    elif "recalibration" in operator:
        try:
            recal = fit_recalibration(truth[cal], corrected[cal])
            loo = recalibration_loo_predictions(truth[cal], corrected[cal], recal)
            corrected = apply_recalibration(corrected, recal)
            fitted.correction = recal
            fitted.applied.append("recalibration")
        except ValueError as exc:
            _guard(fitted, strict, f"recalibration: {exc}")
    before_monotone = corrected
    if "monotone" in operator:
        corrected, info = apply_monotone(adapted, corrected)
        fitted.monotone = True
        fitted.monotone_info = info
        fitted.applied.append("monotone")
    if interval == "conformal":
        try:
            if fitted.calibration_rows == 0:
                raise ValueError("conformal interval needs calibration rows with labels")
            # Carry the (label-free) monotone adjustment realised on each calibration row onto its
            # leave-one-out prediction; without a labelled step the in-sample residuals are the band.
            calibration_pred = corrected[cal] if loo is None else loo + (corrected[cal] - before_monotone[cal])
            conformal = fit_conformal_interval(truth[cal], calibration_pred)
            conformal["residual_source"] = "in_sample" if loo is None else "leave_one_out"
            fitted.conformal = conformal
            fitted.applied.append("conformal")
        except ValueError as exc:
            _guard(fitted, strict, f"conformal: {exc}")
    return fitted


def _operator_interval(fitted: FittedOperator, bounds, corrected: np.ndarray, correction: np.ndarray):
    """The served band for ``corrected`` predictions per ``fitted.interval_source``."""
    if fitted.conformal is not None:
        return apply_conformal_interval(corrected, fitted.conformal)
    if bounds is None:
        return None
    lower = np.asarray(bounds[0], dtype=float).reshape(-1)
    upper = np.asarray(bounds[1], dtype=float).reshape(-1)
    source = fitted.interval_source
    if source == "model_affine":
        lower = apply_recalibration(lower, fitted.correction)
        upper = apply_recalibration(upper, fitted.correction)
        if float(fitted.correction.get("slope", 1.0)) < 0:
            lower, upper = upper, lower
    elif source == "model_shifted":
        lower, upper = lower + correction, upper + correction
    return lower, upper


def apply_operator(
    fitted: FittedOperator, rows: pd.DataFrame, predict, predict_interval=None
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None, dict]:
    """Apply a fitted operator to ``rows`` — the unit's evaluation rows, the calibration rows
    themselves, or new query segments at serving: CORAL on the feature columns, the frozen model
    through ``predict`` (and ``predict_interval(rows) -> (lower, upper)`` for its own band), the
    labelled correction, the monotone projection over each unit's ``cycle_index`` order, and the band
    per ``fitted.interval_source``.  Returns ``(prediction, (lower, upper) or None, record)``; the
    record is :meth:`FittedOperator.record` plus the row count and the mean absolute correction.
    """
    frame = rows.reset_index(drop=True)
    adapted = apply_coral(frame, fitted.coral) if fitted.coral is not None else frame
    pred = np.asarray(predict(adapted), dtype=float).reshape(-1)
    if len(pred) != len(frame):
        raise ValueError("predict must return one value per row")
    bounds = predict_interval(adapted) if predict_interval is not None else None
    corrected = pred.copy()
    if fitted.correction is not None:
        if "residual" in fitted.operator:
            corrected = apply_residual(_feature_matrix(adapted, fitted.correction_columns), corrected, fitted.correction)
        else:
            corrected = apply_recalibration(corrected, fitted.correction)
    if fitted.monotone:
        corrected, _info = apply_monotone(adapted, corrected)
    correction = corrected - pred
    served = _operator_interval(fitted, bounds, corrected, correction)
    finite = np.isfinite(correction)
    record = {
        **fitted.record(), "rows": int(len(frame)),
        "correction_mean_abs": float(np.mean(np.abs(correction[finite]))) if finite.any() else 0.0,
    }
    if served is None and fitted.interval_source != "calibration_conformal":
        record["interval_source"] = "none"
    return corrected, served, record


class InferenceAdapter:
    """Applies a deployment-legal operator inside the audited inference run.

    Constructed by the service when an inference job carries an
    ``adaptation`` spec; the model artifact stays frozen — CORAL rewrites
    the target feature columns before predict, recalibration adjusts the
    outputs using a K-sample calibration budget whose rows are excluded
    from every metric.  The operator, channel, fitted parameters, and
    calibration count all land in the job record and prediction evidence.
    """

    def __init__(
        self,
        service,
        operator: str,
        calibration_budget: int = 32,
        calibration_split: str = "earliest",
        interval: str = "model",
        calibration_unit: str = "rows",
    ):
        if operator not in ADAPTATION_OPERATORS:
            raise ValueError(f"unknown adaptation operator: {operator}")
        if calibration_split not in CALIBRATION_SPLITS:
            raise ValueError(f"unknown calibration split: {calibration_split}")
        if interval not in INTERVAL_MODES:
            raise ValueError(f"unknown interval mode: {interval}")
        if calibration_unit not in CALIBRATION_UNITS:
            raise ValueError(f"unknown calibration unit: {calibration_unit}")
        if calibration_unit == "capacity_tests" and calibration_split != "earliest":
            raise ValueError("calibration unit 'capacity_tests' is a forward cut: only the 'earliest' split applies")
        self.service = service
        self.operator = operator
        self.calibration_budget = max(0, int(calibration_budget))
        self.calibration_split = calibration_split
        self.calibration_unit = calibration_unit
        self.interval = interval
        self._source_cache: dict[str, np.ndarray] = {}
        self._model_columns: list[str] = []
        self._feature_columns: list[str] = []
        self._fitted: FittedOperator | None = None
        self._last_correction: np.ndarray | None = None
        self._last_pred: np.ndarray | None = None
        self._conformal: dict | None = None

    @property
    def _uses_labeled_budget(self) -> bool:
        return (
            "recalibration" in self.operator
            or "residual" in self.operator
            or self.interval == "conformal"
        )

    @property
    def uses_recalibration(self) -> bool:
        """True when a post-predict output step runs (labeled correction
        and/or the label-free monotone projection)."""
        return self._uses_labeled_budget or "monotone" in self.operator

    def prepare(self, loaded, binding: dict, group: pd.DataFrame):
        """Return (adapted_group, calibration_mask, adaptation_info)."""
        channel = OPERATOR_CHANNELS[self.operator]
        if self.interval == "conformal":
            channel += (
                "; interval: split-conformal quantile from the same K "
                "calibration residuals"
            )
        info: dict = {
            "operator": self.operator,
            "channel": channel,
            "calibration_budget": self.calibration_budget,
        }
        if self.interval == "conformal":
            info["interval"] = "conformal"
        adapted = group
        model_columns = [c for c in loaded.feature_columns() if c in group.columns]
        self._model_columns = model_columns
        self._feature_columns = (
            wide_residual_columns(group, model_columns)
            if "residual_wide" in self.operator
            else model_columns
        )
        if "coral" in self.operator:
            columns = model_columns
            source = self._source_features(loaded, binding, columns)
            coral = fit_coral(source, _feature_matrix(group, columns), columns)
            adapted = apply_coral(group, coral)
            info["coral_columns"] = columns
        # The calibration subset is held out for EVERY operator so that a
        # frozen / label-free run and its labeled counterpart are scored on
        # the identical evaluation rows (the search's shared-subset rule).
        budget = self.calibration_budget
        calibration_mask, evaluation_mask = calibration_split(
            adapted, budget, self.calibration_split, unit=self.calibration_unit
        )
        if budget > 0 and int(evaluation_mask.sum()) < _MIN_EVALUATION_ROWS:
            raise ValueError(
                f"calibration budget {budget} leaves {int(evaluation_mask.sum())} "
                f"evaluation rows on {len(adapted)} target samples "
                f"(need at least {_MIN_EVALUATION_ROWS})"
            )
        info["calibration_samples"] = int(calibration_mask.sum())
        if budget > 0:
            info["calibration_split"] = self.calibration_split
            info["calibration_unit"] = self.calibration_unit
            if self.calibration_unit == "capacity_tests":
                info.update(capacity_test_summary(adapted, calibration_mask, budget))
            info["calibration_scenario"] = CALIBRATION_SCENARIOS[self.calibration_split]
            info["supports_forward_claim"] = self.calibration_split == "earliest"
            if not self._uses_labeled_budget:
                info["evaluation_holdout"] = (
                    "calibration rows held out from evaluation for cross-operator "
                    "comparability (labels unused by this operator)"
                )
        return adapted, calibration_mask, info

    def recalibrate(self, group: pd.DataFrame, pred: np.ndarray, calibration_mask) -> tuple[np.ndarray, dict]:
        """The labelled correction, the monotone projection and the conformal band through the
        seam (:func:`fit_operator`, strict: a guard that trips is the rung's error); CORAL was
        applied in :meth:`prepare`.  Returns the corrected predictions and the fitted parameters."""
        raw = np.asarray(pred, dtype=float)
        fitted = fit_operator(
            self.operator, group, calibration_mask, lambda _rows: raw,
            columns=self._model_columns, interval=self.interval,
            correction_columns=self._feature_columns, strict=True, skip_coral=True,
        )
        corrected, _bounds, _record = apply_operator(fitted, group, lambda _rows: raw)
        result: dict = dict(fitted.correction) if fitted.correction is not None else {}
        if fitted.monotone_info is not None:
            result["monotone"] = dict(fitted.monotone_info)
        if fitted.conformal is not None:
            result["conformal"] = dict(fitted.conformal)
        self._fitted = fitted
        self._last_correction = corrected - raw
        self._last_pred = corrected
        self._conformal = fitted.conformal
        return corrected, result

    def recalibrate_interval(self, interval, recal: dict):
        fitted = self._fitted
        if fitted is None or self._last_pred is None:
            return interval
        if self.interval == "conformal" and fitted.conformal is None:
            return interval
        correction = self._last_correction if self._last_correction is not None else np.zeros_like(self._last_pred)
        return _operator_interval(fitted, interval, self._last_pred, correction)

    def _source_features(self, loaded, binding: dict, columns: list[str]) -> np.ndarray:
        from .data import DatasetLoader
        from .transforms import inference_transform_featuresets as _inference_transform_featuresets

        cache_key = str(binding.get("artifact_id") or loaded.name)
        if cache_key in self._source_cache:
            return self._source_cache[cache_key]
        trained_on = [
            str(name) for name in (loaded.manifest.get("trained_on") or []) if str(name)
        ]
        if not trained_on:
            raise ValueError(f"model {loaded.name!r} declares no training datasets for CORAL")
        loader = DatasetLoader(self.service.data_dir)
        names = loader.resolve_datasets(trained_on)
        table, meta, _qc = loader.load(names)
        pseudo_binding = {
            "featureset": str(loaded.manifest.get("featureset") or binding.get("featureset") or "full"),
            "training_featuresets": list(
                loaded.manifest.get("training_featuresets")
                or binding.get("training_featuresets")
                or []
            ),
            "feature_columns": columns,
        }
        transforms = _inference_transform_featuresets([pseudo_binding])
        table = self.service.store.apply_feature_transforms(table, meta, transforms)
        features = _feature_matrix(table, columns)
        self._source_cache[cache_key] = features
        return features
