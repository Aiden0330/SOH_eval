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
comparable.

Design doc: docs/superpowers/specs/2026-08-26-agent-guided-search-design.md §1.
"""

from __future__ import annotations

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

INTERVAL_MODES = ("model", "conformal")

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
    table: pd.DataFrame, budget: int, strategy: str = "earliest"
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic budget split, round-robin across cells.

    ``earliest`` takes the first cycles of each cell (early reference tests
    are what a deployment gets first).  ``spread`` takes cycles spread over
    each cell's observed life (endpoints first, then binary midpoints) —
    the periodic-RPT-checkup scenario, which lets output/residual
    corrections see the degradation range instead of only the fresh plateau.
    """
    if strategy not in CALIBRATION_SPLITS:
        raise ValueError(f"unknown calibration split: {strategy}")
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


# A frozen model that predicts a near-constant on the calibration set gives
# the affine fit almost no prediction variance to lean on, so the fitted
# slope explodes and the "correction" destroys the output scale.  Outside
# these bounds the honest correction is intercept-only.
_RECAL_SLOPE_BOUNDS = (0.1, 10.0)


def fit_recalibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 4:
        raise ValueError("recalibration needs at least 4 finite calibration samples")
    fallback = None
    if float(np.ptp(y_pred[mask])) <= 1e-9:
        slope, intercept = 1.0, float(np.mean(y_true[mask] - y_pred[mask]))
        fallback = "intercept_only"
    else:
        slope, intercept = np.polyfit(y_pred[mask], y_true[mask], 1)
        low, high = _RECAL_SLOPE_BOUNDS
        if not np.isfinite(slope) or not np.isfinite(intercept) or not (low <= abs(slope) <= high):
            slope, intercept = 1.0, float(np.mean(y_true[mask] - y_pred[mask]))
            fallback = "intercept_only"
    result = {"slope": float(slope), "intercept": float(intercept), "samples": int(mask.sum())}
    if fallback:
        result["fallback"] = fallback
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
    this — measured PICP remains the arbiter)."""
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


def apply_residual(features: np.ndarray, y_pred: np.ndarray, model: dict) -> np.ndarray:
    correction = np.full(len(y_pred), float(model["intercept"]))
    if model.get("coef") is not None:
        mean = np.asarray(model["mean"], dtype=float)
        scale = np.asarray(model["scale"], dtype=float)
        coef = np.asarray(model["coef"], dtype=float)
        finite = np.isfinite(features).all(axis=1)
        z = (features[finite] - mean) / scale
        correction[finite] = correction[finite] + z @ coef
    return y_pred + correction


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
    ):
        if operator not in ADAPTATION_OPERATORS:
            raise ValueError(f"unknown adaptation operator: {operator}")
        if calibration_split not in CALIBRATION_SPLITS:
            raise ValueError(f"unknown calibration split: {calibration_split}")
        if interval not in INTERVAL_MODES:
            raise ValueError(f"unknown interval mode: {interval}")
        self.service = service
        self.operator = operator
        self.calibration_budget = max(0, int(calibration_budget))
        self.calibration_split = calibration_split
        self.interval = interval
        self._source_cache: dict[str, np.ndarray] = {}
        self._feature_columns: list[str] = []
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
            "calibration_budget": self.calibration_budget if self._uses_labeled_budget else 0,
        }
        if self.interval == "conformal":
            info["interval"] = "conformal"
        adapted = group
        model_columns = [c for c in loaded.feature_columns() if c in group.columns]
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
        budget = self.calibration_budget if self._uses_labeled_budget else 0
        calibration_mask, _evaluation = calibration_split(
            adapted, budget, self.calibration_split
        )
        info["calibration_samples"] = int(calibration_mask.sum())
        if budget > 0:
            info["calibration_split"] = self.calibration_split
        return adapted, calibration_mask, info

    def recalibrate(self, group: pd.DataFrame, pred: np.ndarray, calibration_mask) -> tuple[np.ndarray, dict]:
        raw = np.asarray(pred, dtype=float)
        corrected = raw
        result: dict = {}
        self._last_correction = None
        if "residual" in self.operator:
            truth = pd.to_numeric(group["soh"], errors="coerce").to_numpy(dtype=float)
            features = _feature_matrix(group, self._feature_columns)
            model = fit_residual(
                features[calibration_mask],
                corrected[calibration_mask],
                truth[calibration_mask],
                self._feature_columns,
            )
            corrected = apply_residual(features, corrected, model)
            result = dict(model)
        elif "recalibration" in self.operator:
            truth = pd.to_numeric(group["soh"], errors="coerce").to_numpy(dtype=float)
            recal = fit_recalibration(truth[calibration_mask], corrected[calibration_mask])
            corrected = apply_recalibration(corrected, recal)
            result = dict(recal)
        if "monotone" in self.operator:
            corrected, monotone_info = apply_monotone(group, corrected)
            result["monotone"] = monotone_info
        self._last_correction = corrected - raw
        self._last_pred = corrected
        self._conformal = None
        if self.interval == "conformal":
            truth = pd.to_numeric(group["soh"], errors="coerce").to_numpy(dtype=float)
            self._conformal = fit_conformal_interval(
                truth[calibration_mask], corrected[calibration_mask]
            )
            result["conformal"] = dict(self._conformal)
        return corrected, result

    def recalibrate_interval(self, interval, recal: dict):
        if self.interval == "conformal":
            # Pure conformal band around the final corrected prediction —
            # replaces (or creates, for models without native intervals)
            # the interval entirely.
            if self._conformal is None or self._last_pred is None:
                return interval
            return apply_conformal_interval(self._last_pred, self._conformal)
        if interval is None:
            return None
        if "residual" in self.operator or "monotone" in self.operator:
            # Feature-level / projection corrections have no closed affine
            # form; shift the bounds by the realized per-row correction
            # (width preserved).
            if self._last_correction is None:
                return interval
            return interval[0] + self._last_correction, interval[1] + self._last_correction
        lower = apply_recalibration(interval[0], recal)
        upper = apply_recalibration(interval[1], recal)
        if float(recal.get("slope", 1.0)) < 0:
            lower, upper = upper, lower
        return lower, upper

    def _source_features(self, loaded, binding: dict, columns: list[str]) -> np.ndarray:
        from .data import DatasetLoader
        from .evaluation_service import _inference_transform_featuresets

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
