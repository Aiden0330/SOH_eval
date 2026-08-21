from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd


PROTECTED_TARGET = "soh"
LABEL_SAFETY_CONTRACT = "sohbench.label-safety.v1"

_DIRECT_LABEL_NAMES = {
    "label",
    "target",
    "true_soh",
    "truth_soh",
    "ground_truth",
    "ground_truth_soh",
    "y",
    "y_true",
}
_DIRECT_LABEL_PATTERNS = (
    re.compile(r"(^|_)fade_from_full($|_)"),
    re.compile(r"(^|_)soh_label($|_)"),
    re.compile(r"(^|_)target_soh($|_)"),
)


class FeatureSafetyError(ValueError):
    """Raised when a feature pipeline can observe or reproduce the label."""


# Quantities the label is computed from (``soh = calibration capacity /
# rated capacity``) plus columns derived from them.  They are structurally
# banned as model inputs on EVERY dataset — even where the per-dataset affine
# scan would not flag them — because they encode the labelling procedure, not
# battery physics.  The affine/value audits stay in place as the dynamic
# backstop for leaks that arrive under other names.
LABEL_SOURCE_FEATURE_COLUMNS = frozenset(
    {
        "charge_capacity_max",
        "capacity_ah",
        "rated_capacity",
        "rated_capacity_ah",
        # derived from charge_capacity_max in the mechanism-proxy transform
        "charge_energy_proxy_wh",
        "cumulative_capacity_proxy_ah",
    }
)


def is_label_source_feature(name: object, *, target: str = PROTECTED_TARGET) -> bool:
    """True when *name* is a banned model input: a label-source capacity
    column, a column derived from one, or a direct alias of the label."""
    lowered = str(name or "").strip().lower()
    if lowered in LABEL_SOURCE_FEATURE_COLUMNS:
        return True
    return bool(suspicious_feature_names([lowered], target=target))


_AFFINE_RECONSTRUCTION_R2 = 0.995
_AFFINE_MIN_SAMPLES = 64


def _affine_reconstruction(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Fit ``y ~ a*x + b`` and return ``(r2, a)``, or None when undecidable.

    A single feature that reconstructs the label affinely to R² >= 0.995 is
    not "highly correlated physics" — it is the quantity the label was
    computed from (e.g. ``soh = capacity / rated_capacity``).  Legitimate
    physical signals in this repo top out around R² ~0.98 per dataset.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < _AFFINE_MIN_SAMPLES:
        return None
    xs = x[mask]
    ys = y[mask]
    if float(np.ptp(xs)) <= 0.0 or float(np.ptp(ys)) <= 0.0:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    residual = ys - (slope * xs + intercept)
    denominator = float(np.sum((ys - float(np.mean(ys))) ** 2))
    if denominator <= 0.0:
        return None
    r2 = 1.0 - float(np.sum(residual ** 2)) / denominator
    return r2, float(slope)


def suspicious_feature_names(
    columns: Iterable[object],
    *,
    target: str = PROTECTED_TARGET,
) -> list[str]:
    """Return feature names that explicitly encode the protected label.

    This deliberately uses a narrow deny-list.  Names such as
    ``battery_soh_cycle_norm`` describe a domain feature and are not rejected;
    direct truth aliases and the known ``fade_from_full = 1 - soh`` leakage
    convention are rejected.
    """
    target_name = str(target or PROTECTED_TARGET).strip().lower()
    output: list[str] = []
    for raw in columns:
        name = str(raw or "").strip()
        lowered = name.lower()
        if (
            lowered == target_name
            or lowered in _DIRECT_LABEL_NAMES
            or any(pattern.search(lowered) for pattern in _DIRECT_LABEL_PATTERNS)
        ):
            output.append(name)
    return sorted(set(output))


def audit_feature_frame(
    table: pd.DataFrame,
    columns: Iterable[object],
    *,
    target: str = PROTECTED_TARGET,
) -> dict[str, Any]:
    """Audit selected columns for direct or exact label reconstruction.

    The value checks intentionally cover only exact identity/complement
    encodings.  They catch accidental copies and ``1 - soh`` without treating
    a legitimately predictive physical signal as leakage merely because it is
    highly correlated with SOH.
    """
    selected = [str(column) for column in columns if str(column)]
    violations: list[dict[str, Any]] = [
        {
            "column": column,
            "kind": "protected_label_name",
            "reason": f"feature column {column!r} names or aliases the protected label",
        }
        for column in suspicious_feature_names(selected, target=target)
    ]
    if target in table.columns:
        truth = pd.to_numeric(table[target], errors="coerce").to_numpy(dtype=float)
        for column in selected:
            if column not in table.columns or any(item["column"] == column for item in violations):
                continue
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values) & np.isfinite(truth)
            if int(finite.sum()) < 8:
                continue
            x = values[finite]
            y = truth[finite]
            if np.max(np.abs(x - y)) <= 1e-12:
                violations.append(
                    {
                        "column": column,
                        "kind": "exact_label_copy",
                        "reason": f"feature column {column!r} exactly reproduces {target}",
                    }
                )
            elif np.max(np.abs((x + y) - 1.0)) <= 1e-12:
                violations.append(
                    {
                        "column": column,
                        "kind": "exact_label_complement",
                        "reason": f"feature column {column!r} exactly reproduces 1 - {target}",
                    }
                )
        # Affine reconstruction: catches scaled/shifted label copies such as
        # the calibration capacity the SOH label is derived from.  Checked
        # per dataset (rated capacity differs across datasets) and pooled.
        groups: list[tuple[str, np.ndarray]] = [("all", np.ones(len(table), dtype=bool))]
        if "dataset" in table.columns:
            dataset_values = table["dataset"].astype(str).to_numpy()
            groups.extend(
                (name, dataset_values == name) for name in pd.unique(dataset_values)
            )
        for column in selected:
            if column not in table.columns or any(item["column"] == column for item in violations):
                continue
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            for group_name, group_mask in groups:
                fit = _affine_reconstruction(values[group_mask], truth[group_mask])
                if fit is None:
                    continue
                r2, slope = fit
                if r2 >= _AFFINE_RECONSTRUCTION_R2:
                    violations.append(
                        {
                            "column": column,
                            "kind": "affine_label_reconstruction",
                            "reason": (
                                f"feature column {column!r} reconstructs {target} affinely "
                                f"on {group_name!r} (R²={r2:.5f}, slope={slope:.6f}) — "
                                "it is the quantity the label is computed from"
                            ),
                        }
                    )
                    break
    return {
        "contract": LABEL_SAFETY_CONTRACT,
        "target": target,
        "status": "blocked" if violations else "passed",
        "checked_columns": selected,
        "violations": violations,
    }


def apply_transform_without_label(
    transform: Callable,
    table: pd.DataFrame,
    meta: pd.DataFrame | None,
    *,
    name: str,
    candidates: Iterable[object] = (),
    target: str = PROTECTED_TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute a feature transform behind a hard label-isolation boundary.

    The transform never receives the target column.  Truth is restored only
    after the transform has returned, so it remains available for loss and
    metric computation but cannot participate in feature construction.
    """
    if target not in table.columns:
        raise FeatureSafetyError(f"protected label column {target!r} is missing")
    truth = table[target].copy(deep=True)
    target_position = int(table.columns.get_loc(target))
    feature_input = table.drop(columns=[target]).copy(deep=True)
    try:
        transformed = transform(feature_input, meta)
    except KeyError as exc:
        missing = str(exc).strip("'\"")
        if missing == target:
            raise FeatureSafetyError(
                f"feature transform {name!r} attempted to read protected label {target!r}"
            ) from exc
        raise
    if not isinstance(transformed, pd.DataFrame):
        raise FeatureSafetyError(f"feature transform {name!r} did not return a DataFrame")
    if target in transformed.columns:
        raise FeatureSafetyError(
            f"feature transform {name!r} attempted to create protected label {target!r}"
        )
    transformed = transformed.copy()
    transformed.insert(min(target_position, len(transformed.columns)), target, truth.to_numpy(copy=True))
    audit = audit_feature_frame(transformed, candidates, target=target)
    if audit["status"] != "passed":
        reasons = "; ".join(item["reason"] for item in audit["violations"])
        raise FeatureSafetyError(f"feature transform {name!r} failed label-safety audit: {reasons}")
    return transformed, audit


def passed_label_safety(*, target: str = PROTECTED_TARGET, checked_columns: Iterable[object] = ()) -> dict:
    return {
        "contract": LABEL_SAFETY_CONTRACT,
        "target": target,
        "status": "passed",
        "checked_columns": [str(column) for column in checked_columns],
        "violations": [],
    }
