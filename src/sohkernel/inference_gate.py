"""Trusted-inference label gate (Spec 01 §2, layer 3; extraction plan Task 8).

A trained artifact's manifest names the columns it consumes.  Before ``predict`` those
columns are audited against the table the model is about to see: label-source columns are
structurally banned, and the dynamic probes (exact copy / complement, affine reconstruction,
Ah-window coverage) run on the actual values.  Moved here from the workbench model store so
any tool that loads a frozen model applies the same gate.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .feature_safety import (
    LABEL_SAFETY_CONTRACT,
    FeatureSafetyError,
    audit_feature_frame,
    is_label_source_feature,
)

__all__ = ["assert_manifest_label_safe"]


def assert_manifest_label_safe(
    manifest_columns: Iterable[object],
    table: pd.DataFrame,
    *,
    target: str = "soh",
    model_name: str = "model",
) -> dict:
    """Raise :class:`FeatureSafetyError` unless every manifest column is label-safe on ``table``.

    Returns the passed audit so callers can record it as provenance.
    """
    columns = [str(column) for column in manifest_columns]
    banned = [column for column in columns if is_label_source_feature(column, target=target)]
    if banned:
        raise FeatureSafetyError(
            f"model {model_name!r} inference blocked by {LABEL_SAFETY_CONTRACT}: "
            f"feature columns are label sources (structurally banned): {banned}"
        )
    audit = audit_feature_frame(table, columns, target=target)
    if audit["status"] == "passed":
        return audit
    reasons = "; ".join(item["reason"] for item in audit["violations"])
    raise FeatureSafetyError(
        f"model {model_name!r} inference blocked by {LABEL_SAFETY_CONTRACT}: {reasons}"
    )
