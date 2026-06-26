"""Named feature-set selectors for the sohbench benchmark.

A ``FeatureSet`` maps a table to the subset of feature columns it represents,
taking the **intersection** with columns actually present in the table so the
same selector works across datasets with different column coverage.

Usage::

    from sohbench.featuresets import project, get_featureset

    projected = project(table, "voltage")   # returns ID cols + voltage features
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from .contracts import ID_COLUMNS, feature_columns
from .registry import register_featureset, get_featureset  # re-export for convenience

__all__ = [
    "FeatureSet",
    "project",
    "register_featureset",
    "get_featureset",
]


@dataclass
class FeatureSet:
    """A named selector for a subset of feature columns.

    Parameters
    ----------
    name:
        Unique identifier used for registry lookup and result tagging.
    candidates:
        Ordered list of candidate column names.  ``columns(table)`` returns the
        intersection of these with the feature columns actually present in
        *table*.
    """

    name: str
    candidates: List[str] = field(default_factory=list)

    def columns(self, table: pd.DataFrame) -> list[str]:
        """Return candidate columns that are present as *feature* columns in *table*.

        ID columns (``dataset``, ``cell_id``, ``cycle_index``, ``soh``) are
        never included even if listed in ``candidates``.
        """
        present_features = set(feature_columns(table))
        return [c for c in self.candidates if c in present_features]


# ---------------------------------------------------------------------------
# project() helper
# ---------------------------------------------------------------------------

def project(table: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return *table* projected to ``ID_COLUMNS`` (present) + selected feature cols.

    If the named feature set selects **zero** columns from *table* (all
    candidates absent), a ``UserWarning`` is emitted and only the ID columns
    are returned.

    Parameters
    ----------
    table:
        Input DataFrame containing at least the ID columns defined in
        ``contracts.ID_COLUMNS``.
    name:
        Name of a registered :class:`FeatureSet`.
    """
    fs = get_featureset(name)
    selected = fs.columns(table)

    id_cols_present = [c for c in ID_COLUMNS if c in table.columns]

    if not selected:
        warnings.warn(
            f"FeatureSet {name!r} selected zero columns from this table "
            f"(no candidates {fs.candidates} are present as feature columns). "
            "Returning ID columns only.",
            UserWarning,
            stacklevel=2,
        )
        return table[id_cols_present].copy()

    keep = id_cols_present + [c for c in selected if c not in id_cols_present]
    return table[keep].copy()


# ---------------------------------------------------------------------------
# Named feature sets (registered at import time)
# ---------------------------------------------------------------------------

register_featureset(FeatureSet(
    name="full",
    candidates=[],  # special sentinel: handled in columns() via override below
))


class _FullFeatureSet(FeatureSet):
    """Returns ALL feature columns present in the table."""

    def columns(self, table: pd.DataFrame) -> list[str]:
        return feature_columns(table)


# Replace the sentinel "full" entry with the proper implementation.
# We use the module-level registry directly to avoid the duplicate-key guard.
from . import registry as _R  # noqa: E402
_R._FEATURESETS.pop("full")  # remove sentinel
_full_fs = _FullFeatureSet(name="full", candidates=[])
_R._FEATURESETS["full"] = _full_fs

register_featureset(FeatureSet(
    name="voltage",
    candidates=["voltage_mean", "voltage_std", "voltage_min", "voltage_max"],
))

register_featureset(FeatureSet(
    name="ic_dva",
    candidates=["ic_peak_ah_per_v", "ic_peak_voltage_v"],
))

register_featureset(FeatureSet(
    name="partial_window",
    candidates=["q_win_ah", "t_win_s"],
))

register_featureset(FeatureSet(
    name="capacity_time",
    candidates=["charge_capacity_max", "time_s_max", "cycle_index_feature", "cycle_fraction"],
))
