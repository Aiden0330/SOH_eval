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
from typing import Callable, List

import numpy as np
import pandas as pd

from .contracts import ID_COLUMNS, feature_columns
from .feature_safety import is_label_source_feature
from .registry import register_featureset, get_featureset  # re-export for convenience

__all__ = [
    "FeatureSet",
    "project",
    "register_featureset",
    "get_featureset",
]

_MECHANISM_DERIVED_COLUMNS = {
    "cycle_sqrt",
    "cycle_log1p",
    "voltage_span_v",
    "current_rms_proxy_a",
    "current_variability_ratio",
    "temperature_over_25c",
    "arrhenius_stress_factor",
}


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
    family: str = "data_driven"
    mechanism_level: str = "none"
    data_granularity: str = "cycle_table"
    description: str = ""
    required_signals: List[str] = field(default_factory=list)
    transform: Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame] | None = None

    def columns(self, table: pd.DataFrame) -> list[str]:
        """Return candidate columns that are present as *feature* columns in *table*.

        ID columns (``dataset``, ``cell_id``, ``cycle_index``, ``soh``) are
        never included even if listed in ``candidates``.  Label-source
        columns (``charge_capacity_max``, ``rated_capacity``/``capacity_ah``
        and their derivatives — the quantities SOH is computed from) are
        banned as model inputs and never selected either.
        """
        present_features = set(feature_columns(table))
        return [
            c for c in self.candidates
            if c in present_features and not is_label_source_feature(c)
        ]


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
        # Keep ``full`` stable when another selected feature profile adds
        # request-scoped derived columns to the shared table.  Otherwise the
        # meaning of ``full`` would silently depend on which sibling checkbox
        # happened to be selected in the same run.
        return [
            column for column in feature_columns(table)
            if column not in _MECHANISM_DERIVED_COLUMNS
            and column not in _PARTIAL_WINDOW_DERIVED_COLUMNS
            and column not in _HI_DERIVED_COLUMNS   # K12 request-scoped derived indicators
            and not is_label_source_feature(column)
        ]


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
    # charge_capacity_max is the calibration capacity the SOH label is
    # computed from and must never be a model input (label-safety v1.1).
    candidates=["time_s_max", "cycle_index_feature", "cycle_fraction"],
))

_PARTIAL_WINDOW_DERIVED_COLUMNS = {"q_win_frac", "t_win_frac"}


def _partial_window_norm_transform(
    table: pd.DataFrame,
    meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dimensionless partial-window health indicators (transferable HIs).

    Absolute window features do not transfer across cell formats and
    C-rates — LODO evidence: extra_trees x partial_window scored 0.069
    inner but 0.401 on the held-out dataset, because q_win_ah carries the
    cell's Ah scale.  These fractions remove the scale:

    - ``q_win_frac``  = q_win_ah / c_rated_ah   (rated capacity from meta;
      rated is nameplate, not the calibration capacity the label uses)
    - ``t_win_frac``  = t_win_s / time_s_max    (fraction of charge time)

    Label-safety note: q_win_frac inherits its window from q_win_ah, whose
    coverage is audited at ingestion and preflight (DOD < 90% rule); a
    full-range window would surface through the per-dataset affine gate on
    q_win_frac itself (rated is constant per cell).
    """
    out = table.copy()
    q_win = _numeric(out, "q_win_ah")

    rated = np.full(len(out), np.nan, dtype=float)
    if meta is not None and "c_rated_ah" in getattr(meta, "columns", ()):
        separator = "\x1f"
        lookup: dict[str, float] = {}
        for dataset, cell, value in zip(
            meta.get("dataset", []), meta.get("cell_id", []), meta["c_rated_ah"]
        ):
            try:
                lookup[f"{dataset}{separator}{cell}"] = float(value)
            except (TypeError, ValueError):
                continue
        if lookup and "dataset" in out.columns and "cell_id" in out.columns:
            keys = out["dataset"].astype(str) + separator + out["cell_id"].astype(str)
            rated = pd.to_numeric(keys.map(lookup), errors="coerce").to_numpy(dtype=float)
    rated = np.where(np.isfinite(rated) & (rated > 0.0), rated, np.nan)
    out["q_win_frac"] = q_win / rated

    t_win = _numeric(out, "t_win_s")
    t_max = _numeric(out, "time_s_max")
    t_max = np.where(np.isfinite(t_max) & (t_max > 0.0), t_max, np.nan)
    out["t_win_frac"] = t_win / t_max
    return out


_HI_DERIVED_COLUMNS = (
    [f"hi_dq_w{k}_frac" for k in range(1, 5)] + ["hi_span_dq_frac"]
    + [f"hi_dt_w{k}_frac" for k in range(1, 5)] + ["hi_span_dt_frac"]
    + ["hi_ic_peak_frac_per_v", "hi_ic_peak_v", "hi_v_q25", "hi_v_q50", "hi_v_q75", "hi_dvdq_v_per_frac",
       "hi_i_mean_c", "hi_i_cv"]
)


def _rated_per_row(table: pd.DataFrame, meta: pd.DataFrame | None) -> np.ndarray:
    """Rated (nameplate) capacity per row from ``meta`` (``dataset, cell_id, c_rated_ah``); NaN when unknown."""
    rated = np.full(len(table), np.nan, dtype=float)
    if meta is not None and "c_rated_ah" in getattr(meta, "columns", ()):
        separator = "\x1f"
        lookup: dict[str, float] = {}
        for dataset, cell, value in zip(meta.get("dataset", []), meta.get("cell_id", []), meta["c_rated_ah"]):
            try:
                lookup[f"{dataset}{separator}{cell}"] = float(value)
            except (TypeError, ValueError):
                continue
        if lookup and "dataset" in table.columns and "cell_id" in table.columns:
            keys = table["dataset"].astype(str) + separator + table["cell_id"].astype(str)
            rated = pd.to_numeric(keys.map(lookup), errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(rated) & (rated > 0.0), rated, np.nan)


def _partial_charge_hi_transform(table: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dimensionless partial-charge health indicators (K12, Wave 6 plan §4 — the honest bar for M2).

    From the raw indicator columns of ``sohkernel.health_indicators`` (Ah, s, V, A) to what transfers across cell
    formats and laboratories: charge fractions of the rated capacity, time fractions of the charge time, C-rates,
    and the voltages and slopes that were dimensionless already.  Rated capacity comes from ``meta`` (nameplate,
    never the calibration capacity the label uses); a row without it degrades to NaN, never to a wrong scale.
    """
    out = table.copy()
    rated = _rated_per_row(out, meta)
    t_max = _numeric(out, "time_s_max")
    t_max = np.where(np.isfinite(t_max) & (t_max > 0.0), t_max, np.nan)
    for k in range(1, 5):
        out[f"hi_dq_w{k}_frac"] = _numeric(out, f"hi_dq_w{k}_ah") / rated
        out[f"hi_dt_w{k}_frac"] = _numeric(out, f"hi_dt_w{k}_s") / t_max
    out["hi_span_dq_frac"] = _numeric(out, "hi_span_dq_ah") / rated
    out["hi_span_dt_frac"] = _numeric(out, "hi_span_dt_s") / t_max
    out["hi_ic_peak_frac_per_v"] = _numeric(out, "hi_ic_peak_ah_per_v") / rated
    out["hi_dvdq_v_per_frac"] = _numeric(out, "hi_dvdq_v_per_ah") * rated
    out["hi_i_mean_c"] = _numeric(out, "hi_i_mean_a") / rated
    for column in ("hi_ic_peak_v", "hi_v_q25", "hi_v_q50", "hi_v_q75", "hi_i_cv"):
        out[column] = _numeric(out, column)
    return out


register_featureset(FeatureSet(
    name="partial_charge_hi",
    candidates=list(_HI_DERIVED_COLUMNS),
    family="literature_hi",
    description=(
        "Literature partial-charging health indicators (K12): charge and time fractions in four fixed plateau "
        "sub-windows and over their span, the in-span incremental-capacity peak, voltages at quarter, half and three "
        "quarters of the span charge, the mean V(Q) slope, the mean C-rate and its variability — the honest bar an "
        "agent-written indicator must beat."
    ),
    transform=_partial_charge_hi_transform,
))


register_featureset(FeatureSet(
    name="partial_window_norm",
    candidates=["q_win_frac", "t_win_frac"],
    description=(
        "无量纲部分窗口健康因子：窗口容量/额定容量、窗口时长/充电时长 —— "
        "跨电芯规格与倍率可迁移（L1 可迁移 HI）。"
    ),
    transform=_partial_window_norm_transform,
))


def _numeric(table: pd.DataFrame, column: str) -> np.ndarray:
    """Return a finite-or-NaN numeric array without changing row identity."""
    if column not in table.columns:
        return np.full(len(table), np.nan, dtype=float)
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _mechanism_proxy_transform(
    table: pd.DataFrame,
    meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive target-free degradation and operating-stress proxies.

    These quantities are intentionally named ``*_proxy`` where the available
    per-cycle aggregate table cannot support identification of a full ECM,
    electrochemical model, or cumulative Ah-throughput state.  Missing source
    signals remain NaN so dataset compatibility stays auditable.
    """
    del meta
    out = table.copy()
    cycle = _numeric(out, "cycle_index_feature")
    if not np.isfinite(cycle).any():
        cycle = _numeric(out, "cycle_index")
    cycle = np.maximum(cycle, 0.0)
    out["cycle_sqrt"] = np.sqrt(cycle)
    out["cycle_log1p"] = np.log1p(cycle)

    voltage_min = _numeric(out, "voltage_min")
    voltage_max = _numeric(out, "voltage_max")
    out["voltage_span_v"] = voltage_max - voltage_min

    # charge_energy_proxy_wh / cumulative_capacity_proxy_ah were derived from
    # charge_capacity_max — the quantity the SOH label is computed from — and
    # are no longer produced (label-safety v1.1 structural ban).

    current_mean = _numeric(out, "current_mean")
    current_std = _numeric(out, "current_std")
    abs_current_mean = _numeric(out, "abs_current_mean")
    out["current_rms_proxy_a"] = np.sqrt(np.square(current_mean) + np.square(current_std))
    denominator = np.where(np.abs(abs_current_mean) > 1e-12, np.abs(abs_current_mean), np.nan)
    out["current_variability_ratio"] = current_std / denominator

    temperature_c = _numeric(out, "temperature_mean")
    temperature_k = temperature_c + 273.15
    out["temperature_over_25c"] = temperature_c - 25.0
    # Fixed 24 kJ/mol activation-energy proxy.  It is a stress descriptor,
    # not an identified cell-specific aging parameter.
    activation_energy = 24_000.0
    gas_constant = 8.314462618
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exponent = (activation_energy / gas_constant) * (1.0 / 298.15 - 1.0 / temperature_k)
    out["arrhenius_stress_factor"] = np.where(
        np.isfinite(exponent) & (temperature_k > 0.0),
        np.exp(np.clip(exponent, -30.0, 30.0)),
        np.nan,
    )
    return out


register_featureset(FeatureSet(
    name="electrochemical_diagnostics",
    candidates=[
        "ic_peak_ah_per_v",
        "ic_peak_voltage_v",
        "q_win_ah",
        "t_win_s",
        "voltage_span_v",
    ],
    family="mechanism_derived",
    mechanism_level="diagnostic_proxy",
    description="IC/DVA 峰、局部充电窗口与电压/能量代理，适合曲线诊断和退化模式比较。",
    required_signals=["cycle_voltage", "cycle_current", "cycle_time"],
    transform=_mechanism_proxy_transform,
))

register_featureset(FeatureSet(
    name="operating_stress",
    candidates=[
        "current_rms_proxy_a",
        "current_variability_ratio",
        "temperature_mean",
        "temperature_over_25c",
        "arrhenius_stress_factor",
        "voltage_span_v",
    ],
    family="mechanism_derived",
    mechanism_level="stress_proxy",
    description="电流波动、温度与 Arrhenius 应力代理，用于跨工况泛化与灰盒退化模型。",
    required_signals=["cycle_current", "cycle_temperature", "cycle_voltage"],
    transform=_mechanism_proxy_transform,
))

register_featureset(FeatureSet(
    name="degradation_kinetics",
    candidates=[
        "cycle_sqrt",
        "cycle_log1p",
        "time_s_max",
        "arrhenius_stress_factor",
    ],
    family="mechanism_derived",
    mechanism_level="semi_empirical",
    description="平方根/对数老化坐标、容量吞吐与温度应力代理，面向半经验退化先验。",
    required_signals=["cycle_index", "cycle_capacity", "cycle_temperature"],
    transform=_mechanism_proxy_transform,
))
