"""Field charging segments — Spec 12 §1–§4 (K13, Wave 6).

Field data (energy-storage strings, forklift cells, portable packs) arrives as one long record per
unit, not as cycles.  This module turns such a record into the kernel's per-row contract:

* :func:`charging_spans` finds the maximal charging segments of a record (Spec 12 §2): current above
  ``charge_threshold_c × C_rated``, rests shorter than ``rest_split_s`` bridged, a segment ended by a
  discharge, by a rest or record gap longer than ``rest_split_s``, or by the end of the record;
* :func:`segment_table` keeps the segments at least ``min_duration_s`` long that charged at least
  ``min_coverage × C_rated`` (the rest are counted in the QC dictionary for the datasheet) and
  computes, per segment, the standard per-cycle columns on the segment's samples, the descriptors
  (``window_coverage``, ``v_start``, ``v_end``, ``current_mean_c``, ``current_cv``,
  ``rest_before_s``, ``rest_after_s``, ``temp_min``, ``temp_max``) and the identity-adjacent
  columns (``segment_start_s``, ``segment_end_s``, ``segment_kind``, ``sample_period_s``);
  ``cycle_index`` is the unit's segment ordinal in time order;
* :func:`attach_forward_labels` labels every segment with the last reference measurement of the
  same unit that precedes it (Spec 12 §3): ``soh = C_measured / C_rated``, ``label_source``,
  ``label_time_s`` and ``label_age_s``.  A later test never labels an earlier segment; segments
  before the unit's first test carry ``NaN``; a task may bound the label age.

Column classes for the leak inventory (Spec 12 §7.5): the identity-adjacent columns are neither
features nor labels (``contracts.IDENTITY_ADJACENT_COLUMNS``); the label columns are label sources
(``feature_safety.LABEL_SOURCE_FEATURE_COLUMNS``); the descriptors are covariates a program may read,
which the auditor's ``protocol_bound_inputs`` flag reports exactly as it does ``current_mean``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .raw_windows import ic_peak, window_integral

SEGMENT_KIND_CHARGE = "charge"
SEGMENT_IDENTITY_COLUMNS: tuple[str, ...] = ("segment_start_s", "segment_end_s", "segment_kind", "sample_period_s")
SEGMENT_DESCRIPTOR_COLUMNS: tuple[str, ...] = (
    "window_coverage", "v_start", "v_end", "current_mean_c", "current_cv",
    "rest_before_s", "rest_after_s", "temp_min", "temp_max",
)
SEGMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "time_s_max", "voltage_mean", "voltage_std", "voltage_min", "voltage_max", "current_mean", "current_std",
    "abs_current_mean", "temperature_mean", "charge_capacity_max", "q_win_ah", "t_win_s", "ic_peak_ah_per_v",
    "ic_peak_voltage_v",
)
LABEL_COLUMNS: tuple[str, ...] = ("label_source", "label_time_s", "label_age_s")
LABEL_SOURCES: tuple[str, ...] = ("capacity_test", "rpt", "cycle_capacity")
SEGMENT_TABLE_COLUMNS: tuple[str, ...] = (
    "dataset", "cell_id", "cycle_index", "soh", *SEGMENT_FEATURE_COLUMNS, *SEGMENT_DESCRIPTOR_COLUMNS,
    *SEGMENT_IDENTITY_COLUMNS, *LABEL_COLUMNS,
)


@dataclass(frozen=True)
class SegmentConfig:
    """Spec 12 §2 defaults: charging above 0.02 C, rests under 600 s bridged, 300 s and 5 % of the
    rated capacity as the acceptance floor."""

    charge_threshold_c: float = 0.02
    rest_split_s: float = 600.0
    min_duration_s: float = 300.0
    min_coverage: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.charge_threshold_c < 1.0:
            raise ValueError("charge_threshold_c must lie in (0, 1)")
        if self.rest_split_s < 0.0 or self.min_duration_s < 0.0 or not 0.0 <= self.min_coverage < 1.0:
            raise ValueError("rest_split_s / min_duration_s must be >= 0 and min_coverage in [0, 1)")


DEFAULT_CONFIG = SegmentConfig()


def _arrays(time_s, current_a) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=float)
    i = np.asarray(current_a, dtype=float)
    if t.ndim != 1 or i.shape != t.shape:
        raise ValueError("time_s and current_a must be one-dimensional arrays of the same length")
    if len(t) and (not np.isfinite(t).all() or np.any(np.diff(t) < 0.0)):
        raise ValueError("time_s must be finite and non-decreasing")
    return t, i


def charging_spans(
    time_s, current_a, c_rated_ah: float, config: SegmentConfig = DEFAULT_CONFIG
) -> list[tuple[int, int]]:
    """Index spans ``(first, last)`` (inclusive) of the maximal charging segments of a record.

    A sample is charging when ``current > charge_threshold_c × C_rated`` and discharging when it is
    below the negative threshold; everything else (including NaN) is rest.  Consecutive charging
    samples belong to one segment unless a discharging sample lies between them or they are more
    than ``rest_split_s`` apart in time (a longer rest, or a gap in the record).
    """
    t, i = _arrays(time_s, current_a)
    if not (float(c_rated_ah) > 0.0):
        raise ValueError("c_rated_ah must be positive")
    threshold = config.charge_threshold_c * float(c_rated_ah)
    with np.errstate(invalid="ignore"):
        charging = i > threshold
        discharging = i < -threshold
    idx = np.flatnonzero(charging)
    if len(idx) == 0:
        return []
    discharging_before = np.concatenate([[0], np.cumsum(discharging)])   # count in [0, k)
    prev, following = idx[:-1], idx[1:]
    gap = t[following] - t[prev]
    between = discharging_before[following] - discharging_before[prev + 1]
    breaks = np.flatnonzero((gap > config.rest_split_s) | (between > 0))
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(idx) - 1]])
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def accumulated_charge_ah(time_s, current_a) -> np.ndarray:
    """Trapezoidal Ah accumulated from the first sample (NaN currents count as zero)."""
    t, i = _arrays(time_s, current_a)
    if len(t) == 0:
        return np.zeros(0, dtype=float)
    i = np.where(np.isfinite(i), i, 0.0)
    increments = 0.5 * (i[1:] + i[:-1]) * np.diff(t) / 3600.0
    return np.concatenate([[0.0], np.cumsum(increments)])


def segment_curve(time_s, voltage_v, current_a, span: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(v, q_ah, t_rel_s)`` of one span — the charge curve a window feature or health indicator
    integrates (time re-based to the segment start, charge accumulated from zero)."""
    first, last = int(span[0]), int(span[1])
    t = np.asarray(time_s, dtype=float)[first:last + 1]
    v = np.asarray(voltage_v, dtype=float)[first:last + 1]
    i = np.asarray(current_a, dtype=float)[first:last + 1]
    return v, accumulated_charge_ah(t, i), t - t[0]


def _rest_durations(t: np.ndarray, active: np.ndarray, spans: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Rest before and after every span: time to the previous / next active sample, NaN at the
    record boundaries (the true rest is unknown there, never shorter than the record shows)."""
    active_idx = np.flatnonzero(active)
    before = np.full(len(spans), np.nan)
    after = np.full(len(spans), np.nan)
    for n, (first, last) in enumerate(spans):
        k = int(np.searchsorted(active_idx, first, side="left")) - 1
        if k >= 0:
            before[n] = t[first] - t[active_idx[k]]
        k = int(np.searchsorted(active_idx, last, side="right"))
        if k < len(active_idx):
            after[n] = t[active_idx[k]] - t[last]
    return before, after


def segment_table(
    unit: str,
    time_s,
    voltage_v,
    current_a,
    temperature_c=None,
    *,
    c_rated_ah: float,
    config: SegmentConfig = DEFAULT_CONFIG,
    cells_in_series: int = 1,
    window: tuple[float, float] | None = None,
    dataset: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """One row per accepted charging segment of a unit's record (:data:`SEGMENT_TABLE_COLUMNS`).

    ``voltage_v`` is the unit's terminal voltage; with ``cells_in_series > 1`` every voltage column
    is the mean cell voltage (pack voltage / cells in series).  ``window`` gives the calibrated
    ``[lo, hi]`` bounds of ``q_win_ah`` / ``t_win_s`` (NaN when absent).  Labels are not attached
    here (:func:`attach_forward_labels`); ``soh`` and the label columns start as NaN.

    Returns the table and a QC dictionary: segments found, accepted, dropped as too short, dropped
    for low coverage, the Ah charged in accepted segments and the record span.
    """
    t, i = _arrays(time_s, current_a)
    v = np.asarray(voltage_v, dtype=float) / float(cells_in_series)
    if v.shape != t.shape:
        raise ValueError("voltage_v must match time_s")
    temp = np.asarray(temperature_c, dtype=float) if temperature_c is not None else np.full(len(t), np.nan)
    if temp.shape != t.shape:
        raise ValueError("temperature_c must match time_s")
    c_rated = float(c_rated_ah)
    threshold = config.charge_threshold_c * c_rated
    with np.errstate(invalid="ignore"):
        active = np.abs(i) > threshold
    spans = charging_spans(t, i, c_rated, config)
    qc = {"segments_found": len(spans), "too_short": 0, "low_coverage": 0, "accepted": 0,
          "accepted_charge_ah": 0.0, "record_start_s": float(t[0]) if len(t) else None,
          "record_end_s": float(t[-1]) if len(t) else None}
    accepted: list[tuple[int, int, float]] = []
    for first, last in spans:
        duration = t[last] - t[first]
        if duration < config.min_duration_s:
            qc["too_short"] += 1
            continue
        delta_q = float(accumulated_charge_ah(t[first:last + 1], i[first:last + 1])[-1])
        if delta_q < config.min_coverage * c_rated:
            qc["low_coverage"] += 1
            continue
        accepted.append((first, last, delta_q))
    rest_before, rest_after = _rest_durations(t, active, [(a, b) for a, b, _ in accepted])
    rows = []
    for ordinal, (first, last, delta_q) in enumerate(accepted):
        sl = slice(first, last + 1)
        with np.errstate(invalid="ignore"):
            charging = i[sl] > threshold
        v_seg, q_seg, t_rel = segment_curve(t, v, i, (first, last))
        v_chg, i_chg, temp_chg = v[sl][charging], i[sl][charging], temp[sl][charging]
        q_win, t_win = window_integral(v_seg, q_seg, t_rel, window[0], window[1]) if window is not None else (np.nan, np.nan)
        ic_value, ic_voltage = ic_peak(v_seg, q_seg)
        current_mean = float(np.mean(i_chg))
        current_std = float(np.std(i_chg))
        temp_finite = temp_chg[np.isfinite(temp_chg)]
        dt = np.diff(t[sl])
        rows.append({
            "dataset": dataset,
            "cell_id": str(unit),
            "cycle_index": ordinal,
            "soh": np.nan,
            "time_s_max": float(t[last] - t[first]),
            "voltage_mean": float(np.mean(v_chg)),
            "voltage_std": float(np.std(v_chg)),
            "voltage_min": float(np.min(v_chg)),
            "voltage_max": float(np.max(v_chg)),
            "current_mean": current_mean,
            "current_std": current_std,
            "abs_current_mean": float(np.mean(np.abs(i_chg))),
            "temperature_mean": float(np.mean(temp_finite)) if len(temp_finite) else np.nan,
            "charge_capacity_max": delta_q,
            "q_win_ah": q_win,
            "t_win_s": t_win,
            "ic_peak_ah_per_v": ic_value,
            "ic_peak_voltage_v": ic_voltage,
            "window_coverage": delta_q / c_rated,
            "v_start": float(v[first]),
            "v_end": float(v[last]),
            "current_mean_c": current_mean / c_rated,
            "current_cv": current_std / current_mean if current_mean > 0.0 else np.nan,
            "rest_before_s": float(rest_before[ordinal]),
            "rest_after_s": float(rest_after[ordinal]),
            "temp_min": float(np.min(temp_finite)) if len(temp_finite) else np.nan,
            "temp_max": float(np.max(temp_finite)) if len(temp_finite) else np.nan,
            "segment_start_s": float(t[first]),
            "segment_end_s": float(t[last]),
            "segment_kind": SEGMENT_KIND_CHARGE,
            "sample_period_s": float(np.median(dt)) if len(dt) else np.nan,
            "label_source": None,
            "label_time_s": np.nan,
            "label_age_s": np.nan,
        })
        qc["accepted_charge_ah"] += delta_q
    qc["accepted"] = len(rows)
    table = pd.DataFrame(rows, columns=list(SEGMENT_TABLE_COLUMNS))
    if dataset is None:
        table = table.drop(columns=["dataset"])
    return table, qc


def attach_forward_labels(
    segments: pd.DataFrame,
    tests: pd.DataFrame,
    *,
    c_rated_ah: float | Mapping[str, float],
    max_label_age_s: float | None = None,
) -> pd.DataFrame:
    """Forward-only labels (Spec 12 §3).

    ``tests`` carries one reference measurement per row: ``cell_id``, ``label_time_s`` (the time the
    measurement is complete — a test still running cannot label anything), ``capacity_ah`` and
    optionally ``label_source`` (default ``capacity_test``).  Every segment takes the last test of
    its unit with ``label_time_s <= segment_start_s``; ``soh = capacity_ah / C_rated`` of the unit;
    segments before the first test, and beyond ``max_label_age_s`` when given, keep ``NaN``.  The
    result never depends on tests later than a segment, so adding a future test leaves every earlier
    segment's label unchanged.
    """
    required = {"cell_id", "label_time_s", "capacity_ah"}
    missing = required - set(tests.columns)
    if missing:
        raise ValueError(f"tests table is missing columns {sorted(missing)}")
    if "segment_start_s" not in segments.columns:
        raise ValueError("segments table needs segment_start_s")
    out = segments.copy()
    n = len(out)
    soh = np.full(n, np.nan)
    label_time = np.full(n, np.nan)
    label_age = np.full(n, np.nan)
    label_source = np.array([None] * n, dtype=object)
    units = out["cell_id"].astype(str).to_numpy()
    starts = pd.to_numeric(out["segment_start_s"], errors="coerce").to_numpy(dtype=float)
    sources = tests["label_source"].astype(str).to_numpy() if "label_source" in tests.columns else np.array(["capacity_test"] * len(tests), dtype=object)
    for unit in pd.unique(units):
        rated = float(c_rated_ah[unit]) if isinstance(c_rated_ah, Mapping) else float(c_rated_ah)
        if not rated > 0.0:
            raise ValueError(f"rated capacity of unit {unit!r} must be positive")
        mask_tests = tests["cell_id"].astype(str).to_numpy() == unit
        if not mask_tests.any():
            continue
        times = pd.to_numeric(tests.loc[mask_tests, "label_time_s"], errors="coerce").to_numpy(dtype=float)
        caps = pd.to_numeric(tests.loc[mask_tests, "capacity_ah"], errors="coerce").to_numpy(dtype=float)
        srcs = sources[mask_tests]
        valid = np.isfinite(times) & np.isfinite(caps)
        times, caps, srcs = times[valid], caps[valid], srcs[valid]
        order = np.argsort(times, kind="stable")
        times, caps, srcs = times[order], caps[order], srcs[order]
        rows = np.flatnonzero(units == unit)
        if not len(times):
            continue
        position = np.searchsorted(times, starts[rows], side="right") - 1
        for row, k in zip(rows, position):
            if k < 0 or not np.isfinite(starts[row]):
                continue
            age = starts[row] - times[k]
            if max_label_age_s is not None and age > float(max_label_age_s):
                continue
            soh[row] = caps[k] / rated
            label_time[row] = times[k]
            label_age[row] = age
            label_source[row] = str(srcs[k])
    out["soh"] = soh
    out["label_source"] = label_source
    out["label_time_s"] = label_time
    out["label_age_s"] = label_age
    return out


def tests_per_unit(table: pd.DataFrame) -> dict[str, int]:
    """Distinct reference measurements that label at least one row, per unit (Spec 12 §3: the
    evidence pack and the leaderboard report tests per unit)."""
    if "label_time_s" not in table.columns:
        return {}
    labeled = table[pd.to_numeric(table["label_time_s"], errors="coerce").notna()]
    return {str(k): int(v) for k, v in labeled.groupby(labeled["cell_id"].astype(str))["label_time_s"].nunique().items()}
