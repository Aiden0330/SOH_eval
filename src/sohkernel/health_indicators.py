"""Literature health indicators from partial charging curves (K12, Wave 6 plan §4) — the honest bar for M2.

The partial-charging literature (fragmented / charge-state-invariant charge capacity, partial incremental-capacity
features, curve-shape features) converges on a small family of hand indicators that transfer across cells and
laboratories.  This module computes them from the **raw charge curve** of every cycle
(``sohkernel.raw_windows.iter_charge_curves`` yields voltage, accumulated charge and time of the charge segment):

- ``hi_dq_w{k}_ah`` / ``hi_dt_w{k}_s`` — charge and time in four fixed sub-windows of the chemistry's indicator span
  (the fragmented-charge-capacity family; invariant to where the segment started as long as the window is covered);
- ``hi_span_dq_ah`` / ``hi_span_dt_s`` — the same over the whole span;
- ``hi_ic_peak_ah_per_v`` / ``hi_ic_peak_v`` — the incremental-capacity peak (height and bin-centre voltage) on
  10 mV bins inside the span;
- ``hi_v_q25`` / ``hi_v_q50`` / ``hi_v_q75`` — voltage at a quarter, half and three quarters of the span's charge
  (curve shape); ``hi_dvdq_v_per_ah`` — the mean slope of V(Q) over the span;
- ``hi_i_mean_a`` / ``hi_i_cv`` — mean charging current over the span and its coefficient of variation across the
  four windows (the variable-current descriptor).

Raw units are stored (Ah, s, V, A); the ``partial_charge_hi`` featureset (``sohkernel.featuresets``) turns them into
dimensionless indicators — fractions of rated capacity, fractions of charge time, volts, C-rates — which is what
transfers.  Charge columns are window capacities and stay under the label-safety DOD rule
(``feature_safety.PARTIAL_WINDOW_CAPACITY_COLUMNS``); the audit checks them on every rebuild.

**Why these spans.** A span must be covered by every laboratory of a chemistry, or the indicator means different
things per lab.  The LFP span is 3.30–3.40 V in four 25 mV windows: UConn's reference charge ends near 3.40 V and
HUST's 5 C charge leaves 3.55 V uncovered in a fifth of its cycles, while 3.30–3.40 V is present in every cycle of
hust, snl_lfp and uconn (rebuild report 2026-09-05).  Fixed voltage windows still see different states of charge at
different C-rates — that is the transfer problem itself; ``hi_i_mean_a`` (as a C-rate) is stored so a model can
condition on it.  A window whose edge lies outside a cycle's voltage range is ``NaN`` for that cycle, as
``q_win_ah`` is for an unmatched cycle.  Constant-voltage-phase and relaxation indicators need signals the charge
segment does not always hold; Spec 12 adds them with field segments (K13).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Fixed indicator windows per chemistry (volts): four windows tiling the span.
HI_SPAN_WINDOWS: dict[str, tuple[tuple[float, float], ...]] = {
    "LFP": ((3.300, 3.325), (3.325, 3.350), (3.350, 3.375), (3.375, 3.400)),
    "NMC": ((3.70, 3.80), (3.80, 3.90), (3.90, 4.00), (4.00, 4.10)),
    "NCA": ((3.70, 3.80), (3.80, 3.90), (3.90, 4.00), (4.00, 4.10)),
    "LCO": ((3.85, 3.90), (3.90, 3.95), (3.95, 4.00), (4.00, 4.05)),
}
N_WINDOWS = 4
IC_BIN_V = 0.010   # incremental-capacity bin width inside the span

WINDOW_CAPACITY_COLUMNS: tuple[str, ...] = tuple(f"hi_dq_w{k}_ah" for k in range(1, N_WINDOWS + 1)) + ("hi_span_dq_ah",)
RAW_COLUMNS: tuple[str, ...] = (
    WINDOW_CAPACITY_COLUMNS
    + tuple(f"hi_dt_w{k}_s" for k in range(1, N_WINDOWS + 1))
    + ("hi_span_dt_s", "hi_ic_peak_ah_per_v", "hi_ic_peak_v", "hi_v_q25", "hi_v_q50", "hi_v_q75",
       "hi_dvdq_v_per_ah", "hi_i_mean_a", "hi_i_cv")
)


def windows_for(chemistry: str) -> tuple[tuple[float, float], ...]:
    """The indicator windows of a chemistry label (``LCO/NCO`` and friends resolve on their first component)."""
    key = str(chemistry).strip().upper().split("/")[0]
    if key not in HI_SPAN_WINDOWS:
        raise KeyError(f"no health-indicator windows for chemistry {chemistry!r}; known: {sorted(HI_SPAN_WINDOWS)}")
    return HI_SPAN_WINDOWS[key]


def _monotone(v: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(v sorted, y made non-decreasing along it) or None when fewer than two finite samples exist."""
    finite = np.isfinite(v) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return None
    order = np.argsort(v[finite], kind="stable")
    return v[finite][order], np.maximum.accumulate(y[finite][order])


def _at(edges: np.ndarray, vs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    values = np.interp(edges, vs, ys)
    values[(edges < vs[0]) | (edges > vs[-1])] = np.nan
    return values


def hi_from_curve(v: np.ndarray, q: np.ndarray, t: np.ndarray | None, windows: Iterable[tuple[float, float]]) -> dict:
    """:data:`RAW_COLUMNS` of one charge curve (voltage, accumulated charge in Ah, time in s)."""
    windows = tuple(windows)
    span_lo, span_hi = windows[0][0], windows[-1][1]
    out = {column: np.nan for column in RAW_COLUMNS}
    v = np.asarray(v, dtype=float)
    q = np.asarray(q, dtype=float)
    mono_q = _monotone(v, q)
    if mono_q is None:
        return out
    vs, qs = mono_q
    mono_t = _monotone(v, np.asarray(t, dtype=float)) if t is not None else None
    edges_w = np.array([w[0] for w in windows] + [span_hi])
    q_w = _at(edges_w, vs, qs)
    t_w = _at(edges_w, *mono_t) if mono_t is not None else np.full(len(edges_w), np.nan)
    currents = []
    for k in range(N_WINDOWS):
        dq, dt = q_w[k + 1] - q_w[k], t_w[k + 1] - t_w[k]
        out[f"hi_dq_w{k + 1}_ah"], out[f"hi_dt_w{k + 1}_s"] = dq, dt
        currents.append(dq / dt * 3600.0 if np.isfinite(dq) and np.isfinite(dt) and dt > 0 else np.nan)
    span_dq, span_dt = q_w[-1] - q_w[0], t_w[-1] - t_w[0]
    out["hi_span_dq_ah"], out["hi_span_dt_s"] = span_dq, span_dt
    # incremental capacity on 10 mV bins inside the span (only bins the curve covers)
    bin_edges = np.round(np.arange(span_lo, span_hi + 1e-9, IC_BIN_V), 4)
    q_bins = _at(bin_edges, vs, qs)
    ic = np.diff(q_bins) / IC_BIN_V
    if np.isfinite(ic).any():
        peak = int(np.nanargmax(np.where(np.isfinite(ic), ic, -np.inf)))
        out["hi_ic_peak_ah_per_v"] = float(ic[peak])
        out["hi_ic_peak_v"] = float((bin_edges[peak] + bin_edges[peak + 1]) / 2.0)
    if np.isfinite(span_dq) and span_dq > 0:
        cumulative = q_bins - q_bins[0]
        if np.isfinite(cumulative).all():
            for fraction, column in ((0.25, "hi_v_q25"), (0.50, "hi_v_q50"), (0.75, "hi_v_q75")):
                out[column] = float(np.interp(fraction * span_dq, cumulative, bin_edges))
        out["hi_dvdq_v_per_ah"] = (span_hi - span_lo) / span_dq
        if np.isfinite(span_dt) and span_dt > 0:
            out["hi_i_mean_a"] = span_dq / span_dt * 3600.0
    finite_currents = np.array([c for c in currents if np.isfinite(c)])
    if len(finite_currents) >= 2 and finite_currents.mean() > 0:
        out["hi_i_cv"] = float(finite_currents.std() / finite_currents.mean())
    return out


def hi_from_curves(curves: Iterable, chemistry: str) -> pd.DataFrame:
    """One row per ``(cell_id, cycle_index)`` with :data:`RAW_COLUMNS` from ``(cell, cycle, v, q, t)`` curves."""
    windows = windows_for(chemistry)
    rows = []
    for cell, cycle, v, q, t in curves:
        record = hi_from_curve(v, q, t, windows)
        record["cell_id"], record["cycle_index"] = str(cell), int(cycle)
        rows.append(record)
    frame = pd.DataFrame(rows, columns=["cell_id", "cycle_index", *RAW_COLUMNS])
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["cycle_index"] = frame["cycle_index"].astype(int)
    return frame


def attach(table: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    """``table`` with :data:`RAW_COLUMNS` joined on ``(cell_id, cycle_index)``; unmatched cycles get ``NaN``."""
    out = table.drop(columns=[c for c in RAW_COLUMNS if c in table.columns]).copy()
    keys = pd.DataFrame({"cell_id": out["cell_id"].astype(str), "cycle_index": out["cycle_index"].astype(int)})
    merged = keys.merge(indicators.drop_duplicates(["cell_id", "cycle_index"]), on=["cell_id", "cycle_index"], how="left")
    for column in RAW_COLUMNS:
        out[column] = merged[column].to_numpy(dtype=float)
    return out


def coverage_summary(table: pd.DataFrame, rated_ah: float | None) -> dict:
    """Per-column availability and the span's median coverage of the rated capacity (for the rebuild report)."""
    summary = {"rows": int(len(table)), "available": {}, "span_coverage_median": None}
    for column in RAW_COLUMNS:
        if column in table.columns:
            summary["available"][column] = round(float(pd.to_numeric(table[column], errors="coerce").notna().mean()), 4)
    if rated_ah and "hi_span_dq_ah" in table.columns:
        span = pd.to_numeric(table["hi_span_dq_ah"], errors="coerce") / float(rated_ah)
        summary["span_coverage_median"] = round(float(span.median()), 4) if span.notna().any() else None
    return summary
