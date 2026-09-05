"""Build ``snl_<chemistry>_features.pkl`` tables from the BatteryLife SNL pickles (D1).

The per-cycle feature extractor that produced ``data/processed/snl_lfp_features.pkl`` was never
committed; this module is its reverse-engineered replacement.  Every statistic is defined on the
**charge segment** of a cycle (``current_in_A > 0.01``, the convention of
``scripts/rebuild_partial_window_features.py`` / ``sohkernel.raw_windows``), with time re-based to
the first charge sample.  ``--verify-against`` proves the reconstruction row by row against an
existing table (see ``compare_tables`` for the tolerances) before the extractor is trusted on the
NMC and NCA subsets.

Columns (in the order of the historical table)::

    dataset cell_id cycle_index soh time_s_max voltage_mean voltage_std voltage_min voltage_max
    current_mean current_std abs_current_mean temperature_mean charge_capacity_max q_win_ah
    t_win_s ic_peak_ah_per_v ic_peak_voltage_v cycle_index_feature cycle_fraction

Definitions (verified against snl_lfp on 2026-09-04 unless marked):

* ``soh``               = max ``discharge_capacity_in_Ah`` of the cycle / rated capacity.  The
                          denominator is ``--rated-ah`` (the datasheet nameplate: LFP 1.1, NMC 3.0,
                          NCA 3.2) and falls back to the pickle's ``nominal_capacity_in_Ah``.  The
                          BatteryLife pickles derate the nominal of the 15 degC NMC/NCA cells to 0.9x
                          (2.7 / 2.88 Ah); Spec 01 anchors soh to one rated capacity per dataset, so
                          the nameplate is used for every cell.  Cycles with a non-positive or
                          missing discharge capacity carry no label and are dropped (the historical
                          snl_lfp table dropped the same 90 zero-capacity cycles);
* ``time_s_max``        = duration of the charge segment (last minus first charge sample);
* ``voltage_*`` / ``current_*`` / ``abs_current_mean`` / ``temperature_mean``
                        = statistics over the charge segment (population std, ``ddof=0``);
* ``charge_capacity_max`` = max ``charge_capacity_in_Ah`` of the cycle (label-source column, banned
                          as a model input by ``sohkernel.feature_safety``);
* ``q_win_ah`` / ``t_win_s`` = charge / time accumulated while the charge-segment voltage sits inside
                          the calibrated ``[lo, hi]`` window (``_window_integral``);
* ``ic_peak_ah_per_v`` / ``ic_peak_voltage_v``
                        = peak of the smoothed incremental-capacity curve of the charge segment
                          (``ic_peak``; the historical recipe could not be reproduced exactly — the
                          definition below is the documented replacement, agreement reported by
                          ``--verify-against``);
* ``cycle_index_feature`` = float(cycle_index); ``cycle_fraction`` = cycle_index / max cycle_index
                          of the cell.

Exact duplicate ``(cell_id, cycle_index)`` rows are dropped (the kernel QC rejects ambiguous cycle
identities).  Cells cycled over a partial SOC interval (pickle ``SOC_interval`` != [0, 1], the four
NCA 20-80 % cells) are skipped: their per-cycle discharge capacity is a partial-cycle throughput, not
a calibration capacity, so no Spec 01 label exists for them.

Usage::

    python scripts/ingest_batterylife_snl.py --chemistry LFP --window 3.30 3.38 \
        --out scratch/snl_lfp_rebuilt.pkl --verify-against data/processed/snl_lfp_features.pkl
    python scripts/ingest_batterylife_snl.py --chemistry NMC --scan-windows 3.4 4.1 0.05
    python scripts/ingest_batterylife_snl.py --chemistry NMC --window 3.75 3.95 \
        --out data/processed/snl_nmc_features.pkl
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sohkernel.raw_windows import (  # noqa: E402
    IC_GRID_STEP_V, IC_MIN_SAMPLES, IC_SMOOTH_POINTS, ic_peak, window_integral as _window_integral,
)

CHARGE_CURRENT_MIN_A = 0.01
# IC_GRID_STEP_V / IC_SMOOTH_POINTS / IC_MIN_SAMPLES and ic_peak live in sohkernel.raw_windows since K13
# (2026-09-05) so field-segment extractors compute the same peak; re-exported here for the D1 test.
__all_ic__ = (IC_GRID_STEP_V, IC_SMOOTH_POINTS, IC_MIN_SAMPLES, ic_peak)
FULL_SOC_INTERVAL = (0.0, 1.0)

COLUMNS = [
    "dataset", "cell_id", "cycle_index", "soh", "time_s_max", "voltage_mean", "voltage_std",
    "voltage_min", "voltage_max", "current_mean", "current_std", "abs_current_mean",
    "temperature_mean", "charge_capacity_max", "q_win_ah", "t_win_s", "ic_peak_ah_per_v",
    "ic_peak_voltage_v", "cycle_index_feature", "cycle_fraction",
]

# Tolerances used by --verify-against (task D1): exact-ish identities to 1e-6, segment statistics
# to 1e-3; ic_peak_* are reported but not gated (definition differs from the historical recipe).
VERIFY_TOLERANCES = {
    "soh": 1e-6, "charge_capacity_max": 1e-6, "cycle_fraction": 1e-6, "cycle_index_feature": 1e-6,
    "time_s_max": 1e-3, "voltage_mean": 1e-3, "voltage_std": 1e-3, "voltage_min": 1e-3,
    "voltage_max": 1e-3, "current_mean": 1e-3, "current_std": 1e-3, "abs_current_mean": 1e-3,
    "temperature_mean": 1e-3, "q_win_ah": 1e-3, "t_win_s": 1e-3,
}
VERIFY_UNGATED = ("ic_peak_ah_per_v", "ic_peak_voltage_v")


def raw_paths(data_dir: Path, chemistry: str) -> list[str]:
    pattern = str(data_dir / "raw" / "snl_batterylife" / "SNL" / f"SNL_18650_{chemistry.upper()}_*.pkl")
    return sorted(glob.glob(pattern))


def charge_segment(cycle: dict) -> dict[str, np.ndarray] | None:
    """Arrays of the charge segment (``current > 0.01 A``), or None when it has < 2 samples."""
    current = np.asarray(cycle["current_in_A"], dtype=float)
    mask = current > CHARGE_CURRENT_MIN_A
    if int(mask.sum()) < 2:
        return None
    out = {"current": current[mask]}
    for key, name in (
        ("voltage_in_V", "voltage"), ("charge_capacity_in_Ah", "charge"),
        ("time_in_s", "time"), ("temperature_in_C", "temperature"),
    ):
        values = cycle.get(key)
        if values is None:
            out[name] = np.full(int(mask.sum()), np.nan)
        else:
            out[name] = np.asarray(values, dtype=float)[mask]
    return out


def full_soc_cell(payload: dict) -> bool:
    interval = payload.get("SOC_interval")
    if interval is None:
        return True
    try:
        lo, hi = (float(x) for x in interval)
    except (TypeError, ValueError):
        return True
    return abs(lo - FULL_SOC_INTERVAL[0]) < 1e-9 and abs(hi - FULL_SOC_INTERVAL[1]) < 1e-9


def cycle_soh(payload: dict, cycle: dict, rated_ah: float | None) -> float:
    rated = float(rated_ah) if rated_ah else float(payload["nominal_capacity_in_Ah"])
    discharge = np.asarray(cycle["discharge_capacity_in_Ah"], dtype=float)
    if not np.isfinite(discharge).any():
        return np.nan
    return float(np.nanmax(discharge)) / rated


def cycle_row(payload: dict, cycle: dict, window: tuple[float, float], dataset: str,
              rated_ah: float | None = None) -> dict | None:
    segment = charge_segment(cycle)
    if segment is None:
        return None
    soh = cycle_soh(payload, cycle, rated_ah)
    if not np.isfinite(soh) or soh <= 0.0:
        return None
    charge_full = np.asarray(cycle["charge_capacity_in_Ah"], dtype=float)
    v, i, q, t, temp = (segment[k] for k in ("voltage", "current", "charge", "time", "temperature"))
    q_win, t_win = _window_integral(v, q, t, window[0], window[1])
    ic_value, ic_voltage = ic_peak(v, q)
    return {
        "dataset": dataset,
        "cell_id": str(payload["cell_id"]),
        "cycle_index": int(cycle["cycle_number"]),
        "soh": soh,
        "time_s_max": float(np.nanmax(t) - np.nanmin(t)),
        "voltage_mean": float(np.mean(v)),
        "voltage_std": float(np.std(v)),
        "voltage_min": float(np.min(v)),
        "voltage_max": float(np.max(v)),
        "current_mean": float(np.mean(i)),
        "current_std": float(np.std(i)),
        "abs_current_mean": float(np.mean(np.abs(i))),
        "temperature_mean": float(np.mean(temp)),
        "charge_capacity_max": float(np.nanmax(charge_full)) if np.isfinite(charge_full).any() else np.nan,
        "q_win_ah": q_win,
        "t_win_s": t_win,
        "ic_peak_ah_per_v": ic_value,
        "ic_peak_voltage_v": ic_voltage,
        "cycle_index_feature": float(cycle["cycle_number"]),
    }


def extract_cell(path: str | Path, window: tuple[float, float], dataset: str,
                 rated_ah: float | None = None) -> pd.DataFrame:
    payload = pd.read_pickle(path)
    rows = [row for row in (cycle_row(payload, cycle, window, dataset, rated_ah) for cycle in payload["cycle_data"]) if row]
    frame = pd.DataFrame(rows, columns=COLUMNS[:-1])
    if len(frame):
        frame["cycle_fraction"] = frame["cycle_index"] / float(frame["cycle_index"].max())
    else:
        frame["cycle_fraction"] = pd.Series(dtype=float)
    return frame


def select_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split the pickles into full-SOC cells (ingested) and partial-SOC cells (skipped)."""
    kept, skipped = [], []
    for path in paths:
        payload = pd.read_pickle(path)
        (kept if full_soc_cell(payload) else skipped).append(path)
    return kept, skipped


def build_table(paths: list[str], dataset: str, window: tuple[float, float],
                rated_ah: float | None = None) -> tuple[pd.DataFrame, dict]:
    paths, skipped = select_paths(paths)
    frames = [extract_cell(path, window, dataset, rated_ah) for path in paths]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    table = table[COLUMNS]
    before = len(table)
    exact_duplicates = int(table.duplicated(keep="first").sum())
    table = table.drop_duplicates(keep="first")
    identity_duplicates = int(table.duplicated(["cell_id", "cycle_index"], keep=False).sum())
    table = table.drop_duplicates(["cell_id", "cycle_index"], keep=False).reset_index(drop=True)
    table = table.sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    stats = {
        "rated_ah_for_soh": rated_ah,
        "partial_soc_cells_skipped": [Path(p).stem for p in skipped],
        "cells": int(table["cell_id"].nunique()),
        "rows": int(len(table)),
        "rows_before_dedup": before,
        "exact_duplicate_rows_dropped": exact_duplicates,
        "ambiguous_identity_rows_dropped": identity_duplicates,
        "soh_min": float(table["soh"].min()) if len(table) else None,
        "soh_max": float(table["soh"].max()) if len(table) else None,
    }
    return table, stats


def compare_tables(new: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Row-by-row match rates on (cell_id, cycle_index); NaN == NaN counts as a match."""
    key = ["cell_id", "cycle_index"]
    merged = ref.merge(new, on=key, how="left", suffixes=("_ref", "_new"), indicator=True)
    report = {
        "reference_rows": int(len(ref)),
        "rebuilt_rows": int(len(new)),
        "reference_rows_found": int((merged["_merge"] == "both").sum()),
        "rebuilt_rows_not_in_reference": int(len(new.merge(ref[key], on=key, how="left", indicator=True).query("_merge == 'left_only'"))),
        "columns": {},
    }
    both = merged[merged["_merge"] == "both"]
    for column, tolerance in {**VERIFY_TOLERANCES, **{c: None for c in VERIFY_UNGATED}}.items():
        a = pd.to_numeric(both[f"{column}_ref"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(both[f"{column}_new"], errors="coerce").to_numpy(dtype=float)
        nan_both = np.isnan(a) & np.isnan(b)
        finite = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a - b)
        entry: dict = {"n": int(len(a)), "nan_in_reference": int(np.isnan(a).sum())}
        if tolerance is not None:
            match = nan_both | (finite & (diff <= tolerance))
            entry.update({"tolerance": tolerance, "match_rate": float(match.mean()) if len(a) else None,
                          "max_abs_diff": float(np.nanmax(diff[finite])) if finite.any() else None})
        else:
            rel = diff[finite] / np.maximum(np.abs(a[finite]), 1e-12)
            corr = float(np.corrcoef(a[finite], b[finite])[0, 1]) if finite.sum() > 2 else None
            entry.update({"gated": False, "pearson_r": corr,
                          "median_abs_diff": float(np.median(diff[finite])) if finite.any() else None,
                          "median_rel_diff": float(np.median(rel)) if finite.any() else None,
                          "within_1pct": float((rel <= 0.01).mean()) if finite.any() else None,
                          "within_5pct": float((rel <= 0.05).mean()) if finite.any() else None,
                          "within_10pct": float((rel <= 0.10).mean()) if finite.any() else None})
        report["columns"][column] = entry
    gated = [c for c in VERIFY_TOLERANCES]
    report["all_gated_columns_at_least_99pct"] = all(
        (report["columns"][c]["match_rate"] or 0.0) >= 0.99 for c in gated
    )
    return report


def scan_windows(paths: list[str], rated_ah: float, lo_min: float, hi_max: float, step: float,
                 *, cycle_stride: int = 10, min_width: float = 0.10) -> pd.DataFrame:
    """Coverage median and affine R² (soh ~ q_win) for a grid of [lo, hi] windows (skill §2).

    Uses every ``cycle_stride``-th cycle of every cell; the same charge-segment convention as the
    extractor.  ``coverage`` = q_win / rated capacity.
    """
    los = np.round(np.arange(lo_min, hi_max - min_width + 1e-9, step), 3)
    his = np.round(np.arange(lo_min + min_width, hi_max + 1e-9, step), 3)
    pairs = [(lo, hi) for lo in los for hi in his if hi - lo >= min_width - 1e-9]
    curves: list[tuple[float, np.ndarray, np.ndarray]] = []
    paths, _skipped = select_paths(paths)
    for path in paths:
        payload = pd.read_pickle(path)
        for cycle in payload["cycle_data"][::cycle_stride]:
            segment = charge_segment(cycle)
            if segment is None:
                continue
            soh = cycle_soh(payload, cycle, rated_ah)
            if not np.isfinite(soh) or soh <= 0.0:
                continue
            curves.append((soh, segment["voltage"], segment["charge"]))
    rows = []
    soh = np.array([c[0] for c in curves])
    for lo, hi in pairs:
        q_win = np.array([_window_integral(v, q, None, lo, hi)[0] for _, v, q in curves])
        ok = np.isfinite(q_win)
        coverage = q_win[ok] / rated_ah
        r2 = np.nan
        if ok.sum() >= 64:
            x = q_win[ok]
            y = soh[ok]
            design = np.column_stack([x, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            resid = y - design @ coef
            r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((y - y.mean()) ** 2))
        rows.append({"lo": lo, "hi": hi, "n": int(ok.sum()), "coverage_median": float(np.median(coverage)) if ok.any() else np.nan,
                     "coverage_p90": float(np.percentile(coverage, 90)) if ok.any() else np.nan,
                     "coverage_max": float(coverage.max()) if ok.any() else np.nan, "affine_r2": r2})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("ingest-batterylife-snl")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--chemistry", required=True, choices=["LFP", "NMC", "NCA"])
    parser.add_argument("--dataset-name", default=None, help="dataset column value (default snl_<chemistry>)")
    parser.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"), default=None)
    parser.add_argument("--out", default=None, help="write the table here (pickle)")
    parser.add_argument("--verify-against", default=None, help="existing table to compare row by row")
    parser.add_argument("--scan-windows", nargs=3, type=float, metavar=("LO_MIN", "HI_MAX", "STEP"), default=None)
    parser.add_argument("--rated-ah", type=float, default=None,
                        help="datasheet rated capacity: soh denominator and coverage reference (default: pickle nominal)")
    parser.add_argument("--report", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    paths = raw_paths(data_dir, args.chemistry)
    if not paths:
        raise SystemExit(f"no SNL {args.chemistry} pickles under {data_dir}")
    dataset = args.dataset_name or f"snl_{args.chemistry.lower()}"
    report: dict = {"chemistry": args.chemistry, "dataset": dataset, "files": len(paths)}

    if args.scan_windows:
        if args.rated_ah is None:
            raise SystemExit("--scan-windows needs --rated-ah")
        lo_min, hi_max, step = args.scan_windows
        scan = scan_windows(paths, args.rated_ah, lo_min, hi_max, step)
        candidates = scan[(scan["coverage_median"] >= 0.40) & (scan["coverage_median"] <= 0.50) & (scan["coverage_max"] < 0.90)]
        with pd.option_context("display.max_rows", 500, "display.width", 200):
            print(scan.to_string(index=False))
            print("\ncandidates with median coverage in [0.40, 0.50] and max < 0.90:")
            print(candidates.sort_values("affine_r2").to_string(index=False))
        report["window_scan"] = scan.to_dict(orient="records")

    if args.window is not None:
        window = (float(args.window[0]), float(args.window[1]))
        table, stats = build_table(paths, dataset, window, args.rated_ah)
        report["window"] = list(window)
        report["stats"] = stats
        rated = args.rated_ah or float(pd.read_pickle(paths[0])["nominal_capacity_in_Ah"])
        coverage = (table["q_win_ah"] / rated).dropna()
        report["coverage_median"] = float(coverage.median()) if len(coverage) else None
        report["coverage_max"] = float(coverage.max()) if len(coverage) else None
        print(json.dumps({k: v for k, v in report.items() if k != "window_scan"}, indent=2))
        if args.verify_against:
            ref = pd.read_pickle(args.verify_against)
            comparison = compare_tables(table, ref)
            report["verification"] = comparison
            print(json.dumps(comparison, indent=2))
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            table.to_pickle(out)
            print(f"wrote {out} ({len(table)} rows, {table['cell_id'].nunique()} cells)")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
