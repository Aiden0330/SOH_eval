"""Build ``turku_rc_<chem>_features.pkl`` from the Turku UAS randomized-current set (D3 re-scoped, 2026-09-05).

Source: Heydarzadeh, M., Toivola, T., Vega-Garita, V., Immonen, E. (2025), Dataset of lithium-ion cell degradation
under randomized current profiles for NMC, NCA, and LFP chemistries, Data in Brief 60, 111531 (CC BY 4.0);
data: FairData 10.23729/b07462e8-9e31-428c-bdd5-deb1890c608b (repository licence CC BY-NC 3.0).  Eight cells at
room temperature: four Samsung ICR18650-26J (NCA, 2.55 Ah, 2.75–4.2 V), two Samsung INR21700-40T (NMC, 4.0 Ah,
2.5–4.2 V), two JGNE JGPFR26650 (LFP, 3.0 Ah, 2.5–3.6 V); one dataset per chemistry.

Rows are the **"Random Walking" cycles**: every such entry discharges the cell with the randomized profile
(0.5–5 C setpoints, one minute each) to the cut-off, rests, and recharges it with the 1C CC-CV charge; the
row's feature source is that charge (``sohkernel.raw_windows.turku_charge_segment``).  Every 20 cycles the
protocol inserts HPPC tests and a **1C reference discharge** ("1C Reference" entries); the paper computes
SOH from those reference capacities.

Labels (Spec 12 §3, forward-only): ``soh`` = the discharged Ah of the last "1C Reference" entry that precedes
the cycle (its first constant-current discharge step, from the full state left by the previous entry's charge
to the cut-off) / nominal capacity; ``label_source`` = ``rpt``; ``label_time_s`` = the test time at the end of
that reference entry; cycles before the first reference carry no label.  ``charge_capacity_max`` = ΔQ of the
row's own charge (label-source column, banned as an input).  The test-time axis counts 1 Hz samples from the
cell's first entry (the files' start times are MATLAB string objects scipy cannot decode).

Usage::

    python scripts/ingest_turku_rc.py --dataset turku_rc_lfp --scan-windows 3.0 3.6 0.05
    python scripts/ingest_turku_rc.py --dataset turku_rc_lfp --window 3.30 3.40 --out data/processed/turku_rc_lfp_features.pkl --report scratch/turku_lfp.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sohkernel.raw_windows import (  # noqa: E402
    TURKU_CYCLE_COMMENT, TURKU_REFERENCE_COMMENT, TURKU_SUBSETS, ic_peak, iter_turku_records, turku_charge_segment,
    window_integral as _window_integral,
)
from sohkernel.segments import attach_forward_labels  # noqa: E402

CHEMISTRY = {"turku_rc_nca": "NCA", "turku_rc_nmc": "NMC", "turku_rc_lfp": "LFP"}
COLUMNS = [
    "dataset", "cell_id", "cycle_index", "soh", "time_s_max", "voltage_mean", "voltage_std",
    "voltage_min", "voltage_max", "current_mean", "current_std", "abs_current_mean",
    "temperature_mean", "charge_capacity_max", "q_win_ah", "t_win_s", "ic_peak_ah_per_v",
    "ic_peak_voltage_v", "cycle_index_feature", "cycle_fraction", "segment_start_s", "segment_end_s",
    "segment_kind", "sample_period_s", "label_source", "label_time_s", "label_age_s",
]
REFERENCE_RATE_TOLERANCE = 0.25   # the reference discharge runs at 1C: mean current within ±25 % of it


def reference_capacity(entry: dict, nominal_ah: float) -> dict:
    """Ah of the entry's first constant-current discharge at ~1C to the cut-off (NaN when absent)."""
    i = entry["current"]
    v = entry["voltage"]
    sign = np.sign(np.round(i, 2))
    breaks = np.flatnonzero(np.diff(sign) != 0) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(i)]])
    for a, b in zip(starts, ends):
        if sign[a] >= 0 or b - a < 60:
            continue
        mean_current = float(np.mean(i[a:b]))
        if abs(-mean_current / nominal_ah - 1.0) > REFERENCE_RATE_TOLERANCE:
            continue
        return {"capacity_ah": float(-np.trapezoid(i[a:b]) / 3600.0), "duration_s": float(b - a), "mean_current_a": mean_current,
                "v_start": float(v[a]), "v_end": float(v[b - 1]), "end_offset_s": float(b - 1)}
    return {"capacity_ah": np.nan}


def build(data_dir: Path, dataset: str, window: tuple[float, float] | None, *, cells: set[str] | None = None,
          verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    nominal = TURKU_SUBSETS[dataset][1]
    rows, tests, comments = [], [], {}
    for record in iter_turku_records(data_dir, dataset, cells):
        cell = record["cell_id"]
        comments.setdefault(cell, {}).setdefault(record["comment"], 0)
        comments[cell][record["comment"]] += 1
        if record["comment"] == TURKU_REFERENCE_COMMENT:
            found = reference_capacity(record, nominal)
            tests.append({"cell_id": cell, "entry": record["index"], "label_source": "rpt",
                          "label_time_s": record["offset_s"] + float(len(record["current"]) - 1), **found})
            continue
        if record["comment"] != TURKU_CYCLE_COMMENT:
            continue
        found = turku_charge_segment(record, nominal)
        if found is None:
            continue
        segment, (first, last) = found
        v, i, q, t, temp = (segment[k] for k in ("voltage", "current", "charge", "time", "temperature"))
        q_win, t_win = _window_integral(v, q, t, window[0], window[1]) if window else (np.nan, np.nan)
        ic_value, ic_voltage = ic_peak(v, q)
        rows.append({
            "dataset": dataset, "cell_id": cell, "cycle_index": -1, "soh": np.nan,
            "time_s_max": float(last - first), "voltage_mean": float(np.mean(v)), "voltage_std": float(np.std(v)),
            "voltage_min": float(np.min(v)), "voltage_max": float(np.max(v)), "current_mean": float(np.mean(i)),
            "current_std": float(np.std(i)), "abs_current_mean": float(np.mean(np.abs(i))),
            "temperature_mean": float(np.nanmean(temp)) if np.isfinite(temp).any() else np.nan,
            "charge_capacity_max": float(q[-1]), "q_win_ah": q_win, "t_win_s": t_win, "ic_peak_ah_per_v": ic_value,
            "ic_peak_voltage_v": ic_voltage, "cycle_index_feature": np.nan, "cycle_fraction": np.nan,
            "segment_start_s": record["offset_s"] + first, "segment_end_s": record["offset_s"] + last, "segment_kind": "charge",
            "sample_period_s": 1.0, "label_source": None, "label_time_s": np.nan, "label_age_s": np.nan, "entry": record["index"],
        })
        if verbose and len(rows) % 100 == 1:
            print(f"  {dataset} {cell}: entry {record['index']}", flush=True)
    table = pd.DataFrame(rows)
    tests_table = pd.DataFrame(tests)
    if len(table):
        usable = tests_table[np.isfinite(tests_table["capacity_ah"])] if len(tests_table) else tests_table
        table = attach_forward_labels(table, usable, c_rated_ah=nominal)
        table = table.sort_values(["cell_id", "segment_start_s"]).reset_index(drop=True)
        table["cycle_index"] = table.groupby("cell_id").cumcount()
        table["cycle_index_feature"] = table["cycle_index"].astype(float)
        table["cycle_fraction"] = table["cycle_index"] / table.groupby("cell_id")["cycle_index"].transform("max")
        entries = table.pop("entry")
        table = table[COLUMNS]
    stats = {
        "nominal_ah": nominal, "chemistry": CHEMISTRY[dataset], "cells": int(table["cell_id"].nunique()) if len(table) else 0,
        "rows": int(len(table)), "labeled_rows": int(table["soh"].notna().sum()) if len(table) else 0,
        "references": int(len(tests_table)), "references_with_capacity": int(np.isfinite(tests_table["capacity_ah"]).sum()) if len(tests_table) else 0,
        "references_per_cell": {c: int(n) for c, n in tests_table.groupby("cell_id").size().items()} if len(tests_table) else {},
        "entries_per_cell": comments,
        "soh_min": float(table["soh"].min()) if len(table) and table["soh"].notna().any() else None,
        "soh_max": float(table["soh"].max()) if len(table) and table["soh"].notna().any() else None,
        "label_age_cycles_median": None,
        "window": list(window) if window else None,
        "coverage_median": float((table["q_win_ah"] / nominal).median()) if len(table) and table["q_win_ah"].notna().any() else None,
        "reference_table": tests_table.to_dict(orient="records"),
    }
    return table, stats


def scan_windows(data_dir: Path, dataset: str, lo_min: float, hi_max: float, step: float, *, cycle_stride: int = 5,
                 min_width: float = 0.10, cells: set[str] | None = None) -> pd.DataFrame:
    nominal = TURKU_SUBSETS[dataset][1]
    table, _ = build(data_dir, dataset, None, cells=cells, verbose=False)
    curves = []
    for record in iter_turku_records(data_dir, dataset, cells):
        if record["comment"] != TURKU_CYCLE_COMMENT or record["index"] % cycle_stride:
            continue
        found = turku_charge_segment(record, nominal)
        if found is None:
            continue
        segment, (first, _last) = found
        match = table[(table["cell_id"] == record["cell_id"]) & (table["segment_start_s"] == record["offset_s"] + first)]
        if not len(match) or not np.isfinite(match["soh"].iloc[0]):
            continue
        curves.append((float(match["soh"].iloc[0]), segment["voltage"], segment["charge"]))
    los = np.round(np.arange(lo_min, hi_max - min_width + 1e-9, step), 3)
    his = np.round(np.arange(lo_min + min_width, hi_max + 1e-9, step), 3)
    pairs = [(lo, hi) for lo in los for hi in his if hi - lo >= min_width - 1e-9]
    soh = np.array([c[0] for c in curves])
    rows = []
    for lo, hi in pairs:
        q_win = np.array([_window_integral(v, q, None, lo, hi)[0] for _, v, q in curves])
        ok = np.isfinite(q_win)
        coverage = q_win[ok] / nominal
        r2 = np.nan
        if ok.sum() >= 64:
            design = np.column_stack([q_win[ok], np.ones(int(ok.sum()))])
            coef, *_ = np.linalg.lstsq(design, soh[ok], rcond=None)
            resid = soh[ok] - design @ coef
            r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((soh[ok] - soh[ok].mean()) ** 2))
        rows.append({"lo": lo, "hi": hi, "n": int(ok.sum()), "coverage_median": float(np.median(coverage)) if ok.any() else np.nan,
                     "coverage_p90": float(np.percentile(coverage, 90)) if ok.any() else np.nan,
                     "coverage_max": float(coverage.max()) if ok.any() else np.nan, "affine_r2": r2})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("ingest-turku-rc")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--dataset", required=True, choices=sorted(TURKU_SUBSETS))
    parser.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"), default=None)
    parser.add_argument("--scan-windows", nargs=3, type=float, metavar=("LO_MIN", "HI_MAX", "STEP"), default=None)
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    cells = set(args.cells) if args.cells else None
    report: dict = {"dataset": args.dataset, "nominal_ah": TURKU_SUBSETS[args.dataset][1], "chemistry": CHEMISTRY[args.dataset]}
    if args.scan_windows:
        lo_min, hi_max, step = args.scan_windows
        scan = scan_windows(data_dir, args.dataset, lo_min, hi_max, step, cells=cells)
        candidates = scan[(scan["coverage_median"] >= 0.40) & (scan["coverage_median"] <= 0.50) & (scan["coverage_max"] < 0.90)]
        with pd.option_context("display.max_rows", 500, "display.width", 200):
            print(scan.to_string(index=False))
            print("\ncandidates with median coverage in [0.40, 0.50] and max < 0.90:")
            print(candidates.sort_values("affine_r2").to_string(index=False))
        report["window_scan"] = scan.to_dict(orient="records")
    if args.window is not None:
        window = (float(args.window[0]), float(args.window[1]))
        table, stats = build(data_dir, args.dataset, window, cells=cells)
        report["window"] = list(window)
        report["stats"] = stats
        print(json.dumps({k: v for k, v in stats.items() if k not in ("reference_table",)}, indent=2, default=float))
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            table.to_pickle(out)
            print(f"wrote {out} ({len(table)} rows, {table['cell_id'].nunique() if len(table) else 0} cells)")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
