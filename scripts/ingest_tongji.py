"""Build ``tongji_<subset>_features.pkl`` tables from the Tongji voltage-relaxation set (D2, 2026-09-05).

Source: Zhu, J. et al. (2022), Data-driven capacity estimation of commercial lithium-ion batteries from
voltage relaxation, Nature Communications 13, 2261 — Zenodo 6405084 (CC BY 4.0).  Three subsets, one
dataset each (the kernel keys chemistry by dataset):

* ``tongji_nca``   Dataset_1: LG INR18650-35E, NCA (LiNi0.86Co0.11Al0.03O2), 3.5 Ah, 2.65–4.2 V, 66 cells;
* ``tongji_ncm``   Dataset_2: Samsung INR18650-MJ1, NCM (LiNi0.83Co0.11Mn0.07O2), 3.5 Ah, 2.5–4.2 V, 55 cells;
* ``tongji_blend`` Dataset_3: Samsung INR18650-25R, NCM+NCA blend cathode, 2.5 Ah, 2.5–4.2 V, 9 cells.

Cells are named ``CY<T>-<charge C-rate>_<discharge C-rate>-#<tag>`` by the authors (``05`` = 0.5C, ``025`` =
0.25C); the chamber temperature T (25/35/45 °C) is the only temperature available — the CSVs carry no
temperature channel — so ``temperature_mean`` is the chamber setpoint.  Every cycle is CC charge (0.25–1C)
to 4.2 V, CV to 0.05C, a 30 min rest (60 min for the blend), CC discharge (1–4C) to the cut-off, rest.

Per-cycle definitions follow ``scripts/ingest_batterylife_snl.py`` (the D1 extractor) on the charge
segment (``<I>/mA`` above ``sohkernel.raw_windows.TONGJI_CHARGE_CURRENT_MIN_MA``, i.e. the CC and CV steps):

* ``soh`` = max ``Q discharge/mA.h`` of the cycle / 1000 / rated capacity (the CC discharge capacity the
  paper treats as the residual capacity); cycles without a positive discharge capacity carry no label and
  are dropped;
* ``charge_capacity_max`` = max ``Q charge/mA.h`` / 1000 (label-source column, banned as an input);
* ``q_win_ah`` / ``t_win_s`` on the calibrated ``[lo, hi]`` window; ``ic_peak_*`` from
  ``sohkernel.raw_windows.ic_peak``; ``cycle_index`` = the file's cycle number.

Usage::

    python scripts/ingest_tongji.py --dataset tongji_nca --scan-windows 3.4 4.2 0.05
    python scripts/ingest_tongji.py --dataset tongji_nca --window 3.70 4.10 --out data/processed/tongji_nca_features.pkl --report scratch/tongji_nca.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sohkernel.raw_windows import (  # noqa: E402
    TONGJI_SUBSETS, ic_peak, read_tongji_cell, tongji_cell_members, tongji_charge_segment,
    window_integral as _window_integral,
)

RATED_AH = {"tongji_nca": 3.5, "tongji_ncm": 3.5, "tongji_blend": 2.5}   # Zhu et al. 2022, Table 1
CHEMISTRY = {"tongji_nca": "NCA", "tongji_ncm": "NCM", "tongji_blend": "NCM+NCA"}
COLUMNS = [
    "dataset", "cell_id", "cycle_index", "soh", "time_s_max", "voltage_mean", "voltage_std",
    "voltage_min", "voltage_max", "current_mean", "current_std", "abs_current_mean",
    "temperature_mean", "charge_capacity_max", "q_win_ah", "t_win_s", "ic_peak_ah_per_v",
    "ic_peak_voltage_v", "cycle_index_feature", "cycle_fraction",
]
_NAME = re.compile(r"^CY(?P<temp>\d+)-(?P<charge>[\d.]+)_(?P<discharge>[\d.]+)-#(?P<tag>\d+)$")


def parse_cell_name(cell_id: str) -> dict:
    """``CY25-05_1-#3`` → chamber 25 °C, 0.5C charge, 1C discharge, tag 3 (the authors' convention)."""
    match = _NAME.match(cell_id)
    if not match:
        raise ValueError(f"unexpected Tongji cell name {cell_id!r}")
    rate = lambda token: float("0." + token[1:]) if token.startswith("0") and "." not in token else float(token)  # noqa: E731
    return {"chamber_temp_c": float(match["temp"]), "charge_c": rate(match["charge"]),
            "discharge_c": rate(match["discharge"]), "tag": int(match["tag"])}


def cycle_row(dataset: str, cell_id: str, cycle_number: int, cycle: pd.DataFrame, window: tuple[float, float],
              rated_ah: float, chamber_temp_c: float) -> dict | None:
    segment = tongji_charge_segment(cycle)
    if segment is None:
        return None
    discharge = cycle["Q discharge/mA.h"].to_numpy(dtype=float)
    if not np.isfinite(discharge).any():
        return None
    soh = float(np.nanmax(discharge)) / 1000.0 / rated_ah
    if not np.isfinite(soh) or soh <= 0.0:
        return None
    charge_full = cycle["Q charge/mA.h"].to_numpy(dtype=float)
    v, i, q, t = (segment[k] for k in ("voltage", "current", "charge", "time"))
    q_win, t_win = _window_integral(v, q, t, window[0], window[1])
    ic_value, ic_voltage = ic_peak(v, q)
    return {
        "dataset": dataset,
        "cell_id": cell_id,
        "cycle_index": int(cycle_number),
        "soh": soh,
        "time_s_max": float(np.nanmax(t) - np.nanmin(t)),
        "voltage_mean": float(np.mean(v)),
        "voltage_std": float(np.std(v)),
        "voltage_min": float(np.min(v)),
        "voltage_max": float(np.max(v)),
        "current_mean": float(np.mean(i)),
        "current_std": float(np.std(i)),
        "abs_current_mean": float(np.mean(np.abs(i))),
        "temperature_mean": chamber_temp_c,
        "charge_capacity_max": float(np.nanmax(charge_full)) / 1000.0 if np.isfinite(charge_full).any() else np.nan,
        "q_win_ah": q_win,
        "t_win_s": t_win,
        "ic_peak_ah_per_v": ic_value,
        "ic_peak_voltage_v": ic_voltage,
        "cycle_index_feature": float(cycle_number),
    }


def extract_cell(dataset: str, cell_id: str, frame: pd.DataFrame, window: tuple[float, float]) -> pd.DataFrame:
    meta = parse_cell_name(cell_id)
    rows = [
        row for row in (
            cycle_row(dataset, cell_id, int(number), cycle, window, RATED_AH[dataset], meta["chamber_temp_c"])
            for number, cycle in frame.groupby("cycle number", sort=True)
        ) if row
    ]
    table = pd.DataFrame(rows, columns=COLUMNS[:-1])
    table["cycle_fraction"] = table["cycle_index"] / float(table["cycle_index"].max()) if len(table) else pd.Series(dtype=float)
    return table


def build_table(data_dir: Path, dataset: str, window: tuple[float, float], *, cells: set[str] | None = None,
                verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    archive, members = tongji_cell_members(data_dir, dataset)
    frames, per_cell = [], []
    for cell_id, member in members:
        if cells is not None and cell_id not in cells:
            continue
        frame = read_tongji_cell(archive, member)
        table = extract_cell(dataset, cell_id, frame, window)
        frames.append(table)
        per_cell.append({"cell_id": cell_id, **parse_cell_name(cell_id), "raw_cycles": int(frame["cycle number"].nunique()),
                         "rows": int(len(table)), "soh_first": float(table["soh"].iloc[0]) if len(table) else None,
                         "soh_last": float(table["soh"].iloc[-1]) if len(table) else None})
        if verbose:
            print(f"  {dataset} {cell_id}: {per_cell[-1]['raw_cycles']} cycles -> {len(table)} rows", flush=True)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    table = table[COLUMNS]
    before = len(table)
    exact_duplicates = int(table.duplicated(keep="first").sum())
    table = table.drop_duplicates(keep="first")
    identity_duplicates = int(table.duplicated(["cell_id", "cycle_index"], keep=False).sum())
    table = table.drop_duplicates(["cell_id", "cycle_index"], keep=False)
    table = table.sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    conditions = pd.DataFrame(per_cell).groupby(["chamber_temp_c", "charge_c", "discharge_c"])["cell_id"].count() if per_cell else pd.Series(dtype=int)
    stats = {
        "rated_ah_for_soh": RATED_AH[dataset], "chemistry": CHEMISTRY[dataset],
        "cells": int(table["cell_id"].nunique()), "rows": int(len(table)), "rows_before_dedup": before,
        "exact_duplicate_rows_dropped": exact_duplicates, "ambiguous_identity_rows_dropped": identity_duplicates,
        "soh_min": float(table["soh"].min()) if len(table) else None, "soh_max": float(table["soh"].max()) if len(table) else None,
        "cells_per_condition": {f"{int(t)}C_{c}C_{d}C": int(n) for (t, c, d), n in conditions.items()},
        "per_cell": per_cell,
    }
    return table, stats


def scan_windows(data_dir: Path, dataset: str, lo_min: float, hi_max: float, step: float, *, cycle_stride: int = 10,
                 min_width: float = 0.10, cells: set[str] | None = None) -> pd.DataFrame:
    """Coverage median and affine R² (soh ~ q_win) for a grid of [lo, hi] windows (feature-window-calibration skill §2)."""
    rated = RATED_AH[dataset]
    los = np.round(np.arange(lo_min, hi_max - min_width + 1e-9, step), 3)
    his = np.round(np.arange(lo_min + min_width, hi_max + 1e-9, step), 3)
    pairs = [(lo, hi) for lo in los for hi in his if hi - lo >= min_width - 1e-9]
    curves = []
    archive, members = tongji_cell_members(data_dir, dataset)
    for cell_id, member in members:
        if cells is not None and cell_id not in cells:
            continue
        frame = read_tongji_cell(archive, member)
        for k, (number, cycle) in enumerate(frame.groupby("cycle number", sort=True)):
            if k % cycle_stride:
                continue
            segment = tongji_charge_segment(cycle)
            if segment is None:
                continue
            discharge = cycle["Q discharge/mA.h"].to_numpy(dtype=float)
            if not np.isfinite(discharge).any():
                continue
            soh = float(np.nanmax(discharge)) / 1000.0 / rated
            if soh <= 0.0:
                continue
            curves.append((soh, segment["voltage"], segment["charge"]))
    soh = np.array([c[0] for c in curves])
    rows = []
    for lo, hi in pairs:
        q_win = np.array([_window_integral(v, q, None, lo, hi)[0] for _, v, q in curves])
        ok = np.isfinite(q_win)
        coverage = q_win[ok] / rated
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
    parser = argparse.ArgumentParser("ingest-tongji")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--dataset", required=True, choices=sorted(TONGJI_SUBSETS))
    parser.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"), default=None)
    parser.add_argument("--out", default=None, help="write the table here (pickle)")
    parser.add_argument("--scan-windows", nargs=3, type=float, metavar=("LO_MIN", "HI_MAX", "STEP"), default=None)
    parser.add_argument("--cells", nargs="+", default=None, help="restrict to these cell ids (smoke tests)")
    parser.add_argument("--report", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    cells = set(args.cells) if args.cells else None
    report: dict = {"dataset": args.dataset, "rated_ah": RATED_AH[args.dataset], "chemistry": CHEMISTRY[args.dataset]}
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
        table, stats = build_table(data_dir, args.dataset, window, cells=cells)
        report["window"] = list(window)
        report["stats"] = stats
        coverage = (table["q_win_ah"] / RATED_AH[args.dataset]).dropna()
        report["coverage_median"] = float(coverage.median()) if len(coverage) else None
        report["coverage_max"] = float(coverage.max()) if len(coverage) else None
        print(json.dumps({k: v for k, v in report.items() if k not in ("window_scan",) and k != "stats"} | {"stats": {k: v for k, v in stats.items() if k != "per_cell"}}, indent=2))
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
