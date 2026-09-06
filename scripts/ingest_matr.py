"""Build ``matr_features.pkl`` from the Severson / Attia MATR batches (D9, 2026-09-05).

Source: Severson, K. A. et al. (2019), Data-driven prediction of battery cycle life before capacity
degradation, Nature Energy 4, 383–391 (batches 2017-05-12, 2017-06-30, 2018-04-12) and Attia, P. M. et al.
(2020), Closed-loop optimization of fast-charging protocols for batteries with machine learning, Nature 578,
397–402 (batch 2019-01-24) — data.matr.io, terms: cite the papers.  A123 APR18650M1A LFP/graphite, 1.1 Ah,
2.0–3.6 V, cycled at 30 °C with 72 (Severson) and 224 (Attia) fast-charging policies to 80 % SOC followed by
1C CC-CV to 3.6 V, and 4C discharge to 2.0 V.

Per-cycle definitions as in ``scripts/ingest_batterylife_snl.py`` (D1) on the charge segment (``I`` above
``sohkernel.raw_windows.MATR_CHARGE_CURRENT_MIN_A``):

* ``soh`` = the batch summary's ``QDischarge`` of the cycle / 1.1 Ah (cycles without a positive value are
  dropped — the first entry of every batch-1 cell is zero);
* ``charge_capacity_max`` = the summary's ``QCharge`` (label-source column, banned as an input);
* ``q_win_ah`` / ``t_win_s`` on the calibrated window; ``ic_peak_*`` from ``sohkernel.raw_windows.ic_peak``;
* cells and cycles follow the authors' cleaning (``sohkernel.raw_windows`` MATR notes): continued batch-1
  cells carry their batch-2 cycles with a running cycle index; excluded cells are absent.

``--verify`` checks the table against the raw batches on two independent quantities: the maximum of every
cycle's raw discharge array against the summary ``QDischarge`` the label uses, and the recomputed cycle
life against the batch ``cycle_life`` field.

Usage::

    python scripts/ingest_matr.py --scan-windows 3.0 3.6 0.05 --cycle-stride 25
    python scripts/ingest_matr.py --window 3.30 3.40 --out data/processed/matr_features.pkl --report scratch/matr.json --verify
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
    MATR_BATCHES, MATR_C_RATED_AH, ic_peak, iter_matr_cycles, matr_cells, matr_charge_segment,
    window_integral as _window_integral,
)

DATASET = "matr"
COLUMNS = [
    "dataset", "cell_id", "cycle_index", "soh", "time_s_max", "voltage_mean", "voltage_std",
    "voltage_min", "voltage_max", "current_mean", "current_std", "abs_current_mean",
    "temperature_mean", "charge_capacity_max", "q_win_ah", "t_win_s", "ic_peak_ah_per_v",
    "ic_peak_voltage_v", "cycle_index_feature", "cycle_fraction",
]


def cycle_row(cycle: dict, window: tuple[float, float]) -> dict | None:
    soh = cycle["q_discharge_ah"] / MATR_C_RATED_AH
    if not np.isfinite(soh) or soh <= 0.0:
        return None
    segment = matr_charge_segment(cycle["arrays"])
    if segment is None:
        return None
    v, i, q, t, temp = (segment[k] for k in ("voltage", "current", "charge", "time", "temperature"))
    q_win, t_win = _window_integral(v, q, t, window[0], window[1])
    ic_value, ic_voltage = ic_peak(v, q)
    return {
        "dataset": DATASET, "cell_id": cycle["cell_id"], "cycle_index": int(cycle["cycle_index"]), "soh": float(soh),
        "time_s_max": float(np.nanmax(t) - np.nanmin(t)),
        "voltage_mean": float(np.mean(v)), "voltage_std": float(np.std(v)), "voltage_min": float(np.min(v)), "voltage_max": float(np.max(v)),
        "current_mean": float(np.mean(i)), "current_std": float(np.std(i)), "abs_current_mean": float(np.mean(np.abs(i))),
        "temperature_mean": float(np.nanmean(temp)) if np.isfinite(temp).any() else np.nan,
        "charge_capacity_max": float(cycle["q_charge_ah"]) if np.isfinite(cycle["q_charge_ah"]) else np.nan,
        "q_win_ah": q_win, "t_win_s": t_win, "ic_peak_ah_per_v": ic_value, "ic_peak_voltage_v": ic_voltage,
        "cycle_index_feature": float(cycle["cycle_index"]),
    }


def build_table(data_dir: Path, window: tuple[float, float], *, cells: set[str] | None = None, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    rows, per_cell = [], {}
    for cycle in iter_matr_cycles(data_dir, cells):
        row = cycle_row(cycle, window)
        per_cell.setdefault(cycle["cell_id"], {"raw_cycles": 0, "rows": 0})["raw_cycles"] += 1
        if row:
            rows.append(row)
            per_cell[cycle["cell_id"]]["rows"] += 1
            if verbose and per_cell[cycle["cell_id"]]["rows"] == 1:
                print(f"  {cycle['cell_id']} ...", flush=True)
    table = pd.DataFrame(rows, columns=COLUMNS[:-1])
    table["cycle_fraction"] = table["cycle_index"] / table.groupby("cell_id")["cycle_index"].transform("max") if len(table) else pd.Series(dtype=float)
    before = len(table)
    exact_duplicates = int(table.duplicated(keep="first").sum())
    table = table.drop_duplicates(keep="first")
    identity_duplicates = int(table.duplicated(["cell_id", "cycle_index"], keep=False).sum())
    table = table.drop_duplicates(["cell_id", "cycle_index"], keep=False)
    table = table.sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    policies = {cell["cell_id"]: cell["policy"] for cell in matr_cells(data_dir)}
    stats = {
        "rated_ah_for_soh": MATR_C_RATED_AH, "cells": int(table["cell_id"].nunique()), "rows": int(len(table)),
        "rows_before_dedup": before, "exact_duplicate_rows_dropped": exact_duplicates, "ambiguous_identity_rows_dropped": identity_duplicates,
        "soh_min": float(table["soh"].min()) if len(table) else None, "soh_max": float(table["soh"].max()) if len(table) else None,
        "cells_per_batch": {batch: int(sum(1 for c in table["cell_id"].unique() if c.startswith(f"b{batch}"))) for batch in MATR_BATCHES},
        "per_cell": {cell: {**counts, "policy": policies.get(cell, "")} for cell, counts in per_cell.items()},
    }
    return table, stats


def verify_against_raw(data_dir: Path, table: pd.DataFrame, *, tolerance_ah: float = 1e-3) -> dict:
    """Two independent checks of the extractor against the raw batches.

    1. ``max(Qd)`` of every cycle's raw discharge array against the summary ``QDischarge`` the label uses
       (Spec 06 §2.4: raw↔processed match on an independent quantity; the writer refuses below 95 %);
    2. the cycle life recomputed from the table (first cycle with soh < 0.8, else the last cycle + 1 — the
       batches end at 80 %) against the batch ``cycle_life`` field.
    """
    import h5py

    from sohkernel.raw_windows import MATR_CONTINUATIONS, matr_cell_id

    cells = set(table["cell_id"].astype(str))
    raw_max = {}
    for cycle in iter_matr_cycles(data_dir, cells):
        qd = cycle["arrays"]["Qd"]
        raw_max[(cycle["cell_id"], int(cycle["cycle_index"]))] = float(np.nanmax(qd)) if np.isfinite(qd).any() else np.nan
    keyed = table.set_index(["cell_id", "cycle_index"])["soh"] * MATR_C_RATED_AH
    raw = np.array([raw_max.get(key, np.nan) for key in keyed.index])
    summary = keyed.to_numpy(dtype=float)
    finite = np.isfinite(raw) & np.isfinite(summary)
    diff = np.abs(raw[finite] - summary[finite])
    recorded = {}
    for batch in MATR_BATCHES:
        path = data_dir / "raw" / "matr" / f"MATR_batch_{batch}.mat"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            group = handle["batch"]
            for index in range(group["summary"].shape[0]):
                if (batch, index) in MATR_CONTINUATIONS:
                    continue
                life = np.asarray(handle[group["cycle_life"][index, 0]][()]).ravel()
                recorded[matr_cell_id(batch, index)] = float(life[0]) if len(life) else np.nan
    # a continued batch-1 cell's life is the batch-1 field plus its batch-2 cycles (the notebook's add_len)
    for (batch, index), (parent_batch, parent_index) in MATR_CONTINUATIONS.items():
        path = data_dir / "raw" / "matr" / f"MATR_batch_{batch}.mat"
        parent = matr_cell_id(parent_batch, parent_index)
        if path.is_file() and parent in recorded:
            with h5py.File(path, "r") as handle:
                recorded[parent] += float(handle[handle["batch"]["cycles"][index, 0]]["I"].shape[0])
    rows = []
    for cell, frame in table.groupby("cell_id"):
        below = frame.loc[frame["soh"] < 0.8, "cycle_index"]
        computed = float(below.min()) if len(below) else float(frame["cycle_index"].max() + 1)
        rows.append({"cell_id": cell, "recorded": recorded.get(cell, np.nan), "computed": computed})
    report = pd.DataFrame(rows)
    both = report[np.isfinite(report["recorded"]) & np.isfinite(report["computed"])]
    life_diff = (both["computed"] - both["recorded"]).abs()
    return {
        "raw_discharge_vs_summary": {"rows": int(len(keyed)), "compared": int(finite.sum()), "tolerance_ah": tolerance_ah,
                                     "match_rate": float((diff <= tolerance_ah).mean()) if finite.any() else None,
                                     "max_abs_diff_ah": float(diff.max()) if finite.any() else None},
        "cycle_life": {"cells": int(len(report)), "compared": int(len(both)), "within_1_cycle": int((life_diff <= 1).sum()),
                       "within_5_cycles": int((life_diff <= 5).sum()), "max_abs_diff": float(life_diff.max()) if len(life_diff) else None,
                       "mismatches": both[life_diff > 1].to_dict(orient="records")},
    }


def scan_windows(data_dir: Path, lo_min: float, hi_max: float, step: float, *, cycle_stride: int = 25, min_width: float = 0.10,
                 cells: set[str] | None = None) -> pd.DataFrame:
    los = np.round(np.arange(lo_min, hi_max - min_width + 1e-9, step), 3)
    his = np.round(np.arange(lo_min + min_width, hi_max + 1e-9, step), 3)
    pairs = [(lo, hi) for lo in los for hi in his if hi - lo >= min_width - 1e-9]
    curves = []
    for cycle in iter_matr_cycles(data_dir, cells):
        if cycle["cycle_index"] % cycle_stride:
            continue
        soh = cycle["q_discharge_ah"] / MATR_C_RATED_AH
        segment = matr_charge_segment(cycle["arrays"])
        if segment is None or not np.isfinite(soh) or soh <= 0.0:
            continue
        curves.append((soh, segment["voltage"], segment["charge"]))
    soh = np.array([c[0] for c in curves])
    rows = []
    for lo, hi in pairs:
        q_win = np.array([_window_integral(v, q, None, lo, hi)[0] for _, v, q in curves])
        ok = np.isfinite(q_win)
        coverage = q_win[ok] / MATR_C_RATED_AH
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
    parser = argparse.ArgumentParser("ingest-matr")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"), default=None)
    parser.add_argument("--scan-windows", nargs=3, type=float, metavar=("LO_MIN", "HI_MAX", "STEP"), default=None)
    parser.add_argument("--cycle-stride", type=int, default=25)
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--verify", action="store_true", help="compare recomputed cycle lives with the batch cycle_life field")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    cells = set(args.cells) if args.cells else None
    report: dict = {"dataset": DATASET, "rated_ah": MATR_C_RATED_AH}
    if args.scan_windows:
        lo_min, hi_max, step = args.scan_windows
        scan = scan_windows(data_dir, lo_min, hi_max, step, cycle_stride=args.cycle_stride, cells=cells)
        candidates = scan[(scan["coverage_median"] >= 0.40) & (scan["coverage_median"] <= 0.50) & (scan["coverage_max"] < 0.90)]
        with pd.option_context("display.max_rows", 500, "display.width", 200):
            print(scan.to_string(index=False))
            print("\ncandidates with median coverage in [0.40, 0.50] and max < 0.90:")
            print(candidates.sort_values("affine_r2").to_string(index=False))
        report["window_scan"] = scan.to_dict(orient="records")
    if args.window is not None:
        window = (float(args.window[0]), float(args.window[1]))
        table, stats = build_table(data_dir, window, cells=cells)
        report["window"] = list(window)
        report["stats"] = stats
        coverage = (table["q_win_ah"] / MATR_C_RATED_AH).dropna()
        report["coverage_median"] = float(coverage.median()) if len(coverage) else None
        report["coverage_max"] = float(coverage.max()) if len(coverage) else None
        if args.verify:
            report["verification"] = verify_against_raw(data_dir, table)
        print(json.dumps({k: v for k, v in report.items() if k not in ("window_scan", "stats")} | {"stats": {k: v for k, v in stats.items() if k != "per_cell"}}, indent=2))
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            table.to_pickle(out)
            print(f"wrote {out} ({len(table)} rows, {table['cell_id'].nunique()} cells)")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
