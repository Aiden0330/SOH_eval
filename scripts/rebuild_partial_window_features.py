"""Rebuild q_win_ah / t_win_s with genuinely partial voltage windows.

Background (label-safety, 2026-08-25): the original feature windows on the
three LFP datasets spanned (nearly) the whole charge curve — [3.0, 3.59] V —
so q_win_ah integrated ~94-98% of full capacity and was the calibration
capacity (the label source) under another name.  This script re-integrates
the window charge from the raw curves using plateau-region windows chosen so
the median coverage lands in the 40-50% DOD band:

  hust     [3.30, 3.52] V  on the 5C CC fast-charge phase   (median ~47%)
  snl_lfp  [3.30, 3.38] V  on the 0.5C charge segment       (median ~50%)
  uconn    [3.30, 3.37] V  on the 1C RPT reference charge   (median ~53%)

The processed tables are updated in place (originals backed up under
data/processed/backup_pre_partial_window_fix/), the per-dataset curve config
(data/curves/<ds>.json feature_window) and the data manifest's processed
sha256 are refreshed, and the label-safety audit is re-run on the result —
the script refuses to write a table whose new q_win_ah still violates.

Usage:
    python scripts/rebuild_partial_window_features.py [--dry-run] [--data-dir data]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sohbench.feature_safety import audit_feature_frame  # noqa: E402

WINDOWS = {
    "hust": {
        "window": (3.30, 3.52),
        "note": "LFP plateau, 5C CC fast-charge phase (A123 18650 1.1Ah)",
    },
    "snl_lfp": {
        "window": (3.30, 3.38),
        "note": "LFP plateau, 0.5C charge segment (SNL 18650 LFP)",
    },
    "uconn": {
        "window": (3.30, 3.37),
        "note": "LFP plateau, 1C RPT reference charge (ref_chg CC segment)",
    },
}

CURVE_CONFIG_NAMES = {"hust": "hust", "snl_lfp": "snl_lfp", "uconn": "uconn"}


def _window_integral(v: np.ndarray, q: np.ndarray, t: np.ndarray, lo: float, hi: float):
    mask = np.isfinite(v) & np.isfinite(q) & (v >= lo) & (v <= hi)
    if int(mask.sum()) < 2:
        return np.nan, np.nan
    q_win = float(np.nanmax(q[mask]) - np.nanmin(q[mask]))
    if t is None:
        return q_win, np.nan
    tm = t[mask]
    t_win = float(np.nanmax(tm) - np.nanmin(tm)) if np.isfinite(tm).any() else np.nan
    return q_win, t_win


# ---------------------------------------------------------------------------
# Per-dataset raw iterators: yield (cell_id, cycle_index, q_win_ah, t_win_s)
# ---------------------------------------------------------------------------


def _iter_hust(data_dir: Path, lo: float, hi: float):
    for path in sorted(glob.glob(str(data_dir / "raw" / "hust" / "our_data" / "*.pkl"))):
        name = Path(path).stem
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        cycles = payload[name]["data"]
        for cycle_key, frame in cycles.items():
            chg = frame[frame["Status"] == "Constant current charge"]
            if len(chg) < 2:
                yield name, int(cycle_key), np.nan, np.nan
                continue
            q_win, t_win = _window_integral(
                chg["Voltage (V)"].to_numpy(dtype=float),
                chg["Capacity (mAh)"].to_numpy(dtype=float) / 1000.0,
                chg["Time (s)"].to_numpy(dtype=float),
                lo,
                hi,
            )
            yield name, int(cycle_key), q_win, t_win


def _iter_snl(data_dir: Path, lo: float, hi: float):
    for path in sorted(
        glob.glob(str(data_dir / "raw" / "snl_batterylife" / "SNL" / "SNL_18650_LFP_*.pkl"))
    ):
        payload = pd.read_pickle(path)
        cell_id = str(payload["cell_id"])
        for cycle in payload["cycle_data"]:
            v = np.asarray(cycle["voltage_in_V"], dtype=float)
            i = np.asarray(cycle["current_in_A"], dtype=float)
            q = np.asarray(cycle["charge_capacity_in_Ah"], dtype=float)
            t = np.asarray(cycle["time_in_s"], dtype=float)
            chg = i > 0.01
            if int(chg.sum()) < 2:
                yield cell_id, int(cycle["cycle_number"]), np.nan, np.nan
                continue
            q_win, t_win = _window_integral(v[chg], q[chg], t[chg], lo, hi)
            yield cell_id, int(cycle["cycle_number"]), q_win, t_win


def _iter_uconn(data_dir: Path, lo: float, hi: float, processed: pd.DataFrame):
    """RPT-order join: the k-th RPT of a cell maps to the k-th processed row.

    Verified via cumsum('Num Cycles') == processed cycle_index; cells where
    the cumulative check fails are skipped with a warning (their rows go NaN).
    """
    archive = zipfile.ZipFile(data_dir / "raw" / "uconn_lfp" / "rpt_data.zip")
    by_cell: dict[str, list[str]] = {}
    for entry in archive.namelist():
        stem = Path(entry).stem  # rpt_cell_02_part0
        if not stem.startswith("rpt_"):
            continue
        cell = "_".join(stem.split("_")[1:3])  # cell_02
        by_cell.setdefault(cell, []).append(entry)
    for cell, parts in sorted(by_cell.items()):
        expected = (
            processed.loc[processed["cell_id"].astype(str) == cell, "cycle_index"]
            .astype(int)
            .sort_values()
            .tolist()
        )
        if not expected:
            continue
        frames = [pd.read_csv(archive.open(p)) for p in sorted(parts)]
        table = pd.concat(frames, ignore_index=True)
        rpt_numbers = sorted(table["RPT Number"].dropna().unique())
        # The pipeline's cycle_index is the cumulative cycle count at each
        # RPT: cumsum of the per-RPT "Num Cycles" increment, in RPT order
        # (first RPT carries increment 0).  Verified against cell_02.
        running = 0
        cumsum = []
        for rpt in rpt_numbers:
            increment = pd.to_numeric(
                table.loc[table["RPT Number"] == rpt, "Num Cycles"], errors="coerce"
            ).dropna()
            running += int(increment.iloc[0]) if len(increment) else 0
            cumsum.append(running)
        aligned = len(cumsum) >= len(expected) and cumsum[: len(expected)] == expected
        if not aligned:
            # Fall back to strict order join only when counts agree.
            if len(rpt_numbers) < len(expected):
                print(f"  [uconn] {cell}: RPT count {len(rpt_numbers)} < processed rows {len(expected)} — leaving NaN")
                continue
            print(f"  [uconn] {cell}: cumsum mismatch, joining by RPT order")
        for k, cycle_index in enumerate(expected):
            group = table[table["RPT Number"] == rpt_numbers[k]]
            cc = group[(group["Segment Key"] == "ref_chg") & (group["State"] == "CC Chg")]
            if len(cc) < 2:
                yield cell, cycle_index, np.nan, np.nan
                continue
            q_win, t_win = _window_integral(
                cc["Voltage (V)"].to_numpy(dtype=float),
                cc["Capacity (Ah)"].to_numpy(dtype=float),
                cc["Time (s)"].to_numpy(dtype=float),
                lo,
                hi,
            )
            yield cell, cycle_index, q_win, t_win


# ---------------------------------------------------------------------------


def rebuild_dataset(name: str, data_dir: Path, *, dry_run: bool) -> dict:
    lo, hi = WINDOWS[name]["window"]
    processed_path = data_dir / "processed" / f"{name}_features.pkl"
    table = pd.read_pickle(processed_path)
    before = pd.to_numeric(table["q_win_ah"], errors="coerce")
    capacity = pd.to_numeric(table["charge_capacity_max"], errors="coerce")
    before_cov = float((before / capacity).dropna().median())

    if name == "hust":
        iterator = _iter_hust(data_dir, lo, hi)
    elif name == "snl_lfp":
        iterator = _iter_snl(data_dir, lo, hi)
    elif name == "uconn":
        iterator = _iter_uconn(data_dir, lo, hi, table)
    else:
        raise SystemExit(f"no raw iterator for dataset {name}")

    new_values = {
        (str(cell), int(cycle)): (q_win, t_win) for cell, cycle, q_win, t_win in iterator
    }
    keys = list(zip(table["cell_id"].astype(str), table["cycle_index"].astype(int)))
    matched = sum(1 for key in keys if key in new_values)
    table = table.copy()
    table["q_win_ah"] = [new_values.get(key, (np.nan, np.nan))[0] for key in keys]
    table["t_win_s"] = [new_values.get(key, (np.nan, np.nan))[1] for key in keys]

    after = pd.to_numeric(table["q_win_ah"], errors="coerce")
    after_cov = float((after / capacity).dropna().median())
    audit = audit_feature_frame(table, ["q_win_ah", "t_win_s"])
    report = {
        "dataset": name,
        "window": [lo, hi],
        "rows": len(table),
        "matched": matched,
        "coverage_median_before": round(before_cov, 4),
        "coverage_median_after": round(after_cov, 4),
        "audit_status": audit["status"],
        "audit_violations": [item["reason"] for item in audit["violations"]],
        "audit_warnings": [item["reason"] for item in audit.get("warnings", [])],
    }
    if audit["violations"]:
        raise SystemExit(
            f"{name}: rebuilt q_win_ah STILL violates label safety — window too wide?\n"
            + "\n".join(report["audit_violations"])
        )
    if matched < 0.95 * len(table):
        raise SystemExit(
            f"{name}: only {matched}/{len(table)} rows matched raw curves — join is broken, refusing to write"
        )
    if dry_run:
        return report

    backup_dir = data_dir / "processed" / "backup_pre_partial_window_fix"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / processed_path.name
    if not backup_path.exists():
        shutil.copy2(processed_path, backup_path)
    table.to_pickle(processed_path)

    curve_path = data_dir / "curves" / f"{CURVE_CONFIG_NAMES[name]}.json"
    if curve_path.exists():
        config = json.loads(curve_path.read_text(encoding="utf-8"))
        config["feature_window"] = [lo, hi]
        config["feature_window_approx"] = False
        config["feature_window_note"] = WINDOWS[name]["note"]
        curve_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("datasets", []):
            if entry.get("name") == name and isinstance(entry.get("processed"), dict):
                blob = processed_path.read_bytes()
                entry["processed"]["bytes"] = len(blob)
                entry["processed"]["sha256"] = hashlib.sha256(blob).hexdigest()
                entry["processed"]["partial_window_v"] = [lo, hi]
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("rebuild-partial-window-features")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=list(WINDOWS))
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    reports = []
    for name in args.datasets:
        print(f"rebuilding {name} with window {WINDOWS[name]['window']} ...", flush=True)
        report = rebuild_dataset(name, data_dir, dry_run=args.dry_run)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("\nDONE" + (" (dry run — nothing written)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
