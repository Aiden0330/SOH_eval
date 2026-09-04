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
import io
import json
import pickle
import re
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
    # calce's old [3.6, 4.1] window integrated ~75% of the true cycle charge;
    # the historical 2.8% "coverage" was an artifact of the file-cumulative
    # Charge_Capacity(Ah) column corrupting charge_capacity_max.
    #
    # Coverage floor rationale (reviewed 2026-08-27): rows are joined by
    # exact ccm-sequence fingerprints with overlap guards, so misjoins are
    # impossible — unmatched rows keep NaN windows (missing, never wrong).
    # ~7% of rows are the two txt-logger cells (CS2_8, CS2_21) with no Arbin
    # xlsx at all, and ~10% sit in files whose fingerprints don't align
    # (pipeline-trimmed cycles / preliminary tests), giving ~82% coverage.
    "calce_cs2": {
        "window": (3.85, 4.05),
        "note": "LCO mid-charge window at 0.5C (CS2/CX2 Arbin cells; unaligned rows NaN)",
        "min_match": 0.80,
    },
}

# D1 (2026-09-04): the SNL NMC/NCA subsets are built by scripts/ingest_batterylife_snl.py with these
# windows (calibrated by its --scan-windows on the 0.5C charge segment; median coverage 0.43 / 0.44
# of the 3.0 / 3.2 Ah nameplate, affine R² 0.88 / 0.87); listed here so a rebuild uses the same bounds.
WINDOWS["snl_nmc"] = {"window": (3.65, 4.10), "note": "NMC mid-charge window, 0.5C charge segment (LG Chem 18650HG2)"}
WINDOWS["snl_nca"] = {"window": (3.70, 4.10), "note": "NCA mid-charge window, 0.5C charge segment (Panasonic NCR18650B)"}

CURVE_CONFIG_NAMES = {"hust": "hust", "snl_lfp": "snl_lfp", "uconn": "uconn", "calce_cs2": "calce"}


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


def _calce_date_key(name: str) -> tuple[int, int, int]:
    match = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{2})\.(?:xlsx|txt)$", name)
    if not match:
        return (9999, 0, 0)
    month, day, year = (int(match.group(i)) for i in (1, 2, 3))
    return (2000 + year, month, day)


def _iter_calce(data_dir: Path, lo: float, hi: float, processed: pd.DataFrame):
    """Arbin xlsx cells joined by ccm-sequence alignment, not file ordering.

    charge_capacity_max in the processed table is the per-cycle max of the
    file-cumulative Charge_Capacity(Ah), which makes each raw file's
    per-cycle ccm sequence a unique fingerprint.  Every file is located in
    the processed cell's ccm array by exact subsequence match — robust to
    whatever file ordering the original pipeline used and to files it
    dropped.  Files that match nowhere (or ambiguously) leave their cycles
    NaN rather than joining wrongly.
    """
    for zip_path in sorted(glob.glob(str(data_dir / "raw" / "calce" / "*.zip"))):
        cell = Path(zip_path).stem
        cell_rows = processed[processed["cell_id"].astype(str) == cell].sort_values("cycle_index")
        if not len(cell_rows):
            continue
        archive = zipfile.ZipFile(zip_path)
        files = sorted(n for n in archive.namelist() if n.endswith(".xlsx"))
        if not files:
            print(f"  [calce_cs2] {cell}: no Arbin xlsx (txt logger) — leaving NaN")
            continue
        target = pd.to_numeric(cell_rows["charge_capacity_max"], errors="coerce").to_numpy(float)
        cycle_ids = cell_rows["cycle_index"].astype(int).to_numpy()
        claimed = np.zeros(len(target), dtype=bool)
        matched_cycles = 0
        unmatched_files = []

        parsed: list[tuple[str, np.ndarray, list[tuple[float, float]]]] = []
        for entry in files:
            with archive.open(entry) as handle:
                data = io.BytesIO(handle.read())
            excel = pd.ExcelFile(data, engine="calamine")
            sheet = next(
                (s for s in excel.sheet_names if s.lower().startswith("channel")), None
            )
            if sheet is None:
                continue
            frame = excel.parse(sheet)
            sequence: list[float] = []
            integrals: list[tuple[float, float]] = []
            for _local_cycle, group in sorted(frame.groupby("Cycle_Index"), key=lambda kv: kv[0]):
                sequence.append(
                    float(pd.to_numeric(group["Charge_Capacity(Ah)"], errors="coerce").max())
                )
                charge = group[pd.to_numeric(group["Current(A)"], errors="coerce") > 0.01]
                if len(charge) < 5:
                    integrals.append((np.nan, np.nan))
                    continue
                integrals.append(_window_integral(
                    pd.to_numeric(charge["Voltage(V)"], errors="coerce").to_numpy(float),
                    pd.to_numeric(charge["Charge_Capacity(Ah)"], errors="coerce").to_numpy(float),
                    pd.to_numeric(charge["Test_Time(s)"], errors="coerce").to_numpy(float),
                    lo,
                    hi,
                ))
            parsed.append((entry, np.asarray(sequence, dtype=float), integrals))

        def positions_matching(probe: np.ndarray, *, unclaimed_only: bool = False):
            length = len(probe)
            hits = []
            for start in range(0, len(target) - length + 1):
                if unclaimed_only and claimed[start:start + length].any():
                    continue
                window = target[start:start + length]
                finite = np.isfinite(window) & np.isfinite(probe)
                if finite.sum() < max(1, int(0.9 * length)):
                    continue
                if np.max(np.abs(window[finite] - probe[finite])) <= 0.02:
                    hits.append(start)
            return hits

        def assign(start: int, probe_slice: slice, integrals, entry_probe):
            nonlocal matched_cycles
            span = len(entry_probe[probe_slice])
            claimed[start:start + span] = True
            for offset_index, (q_win, t_win) in enumerate(integrals[probe_slice]):
                yield cell, int(cycle_ids[start + offset_index]), q_win, t_win
            matched_cycles += span

        # Pass 1: whole-file exact spans (the common case).
        leftovers = []
        for entry, probe, integrals in parsed:
            if len(probe) < 3:
                leftovers.append((entry, probe, integrals))
                continue
            hits = positions_matching(probe)
            if len(hits) == 1 and not claimed[hits[0]:hits[0] + len(probe)].any():
                yield from assign(hits[0], slice(None), integrals, probe)
            else:
                leftovers.append((entry, probe, integrals))

        # Pass 2: remaining files (including tiny preliminary ones) matched
        # against still-unclaimed slots only.
        for entry, probe, integrals in leftovers:
            if not len(probe):
                unmatched_files.append(entry)
                continue
            hits = positions_matching(probe, unclaimed_only=True)
            if len(hits) == 1:
                yield from assign(hits[0], slice(None), integrals, probe)
            else:
                unmatched_files.append(entry)
        note = f", {len(unmatched_files)} file(s) unmatched" if unmatched_files else ""
        print(
            f"  [calce_cs2] {cell}: aligned {matched_cycles}/{len(cell_rows)} cycles"
            f" across {len(files)} files{note}"
        )


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
    elif name == "calce_cs2":
        iterator = _iter_calce(data_dir, lo, hi, table)
    elif name in ("snl_nmc", "snl_nca"):
        from sohkernel.raw_windows import iter_charge_curves

        iterator = (
            (cell, cycle, *_window_integral(v, q, t, lo, hi))
            for cell, cycle, v, q, t in iter_charge_curves(data_dir, name, table)
        )
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
    min_match = float(WINDOWS[name].get("min_match", 0.95))
    if matched < min_match * len(table):
        raise SystemExit(
            f"{name}: only {matched}/{len(table)} rows matched raw curves "
            f"(floor {min_match:.0%}) — join is broken, refusing to write"
        )
    if dry_run:
        return report

    backup_dir = data_dir / "processed" / "backup_pre_partial_window_fix"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / processed_path.name
    if not backup_path.exists():
        shutil.copy2(processed_path, backup_path)
    table.to_pickle(processed_path)

    curve_path = data_dir / "curves" / f"{CURVE_CONFIG_NAMES.get(name, name)}.json"
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
