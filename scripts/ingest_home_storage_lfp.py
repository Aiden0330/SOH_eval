"""Build ``home_storage_lfp_features.pkl`` — charging segments of the RWTH ISEA home-storage LFP systems with
field capacity tests as labels (D7, 2026-09-05; Spec 12).

Source: Figgener, J. et al. (2024), Multi-year field measurements of home storage systems and their use in
capacity estimation, Nature Energy 9, 1438–1447 (DOI 10.1038/s41560-024-01620-9); data Zenodo 12091223
(CC BY 4.0): 21 PV home storage systems in Germany, 2015–2022, 1 s system-level current, voltage, power and
temperatures.  The eight LFP systems (manufacturer E, IDs 14–21; 158–300 Ah, 14 or 16 cells in series) are
onboarded; a system is the unit (``sys14`` …), its rated capacity and cells in series come from
``Metadata_Systems.xlsx``, and every voltage column is the mean cell voltage (pack voltage / cells in series).

Rows are the charging segments of the monthly files (``sohkernel.segments``, Spec 12 §2 defaults).

Labels (Spec 12 §3, forward-only): the field capacity tests.  ``Capacity_Tests.xlsx`` (the dataset) gives
the start time of every test; the paper's Source Data for Fig. 2 (``nature_energy_2024_source_data_fig2.xlsx``,
the authors' measured ``Capacity_Ah`` per test) gives the measured usable capacity, which is the label:
``soh = Capacity_Ah / Capacity_nominal_in_Ah``.  ``label_time_s`` is the end of the test discharge found in
the raw data (the full-power discharge that starts at the test's start time and runs to the end-of-discharge
voltage), so no segment before the end of a test carries that test.  ``--verify`` Coulomb-counts that
discharge from the raw data and compares it with the paper's value (the raw↔label check of Spec 06 §2.4).

Usage::

    python scripts/ingest_home_storage_lfp.py --systems 17 19 --window 3.30 3.40 --verify --out scratch/hs.pkl --report scratch/hs.json
    python scripts/ingest_home_storage_lfp.py --window 3.30 3.40 --verify --out data/processed/home_storage_lfp_features.pkl --report scratch/hs.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sohkernel.raw_windows import (  # noqa: E402
    HOME_STORAGE_LFP_SYSTEMS, HOME_STORAGE_RAW, home_storage_members, home_storage_metadata, home_storage_unit,
    read_home_storage_month,
)
from sohkernel.segments import DEFAULT_CONFIG, SegmentConfig, attach_forward_labels, finalize_table, segment_table  # noqa: E402

DATASET = "home_storage_lfp"
SOURCE_DATA_FILE = "nature_energy_2024_source_data_fig2.xlsx"   # Source Data Fig. 2 of the Nature Energy paper (MOESM4)
TEST_DISCHARGE_C = 0.10          # the test discharge runs at full inverter power: well above 0.1 C
TEST_REST_SPLIT_S = 900.0        # a pause longer than this ends the test discharge
TEST_SEARCH_WINDOW_S = 6 * 3600  # the discharge starts within six hours of the recorded start time
DEFAULT_MAX_LABEL_AGE_S = 180 * 86400.0   # Spec 12 §3 default for field capacity tests


def capacity_tests(data_dir: Path) -> pd.DataFrame:
    """Start times (Capacity_Tests.xlsx) joined with the paper's measured capacities (Source Data Fig. 2)."""
    archive = zipfile.ZipFile(data_dir / HOME_STORAGE_RAW / "Metadata_and_Code.zip")
    with archive.open("00_Data/00_Metadata/Capacity_Tests.xlsx") as handle:
        starts = pd.read_excel(io.BytesIO(handle.read()))
    starts = starts.rename(columns={"Start Time": "start", "ID": "system_id"})
    starts["start"] = pd.to_datetime(starts["start"])
    source = pd.read_excel(data_dir / HOME_STORAGE_RAW / SOURCE_DATA_FILE, sheet_name="a)-c)")
    source = source.rename(columns={"Time": "start", "ID": "system_id"})
    source["start"] = pd.to_datetime(source["start"])
    merged = starts.merge(source[["start", "system_id", "Chemistry", "Capacity_Ah", "Energy_kWh", "Age_a", "EFC_count"]],
                          on=["start", "system_id"], how="outer", indicator="merge_source")
    merged["cell_id"] = merged["system_id"].map(home_storage_unit)
    merged["start_s"] = (merged["start"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    return merged.sort_values(["system_id", "start"]).reset_index(drop=True)


def locate_test_discharge(frame: pd.DataFrame, start_s: float, rated_ah: float) -> dict:
    """The test discharge in one monthly frame: the discharging span (current below -TEST_DISCHARGE_C × rated,
    pauses under TEST_REST_SPLIT_S bridged) that begins within TEST_SEARCH_WINDOW_S after ``start_s``; returns
    its bounds, the Coulomb-counted Ah and the pack voltages at its ends."""
    t = frame["time_s"].to_numpy(dtype=float)
    i = frame["I_in_A"].to_numpy(dtype=float)
    v = frame["V_in_V"].to_numpy(dtype=float)
    window = (t >= start_s) & (t <= start_s + TEST_SEARCH_WINDOW_S)
    idx = np.flatnonzero(window & (i < -TEST_DISCHARGE_C * rated_ah))
    if len(idx) == 0:
        return {"found": False}
    first = int(idx[0])
    discharging = np.flatnonzero(i < -TEST_DISCHARGE_C * rated_ah)
    discharging = discharging[discharging >= first]
    gaps = np.diff(t[discharging])
    stop = np.flatnonzero(gaps > TEST_REST_SPLIT_S)
    last = int(discharging[stop[0]]) if len(stop) else int(discharging[-1])
    ah = float(-np.trapezoid(i[first:last + 1], t[first:last + 1]) / 3600.0)
    return {"found": True, "discharge_start_s": float(t[first]), "discharge_end_s": float(t[last]),
            "duration_s": float(t[last] - t[first]), "ah_counted": ah, "mean_current_a": float(np.mean(i[first:last + 1])),
            "v_pack_start": float(v[first]), "v_pack_end": float(v[last]),
            "v_pack_max_24h_before": float(np.max(v[(t >= start_s - 86400.0) & (t <= start_s)])) if np.any((t >= start_s - 86400.0) & (t <= start_s)) else np.nan}


def build(data_dir: Path, *, window: tuple[float, float] | None, config: SegmentConfig = DEFAULT_CONFIG,
          systems=HOME_STORAGE_LFP_SYSTEMS, max_label_age_s: float | None = DEFAULT_MAX_LABEL_AGE_S,
          verify: bool = True, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    meta = home_storage_metadata(data_dir)
    tests = capacity_tests(data_dir)
    tests = tests[tests["system_id"].isin(systems)].copy()
    members = home_storage_members(data_dir, systems)
    tables, qc_rows = [], []
    located: dict[int, dict] = {}
    for item in members:
        unit = item["cell_id"]
        rated = float(meta.loc[unit, "Capacity_nominal_in_Ah"])
        cells_in_series = int(meta.loc[unit, "Cell_number_in_series"])
        frame = read_home_storage_month(item)
        month_start, month_end = float(frame["time_s"].iloc[0]), float(frame["time_s"].iloc[-1])
        for row in tests[(tests["cell_id"] == unit) & (tests["start_s"] >= month_start) & (tests["start_s"] <= month_end)].itertuples():
            located[row.Index] = locate_test_discharge(frame, float(row.start_s), rated)
        table, qc = segment_table(unit, frame["time_s"].to_numpy(dtype=float), frame["V_in_V"].to_numpy(dtype=float),
                                  frame["I_in_A"].to_numpy(dtype=float), frame["T_Bat_in_C"].to_numpy(dtype=float),
                                  c_rated_ah=rated, config=config, cells_in_series=cells_in_series, window=window, dataset=DATASET)
        tables.append(table)
        qc_rows.append({"cell_id": unit, "year": item["year"], "month": item["month"], "rows": int(len(frame)),
                        "interpolated_fraction": float(frame["Interpolated"].mean()), **qc})
        if verbose:
            print(f"  {unit} {item['year']}-{item['month']:02d}: {qc['accepted']} segments ({qc['too_short']} short, "
                  f"{qc['low_coverage']} low coverage), {qc['accepted_charge_ah']:.0f} Ah", flush=True)
    segments = finalize_table(pd.concat([t for t in tables if len(t)], ignore_index=True)) if any(len(t) for t in tables) else pd.DataFrame()
    # label table: the paper's measured capacity, timed at the end of the located test discharge
    label_rows = []
    for row in tests.itertuples():
        found = located.get(row.Index, {"found": False})
        label_rows.append({"cell_id": row.cell_id, "system_id": int(row.system_id), "start": str(row.start), "start_s": float(row.start_s),
                           "capacity_ah": float(row.Capacity_Ah) if pd.notna(row.Capacity_Ah) else np.nan,
                           "in_capacity_tests_xlsx": row.merge_source in ("both", "left_only"), "in_source_data": row.merge_source in ("both", "right_only"),
                           "label_time_s": found.get("discharge_end_s", np.nan), "label_source": "capacity_test", **{k: v for k, v in found.items() if k != "found"},
                           "located": bool(found.get("found", False)),
                           "rated_ah": float(meta.loc[row.cell_id, "Capacity_nominal_in_Ah"]) if row.cell_id in meta.index else np.nan})
    labels = pd.DataFrame(label_rows)
    if len(labels):
        labels["ah_counted_vs_paper"] = labels["ah_counted"] / labels["capacity_ah"] if "ah_counted" in labels else np.nan
    label_columns = ["cell_id", "label_time_s", "capacity_ah", "label_source"]
    if len(labels):
        usable = labels[labels["located"] & np.isfinite(labels["capacity_ah"]) & np.isfinite(labels["label_time_s"])]
    else:   # a system without any capacity test (IDs 16 and 18): every segment stays unlabeled
        usable = pd.DataFrame(columns=label_columns)
    rated_by_unit = {unit: float(meta.loc[unit, "Capacity_nominal_in_Ah"]) for unit in meta.index}
    if len(segments):
        segments = attach_forward_labels(segments, usable[label_columns], c_rated_ah=rated_by_unit, max_label_age_s=max_label_age_s)
        segments = segments.sort_values(["cell_id", "segment_start_s"]).reset_index(drop=True)
        segments["cycle_index"] = segments.groupby("cell_id").cumcount()
    per_month = pd.DataFrame(qc_rows)
    stats = {
        "systems": int(segments["cell_id"].nunique()) if len(segments) else 0, "rows": int(len(segments)),
        "labeled_rows": int(segments["soh"].notna().sum()) if len(segments) else 0,
        "tests_listed": int(len(labels)), "tests_located": int(usable.shape[0]),
        "tests_per_unit": {u: int(n) for u, n in usable.groupby("cell_id").size().items()} if len(usable) else {},
        "months_per_unit": {u: int(n) for u, n in per_month.groupby("cell_id").size().items()} if len(per_month) else {},
        "qc_totals": {k: int(per_month[k].sum()) for k in ("segments_found", "too_short", "low_coverage", "accepted")} if len(per_month) else {},
        "soh_min": float(segments["soh"].min()) if len(segments) and segments["soh"].notna().any() else None,
        "soh_max": float(segments["soh"].max()) if len(segments) and segments["soh"].notna().any() else None,
        "label_age_days_median": float(segments["label_age_s"].median() / 86400.0) if len(segments) and segments["label_age_s"].notna().any() else None,
        "max_label_age_days": max_label_age_s / 86400.0 if max_label_age_s else None,
        "window": list(window) if window else None,
        "coverage_median": float((segments["q_win_ah"] / segments["cell_id"].map(rated_by_unit)).median()) if len(segments) and segments["q_win_ah"].notna().any() else None,
        "segment_config": vars(config),
        "verification": {
            "tests_compared": int(np.isfinite(usable["ah_counted_vs_paper"]).sum()) if len(usable) else 0,
            "ratio_counted_over_paper_median": float(usable["ah_counted_vs_paper"].median()) if len(usable) else None,
            "ratio_min": float(usable["ah_counted_vs_paper"].min()) if len(usable) else None,
            "ratio_max": float(usable["ah_counted_vs_paper"].max()) if len(usable) else None,
        } if verify else None,
        "label_table": labels.to_dict(orient="records"),
        "per_month_qc": qc_rows,
        "metadata": meta.reset_index().to_dict(orient="records"),
    }
    return segments, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("ingest-home-storage-lfp")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--systems", nargs="+", type=int, default=list(HOME_STORAGE_LFP_SYSTEMS))
    parser.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"), default=None)
    parser.add_argument("--max-label-age-days", type=float, default=180.0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    window = (float(args.window[0]), float(args.window[1])) if args.window else None
    table, stats = build(Path(args.data_dir), window=window, systems=tuple(args.systems),
                         max_label_age_s=args.max_label_age_days * 86400.0 if args.max_label_age_days else None, verify=args.verify)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("label_table", "per_month_qc", "metadata")}, indent=2, default=float))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_pickle(out)
        print(f"wrote {out} ({len(table)} rows, {table['cell_id'].nunique() if len(table) else 0} systems)")
    if args.report:
        Path(args.report).write_text(json.dumps({"dataset": DATASET, **stats}, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
