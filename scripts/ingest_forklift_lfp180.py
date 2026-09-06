"""Build ``forklift_lfp180_features.pkl`` — charging segments with RPT labels (D8, 2026-09-05; Spec 12).

Source: Vilsen, S. B. & Stroe, D.-I. (2024), Dataset of lithium-ion battery degradation based on a forklift
mission profile for state-of-health estimation and lifetime prediction, Data in Brief 52, 109861 — Mendeley
Data yz4pttm73n v2 (CC BY 4.0).  Three prismatic LFP/graphite cells, 180 Ah / 3.2 V nominal, 2.5–3.65 V,
aged with a two-week realistic forklift profile (currents limited to 50 A, idle periods removed) at 45 / 40 /
35 °C (Cell 1 / 2 / 3) with a reference performance test (RPT, 25 °C) after every round; 58 / 58 / 53 rounds.

Rows are the **charging segments of the ageing files** (``sohkernel.segments``, Spec 12 §2 defaults: current
above 0.02 C = 3.6 A, rests under 600 s bridged, at least 300 s and 5 % of 180 Ah = 9 Ah).  The unit is the
cell; ``cycle_index`` is the segment ordinal on the cell's test-time axis (``sohkernel.raw_windows.
iter_forklift_records``: files concatenated in round / part / RPT order, seconds since the record start).

Labels (Spec 12 §3, forward-only): ``soh`` = the C/4 capacity of the last RPT that precedes the segment /
180 Ah.  The C/4 capacity is the Ah of the RPT's constant-current 45 A discharge from the fully charged
state to 2.5 V (the paper's capacity measurement at C/4; the 90 A / C/2 discharge is recorded alongside as
``capacity_0p5c_ah`` in the report).  ``label_time_s`` is the test time at the end of that discharge, so
the segments of round N carry the RPT of round N-1 (or the beginning-of-life RPT of Round00 for round 1),
and a lost RPT (Table 2 of the paper) makes the following round's label one round older — never newer.

Usage::

    python scripts/ingest_forklift_lfp180.py --window 3.30 3.40 --out data/processed/forklift_lfp180_features.pkl --report scratch/forklift.json
    python scripts/ingest_forklift_lfp180.py --cells cell1 --rounds 0 5 --window 3.30 3.40 --out scratch/smoke.pkl
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

from sohkernel.raw_windows import FORKLIFT_C_RATED_AH, FORKLIFT_CHAMBER_C, iter_forklift_records  # noqa: E402
from sohkernel.segments import DEFAULT_CONFIG, SegmentConfig, attach_forward_labels, finalize_table, segment_table  # noqa: E402

DATASET = "forklift_lfp180"
V_MIN, V_MAX = 2.5, 3.65
RPT_CURRENT_TOLERANCE_A = 5.0
RPT_MIN_DURATION_S = 3600.0
RPT_FULL_CHARGE_V = 3.60  # a capacity discharge follows a CC-CV charge that reached the 3.65 V limit
RPT_END_V = 2.55          # ... and ends at the 2.5 V cut-off


def rpt_capacities(frame: pd.DataFrame) -> dict:
    """C/4 (45 A) and C/2 (90 A) capacities of one RPT file: the Ah of the constant-current discharge
    steps that run from the fully charged state (the preceding charge step reached ≥ 3.60 V, i.e. the CC-CV
    charge to 3.65 V — the RPT's first 45 A step only empties the cell from its ageing state and is not a
    capacity) to the 2.5 V cut-off; ``label_time_s`` is the local end time of the C/4 discharge.  NaN when a
    step is not found (the datasheet lists these rounds)."""
    t = frame["Time"].to_numpy(dtype=float)
    i = frame["Current"].to_numpy(dtype=float)
    v = frame["Voltage"].to_numpy(dtype=float)
    sign = np.sign(np.round(i, 1))
    breaks = np.flatnonzero(np.diff(sign) != 0) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(t)]])
    out = {"capacity_0p25c_ah": np.nan, "capacity_0p5c_ah": np.nan, "label_time_local_s": np.nan, "discharge_steps": []}
    last_charge_vmax = np.nan
    for a, b in zip(starts, ends):
        if sign[a] > 0:
            last_charge_vmax = float(np.max(v[a:b]))
            continue
        if sign[a] == 0 or t[b - 1] - t[a] < RPT_MIN_DURATION_S:
            continue
        mean_current = float(np.mean(i[a:b]))
        ah = float(-np.trapezoid(i[a:b], t[a:b]) / 3600.0)
        step = {"t0": float(t[a]), "duration_s": float(t[b - 1] - t[a]), "mean_current_a": mean_current,
                "v_start": float(v[a]), "v_end": float(v[b - 1]), "ah": ah, "preceding_charge_vmax": last_charge_vmax}
        out["discharge_steps"].append(step)
        if not (last_charge_vmax >= RPT_FULL_CHARGE_V) or v[b - 1] > RPT_END_V:
            continue
        if abs(mean_current + 45.0) <= RPT_CURRENT_TOLERANCE_A and not np.isfinite(out["capacity_0p25c_ah"]):
            out["capacity_0p25c_ah"] = ah
            out["label_time_local_s"] = float(t[b - 1])
        elif abs(mean_current + 90.0) <= RPT_CURRENT_TOLERANCE_A and not np.isfinite(out["capacity_0p5c_ah"]):
            out["capacity_0p5c_ah"] = ah
    return out


def build(data_dir: Path, *, window: tuple[float, float] | None, config: SegmentConfig = DEFAULT_CONFIG,
          cells: set[str] | None = None, rounds: tuple[int, int] | None = None, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    tables, tests, qc_rows = [], [], []
    for record in iter_forklift_records(data_dir, cells):
        if rounds is not None and not (rounds[0] <= record["round"] <= rounds[1]):
            continue
        frame = record["frame"]
        cell = record["cell_id"]
        if record["kind"] == "rpt":
            caps = rpt_capacities(frame)
            tests.append({"cell_id": cell, "round": record["round"], "capacity_ah": caps["capacity_0p25c_ah"],
                          "capacity_0p5c_ah": caps["capacity_0p5c_ah"],
                          "label_time_s": record["offset_s"] + caps["label_time_local_s"] if np.isfinite(caps["label_time_local_s"]) else np.nan,
                          "label_source": "rpt", "rpt_start_s": record["offset_s"], "rpt_duration_s": float(frame["Time"].iloc[-1]),
                          "rpt_temperature_mean_c": float(frame["Temperature"].mean()), "discharge_steps": caps["discharge_steps"]})
            if verbose:
                print(f"  {cell} round {record['round']:02d} RPT: C/4 {caps['capacity_0p25c_ah']:.2f} Ah, C/2 {caps['capacity_0p5c_ah']:.2f} Ah", flush=True)
            continue
        table, qc = segment_table(
            cell, frame["Time"].to_numpy(dtype=float), frame["Voltage"].to_numpy(dtype=float),
            frame["Current"].to_numpy(dtype=float), frame["Temperature"].to_numpy(dtype=float),
            c_rated_ah=FORKLIFT_C_RATED_AH, config=config, window=window, dataset=DATASET,
        )
        table["segment_start_s"] += record["offset_s"]
        table["segment_end_s"] += record["offset_s"]
        table["round"] = record["round"]
        table["part"] = record["part"]
        tables.append(table)
        qc_rows.append({"cell_id": cell, "round": record["round"], "part": record["part"], **qc,
                        "duration_s": float(frame["Time"].iloc[-1]), "chamber_c": FORKLIFT_CHAMBER_C[cell]})
        if verbose:
            print(f"  {cell} round {record['round']:02d} part {record['part']}: {qc['accepted']} segments "
                  f"({qc['too_short']} short, {qc['low_coverage']} low coverage) {qc['accepted_charge_ah']:.0f} Ah", flush=True)
    segments = finalize_table(pd.concat([t for t in tables if len(t)], ignore_index=True)) if any(len(t) for t in tables) else pd.DataFrame()
    tests_table = pd.DataFrame(tests)
    if len(segments):
        usable = tests_table[np.isfinite(tests_table["label_time_s"]) & np.isfinite(tests_table["capacity_ah"])]
        segments = attach_forward_labels(segments, usable, c_rated_ah=FORKLIFT_C_RATED_AH)
        segments = segments.sort_values(["cell_id", "segment_start_s"]).reset_index(drop=True)
        segments["cycle_index"] = segments.groupby("cell_id").cumcount()
        round_part = segments[["round", "part"]]
        segments = segments.drop(columns=["round", "part"])
    per_round = pd.DataFrame(qc_rows)
    stats = {
        "cells": int(segments["cell_id"].nunique()) if len(segments) else 0,
        "rows": int(len(segments)),
        "labeled_rows": int(segments["soh"].notna().sum()) if len(segments) else 0,
        "rpts": int(len(tests_table)), "rpts_with_c4_capacity": int(tests_table["capacity_ah"].notna().sum()) if len(tests_table) else 0,
        "rounds_per_cell": {c: int(n) for c, n in per_round.groupby("cell_id")["round"].nunique().items()} if len(per_round) else {},
        "rpts_per_cell": {c: int(n) for c, n in tests_table.groupby("cell_id")["round"].nunique().items()} if len(tests_table) else {},
        "segments_per_round": {c: float(v) for c, v in per_round.groupby("cell_id")["accepted"].sum().div(per_round.groupby("cell_id")["round"].nunique()).round(2).items()} if len(per_round) else {},
        "qc_totals": {k: int(per_round[k].sum()) for k in ("segments_found", "too_short", "low_coverage", "accepted")} if len(per_round) else {},
        "soh_min": float(segments["soh"].min()) if len(segments) else None, "soh_max": float(segments["soh"].max()) if len(segments) else None,
        "label_age_days_median": float(segments["label_age_s"].median() / 86400.0) if len(segments) else None,
        "window": list(window) if window else None,
        "coverage_median": float((segments["q_win_ah"] / FORKLIFT_C_RATED_AH).median()) if len(segments) and segments["q_win_ah"].notna().any() else None,
        "segment_config": vars(config),
        "rpt_table": [{k: (v if k != "discharge_steps" else v) for k, v in row.items()} for row in tests],
        "per_round_qc": qc_rows,
    }
    return segments, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("ingest-forklift-lfp180")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"), default=None)
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--rounds", nargs=2, type=int, metavar=("FIRST", "LAST"), default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    window = (float(args.window[0]), float(args.window[1])) if args.window else None
    table, stats = build(Path(args.data_dir), window=window, cells=set(args.cells) if args.cells else None,
                         rounds=tuple(args.rounds) if args.rounds else None)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("rpt_table", "per_round_qc")}, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_pickle(out)
        print(f"wrote {out} ({len(table)} rows, {table['cell_id'].nunique() if len(table) else 0} cells)")
    if args.report:
        Path(args.report).write_text(json.dumps({"dataset": DATASET, **stats}, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
