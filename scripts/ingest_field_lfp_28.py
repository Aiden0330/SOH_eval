"""Build ``field_lfp_28_features.pkl`` — charging segments of the 28 portable LFP systems, unlabeled (D10, 2026-09-05).

Source: Schaeffer, J., Lenz, E., Gulla, D., Bazant, M. Z., Braatz, R. D., Findeisen, R. (2024), Gaussian
process-based online health monitoring and fault analysis of lithium-ion battery systems from field data,
Cell Reports Physical Science 5, 102258; data Zenodo 13715694, **CC BY-NC 4.0** (non-commercial).  28 portable
24 V systems of eight prismatic LFP cells in series (≈ 160 Ah), one load-current sensor, eight cell voltages,
four temperature sensors between cell pairs, active balancing; a month to five years of operation; all systems
were returned to the manufacturer for unsatisfactory behaviour (a biased sample).

The unit is the cell (``sys<ID>/cell<k>``): cell voltage from ``U_Cell_k``, current from the pack sensor (shared
by the string), temperature from the sensor of the cell's pair.  Rows are the charging segments of the pack
current (``sohkernel.segments``, Spec 12 §2 defaults with the ≈ 160 Ah nominal), one row per cell per segment.

There are **no labels** (no capacity tests): ``soh`` and the label columns are NaN in every row, the dataset's
supervision kind is ``unlabeled_field`` (Spec 12 §4) — inference, audit and drift only; the training preflight
refuses it and no claim can cite it.

Usage::

    python scripts/ingest_field_lfp_28.py --systems 3 14 --window 3.30 3.40 --out scratch/f28.pkl --report scratch/f28.json
    python scripts/ingest_field_lfp_28.py --window 3.30 3.40 --out data/processed/field_lfp_28_features.pkl --report scratch/f28.json
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
    FIELD28_C_RATED_AH, FIELD28_CELLS, field28_cell_temperature, field28_members, field28_unit, read_field28_system,
)
from sohkernel.segments import DEFAULT_CONFIG, SegmentConfig, finalize_table, segment_table  # noqa: E402

DATASET = "field_lfp_28"


def build(data_dir: Path, *, window: tuple[float, float] | None, config: SegmentConfig = DEFAULT_CONFIG,
          systems: set[int] | None = None, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    archive, members = field28_members(data_dir)
    tables, qc_rows = [], []
    for item in members:
        if systems is not None and item["system_id"] not in systems:
            continue
        frame = read_field28_system(archive, item["member"])
        t = frame["time_s"].to_numpy(dtype=float)
        i = frame["I_Battery"].to_numpy(dtype=float)
        dt = np.diff(t)
        for cell in range(1, FIELD28_CELLS + 1):
            column = f"U_Cell_{cell}"
            if column not in frame.columns:
                continue
            unit = field28_unit(item["system_id"], cell)
            table, qc = segment_table(unit, t, frame[column].to_numpy(dtype=float), i, field28_cell_temperature(frame, cell),
                                      c_rated_ah=FIELD28_C_RATED_AH, config=config, window=window, dataset=DATASET)
            tables.append(table)
            qc_rows.append({"cell_id": unit, "system_id": item["system_id"], "cell": cell, "rows": int(len(frame)), **qc,
                            "record_days": float((t[-1] - t[0]) / 86400.0) if len(t) > 1 else 0.0,
                            "sample_period_median_s": float(np.median(dt)) if len(dt) else np.nan})
        if verbose:
            last = [q for q in qc_rows if q["system_id"] == item["system_id"]]
            print(f"  sys{item['system_id']:02d}: {len(frame)} rows, {last[0]['record_days']:.0f} days, "
                  f"{last[0]['accepted']} segments per cell ({last[0]['too_short']} short, {last[0]['low_coverage']} low coverage)", flush=True)
    segments = finalize_table(pd.concat([t for t in tables if len(t)], ignore_index=True)) if any(len(t) for t in tables) else pd.DataFrame()
    if len(segments):
        segments = segments.sort_values(["cell_id", "segment_start_s"]).reset_index(drop=True)
        segments["cycle_index"] = segments.groupby("cell_id").cumcount()
    per_unit = pd.DataFrame(qc_rows)
    stats = {
        "systems": int(per_unit["system_id"].nunique()) if len(per_unit) else 0, "cells": int(segments["cell_id"].nunique()) if len(segments) else 0,
        "rows": int(len(segments)), "labeled_rows": int(segments["soh"].notna().sum()) if len(segments) else 0,
        "qc_totals": {k: int(per_unit[k].sum()) for k in ("segments_found", "too_short", "low_coverage", "accepted")} if len(per_unit) else {},
        "segments_per_system": {int(s): int(n) for s, n in per_unit.groupby("system_id")["accepted"].first().items()} if len(per_unit) else {},
        "record_days_per_system": {int(s): round(float(d), 1) for s, d in per_unit.groupby("system_id")["record_days"].first().items()} if len(per_unit) else {},
        "sample_period_median_s": {int(s): float(d) for s, d in per_unit.groupby("system_id")["sample_period_median_s"].first().items()} if len(per_unit) else {},
        "window": list(window) if window else None,
        "coverage_median": float((segments["q_win_ah"] / FIELD28_C_RATED_AH).median()) if len(segments) and segments["q_win_ah"].notna().any() else None,
        "segment_config": vars(config), "per_unit_qc": qc_rows,
    }
    return segments, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("ingest-field-lfp-28")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--systems", nargs="+", type=int, default=None)
    parser.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"), default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    window = (float(args.window[0]), float(args.window[1])) if args.window else None
    table, stats = build(Path(args.data_dir), window=window, systems=set(args.systems) if args.systems else None)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_unit_qc"}, indent=2, default=float))
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
