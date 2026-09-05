"""Canonical raw charge-curve access for window features.

``iter_charge_curves(data_dir, dataset, processed)`` yields
``(cell_id, cycle_index, v, q, t)`` for the charge segment of every raw
cycle that can be joined to the processed table.  Consumers integrate
whatever voltage windows they need:

- ``scripts/rebuild_partial_window_features.py`` — one calibrated window,
  written back into the processed tables;
- ``sohbench.window_bank`` — a full bounds lattice for the tuned-window
  search dimension.

Join guarantees (see the label-safety doc §6.3): hust/snl join by native
cycle ids; uconn by RPT-order cumsum verification; calce by exact
ccm-sequence fingerprint alignment with overlap guards — misjoins are
structurally impossible, unmatched cycles are simply not yielded.
"""

from __future__ import annotations

import glob
import io
import pickle
import re
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

Curve = tuple[str, int, np.ndarray, np.ndarray, np.ndarray]

IC_GRID_STEP_V = 0.001   # incremental-capacity curve on a uniform 1 mV voltage grid
IC_SMOOTH_POINTS = 61    # centred moving average, 61 mV wide (best agreement with the snl_lfp values)
IC_MIN_SAMPLES = 8


def ic_peak(voltage: np.ndarray, charge: np.ndarray) -> tuple[float, float]:
    """Peak of the smoothed dQ/dV curve of a charge segment (the per-cycle ``ic_peak_*`` columns).

    The charge curve is reduced to its monotone part (voltage running maximum; last charge value
    at each distinct voltage), resampled onto a uniform ``IC_GRID_STEP_V`` grid, differentiated
    (``np.gradient``) and smoothed with a centred ``IC_SMOOTH_POINTS``-wide moving average.  The
    peak is searched on the interior of the grid so the CC->CV corner cannot win.  Returns
    ``(ic_peak_ah_per_v, ic_peak_voltage_v)`` or NaNs when the segment is too short.  Defined here
    (D1, 2026-09-04, ``scripts/ingest_batterylife_snl.py``) so every extractor — per-cycle or per
    field segment (K13) — computes the same quantity.
    """
    voltage = np.asarray(voltage, dtype=float)
    charge = np.asarray(charge, dtype=float)
    finite = np.isfinite(voltage) & np.isfinite(charge)
    v = voltage[finite]
    q = charge[finite]
    if len(v) < IC_MIN_SAMPLES:
        return np.nan, np.nan
    v_mono = np.maximum.accumulate(v)
    frame = pd.DataFrame({"v": np.round(v_mono, 6), "q": q}).groupby("v", sort=True)["q"].last()
    v_u = frame.index.to_numpy(dtype=float)
    q_u = np.maximum.accumulate(frame.to_numpy(dtype=float))
    if len(v_u) < IC_MIN_SAMPLES or v_u[-1] - v_u[0] < IC_SMOOTH_POINTS * IC_GRID_STEP_V:
        return np.nan, np.nan
    grid = np.arange(v_u[0], v_u[-1] + IC_GRID_STEP_V / 2, IC_GRID_STEP_V)
    q_grid = np.interp(grid, v_u, q_u)
    ic = np.gradient(q_grid, IC_GRID_STEP_V)
    kernel = np.ones(IC_SMOOTH_POINTS) / IC_SMOOTH_POINTS
    smooth = np.convolve(ic, kernel, mode="same")
    half = IC_SMOOTH_POINTS // 2
    interior = slice(half, len(smooth) - half)
    if len(smooth[interior]) < 1:
        return np.nan, np.nan
    k = int(np.argmax(smooth[interior])) + half
    return float(smooth[k]), float(grid[k])


def window_integral(
    v: np.ndarray, q: np.ndarray, t: np.ndarray | None, lo: float, hi: float
) -> tuple[float, float]:
    """Charge and time accumulated while voltage sits inside [lo, hi]."""
    mask = np.isfinite(v) & np.isfinite(q) & (v >= lo) & (v <= hi)
    if int(mask.sum()) < 2:
        return np.nan, np.nan
    q_win = float(np.nanmax(q[mask]) - np.nanmin(q[mask]))
    if t is None:
        return q_win, np.nan
    tm = t[mask]
    t_win = float(np.nanmax(tm) - np.nanmin(tm)) if np.isfinite(tm).any() else np.nan
    return q_win, t_win


def _iter_hust(data_dir: Path, processed: pd.DataFrame | None) -> Iterator[Curve]:
    for path in sorted(glob.glob(str(data_dir / "raw" / "hust" / "our_data" / "*.pkl"))):
        name = Path(path).stem
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        cycles = payload[name]["data"]
        for cycle_key, frame in cycles.items():
            charge = frame[frame["Status"] == "Constant current charge"]
            if len(charge) < 2:
                continue
            yield (
                name,
                int(cycle_key),
                charge["Voltage (V)"].to_numpy(dtype=float),
                charge["Capacity (mAh)"].to_numpy(dtype=float) / 1000.0,
                charge["Time (s)"].to_numpy(dtype=float),
            )


def _iter_snl(data_dir: Path, processed: pd.DataFrame | None, chemistry: str = "LFP") -> Iterator[Curve]:
    """BatteryLife SNL pickles of one chemistry; ``processed`` (when given) restricts the cells to
    those in the table, so cells the ingestion skipped (partial-SOC NCA cells) are not yielded."""
    cells = None
    if processed is not None and "cell_id" in processed:
        cells = set(processed["cell_id"].astype(str))
    for path in sorted(
        glob.glob(str(data_dir / "raw" / "snl_batterylife" / "SNL" / f"SNL_18650_{chemistry}_*.pkl"))
    ):
        payload = pd.read_pickle(path)
        cell_id = str(payload["cell_id"])
        if cells is not None and cell_id not in cells:
            continue
        for cycle in payload["cycle_data"]:
            v = np.asarray(cycle["voltage_in_V"], dtype=float)
            i = np.asarray(cycle["current_in_A"], dtype=float)
            q = np.asarray(cycle["charge_capacity_in_Ah"], dtype=float)
            t = np.asarray(cycle["time_in_s"], dtype=float)
            charging = i > 0.01
            if int(charging.sum()) < 2:
                continue
            yield cell_id, int(cycle["cycle_number"]), v[charging], q[charging], t[charging]


def _iter_uconn(data_dir: Path, processed: pd.DataFrame | None) -> Iterator[Curve]:
    if processed is None:
        raise ValueError("uconn curves require the processed table for the RPT join")
    archive = zipfile.ZipFile(data_dir / "raw" / "uconn_lfp" / "rpt_data.zip")
    by_cell: dict[str, list[str]] = {}
    for entry in archive.namelist():
        stem = Path(entry).stem
        if not stem.startswith("rpt_"):
            continue
        cell = "_".join(stem.split("_")[1:3])
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
            if len(rpt_numbers) < len(expected):
                print(
                    f"  [uconn] {cell}: RPT count {len(rpt_numbers)} < processed rows "
                    f"{len(expected)} — skipping"
                )
                continue
            print(f"  [uconn] {cell}: cumsum mismatch, joining by RPT order")
        for k, cycle_index in enumerate(expected):
            group = table[table["RPT Number"] == rpt_numbers[k]]
            cc = group[(group["Segment Key"] == "ref_chg") & (group["State"] == "CC Chg")]
            if len(cc) < 2:
                continue
            yield (
                cell,
                int(cycle_index),
                cc["Voltage (V)"].to_numpy(dtype=float),
                cc["Capacity (Ah)"].to_numpy(dtype=float),
                cc["Time (s)"].to_numpy(dtype=float),
            )


def _iter_calce(data_dir: Path, processed: pd.DataFrame | None) -> Iterator[Curve]:
    """Arbin xlsx cells joined by ccm-sequence fingerprint alignment."""
    if processed is None:
        raise ValueError("calce curves require the processed table for the fingerprint join")
    for zip_path in sorted(glob.glob(str(data_dir / "raw" / "calce" / "*.zip"))):
        cell = Path(zip_path).stem
        cell_rows = processed[processed["cell_id"].astype(str) == cell].sort_values("cycle_index")
        if not len(cell_rows):
            continue
        archive = zipfile.ZipFile(zip_path)
        files = sorted(n for n in archive.namelist() if n.endswith(".xlsx"))
        if not files:
            print(f"  [calce_cs2] {cell}: no Arbin xlsx (txt logger) — skipping")
            continue
        target = pd.to_numeric(cell_rows["charge_capacity_max"], errors="coerce").to_numpy(float)
        cycle_ids = cell_rows["cycle_index"].astype(int).to_numpy()
        claimed = np.zeros(len(target), dtype=bool)
        matched_cycles = 0
        unmatched_files: list[str] = []

        parsed: list[tuple[str, np.ndarray, list[Curve]]] = []
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
            curves: list[Curve | None] = []
            for _local_cycle, group in sorted(frame.groupby("Cycle_Index"), key=lambda kv: kv[0]):
                sequence.append(
                    float(pd.to_numeric(group["Charge_Capacity(Ah)"], errors="coerce").max())
                )
                charge = group[pd.to_numeric(group["Current(A)"], errors="coerce") > 0.01]
                if len(charge) < 5:
                    curves.append(None)
                    continue
                curves.append((
                    cell,
                    -1,  # cycle id assigned after alignment
                    pd.to_numeric(charge["Voltage(V)"], errors="coerce").to_numpy(float),
                    pd.to_numeric(charge["Charge_Capacity(Ah)"], errors="coerce").to_numpy(float),
                    pd.to_numeric(charge["Test_Time(s)"], errors="coerce").to_numpy(float),
                ))
            parsed.append((entry, np.asarray(sequence, dtype=float), curves))

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

        def emit(start: int, curves):
            nonlocal matched_cycles
            claimed[start:start + len(curves)] = True
            for offset_index, curve in enumerate(curves):
                if curve is None:
                    continue
                yield (curve[0], int(cycle_ids[start + offset_index]), curve[2], curve[3], curve[4])
            matched_cycles += len(curves)

        leftovers = []
        for entry, probe, curves in parsed:
            if len(probe) < 3:
                leftovers.append((entry, probe, curves))
                continue
            hits = positions_matching(probe)
            if len(hits) == 1 and not claimed[hits[0]:hits[0] + len(probe)].any():
                yield from emit(hits[0], curves)
            else:
                leftovers.append((entry, probe, curves))
        for entry, probe, curves in leftovers:
            if not len(probe):
                unmatched_files.append(entry)
                continue
            hits = positions_matching(probe, unclaimed_only=True)
            if len(hits) == 1:
                yield from emit(hits[0], curves)
            else:
                unmatched_files.append(entry)
        note = f", {len(unmatched_files)} file(s) unmatched" if unmatched_files else ""
        print(
            f"  [calce_cs2] {cell}: aligned {matched_cycles}/{len(cell_rows)} cycles"
            f" across {len(files)} files{note}"
        )


# ---------------------------------------------------------------------------
# Tongji voltage-relaxation set (D2, 2026-09-05): Zhu et al. 2022, Zenodo 6405084.  One BioLogic CSV per
# cell inside the three dataset zips; columns time/s, control/V/mA, Ecell/V, <I>/mA, Q discharge/mA.h,
# Q charge/mA.h, control/V, control/mA, cycle number.  Charge and discharge quantities accumulate within a
# cycle.  The charge segment is every sample with <I>/mA above TONGJI_CHARGE_CURRENT_MIN_MA (CC and CV
# steps; the relaxation samples carry ~0 mA and are excluded).
# ---------------------------------------------------------------------------
TONGJI_SUBSETS = {
    "tongji_nca": ("Dataset_1_NCA_battery.zip", "Dataset_1_NCA_battery"),
    "tongji_ncm": ("Dataset_2_NCM_battery.zip", "Dataset_2_NCM_battery"),
    "tongji_blend": ("Dataset_3_NCM_NCA_battery.zip", "Dataset_3_NCM_NCA_battery"),
}
TONGJI_CHARGE_CURRENT_MIN_MA = 10.0
TONGJI_COLUMNS = ("time/s", "Ecell/V", "<I>/mA", "Q discharge/mA.h", "Q charge/mA.h", "cycle number")


def tongji_cell_members(data_dir: str | Path, dataset: str) -> tuple[zipfile.ZipFile, list[tuple[str, str]]]:
    """The dataset's zip and its ``(cell_id, member name)`` pairs, sorted by cell id."""
    archive_name, folder = TONGJI_SUBSETS[dataset]
    archive = zipfile.ZipFile(Path(data_dir) / "raw" / "tongji" / archive_name)
    members = []
    for name in archive.namelist():
        if name.startswith(folder + "/") and name.endswith(".csv"):
            members.append((Path(name).stem, name))
    return archive, sorted(members)


def read_tongji_cell(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as handle:
        frame = pd.read_csv(handle, usecols=list(TONGJI_COLUMNS))
    frame = frame.dropna(subset=["cycle number"])
    frame["cycle number"] = frame["cycle number"].astype(int)
    return frame


def tongji_charge_segment(cycle: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """Arrays of one cycle's charge segment (``<I>/mA`` above the threshold) or None below 2 samples."""
    current_ma = cycle["<I>/mA"].to_numpy(dtype=float)
    mask = current_ma > TONGJI_CHARGE_CURRENT_MIN_MA
    if int(mask.sum()) < 2:
        return None
    return {
        "voltage": cycle["Ecell/V"].to_numpy(dtype=float)[mask],
        "current": current_ma[mask] / 1000.0,
        "charge": cycle["Q charge/mA.h"].to_numpy(dtype=float)[mask] / 1000.0,
        "time": cycle["time/s"].to_numpy(dtype=float)[mask],
    }


def _iter_tongji(data_dir: Path, processed: pd.DataFrame | None, dataset: str) -> Iterator[Curve]:
    cells = set(processed["cell_id"].astype(str)) if processed is not None and "cell_id" in processed else None
    archive, members = tongji_cell_members(data_dir, dataset)
    for cell_id, member in members:
        if cells is not None and cell_id not in cells:
            continue
        frame = read_tongji_cell(archive, member)
        for cycle_number, cycle in frame.groupby("cycle number", sort=True):
            segment = tongji_charge_segment(cycle)
            if segment is None:
                continue
            yield cell_id, int(cycle_number), segment["voltage"], segment["charge"], segment["time"]


def _iter_tongji_nca(data_dir: Path, processed: pd.DataFrame | None) -> Iterator[Curve]:
    return _iter_tongji(data_dir, processed, "tongji_nca")


def _iter_tongji_ncm(data_dir: Path, processed: pd.DataFrame | None) -> Iterator[Curve]:
    return _iter_tongji(data_dir, processed, "tongji_ncm")


def _iter_tongji_blend(data_dir: Path, processed: pd.DataFrame | None) -> Iterator[Curve]:
    return _iter_tongji(data_dir, processed, "tongji_blend")


# ---------------------------------------------------------------------------
# Aalborg forklift-profile 180 Ah prismatic LFP set (D8, 2026-09-05): Vilsen & Stroe, Data in Brief 52 (2024)
# 109861, Mendeley yz4pttm73n v2.  One zip; Cell{1,2,3}/Round{00..58}/{Ageing,RPT}.csv with columns
# Cell, Round, [Part], Time (s, restarting in every file and every Part), Current (A, charge positive),
# Voltage (V), Energy (Wh), Temperature (°C).  A round is the two-week ageing profile (two parts) followed
# by an RPT; Round00 is the beginning-of-life RPT.  The kernel builds one monotone test-time axis per cell
# (seconds since the start of the cell's record) by concatenating the files in round / part / RPT order,
# so segments and labels order forward in time; wall-clock dates are not in the data (datasheet VERIFY).
# ---------------------------------------------------------------------------
FORKLIFT_ARCHIVE = Path("raw") / "forklift_lfp180" / "yz4pttm73n-2.zip"
FORKLIFT_C_RATED_AH = 180.0
FORKLIFT_CHAMBER_C = {"cell1": 45.0, "cell2": 40.0, "cell3": 35.0}
_FORKLIFT_MEMBER = re.compile(r"/Cell(?P<cell>\d+)/Round(?P<round>\d+)/(?P<kind>Ageing|RPT)\.csv$")


def forklift_members(data_dir: str | Path) -> tuple[zipfile.ZipFile, list[dict]]:
    """The archive and its members in record order: per cell, per round, Ageing before RPT."""
    archive = zipfile.ZipFile(Path(data_dir) / FORKLIFT_ARCHIVE)
    members = []
    for name in archive.namelist():
        match = _FORKLIFT_MEMBER.search(name)
        if match:
            members.append({"cell_id": f"cell{int(match['cell'])}", "round": int(match["round"]),
                            "kind": match["kind"].lower(), "member": name})
    order = {"ageing": 0, "rpt": 1}
    members.sort(key=lambda m: (m["cell_id"], m["round"], order[m["kind"]]))
    return archive, members


def read_forklift_csv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as handle:
        frame = pd.read_csv(handle)
    if "Part" not in frame.columns:
        frame["Part"] = 0
    return frame


def iter_forklift_records(data_dir: str | Path, cells: set[str] | None = None) -> Iterator[dict]:
    """Yield ``{cell_id, round, kind, part, offset_s, frame}`` per file part in record order.

    ``offset_s`` is the running test time at the part's first sample: every part is placed after the
    previous one (its local ``Time`` re-based to zero), with a one-second step between parts.  The gap
    between parts is unknown, so segments never span parts and rests at part boundaries are NaN.
    """
    archive, members = forklift_members(data_dir)
    offsets: dict[str, float] = {}
    for item in members:
        if cells is not None and item["cell_id"] not in cells:
            continue
        frame = read_forklift_csv(archive, item["member"])
        for part, group in frame.groupby("Part", sort=True):
            group = group.sort_values("Time", kind="stable").reset_index(drop=True)
            group["Time"] = group["Time"] - float(group["Time"].iloc[0])
            offset = offsets.get(item["cell_id"], 0.0)
            yield {"cell_id": item["cell_id"], "round": item["round"], "kind": item["kind"], "part": int(part),
                   "offset_s": offset, "frame": group}
            offsets[item["cell_id"]] = offset + float(group["Time"].iloc[-1]) + 1.0


def _iter_forklift(data_dir: Path, processed: pd.DataFrame | None) -> Iterator[Curve]:
    """Charge curves of the processed segments (matched by cell and segment start on the test-time axis)."""
    from .segments import DEFAULT_CONFIG, charging_spans, segment_curve

    if processed is None:
        raise ValueError("forklift_lfp180 curves require the processed table (segment bounds)")
    wanted = {(str(c), float(s)): int(k) for c, s, k in zip(processed["cell_id"], processed["segment_start_s"], processed["cycle_index"])}
    cells = {c for c, _ in wanted}
    for record in iter_forklift_records(data_dir, cells):
        if record["kind"] != "ageing":
            continue
        frame = record["frame"]
        t = frame["Time"].to_numpy(dtype=float)
        i = frame["Current"].to_numpy(dtype=float)
        v = frame["Voltage"].to_numpy(dtype=float)
        for first, last in charging_spans(t, i, FORKLIFT_C_RATED_AH, DEFAULT_CONFIG):
            key = (record["cell_id"], record["offset_s"] + t[first])
            if key not in wanted:
                continue
            v_seg, q_seg, t_rel = segment_curve(t, v, i, (first, last))
            yield record["cell_id"], wanted[key], v_seg, q_seg, t_rel


def _iter_snl_nmc(data_dir: Path, processed: pd.DataFrame | None) -> Iterator[Curve]:
    return _iter_snl(data_dir, processed, "NMC")


def _iter_snl_nca(data_dir: Path, processed: pd.DataFrame | None) -> Iterator[Curve]:
    return _iter_snl(data_dir, processed, "NCA")


CHARGE_CURVE_SOURCES = {
    "hust": _iter_hust,
    "snl_lfp": _iter_snl,
    "snl_nmc": _iter_snl_nmc,   # D1 (2026-09-04): scripts/ingest_batterylife_snl.py
    "snl_nca": _iter_snl_nca,
    "uconn": _iter_uconn,
    "calce_cs2": _iter_calce,
    "tongji_nca": _iter_tongji_nca,     # D2 (2026-09-05): scripts/ingest_tongji.py
    "tongji_ncm": _iter_tongji_ncm,
    "tongji_blend": _iter_tongji_blend,
    "forklift_lfp180": _iter_forklift,   # D8 (2026-09-05): scripts/ingest_forklift_lfp180.py (K13 segments)
}


def iter_charge_curves(
    data_dir: str | Path, dataset: str, processed: pd.DataFrame | None = None
) -> Iterator[Curve]:
    source = CHARGE_CURVE_SOURCES.get(dataset)
    if source is None:
        raise KeyError(f"no raw charge-curve source registered for dataset {dataset!r}")
    return source(Path(data_dir), processed)
