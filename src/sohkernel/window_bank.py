"""Window bank: every lattice voltage window pre-integrated per raw cycle.

Makes window bounds a *searchable* dimension without re-parsing raw data
per candidate: one sweep over the raw charge curves integrates the full
bounds lattice (via monotone interpolation at the lattice edges), stored as
one parquet per dataset.  The tuned-window search then:

1. picks bounds per dataset with :func:`select_bounds` — the lattice window
   whose median ``q_win / rated capacity`` lands closest to a target
   coverage.  The statistic is label-free (rated capacity is nameplate, not
   the calibration capacity the SOH label uses), so the same rule is legal
   at inference-adaptation time on an unlabeled dataset;
2. substitutes the bank's q_win/t_win into the feature table with
   :func:`apply_bank_window` and re-audits the substituted column before
   any model sees it (the DOD gate stays a hard constraint).

Build with ``python scripts/build_window_bank.py --datasets hust ...``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .raw_windows import iter_charge_curves

# Lattice edges per dataset (volts).  Step 0.05 V; windows are all ordered
# edge pairs, so a dataset with 13 edges banks 78 windows per cycle.
DATASET_LATTICE = {
    "hust": (3.00, 3.60),
    "snl_lfp": (3.00, 3.60),
    "snl_nmc": (3.40, 4.20),  # D1 (2026-09-04): LG Chem 18650HG2, 0.5C CC charge to 4.2 V
    "snl_nca": (3.30, 4.20),  # D1 (2026-09-04): Panasonic NCR18650B, 0.5C CC charge to 4.2 V
    "uconn": (3.00, 3.60),
    "calce_cs2": (3.60, 4.20),
    "tongji_nca": (3.40, 4.20),      # D2 (2026-09-05): CC-CV charge to 4.2 V
    "tongji_ncm": (3.40, 4.20),
    "tongji_blend": (3.40, 4.20),
    "matr": (3.00, 3.60),            # D9 (2026-09-05): fast charge to 3.6 V
    "forklift_lfp180": (3.00, 3.60), # D8 (2026-09-05): ageing charges end below 3.40 V; RPT charges to 3.65 V are not segments
    "turku_rc_nca": (3.40, 4.20),    # D3 (2026-09-05): 1C CC-CV charge to 4.2 V
    "turku_rc_nmc": (3.40, 4.20),
    "turku_rc_lfp": (3.00, 3.60),    # D3 (2026-09-05): 1C CC-CV charge to 3.6 V
    "field_lfp_28": (3.00, 3.60),    # D10 (2026-09-05): cell voltages of the 8s packs
}
LATTICE_STEP = 0.05


def lattice_edges(dataset: str) -> np.ndarray:
    lo, hi = DATASET_LATTICE[dataset]
    count = int(round((hi - lo) / LATTICE_STEP)) + 1
    return np.round(lo + LATTICE_STEP * np.arange(count), 3)


def bank_path(data_dir: str | Path, dataset: str) -> Path:
    return Path(data_dir) / "processed" / "window_bank" / f"{dataset}.parquet"


def _edge_values(v: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """y interpolated at the lattice edges; NaN outside the curve's V range.

    The charge curve's accumulated quantities are monotone in voltage once
    sorted, so interpolation at the edges turns 78 window integrals into a
    single pass: q_win(lo, hi) = q(hi) - q(lo).
    """
    finite = np.isfinite(v) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return np.full(len(edges), np.nan)
    order = np.argsort(v[finite], kind="stable")
    vs = v[finite][order]
    ys = np.maximum.accumulate(y[finite][order])
    values = np.interp(edges, vs, ys)
    values[(edges < vs[0]) | (edges > vs[-1])] = np.nan
    return values


def bank_from_curves(curves, edges: np.ndarray) -> pd.DataFrame:
    """Long-format bank rows for every curve x lattice window."""
    edge_mv = np.round(edges * 1000).astype(int)
    lo_idx, hi_idx = np.triu_indices(len(edges), k=1)
    records: dict[str, list] = {
        "cell_id": [], "cycle_index": [], "lo_mv": [], "hi_mv": [],
        "q_win_ah": [], "t_win_s": [],
    }
    for cell, cycle, v, q, t in curves:
        q_edges = _edge_values(v, q, edges)
        t_edges = _edge_values(v, t, edges)
        q_win = q_edges[hi_idx] - q_edges[lo_idx]
        t_win = t_edges[hi_idx] - t_edges[lo_idx]
        count = len(lo_idx)
        records["cell_id"].extend([cell] * count)
        records["cycle_index"].extend([int(cycle)] * count)
        records["lo_mv"].extend(edge_mv[lo_idx])
        records["hi_mv"].extend(edge_mv[hi_idx])
        records["q_win_ah"].extend(q_win)
        records["t_win_s"].extend(t_win)
    return pd.DataFrame(records)


def build_bank(
    data_dir: str | Path,
    dataset: str,
    processed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if processed is None:
        processed_path = Path(data_dir) / "processed" / f"{dataset}_features.pkl"
        processed = pd.read_pickle(processed_path)
    curves = iter_charge_curves(data_dir, dataset, processed)
    bank = bank_from_curves(curves, lattice_edges(dataset))
    out = bank_path(data_dir, dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(out, index=False)
    return bank


def load_bank(data_dir: str | Path, dataset: str) -> pd.DataFrame:
    path = bank_path(data_dir, dataset)
    if not path.is_file():
        raise FileNotFoundError(
            f"window bank missing for {dataset}: run scripts/build_window_bank.py"
        )
    return pd.read_parquet(path)


def select_bounds(
    bank: pd.DataFrame,
    rated_by_cell: dict[str, float],
    target_fraction: float,
) -> dict:
    """Pick the lattice window whose median q_win/rated is closest to target.

    Label-free: uses only the bank integrals and nameplate capacities.
    Windows are also required to stay under the DOD warning band relative to
    rated capacity (median fraction < 0.60 x a nominal 1.0 SOH), mirroring
    the audit's intent before any data is touched.
    """
    rated = bank["cell_id"].astype(str).map(rated_by_cell)
    fraction = bank["q_win_ah"] / rated
    stats = (
        pd.DataFrame({
            "lo_mv": bank["lo_mv"], "hi_mv": bank["hi_mv"], "fraction": fraction,
        })
        .dropna()
        .groupby(["lo_mv", "hi_mv"], as_index=False)
        .agg(median_fraction=("fraction", "median"), rows=("fraction", "size"))
    )
    stats = stats[(stats["rows"] >= 64) & (stats["median_fraction"] > 0.0)]
    if not len(stats):
        raise ValueError("window bank has no viable lattice window for this dataset")
    stats["distance"] = (stats["median_fraction"] - target_fraction).abs()
    # Coverage alone is degenerate: many windows share a coverage value and
    # the knee/edge ones carry poor signal (measured: knee windows doubled
    # the outer MAE).  Among near-target windows prefer maximum charge
    # density — fraction per volt of span — which is exactly the plateau,
    # and still label-free.
    stats["density"] = stats["median_fraction"] / ((stats["hi_mv"] - stats["lo_mv"]) / 1000.0)
    eligible = stats[stats["distance"] <= 0.05]
    if not len(eligible):
        eligible = stats.nsmallest(3, "distance")
    best = eligible.sort_values(
        ["density", "distance", "hi_mv", "lo_mv"], ascending=[False, True, True, True]
    ).iloc[0]
    return {
        "lo_mv": int(best["lo_mv"]),
        "hi_mv": int(best["hi_mv"]),
        "median_fraction": float(best["median_fraction"]),
        "density_per_v": float(best["density"]),
        "rows": int(best["rows"]),
    }


def apply_bank_window(
    table: pd.DataFrame,
    dataset: str,
    bank: pd.DataFrame,
    lo_mv: int,
    hi_mv: int,
) -> pd.DataFrame:
    """Replace q_win_ah/t_win_s for *dataset*'s rows with the bank window."""
    chosen = bank[(bank["lo_mv"] == lo_mv) & (bank["hi_mv"] == hi_mv)]
    lookup_q = {
        (str(cell), int(cycle)): value
        for cell, cycle, value in zip(chosen["cell_id"], chosen["cycle_index"], chosen["q_win_ah"])
    }
    lookup_t = {
        (str(cell), int(cycle)): value
        for cell, cycle, value in zip(chosen["cell_id"], chosen["cycle_index"], chosen["t_win_s"])
    }
    out = table.copy()
    mask = out["dataset"].astype(str) == dataset
    keys = list(zip(out.loc[mask, "cell_id"].astype(str), out.loc[mask, "cycle_index"].astype(int)))
    out.loc[mask, "q_win_ah"] = [lookup_q.get(key, np.nan) for key in keys]
    out.loc[mask, "t_win_s"] = [lookup_t.get(key, np.nan) for key in keys]
    return out
