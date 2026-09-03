from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from typing import Iterator
from .contracts import Fold

def _folds_by_group(table: pd.DataFrame, group_vals: pd.Series) -> Iterator[Fold]:
    idx = np.arange(len(table))
    for key in pd.unique(group_vals):
        test_mask = (group_vals == key).to_numpy()
        yield Fold(train_idx=idx[~test_mask], test_idx=idx[test_mask], held_out_key=str(key))

class LeaveOneCellOut:
    cross_domain: bool = False

    def generate(self, table, meta) -> Iterator[Fold]:
        keyed = table["dataset"].astype(str) + "/" + table["cell_id"].astype(str)
        yield from _folds_by_group(table, keyed)

class LeaveOneDatasetOut:
    cross_domain: bool = True

    def generate(self, table, meta) -> Iterator[Fold]:
        yield from _folds_by_group(table, table["dataset"].astype(str))

class LeaveOneChemistryOut:
    cross_domain: bool = True

    def generate(self, table, meta) -> Iterator[Fold]:
        # Merge chemistry onto table rows
        chem_col = table.merge(
            meta[["dataset", "cell_id", "chemistry"]],
            on=["dataset", "cell_id"], how="left"
        )["chemistry"]

        # All unique chemistries
        all_chemistries = pd.unique(chem_col)

        if len(all_chemistries) < 2:
            warnings.warn(
                "chemistry-out: only one chemistry present; cannot hold any out — yielding nothing"
            )
            return

        # Count datasets per chemistry (from meta, not table)
        datasets_per_chem = (
            meta.groupby("chemistry")["dataset"].nunique()
        )

        idx = np.arange(len(table))
        for chem in all_chemistries:
            test_mask = (chem_col == chem).to_numpy()
            n_datasets = int(datasets_per_chem.get(chem, 0))
            if n_datasets < 2:
                held_out_key = f"{chem} (confounded: 1 dataset)"
                warnings.warn(
                    f"chemistry-out: held-out chemistry '{chem}' spans only 1 dataset "
                    f"— lab/chemistry confound; fold flagged as '{held_out_key}'"
                )
            else:
                held_out_key = chem
            yield Fold(
                train_idx=idx[~test_mask],
                test_idx=idx[test_mask],
                held_out_key=held_out_key,
            )


class ForwardChaining:
    """Per-cell forward chaining (Spec 01 §3, Spec 04 §1): the earliest cycles train, the rest test.

    For every cell the earliest ``max(min_train_cycles, round(train_frac * n))`` cycles are its
    *early life*.  One fold is produced per cell that still has cycles left after that cut:
    the fold tests on that cell's remaining (later) cycles and trains on the early life of
    **all** cells when ``pooled=True`` (default — the fleet model must extrapolate in time), or
    only on the held-out cell's own early cycles when ``pooled=False`` (a per-cell tracker).
    No future cycle of any cell ever reaches training, and cycles of one cell are never split
    at random.  Cells too short to hold out a future segment contribute their cycles to the
    pooled early life only and are reported with a warning.  ``held_out_key`` is
    ``"<dataset>/<cell>@forward"``.  Required for temporal and adaptation claims.
    """

    name = "forward"
    cross_domain: bool = False

    def __init__(self, train_frac: float = 0.6, min_train_cycles: int = 20, *, pooled: bool = True):
        if not 0.0 < float(train_frac) < 1.0:
            raise ValueError("train_frac must lie strictly between 0 and 1")
        if int(min_train_cycles) < 1:
            raise ValueError("min_train_cycles must be at least 1")
        self.train_frac = float(train_frac)
        self.min_train_cycles = int(min_train_cycles)
        self.pooled = bool(pooled)

    def plan(self, table: pd.DataFrame) -> dict[str, tuple[int, int]]:
        """Per cell key → ``(n_cycles, n_train)``; a cell is eligible when ``n_train < n_cycles``."""
        keyed = table["dataset"].astype(str) + "/" + table["cell_id"].astype(str)
        out: dict[str, tuple[int, int]] = {}
        for key, count in keyed.value_counts(sort=False).items():
            n = int(count)
            out[str(key)] = (n, max(self.min_train_cycles, int(round(self.train_frac * n))))
        return out

    def eligible_cells(self, table: pd.DataFrame) -> list[str]:
        return [key for key, (n, n_train) in self.plan(table).items() if n_train < n]

    def generate(self, table, meta) -> Iterator[Fold]:
        idx = np.arange(len(table))
        keyed = (table["dataset"].astype(str) + "/" + table["cell_id"].astype(str)).to_numpy()
        cycles = pd.to_numeric(table["cycle_index"], errors="coerce").to_numpy(dtype=float)
        plan = self.plan(table)
        early: dict[str, np.ndarray] = {}
        late: dict[str, np.ndarray] = {}
        for key, (n, n_train) in plan.items():
            rows = idx[keyed == key]
            ordered = rows[np.argsort(cycles[rows], kind="stable")]
            early[key] = ordered[:n_train]
            late[key] = ordered[n_train:]
            if n_train >= n:
                warnings.warn(
                    f"forward: cell '{key}' has {n} cycles, not more than its {n_train}-cycle "
                    "early life — no future segment to hold out; it only feeds the pooled early life"
                )
        pooled_early = np.concatenate([early[key] for key in plan]) if plan else idx[:0]
        for key in plan:
            if len(late[key]) == 0:
                continue
            train_idx = pooled_early if self.pooled else early[key]
            yield Fold(train_idx=np.sort(train_idx), test_idx=late[key], held_out_key=f"{key}@forward")


# The protocol menu any tool may request by name (Spec 01 §3, §9).
PROTOCOLS = {
    "loco": LeaveOneCellOut,
    "lodo": LeaveOneDatasetOut,
    "lcho": LeaveOneChemistryOut,
    "forward": ForwardChaining,
}
