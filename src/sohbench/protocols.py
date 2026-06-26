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
