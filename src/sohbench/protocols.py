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
    def generate(self, table, meta) -> Iterator[Fold]:
        keyed = table["dataset"].astype(str) + "/" + table["cell_id"].astype(str)
        yield from _folds_by_group(table, keyed)

class LeaveOneDatasetOut:
    def generate(self, table, meta) -> Iterator[Fold]:
        yield from _folds_by_group(table, table["dataset"].astype(str))

class LeaveOneChemistryOut:
    def generate(self, table, meta) -> Iterator[Fold]:
        chem = table.merge(meta[["dataset","cell_id","chemistry"]],
                           on=["dataset","cell_id"], how="left")["chemistry"]
        datasets_per_chem = (meta.groupby("chemistry")["dataset"].nunique())
        if (datasets_per_chem < 2).any():
            warnings.warn("chemistry-out downgrade: <2 datasets per chemistry; "
                          "using dataset-out to avoid lab/chemistry confound")
            yield from _folds_by_group(table, table["dataset"].astype(str))
        else:
            yield from _folds_by_group(table, chem)
