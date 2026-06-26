import numpy as np, pandas as pd
from sohbench.protocols import LeaveOneCellOut, LeaveOneDatasetOut, LeaveOneChemistryOut
from sohbench.metadata import build_meta

def _table():
    rows = []
    for ds, cell, chem in [("nasa","c1","LCO"),("nasa","c2","LCO"),
                           ("hust","h1","LFP"),("oxford","o1","NMC")]:
        for k in range(3):
            rows.append((ds, cell, k, 1.0 - 0.01*k))
    return pd.DataFrame(rows, columns=["dataset","cell_id","cycle_index","soh"])

def test_leave_one_cell_no_leakage():
    t = _table(); meta = build_meta(t)
    for fold in LeaveOneCellOut().generate(t, meta):
        train_cells = set(t.loc[fold.train_idx, "cell_id"])
        test_cells = set(t.loc[fold.test_idx, "cell_id"])
        assert train_cells.isdisjoint(test_cells)           # no cell in both
        assert len(test_cells) == 1

def test_leave_one_dataset_out():
    t = _table(); meta = build_meta(t)
    keys = [f.held_out_key for f in LeaveOneDatasetOut().generate(t, meta)]
    assert set(keys) == {"nasa", "hust", "oxford"}

def test_chemistry_out_downgrades_when_single_dataset(recwarn):
    t = _table(); meta = build_meta(t)
    # LCO has 1 dataset (nasa), LFP 1 (hust), NMC 1 (oxford) -> all single -> downgrade
    folds = list(LeaveOneChemistryOut().generate(t, meta))
    assert any("downgrade" in str(w.message).lower() for w in recwarn.list)
    # downgraded to dataset-out keys
    assert {f.held_out_key for f in folds} == {"nasa", "hust", "oxford"}
