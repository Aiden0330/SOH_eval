import warnings
import numpy as np, pandas as pd
from sohbench.protocols import LeaveOneCellOut, LeaveOneDatasetOut, LeaveOneChemistryOut
from sohbench.metadata import build_meta

def _table():
    """4-dataset fixture: nasa(LCO), calce_cs2(LCO), hust(LFP), oxford(LCO/NCO)."""
    rows = []
    for ds, cell in [("nasa","c1"),("calce_cs2","c2"),("hust","h1"),("oxford","o1")]:
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
    assert set(keys) == {"nasa", "calce_cs2", "hust", "oxford"}

# H5: cross_domain attributes
def test_cross_domain_attrs():
    assert LeaveOneCellOut.cross_domain == False
    assert LeaveOneDatasetOut.cross_domain == True
    assert LeaveOneChemistryOut.cross_domain == True

# H5: per-chemistry folds — 3 folds for 3 chemistries
def test_chemistry_out_three_folds():
    t = _table(); meta = build_meta(t)
    folds = list(LeaveOneChemistryOut().generate(t, meta))
    keys = [f.held_out_key for f in folds]
    assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}: {keys}"

# H5: LCO fold is unflagged (2 datasets: nasa + calce_cs2)
def test_lco_fold_unflagged():
    t = _table(); meta = build_meta(t)
    folds = list(LeaveOneChemistryOut().generate(t, meta))
    lco_keys = [f.held_out_key for f in folds if "LCO" in f.held_out_key]
    assert any(k == "LCO" for k in lco_keys), f"Expected unflagged 'LCO' fold, got: {lco_keys}"

# H5: singleton chemistries are flagged "confounded: 1 dataset"
def test_singleton_chemistries_flagged():
    t = _table(); meta = build_meta(t)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folds = list(LeaveOneChemistryOut().generate(t, meta))
    keys = [f.held_out_key for f in folds]
    # LFP (hust only) and LCO/NCO (oxford only) must be flagged
    assert any("LFP" in k and "confounded" in k for k in keys), f"LFP should be flagged: {keys}"
    assert any("LCO/NCO" in k and "confounded" in k for k in keys), f"LCO/NCO should be flagged: {keys}"
    # Warnings emitted for singletons
    warn_msgs = [str(warning.message) for warning in w]
    assert any("LFP" in msg or "LCO/NCO" in msg for msg in warn_msgs), f"Expected singleton warnings: {warn_msgs}"

# H5: single-chemistry dataset yields no folds (with warning)
def test_single_chemistry_yields_no_folds():
    t = pd.DataFrame({
        "dataset": ["nasa"]*6,
        "cell_id": ["c1","c1","c1","c2","c2","c2"],
        "cycle_index": [0,1,2,0,1,2],
        "soh": [1.0,0.99,0.98,1.0,0.99,0.98],
    })
    meta = build_meta(t)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folds = list(LeaveOneChemistryOut().generate(t, meta))
    assert len(folds) == 0
    assert any("one chemistry" in str(warning.message).lower() or
               "single" in str(warning.message).lower()
               for warning in w)
