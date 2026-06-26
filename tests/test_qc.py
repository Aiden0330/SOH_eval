import numpy as np, pandas as pd
from sohbench.qc import qc_gate
from sohbench.metadata import build_meta

def _table():
    return pd.DataFrame({
        "dataset": ["nasa"]*5,
        "cell_id": ["B0005"]*5,
        "cycle_index": [0,1,2,3,4],
        "soh": [1.25, 0.95, 1.35, 0.0, 0.80],   # 1.35 and 0.0 are out of [0,1.3)
        "f1": [1,2,3,4,5],
        "f_sparse": [np.nan, np.nan, np.nan, np.nan, 1.0],  # 20% available -> dropped (<20% strictly? keep boundary)
    })

def test_value_range_drop():
    t = _table(); meta = build_meta(t)
    clean, rep = qc_gate(t, meta, lower=0.0, upper=1.3)
    # 1.35 and 0.0 dropped by SOH range; f_sparse NaN rows (soh=1.25,0.95) also dropped
    assert set(clean["soh"]) <= {1.25, 0.95, 0.80}      # 1.35 and 0.0 dropped by range
    assert rep.drop_reasons["soh_out_of_range"] == 2
    # n_out now includes NaN-feature-row drops in addition to SOH range drops
    assert rep.n_out >= 2

def test_soh_above_one_is_kept():
    # Use a table without NaN in feature columns to isolate the SOH > 1.0 retention test
    t = pd.DataFrame({
        "dataset": ["nasa"]*3, "cell_id": ["B0005"]*3,
        "cycle_index": [0,1,2], "soh": [1.25, 0.95, 0.80],
        "f1": [1.0, 2.0, 3.0],
    })
    meta = build_meta(t)
    clean, _ = qc_gate(t, meta, lower=0.0, upper=1.3)
    assert (clean["soh"] > 1.0).any()                    # 1.25 retained (D2)

def test_availability_drop_no_fill():
    t = _table(); meta = build_meta(t)
    clean, rep = qc_gate(t, meta, availability_min=0.5)
    assert "f_sparse" in rep.dropped_columns
    assert "f_sparse" not in clean.columns
    assert not clean["f1"].isna().any()                  # no synthetic fill introduced

def test_reject_bms_label_source():
    t = _table()
    meta = build_meta(t); meta.loc[:, "label_source"] = "bms_reported"
    import pytest
    with pytest.raises(ValueError):
        qc_gate(t, meta, reject_bms=True)
