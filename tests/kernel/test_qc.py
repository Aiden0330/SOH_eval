import numpy as np, pandas as pd
from sohkernel.qc import qc_gate
from sohkernel.metadata import build_meta

def _table():
    return pd.DataFrame({
        "dataset": ["nasa"]*5,
        "cell_id": ["B0005"]*5,
        "cycle_index": [0,1,2,3,4],
        "soh": [1.25, 0.95, 1.35, 0.0, 0.80],   # 1.35 out of [0,1.3); 0.0 non-positive
        "f1": [1,2,3,4,5],
        "f_sparse": [np.nan, np.nan, np.nan, np.nan, 1.0],  # 20% available -> dropped
    })

def test_value_range_drop():
    t = _table(); meta = build_meta(t)
    clean, rep = qc_gate(t, meta, lower=0.0, upper=1.3)
    # 1.35 dropped by soh_out_of_range; 0.0 dropped by soh_nonpositive
    assert 1.35 not in list(clean["soh"])
    assert 0.0 not in list(clean["soh"])
    # Total out-of-range (1.35 only, since 0.0 goes to nonpositive)
    assert rep.drop_reasons["soh_out_of_range"] == 1
    # Non-positive (0.0)
    assert rep.drop_reasons["soh_nonpositive"] == 1
    # Total dropped >= 2 (range + nonpositive, possibly more from NaN rows)
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


def test_structural_feature_gaps_do_not_drop_whole_datasets():
    t = pd.DataFrame({
        "dataset": ["alpha", "alpha", "beta", "beta"],
        "cell_id": ["a1", "a1", "b1", "b1"],
        "cycle_index": [0, 1, 0, 1],
        "soh": [1.0, 0.98, 1.0, 0.97],
        "alpha_only_feature": [0.1, 0.2, np.nan, np.nan],
        "beta_only_feature": [np.nan, np.nan, 0.3, 0.4],
    })
    meta = build_meta(t)

    clean, rep = qc_gate(t, meta, availability_min=0.2)

    assert set(clean["dataset"]) == {"alpha", "beta"}
    assert len(clean) == len(t)
    assert rep.drop_reasons["nan_feature_row"] == 0

def test_reject_bms_label_source():
    t = _table()
    meta = build_meta(t); meta.loc[:, "label_source"] = "bms_reported"
    import pytest
    with pytest.raises(ValueError):
        qc_gate(t, meta, reject_bms=True)

# H4: lower-inclusive — soh==lower is KEPT when lower>0
def test_lower_inclusive_boundary_kept():
    t = pd.DataFrame({
        "dataset": ["nasa"]*3, "cell_id": ["B0005"]*3,
        "cycle_index": [0,1,2], "soh": [0.6, 0.8, 1.0],
        "f1": [1.0, 2.0, 3.0],
    })
    meta = build_meta(t)
    clean, rep = qc_gate(t, meta, lower=0.6, upper=1.3)
    assert 0.6 in list(clean["soh"]), "soh==lower should be retained (lower-inclusive)"
    assert rep.drop_reasons.get("soh_out_of_range", 0) == 0

# H4: soh=0.0 dropped via soh_nonpositive, not out_of_range
def test_zero_soh_dropped_via_nonpositive():
    t = pd.DataFrame({
        "dataset": ["nasa"]*3, "cell_id": ["B0005"]*3,
        "cycle_index": [0,1,2], "soh": [0.0, 0.8, 1.0],
        "f1": [1.0, 2.0, 3.0],
    })
    meta = build_meta(t)
    clean, rep = qc_gate(t, meta, lower=0.0, upper=1.3)
    assert 0.0 not in list(clean["soh"])
    assert rep.drop_reasons.get("soh_nonpositive", 0) == 1
