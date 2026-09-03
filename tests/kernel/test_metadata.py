import pandas as pd
from sohkernel.metadata import build_meta, DATASET_CHEMISTRY, DATASET_C_RATED_AH

def test_known_datasets_have_chemistry():
    for ds in ["nasa", "calce_cs2", "hust", "oxford"]:
        assert ds in DATASET_CHEMISTRY

def test_build_meta_one_row_per_cell():
    table = pd.DataFrame({
        "dataset": ["nasa","nasa","hust"], "cell_id": ["B0005","B0005","H1"],
        "cycle_index": [0,1,0], "soh": [1.0,0.99,1.0],
    })
    meta = build_meta(table)
    assert set(meta["cell_id"]) == {"B0005", "H1"}
    assert meta.loc[meta.cell_id=="H1","chemistry"].iloc[0] == "LFP"
    assert {"dataset","cell_id","chemistry","c_rated_ah","label_source"} <= set(meta.columns)

# H3: verified chemistry values
def test_oxford_chemistry_is_lco_nco():
    assert DATASET_CHEMISTRY["oxford"] == "LCO/NCO"

def test_hust_chemistry_is_lfp():
    assert DATASET_CHEMISTRY["hust"] == "LFP"

def test_no_verify_markers_in_values():
    for v in DATASET_CHEMISTRY.values():
        assert "VERIFY" not in str(v)
    for v in DATASET_C_RATED_AH.values():
        assert "VERIFY" not in str(v)


def test_canonical_dataset_name_accepts_raw_source_labels():
    from sohkernel.metadata import canonical_dataset_name
    assert canonical_dataset_name("CALCE") == "calce_cs2"
    assert canonical_dataset_name("SNL") == "snl_lfp"
    assert canonical_dataset_name("HUST") == "hust"
    assert canonical_dataset_name("snl_lfp") == "snl_lfp"
