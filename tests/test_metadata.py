import pandas as pd
from sohbench.metadata import build_meta, DATASET_CHEMISTRY

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
