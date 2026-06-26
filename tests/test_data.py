import pandas as pd, numpy as np
from pathlib import Path
from sohbench.data import DatasetLoader

def _write_fixture(tmp_path):
    proc = tmp_path / "processed"; proc.mkdir()
    df = pd.DataFrame({
        "dataset": ["hust"]*4, "cell_id": ["h1"]*4, "cycle_index":[0,1,2,3],
        "soh":[1.02, 0.99, 1.35, 0.90],  # 1.35 dropped by QC
        "voltage_mean":[3.3,3.29,3.28,3.27],
    })
    df.to_pickle(proc / "hust_features.pkl")
    return tmp_path

def test_load_builds_table_meta_and_qc(tmp_path):
    root = _write_fixture(tmp_path)
    loader = DatasetLoader(data_dir=root)
    table, meta, rep = loader.load(["hust"])
    assert (table["soh"] < 1.3).all()                # 1.35 dropped
    assert (table["soh"] > 1.0).any()                # 1.02 retained
    assert meta.loc[meta.cell_id=="h1","chemistry"].iloc[0] == "LFP"
    assert rep.n_out == 1
