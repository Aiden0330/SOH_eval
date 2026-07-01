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

def _feature_frame(dataset):
    return pd.DataFrame({
        "dataset": [dataset] * 4,
        "cell_id": ["c1"] * 4,
        "cycle_index": [0, 1, 2, 3],
        "soh": [1.0, 0.98, 0.96, 0.94],
        "voltage_mean": [3.3, 3.29, 3.28, 3.27],
    })

def test_load_builds_table_meta_and_qc(tmp_path):
    root = _write_fixture(tmp_path)
    loader = DatasetLoader(data_dir=root)
    table, meta, rep = loader.load(["hust"])
    assert (table["soh"] < 1.3).all()                # 1.35 dropped
    assert (table["soh"] > 1.0).any()                # 1.02 retained
    assert meta.loc[meta.cell_id=="h1","chemistry"].iloc[0] == "LFP"
    assert rep.n_out == 1


def test_discover_datasets_finds_valid_feature_tables(tmp_path):
    proc = tmp_path / "processed"; proc.mkdir()
    _feature_frame("alpha").to_pickle(proc / "alpha_features.pkl")
    _feature_frame("beta").to_pickle(proc / "beta_features.pkl")
    _feature_frame("event").drop(columns=["soh"]).to_pickle(proc / "event_features.pkl")
    pd.DataFrame({"dataset": ["ignored"]}).to_pickle(proc / "rwth_id14_events.pkl")

    loader = DatasetLoader(data_dir=tmp_path)

    assert loader.discover_datasets() == ["alpha", "beta"]


def test_load_all_uses_discovered_datasets(tmp_path):
    proc = tmp_path / "processed"; proc.mkdir()
    _feature_frame("alpha").to_pickle(proc / "alpha_features.pkl")
    _feature_frame("beta").to_pickle(proc / "beta_features.pkl")

    table, meta, report = DatasetLoader(tmp_path).load(["all"], qc=False)

    assert set(table["dataset"]) == {"alpha", "beta"}
    assert set(meta["dataset"]) == {"alpha", "beta"}
    assert report.n_in == len(table)
