import pandas as pd, numpy as np
import pytest
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


def test_unlabeled_tables_are_discoverable_and_loadable_only_for_inference(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    unlabeled = _feature_frame("event").drop(columns=["soh"])
    unlabeled.to_pickle(proc / "event_features.pkl")
    loader = DatasetLoader(tmp_path)

    assert loader.discover_datasets() == []
    assert loader.discover_datasets(require_labels=False) == ["event"]
    with pytest.raises(ValueError, match="missing required columns"):
        loader.load(["event"], qc=False)

    table, meta, report = loader.load(
        ["event"], require_labels=False, qc=False,
    )

    assert "soh" not in table.columns
    assert set(meta["dataset"]) == {"event"}
    assert report.n_in == len(table)


@pytest.mark.parametrize(
    "bad_cycle",
    [1.2, np.nan, np.inf, -np.inf, True, -1, 2**63, float(2**63)],
)
def test_loader_rejects_non_integral_cycle_index(tmp_path, bad_cycle):
    proc = tmp_path / "processed"
    proc.mkdir()
    frame = _feature_frame("invalid")
    frame["cycle_index"] = frame["cycle_index"].astype(object)
    frame.loc[1, "cycle_index"] = bad_cycle
    frame.to_pickle(proc / "invalid_features.pkl")

    with pytest.raises(ValueError, match="cycle_index must contain finite integers"):
        DatasetLoader(tmp_path).load(["invalid"], qc=False)


def test_loader_normalizes_integral_float_cycle_index(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    frame = _feature_frame("float_cycles")
    frame["cycle_index"] = frame["cycle_index"].astype(float)
    frame.to_pickle(proc / "float_cycles_features.pkl")

    table, _meta, _report = DatasetLoader(tmp_path).load(
        ["float_cycles"], qc=False,
    )

    assert pd.api.types.is_integer_dtype(table["cycle_index"])
    assert table["cycle_index"].tolist() == [0, 1, 2, 3]
