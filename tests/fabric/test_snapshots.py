import numpy as np
import pandas as pd

from sohbench.fabric.snapshots import DatasetSnapshot, frame_digest


def table():
    return pd.DataFrame({
        "dataset": ["d"] * 4,
        "cell_id": ["a", "a", "b", "b"],
        "cycle_index": [0, 1, 0, 1],
        "soh": [1.0, .99, 1.0, .98],
        "voltage_mean": [3.5, 3.4, 3.6, 3.3],
    })


def test_frame_digest_is_stable_after_row_reordering():
    data = table()
    assert frame_digest(data) == frame_digest(data.sample(frac=1, random_state=4))


def test_frame_digest_is_stable_when_duplicate_sort_keys_are_reordered():
    data = table().iloc[[0, 0, 1, 1]].reset_index(drop=True)
    reordered = data.iloc[[1, 0, 3, 2]].reset_index(drop=True)
    assert frame_digest(data) == frame_digest(reordered)


def test_frame_digest_changes_for_values_dtypes_and_column_order():
    data = table()
    changed_value = data.copy()
    changed_value.loc[0, "soh"] = .97
    changed_dtype = data.astype({"cycle_index": "int64"})
    changed_dtype["cycle_index"] = changed_dtype["cycle_index"].astype("float64")
    changed_order = data[["dataset", "cell_id", "cycle_index", "voltage_mean", "soh"]]
    digest = frame_digest(data)
    assert frame_digest(changed_value) != digest
    assert frame_digest(changed_dtype) != digest
    assert frame_digest(changed_order) != digest


def test_dataset_snapshot_copies_inputs_and_nested_qc():
    data = table()
    qc = {"checks": {"pass": True}}
    snapshot = DatasetSnapshot.from_frame(data, ["d"], "labels", qc)
    qc["checks"]["pass"] = False
    assert snapshot.qc == {"checks": {"pass": True}}
    assert dict(snapshot.qc) == {"checks": {"pass": True}}
    assert snapshot.datasets == ("d",)
    assert snapshot.assets == 2
    assert snapshot.rows == 4
