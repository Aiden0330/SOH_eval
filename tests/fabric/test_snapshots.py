import numpy as np
import pandas as pd
import json
import pickle
import pytest

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


def test_frame_digest_preserves_real_helper_named_column():
    data = table()
    data["__fabric_row_hash__"] = ["left", "left", "right", "right"]
    original = data.copy(deep=True)
    changed = data.copy()
    changed.loc[0, "__fabric_row_hash__"] = "changed"
    assert frame_digest(changed) != frame_digest(data)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("value", [["a", "b"], {"a": 1}])
def test_frame_digest_rejects_unhashable_cell_values(value):
    data = table()
    data["voltage_mean"] = data["voltage_mean"].astype(object)
    data.at[0, "voltage_mean"] = value
    with pytest.raises(ValueError, match="hashable"):
        frame_digest(data)


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


def test_dataset_snapshot_qc_is_json_pickle_compatible_and_deeply_immutable():
    snapshot = DatasetSnapshot.from_frame(
        table(), ["d"], "labels", {"checks": ({"pass": True}, ["ok"])}
    )
    json.dumps(snapshot.qc)
    restored = pickle.loads(pickle.dumps(snapshot))
    assert restored == snapshot
    assert dict(snapshot.qc) == {"checks": ({"pass": True}, ("ok",))}
    with pytest.raises(TypeError):
        snapshot.qc["checks"][0]["pass"] = False
    with pytest.raises(TypeError):
        snapshot.qc["checks"][1] += ("bad",)
