import numpy as np
import pandas as pd
import pytest

from sohbench.contracts import Fold
from sohbench.fabric.views import assert_fold_integrity, make_fold_views


def table():
    return pd.DataFrame({
        "dataset": ["d"] * 4,
        "cell_id": ["a", "a", "b", "b"],
        "cycle_index": [0, 1, 0, 1],
        "soh": [1.0, .99, 1.0, .98],
        "voltage_mean": [3.5, 3.4, 3.6, 3.3],
    })


def test_fold_views_strip_target_from_both_algorithm_views():
    fold = Fold(np.array([0, 1]), np.array([2, 3]), "d/b")
    views = make_fold_views(table(), fold)
    assert "soh" not in views.train_x
    assert "soh" not in views.test_x
    assert views.train_y.tolist() == [1.0, .99]
    assert views.test_y.tolist() == [1.0, .98]


def test_fold_views_preserve_fold_order_and_are_independent():
    fold = Fold(np.array([1, 0]), np.array([3, 2]), "d/b")
    views = make_fold_views(table(), fold)
    assert views.train_y.tolist() == [.99, 1.0]
    views.train_x.loc[1, "voltage_mean"] = -1
    assert table().loc[1, "voltage_mean"] == 3.4


def test_fold_views_reject_missing_target():
    with pytest.raises(ValueError, match="target column 'capacity' not found"):
        make_fold_views(table(), Fold(np.array([0, 1]), np.array([2, 3]), "d/b"), "capacity")


def test_fold_integrity_rejects_asset_overlap():
    fold = Fold(np.array([0]), np.array([1]), "bad")
    with pytest.raises(ValueError, match="asset overlap"):
        assert_fold_integrity(table(), fold)


@pytest.mark.parametrize(
    ("train_idx", "test_idx", "message"),
    [
        ([], [2], "empty train or test"),
        ([0], [], "empty train or test"),
        ([0, 0], [2], "duplicate index"),
        ([0], [0, 2], "index overlap"),
        ([-1], [2], "negative index"),
        ([4], [2], "out of bounds"),
    ],
)
def test_fold_integrity_rejects_invalid_partitions(train_idx, test_idx, message):
    fold = Fold(np.array(train_idx), np.array(test_idx), "bad")
    with pytest.raises(ValueError, match=message):
        assert_fold_integrity(table(), fold)


@pytest.mark.parametrize("missing", ["dataset", "cell_id"])
def test_fold_integrity_rejects_missing_asset_columns(missing):
    data = table().drop(columns=[missing])
    fold = Fold(np.array([0, 1]), np.array([2, 3]), "bad")
    with pytest.raises(ValueError, match=missing):
        assert_fold_integrity(data, fold)
