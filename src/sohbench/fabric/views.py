from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numbers import Integral

from sohbench.contracts import Fold


@dataclass(frozen=True)
class FoldViews:
    train_x: pd.DataFrame
    train_y: np.ndarray
    test_x: pd.DataFrame
    test_y: np.ndarray


def _validate_indices(table: pd.DataFrame, fold: Fold) -> tuple[list[int], list[int]]:
    raw_indices = [*fold.train_idx, *fold.test_idx]
    if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) for index in raw_indices):
        raise ValueError("fold indices must be integer indices")
    train = [int(index) for index in fold.train_idx]
    test = [int(index) for index in fold.test_idx]
    if not train or not test:
        raise ValueError("fold contains an empty train or test partition")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("fold contains a duplicate index")
    if set(train) & set(test):
        raise ValueError("fold index overlap")
    row_count = len(table)
    if any(index < 0 for index in [*train, *test]):
        raise ValueError("fold contains a negative index")
    if any(index >= row_count for index in [*train, *test]):
        raise ValueError("fold index out of bounds")
    return train, test


def assert_fold_integrity(table: pd.DataFrame, fold: Fold) -> None:
    train_idx, test_idx = _validate_indices(table, fold)
    missing = [column for column in ("dataset", "cell_id") if column not in table.columns]
    if missing:
        raise ValueError(f"fold integrity requires columns: {', '.join(missing)}")
    train_assets = set(map(tuple, table.iloc[train_idx][["dataset", "cell_id"]].astype(str).to_numpy()))
    test_assets = set(map(tuple, table.iloc[test_idx][["dataset", "cell_id"]].astype(str).to_numpy()))
    if train_assets & test_assets:
        raise ValueError("fold asset overlap")


def make_fold_views(table: pd.DataFrame, fold: Fold, target: str = "soh") -> FoldViews:
    assert_fold_integrity(table, fold)
    if target not in table.columns:
        raise ValueError(f"target column '{target}' not found")
    train_idx = [int(index) for index in fold.train_idx]
    test_idx = [int(index) for index in fold.test_idx]
    train = table.iloc[train_idx].copy()
    test = table.iloc[test_idx].copy()
    return FoldViews(
        train_x=train.drop(columns=[target]).copy(),
        train_y=train[target].to_numpy(copy=True),
        test_x=test.drop(columns=[target]).copy(),
        test_y=test[target].to_numpy(copy=True),
    )
