"""Fold-local feature pipelines (Spec 01 §2 layer 3; extraction plan Task 7d).

A *pipeline* is a feature transform with fitted state.  Inside an evaluation it is fitted on
the training rows of a fold only and applied to both the training and the test rows, so no
statistic of a test row — and no test label — can reach the features a model is scored on.
The transformed training frame is audited (label-safety) before any estimator sees it.

Port of the archived Codex line (``1fdf6ed`` / ``320b250``), reduced to the kernel contract::

    class MyPipeline:
        def fit(self, train_x, train_y, meta) -> state: ...
        def transform(self, frame, state, meta) -> pd.DataFrame: ...

``train_x``/``frame`` never contain the label column; ID columns are present and must be
returned unchanged, row count and index must be preserved.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .contracts import ID_COLUMNS
from .feature_safety import FeatureSafetyError, audit_feature_frame

IDENTITY_COLUMNS = ("dataset", "cell_id", "cycle_index")


@runtime_checkable
class FeaturePipeline(Protocol):
    def fit(self, train_x: pd.DataFrame, train_y: np.ndarray, meta: pd.DataFrame | None) -> Any: ...

    def transform(self, frame: pd.DataFrame, state: Any, meta: pd.DataFrame | None) -> pd.DataFrame: ...


class StatelessPipeline:
    """Adapter for a legacy stateless ``transform(table, meta)`` callable: nothing is fitted,
    the transform simply runs per fold on label-free frames."""

    def __init__(self, transform: Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame], *, name: str = "stateless"):
        self._transform = transform
        self.name = str(name)

    def fit(self, train_x: pd.DataFrame, train_y: np.ndarray, meta: pd.DataFrame | None) -> None:
        return None

    def transform(self, frame: pd.DataFrame, state: Any, meta: pd.DataFrame | None) -> pd.DataFrame:
        return self._transform(frame, meta)


def _check_output(name: str, source: pd.DataFrame, transformed: Any, *, target: str) -> pd.DataFrame:
    if not isinstance(transformed, pd.DataFrame):
        raise FeatureSafetyError(f"feature pipeline {name!r} did not return a DataFrame")
    if len(transformed) != len(source) or not transformed.index.equals(source.index):
        raise FeatureSafetyError(f"feature pipeline {name!r} changed the rows or index of the fold")
    if target in transformed.columns:
        raise FeatureSafetyError(f"feature pipeline {name!r} attempted to create protected label {target!r}")
    for column in IDENTITY_COLUMNS:
        if column in source.columns and (
            column not in transformed.columns or not transformed[column].equals(source[column])
        ):
            raise FeatureSafetyError(f"feature pipeline {name!r} changed identity column {column!r}")
    return transformed


def prepare_fold(
    pipelines: Mapping[str, FeaturePipeline],
    train: pd.DataFrame,
    test: pd.DataFrame,
    meta: pd.DataFrame | None,
    *,
    target: str = "soh",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit every pipeline on the fold's training rows and apply it to both partitions.

    Returns ``(train, test, states)`` with the label column restored on both frames.  The
    columns the pipelines created are audited on the training frame; a violation raises
    :class:`FeatureSafetyError` so the fold fails instead of scoring a leak.
    """
    train_y = train[target].to_numpy(copy=True)
    test_y = test[target].to_numpy(copy=True)
    position = int(train.columns.get_loc(target))
    train_x = train.drop(columns=[target]).copy(deep=True)
    test_x = test.drop(columns=[target]).copy(deep=True)
    states: dict[str, Any] = {}
    for name, pipeline in pipelines.items():
        state = pipeline.fit(train_x.copy(deep=True), train_y.copy(), meta)
        states[name] = state
        train_x = _check_output(name, train_x, pipeline.transform(train_x.copy(deep=True), state, meta), target=target)
        test_x = _check_output(name, test_x, pipeline.transform(test_x.copy(deep=True), state, meta), target=target)
    train_out = train_x.copy()
    train_out.insert(min(position, len(train_out.columns)), target, train_y)
    test_out = test_x.copy()
    test_out.insert(min(position, len(test_out.columns)), target, test_y)
    created = [c for c in train_out.columns if c not in train.columns and c not in ID_COLUMNS]
    audit = audit_feature_frame(train_out, created, target=target)
    if audit["status"] != "passed":
        reasons = "; ".join(item["reason"] for item in audit["violations"])
        raise FeatureSafetyError(f"fold-local feature pipeline failed label-safety audit: {reasons}")
    return train_out, test_out, states
