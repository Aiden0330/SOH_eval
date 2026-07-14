from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import pandas as pd

from .ids import canonical_json, sha256_file, sha256_json


class _FrozenDict(dict):
    def __init__(self, value=()):
        dict.__init__(self, {str(key): _freeze(item) for key, item in dict(value).items()})

    def __reduce__(self):
        return (_FrozenDict, (dict(self),))

    def _immutable(self, *args, **kwargs):
        raise TypeError("FeatureState payload is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"FeatureState payload contains non-JSON value of type {type(value).__name__}")


@dataclass(frozen=True)
class FeatureState:
    """Immutable, serializable fitting state for a feature pipeline."""

    payload: Mapping[str, Any]
    digest: str

    def __post_init__(self) -> None:
        clean = json.loads(canonical_json(dict(self.payload)).decode("utf-8"))
        digest = sha256_json(clean)
        if self.digest != digest:
            raise ValueError("FeatureState digest does not match canonical payload")
        object.__setattr__(self, "payload", _FrozenDict(clean))

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "FeatureState":
        # Canonical JSON both validates and cuts aliases from caller-owned inputs.
        clean = json.loads(canonical_json(dict(payload)).decode("utf-8"))
        return cls(payload=clean, digest=sha256_json(clean))

    def to_dict(self) -> dict:
        return {"payload": json.loads(canonical_json(self.payload).decode("utf-8")), "digest": self.digest}

    def to_json(self) -> str:
        return canonical_json(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureState":
        state = cls.create(value.get("payload", {}))
        if value.get("digest") != state.digest:
            raise ValueError("FeatureState digest does not match canonical payload")
        return state

    @classmethod
    def from_json(cls, value: str) -> "FeatureState":
        return cls.from_dict(json.loads(value))


class FeaturePipeline(Protocol):
    def fit(self, train_x: pd.DataFrame, train_y: np.ndarray, meta: pd.DataFrame | None) -> FeatureState: ...

    def transform(
        self,
        frame: pd.DataFrame,
        state: FeatureState,
        meta: pd.DataFrame | None,
    ) -> pd.DataFrame: ...


class IdentityFeaturePipeline:
    def fit(self, train_x: pd.DataFrame, train_y: np.ndarray, meta: pd.DataFrame | None) -> FeatureState:
        return FeatureState.create({"kind": "identity"})

    def transform(
        self,
        frame: pd.DataFrame,
        state: FeatureState,
        meta: pd.DataFrame | None,
    ) -> pd.DataFrame:
        return frame.copy(deep=True)


class LegacyStatelessPipeline:
    """Safe fold-scoped adapter for existing transform-only trusted plugins."""

    def __init__(self, name: str, candidates: list[str], transform: Callable, path: str | Path):
        self.name = str(name)
        self.candidates = tuple(map(str, candidates))
        self._transform = transform
        self._path = Path(path)

    def fit(self, train_x: pd.DataFrame, train_y: np.ndarray, meta: pd.DataFrame | None) -> FeatureState:
        return FeatureState.create(
            {
                "kind": "legacy_stateless",
                "plugin_name": self.name,
                "plugin_digest": sha256_file(self._path),
                "candidates": list(self.candidates),
            }
        )

    def transform(
        self,
        frame: pd.DataFrame,
        state: FeatureState,
        meta: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if state.payload.get("plugin_digest") != sha256_file(self._path):
            raise ValueError(f"feature transform {self.name!r} plugin digest does not match fitted state")
        source = _target_stripped(frame)
        transformed = self._transform(source.copy(deep=True), _copy_meta(meta))
        _validate_legacy_output(self.name, source, transformed, self.candidates)
        return transformed.copy(deep=True)


def _target_stripped(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("feature pipeline input must be a DataFrame")
    return frame.drop(columns=["soh"], errors="ignore").copy(deep=True)


def _copy_meta(meta: pd.DataFrame | None) -> pd.DataFrame | None:
    return meta.copy(deep=True) if isinstance(meta, pd.DataFrame) else meta


def _validate_legacy_output(name: str, source: pd.DataFrame, transformed: Any, candidates: tuple[str, ...]) -> None:
    if not isinstance(transformed, pd.DataFrame):
        raise ValueError(f"feature transform {name!r} did not return a DataFrame")
    if len(transformed) != len(source):
        raise ValueError(f"feature transform {name!r} changed row count")
    if not transformed.index.equals(source.index):
        raise ValueError(f"feature transform {name!r} changed row index")
    identity_columns = [column for column in ("dataset", "cell_id", "cycle_index") if column in source.columns]
    for column in identity_columns:
        if column not in transformed.columns or not transformed[column].equals(source[column]):
            raise ValueError(f"feature transform {name!r} changed identity column {column!r}")
    missing = [column for column in candidates if column not in transformed.columns]
    if missing:
        raise ValueError(f"feature transform {name!r} did not create columns: {missing}")


def assert_prefix_invariant(
    pipeline: FeaturePipeline,
    frame: pd.DataFrame,
    meta: pd.DataFrame | None,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Reject feature outputs that change when a later cycle is appended."""
    if not {"dataset", "cell_id", "cycle_index"}.issubset(frame.columns):
        raise ValueError("prefix invariant check requires dataset, cell_id, and cycle_index")
    _assert_unique_identity(frame)
    x = _target_stripped(frame)
    y = frame["soh"].to_numpy(copy=True) if "soh" in frame.columns else np.empty(len(frame))
    full = pipeline.transform(x.copy(deep=True), pipeline.fit(x.copy(deep=True), y.copy(), _copy_meta(meta)), _copy_meta(meta))
    if not isinstance(full, pd.DataFrame):
        raise ValueError("prefix invariant check requires transform to return a DataFrame")
    created = [column for column in full.columns if column not in x.columns]
    if not created:
        return

    key_columns = ["dataset", "cell_id", "cycle_index"]
    for _, group in frame.groupby(["dataset", "cell_id"], sort=False):
        ordered = group.sort_values("cycle_index", kind="mergesort")
        for stop in range(1, len(ordered) + 1):
            prefix = ordered.iloc[:stop]
            prefix_x = _target_stripped(prefix)
            prefix_y = prefix["soh"].to_numpy(copy=True) if "soh" in prefix.columns else np.empty(len(prefix))
            prefix_out = pipeline.transform(
                prefix_x.copy(deep=True),
                pipeline.fit(prefix_x.copy(deep=True), prefix_y.copy(), _copy_meta(meta)),
                _copy_meta(meta),
            )
            _compare_prefix_features(full, prefix_out, key_columns, created, rtol, atol)


def _assert_unique_identity(frame: pd.DataFrame) -> None:
    keys = ["dataset", "cell_id", "cycle_index"]
    if frame.duplicated(keys).any():
        raise ValueError("prefix invariant check requires unique dataset/cell_id/cycle_index rows")


def _compare_prefix_features(
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    keys: list[str],
    columns: list[str],
    rtol: float,
    atol: float,
) -> None:
    missing = [column for column in columns if column not in prefix.columns]
    if missing:
        raise ValueError(f"feature pipeline is not prefix invariant: missing columns {missing}")
    full_indexed = full.set_index(keys)[columns]
    prefix_indexed = prefix.set_index(keys)[columns]
    expected = full_indexed.loc[prefix_indexed.index]
    for column in columns:
        before = expected[column].to_numpy()
        after = prefix_indexed[column].to_numpy()
        if pd.api.types.is_numeric_dtype(expected[column]) and pd.api.types.is_numeric_dtype(prefix_indexed[column]):
            same = np.allclose(before, after, rtol=rtol, atol=atol, equal_nan=True)
        else:
            same = pd.Series(before).equals(pd.Series(after))
        if not same:
            raise ValueError(f"feature pipeline is not prefix invariant for column {column!r}")
