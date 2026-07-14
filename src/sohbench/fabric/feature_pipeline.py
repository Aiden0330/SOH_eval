from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Protocol

import numpy as np
import pandas as pd

from .ids import canonical_json, sha256_file, sha256_json


class _FrozenMapping(Mapping[str, Any]):
    """Pickle-friendly immutable mapping that does not inherit ``dict``."""

    def __init__(self, value: Mapping[str, Any] = ()): 
        self._items = tuple((str(key), _freeze(item)) for key, item in dict(value).items())

    def __getitem__(self, key: str) -> Any:
        for stored_key, value in self._items:
            if stored_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(self):
        return (_FrozenMapping, (_thaw(self),))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"FeatureState payload contains non-JSON value of type {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class FeatureState:
    """Immutable, serializable fitting state for a feature pipeline."""

    payload: Mapping[str, Any]
    digest: str

    def __post_init__(self) -> None:
        clean = json.loads(canonical_json(_thaw(self.payload)).decode("utf-8"))
        digest = sha256_json(clean)
        if self.digest != digest:
            raise ValueError("FeatureState digest does not match canonical payload")
        object.__setattr__(self, "payload", _FrozenMapping(clean))

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "FeatureState":
        clean = json.loads(canonical_json(dict(payload)).decode("utf-8"))
        return cls(payload=clean, digest=sha256_json(clean))

    def to_dict(self) -> dict:
        return {"payload": _thaw(self.payload), "digest": self.digest}

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

    def transform(self, frame: pd.DataFrame, state: FeatureState, meta: pd.DataFrame | None) -> pd.DataFrame: ...


def clone_for_fold(pipeline: FeaturePipeline) -> FeaturePipeline:
    clone = getattr(pipeline, "clone_for_fold", None)
    if callable(clone):
        cloned = clone()
    else:
        try:
            cloned = copy.deepcopy(pipeline)
        except Exception as exc:
            raise ValueError("feature pipeline must support deepcopy or clone_for_fold()") from exc
    if cloned is pipeline:
        raise ValueError("feature pipeline clone_for_fold() must return a distinct instance")
    return cloned


def scoped_meta_for_frame(meta: pd.DataFrame | None, frame: pd.DataFrame, target: str = "soh") -> pd.DataFrame | None:
    """Return independent, target-free meta limited to assets present in ``frame``."""
    if not isinstance(meta, pd.DataFrame):
        return copy.deepcopy(meta)
    clean = meta.drop(columns=[target], errors="ignore").copy(deep=True)
    if clean.empty:
        return clean
    keys = ["dataset", "cell_id"]
    if not set(keys).issubset(clean.columns):
        # Dropping unkeyed metadata is safer than handing a global table to an algorithm.
        return clean.iloc[0:0].copy()
    if not set(keys).issubset(frame.columns):
        raise ValueError("meta scoping requires dataset and cell_id in the current frame")
    assets = frame[keys].drop_duplicates().copy()
    return clean.merge(assets, on=keys, how="inner", sort=False).copy(deep=True)


class IdentityFeaturePipeline:
    def clone_for_fold(self):
        return IdentityFeaturePipeline()

    def fit(self, train_x: pd.DataFrame, train_y: np.ndarray, meta: pd.DataFrame | None) -> FeatureState:
        return FeatureState.create({"kind": "identity"})

    def transform(self, frame: pd.DataFrame, state: FeatureState, meta: pd.DataFrame | None) -> pd.DataFrame:
        return frame.copy(deep=True)


class LegacyStatelessPipeline:
    """Safe fold-scoped adapter that reloads trusted transform code per invocation."""

    def __init__(self, name: str, candidates: list[str], transform: Callable, path: str | Path):
        self.name = str(name)
        self.candidates = tuple(map(str, candidates))
        self._path = Path(path)

    def clone_for_fold(self):
        return LegacyStatelessPipeline(self.name, list(self.candidates), lambda *_: None, self._path)

    def fit(self, train_x: pd.DataFrame, train_y: np.ndarray, meta: pd.DataFrame | None) -> FeatureState:
        transform, callable_digest = _load_isolated_transform(self._path)
        del transform
        return FeatureState.create(
            {
                "kind": "legacy_stateless",
                "plugin_name": self.name,
                "plugin_digest": sha256_file(self._path),
                "callable_digest": callable_digest,
                "candidates": list(self.candidates),
            }
        )

    def transform(self, frame: pd.DataFrame, state: FeatureState, meta: pd.DataFrame | None) -> pd.DataFrame:
        transform, callable_digest = _load_isolated_transform(self._path)
        if state.payload.get("plugin_digest") != sha256_file(self._path):
            raise ValueError(f"feature transform {self.name!r} plugin digest does not match fitted state")
        if state.payload.get("callable_digest") != callable_digest:
            raise ValueError(f"feature transform {self.name!r} callable digest does not match fitted state")
        source = _target_stripped(frame)
        safe_meta = scoped_meta_for_frame(meta, source)
        transformed = transform(source.copy(deep=True), safe_meta)
        _validate_legacy_output(self.name, source, transformed, self.candidates)
        return transformed.copy(deep=True)


def _load_isolated_transform(path: Path) -> tuple[Callable, str]:
    module_name = f"sohbench_feature_runtime_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load feature transform from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transform = getattr(module, "transform", None)
    if not callable(transform):
        raise ValueError(f"feature transform {path.name!r} does not define transform()")
    try:
        source = inspect.getsource(transform)
    except (OSError, TypeError):
        source = path.read_text(encoding="utf-8")
    return transform, hashlib.sha256(source.encode("utf-8")).hexdigest()


def _target_stripped(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("feature pipeline input must be a DataFrame")
    return frame.drop(columns=["soh"], errors="ignore").copy(deep=True)


def _validate_legacy_output(name: str, source: pd.DataFrame, transformed: Any, candidates: tuple[str, ...]) -> None:
    if not isinstance(transformed, pd.DataFrame):
        raise ValueError(f"feature transform {name!r} did not return a DataFrame")
    if len(transformed) != len(source):
        raise ValueError(f"feature transform {name!r} changed row count")
    if not transformed.index.equals(source.index):
        raise ValueError(f"feature transform {name!r} changed row index")
    for column in (column for column in ("dataset", "cell_id", "cycle_index") if column in source.columns):
        if column not in transformed.columns or not transformed[column].equals(source[column]):
            raise ValueError(f"feature transform {name!r} changed identity column {column!r}")
    missing = [column for column in candidates if column not in transformed.columns]
    if missing:
        raise ValueError(f"feature transform {name!r} did not create columns: {missing}")


def assert_prefix_invariant(pipeline: FeaturePipeline, frame: pd.DataFrame, meta: pd.DataFrame | None, *, rtol: float = 1e-9, atol: float = 1e-12) -> None:
    if not {"dataset", "cell_id", "cycle_index"}.issubset(frame.columns):
        raise ValueError("prefix invariant check requires dataset, cell_id, and cycle_index")
    _assert_unique_identity(frame)
    x = _target_stripped(frame)
    y = frame["soh"].to_numpy(copy=True) if "soh" in frame.columns else np.empty(len(frame))
    fitted = clone_for_fold(pipeline)
    state = fitted.fit(x.copy(deep=True), y.copy(), scoped_meta_for_frame(meta, x))
    full = clone_for_fold(pipeline).transform(x.copy(deep=True), state, scoped_meta_for_frame(meta, x))
    if not isinstance(full, pd.DataFrame):
        raise ValueError("prefix invariant check requires transform to return a DataFrame")
    keys = ["dataset", "cell_id", "cycle_index"]
    compared = [column for column in full.columns if column not in keys]
    for _, group in frame.groupby(["dataset", "cell_id"], sort=False):
        ordered = group.sort_values("cycle_index", kind="mergesort")
        for stop in range(1, len(ordered) + 1):
            prefix_x = _target_stripped(ordered.iloc[:stop])
            prefix = clone_for_fold(pipeline).transform(prefix_x.copy(deep=True), state, scoped_meta_for_frame(meta, prefix_x))
            _compare_prefix_features(full, prefix, keys, compared, rtol, atol)


def _assert_unique_identity(frame: pd.DataFrame) -> None:
    if frame.duplicated(["dataset", "cell_id", "cycle_index"]).any():
        raise ValueError("prefix invariant check requires unique dataset/cell_id/cycle_index rows")


def _compare_prefix_features(full: pd.DataFrame, prefix: pd.DataFrame, keys: list[str], columns: list[str], rtol: float, atol: float) -> None:
    missing = [column for column in columns if column not in prefix.columns]
    if missing:
        raise ValueError(f"feature pipeline is not prefix invariant: missing columns {missing}")
    expected = full.set_index(keys)[columns].loc[prefix.set_index(keys).index]
    actual = prefix.set_index(keys)[columns]
    for column in columns:
        before, after = expected[column].to_numpy(), actual[column].to_numpy()
        same = np.allclose(before, after, rtol=rtol, atol=atol, equal_nan=True) if pd.api.types.is_numeric_dtype(expected[column]) and pd.api.types.is_numeric_dtype(actual[column]) else pd.Series(before).equals(pd.Series(after))
        if not same:
            raise ValueError(f"feature pipeline is not prefix invariant for column {column!r}")
