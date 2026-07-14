from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


_SORT_KEYS = ("dataset", "cell_id", "cycle_index")


class _FrozenDict(dict):
    def __init__(self, value=()):
        dict.__init__(self, {key: _frozen(item) for key, item in dict(value).items()})

    def __reduce__(self):
        return (_FrozenDict, (dict(self),))

    def _immutable(self, *args, **kwargs):
        raise TypeError("FrozenDict is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, set):
        raise ValueError("QC contains non-JSON set value")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"QC contains non-JSON value of type {type(value).__name__}")


def _row_hashes(frame: pd.DataFrame) -> pd.Series:
    try:
        return pd.util.hash_pandas_object(frame, index=False).astype("uint64")
    except (TypeError, ValueError) as exc:
        raise ValueError("frame contains unhashable cell values") from exc


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    row_hash = _row_hashes(ordered).to_numpy()
    sort_keys = [column for column in _SORT_KEYS if column in ordered.columns]
    sort_positions = [ordered.columns.get_loc(column) for column in sort_keys]
    helper = ordered.iloc[:, sort_positions].copy()
    helper.columns = range(len(sort_positions))
    helper["__row_hash"] = row_hash
    helper["__position"] = range(len(ordered))
    positions = helper.sort_values(
        by=[*range(len(sort_positions)), "__row_hash", "__position"],
        kind="mergesort",
        na_position="first",
    )["__position"].to_numpy()
    return ordered.iloc[positions].reset_index(drop=True)


def frame_digest(frame: pd.DataFrame) -> str:
    ordered = _ordered_frame(frame)
    digest = hashlib.sha256()
    for column in ordered.columns:
        encoded = f"{len(str(column))}:{column}\0{str(ordered[column].dtype)}\0".encode()
        digest.update(encoded)
    digest.update(_row_hashes(ordered).values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSnapshot:
    snapshot_id: str
    digest: str
    datasets: tuple[str, ...]
    rows: int
    assets: int
    label_digest: str
    qc: Mapping[str, Any]

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        datasets,
        label_digest,
        qc,
    ) -> "DatasetSnapshot":
        digest = frame_digest(frame)
        if not {"dataset", "cell_id"}.issubset(frame.columns):
            raise ValueError("snapshot frame requires dataset and cell_id columns")
        return cls(
            snapshot_id=f"snapshot-{digest[:16]}",
            digest=digest,
            datasets=tuple(sorted(map(str, datasets))),
            rows=len(frame),
            assets=frame[["dataset", "cell_id"]].drop_duplicates().shape[0],
            label_digest=str(label_digest),
            qc=_frozen(qc),
        )
