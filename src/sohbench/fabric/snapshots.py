from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd


_SORT_KEYS = ("dataset", "cell_id", "cycle_index")


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, set):
        return frozenset(_frozen(item) for item in value)
    return value


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    row_hash = pd.util.hash_pandas_object(ordered, index=False).astype("uint64")
    sort_keys = [column for column in _SORT_KEYS if column in ordered.columns]
    ordered["__fabric_row_hash__"] = row_hash.to_numpy()
    ordered = ordered.sort_values(
        by=[*sort_keys, "__fabric_row_hash__"],
        kind="mergesort",
        na_position="first",
    )
    return ordered.drop(columns=["__fabric_row_hash__"])


def frame_digest(frame: pd.DataFrame) -> str:
    ordered = _ordered_frame(frame)
    digest = hashlib.sha256()
    for column in ordered.columns:
        encoded = f"{len(str(column))}:{column}\0{str(ordered[column].dtype)}\0".encode()
        digest.update(encoded)
    digest.update(pd.util.hash_pandas_object(ordered, index=False).values.tobytes())
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
