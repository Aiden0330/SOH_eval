from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Literal, Optional
import numpy as np
import pandas as pd

ID_COLUMNS = ["dataset", "cell_id", "cycle_index", "soh"]

def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ID_COLUMNS]

@dataclass(frozen=True)
class CellMeta:
    dataset: str
    cell_id: str
    chemistry: str
    c_rated_ah: float
    label_source: str  # "calibration_capacity" | "bms_reported" | "unknown"
    cathode: Optional[str] = None
    anode: Optional[str] = None
    cell_format: Optional[str] = None
    station: Optional[str] = None

@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    held_out_key: str

@dataclass
class QCReport:
    pass_rate: float
    n_in: int
    n_out: int
    drop_reasons: dict[str, int] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    per_dataset: dict[str, dict] = field(default_factory=dict)

@dataclass
class Verdict:
    name: str  # "generalization" | "calibration" | "physical_consistency"
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

@runtime_checkable
class Estimator(Protocol):
    name: str
    feature_mode: Literal["unified", "byo"]
    backend: Literal["sklearn", "pytorch", "external"]
    def fit(self, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame) -> None: ...
    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray: ...
