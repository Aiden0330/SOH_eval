from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


EXECUTION_UNIT_COLUMNS = (
    "estimator",
    "featureset",
    "protocol",
    "held_out_key",
    "fold",
    "seed",
)


def iter_execution_units(results: pd.DataFrame) -> Iterator[pd.DataFrame]:
    if results.empty:
        return
    unit_columns = [column for column in EXECUTION_UNIT_COLUMNS if column in results.columns]
    if not unit_columns:
        yield results
        return
    grouper = unit_columns[0] if len(unit_columns) == 1 else unit_columns
    for _, unit in results.groupby(grouper, dropna=False, sort=False):
        yield unit


def execution_unit_succeeded(unit: pd.DataFrame) -> bool:
    if unit.empty:
        return False
    if "unit_status" in unit.columns:
        statuses = unit["unit_status"]
        return bool(statuses.notna().all() and statuses.eq("succeeded").all())
    if "value" not in unit.columns:
        return False
    numeric_values = pd.to_numeric(unit["value"], errors="coerce")
    return bool(np.isfinite(numeric_values.to_numpy(dtype=float, na_value=np.nan)).any())


def successful_result_rows(results: pd.DataFrame) -> pd.DataFrame:
    successful_units = [unit for unit in iter_execution_units(results) if execution_unit_succeeded(unit)]
    if not successful_units:
        return results.iloc[0:0].copy()
    return pd.concat(successful_units).copy()


@dataclass(frozen=True)
class RunOutcome:
    outcome: str
    attempted: int
    succeeded: int
    failed: int

    @classmethod
    def from_results(cls, results: pd.DataFrame) -> "RunOutcome":
        if results.empty:
            return cls(outcome="failed", attempted=0, succeeded=0, failed=0)

        unit_successes = [execution_unit_succeeded(unit) for unit in iter_execution_units(results)]
        attempted = len(unit_successes)
        succeeded = sum(unit_successes)

        failed = attempted - succeeded
        outcome = "complete" if attempted and failed == 0 else "partial" if succeeded else "failed"
        return cls(outcome=outcome, attempted=attempted, succeeded=succeeded, failed=failed)
