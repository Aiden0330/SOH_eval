from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


EXECUTION_UNIT_COLUMNS = (
    "estimator",
    "featureset",
    "protocol",
    "held_out_key",
    "fold",
    "seed",
)


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

        unit_columns = [column for column in EXECUTION_UNIT_COLUMNS if column in results.columns]
        if unit_columns:
            grouper = unit_columns[0] if len(unit_columns) == 1 else unit_columns
            groups = (
                group
                for _, group in results.groupby(grouper, dropna=False, sort=False)
            )
        else:
            groups = iter((results,))

        attempted = 0
        succeeded = 0
        for unit in groups:
            attempted += 1
            if "unit_status" in unit.columns:
                statuses = unit["unit_status"].dropna().astype(str)
                unit_succeeded = bool(len(statuses)) and statuses.eq("succeeded").all()
            else:
                unit_succeeded = "value" in unit.columns and unit["value"].notna().any()
            succeeded += int(unit_succeeded)

        failed = attempted - succeeded
        outcome = "complete" if attempted and failed == 0 else "partial" if succeeded else "failed"
        return cls(outcome=outcome, attempted=attempted, succeeded=succeeded, failed=failed)
