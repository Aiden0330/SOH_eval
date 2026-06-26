from __future__ import annotations
import numpy as np, pandas as pd
from .contracts import QCReport, ID_COLUMNS

def qc_gate(table: pd.DataFrame, meta: pd.DataFrame, *, lower: float = 0.0,
            upper: float = 1.3, availability_min: float = 0.2,
            reject_bms: bool = True) -> tuple[pd.DataFrame, QCReport]:
    n_in = len(table)
    drop_reasons: dict[str, int] = {}

    if reject_bms:
        bad = meta.loc[meta["label_source"] == "bms_reported", "dataset"].unique().tolist()
        if bad:
            raise ValueError(f"datasets with bms_reported labels rejected: {bad}")

    # value-range
    in_range = (table["soh"] > lower) & (table["soh"] < upper)
    drop_reasons["soh_out_of_range"] = int((~in_range).sum())
    clean = table.loc[in_range].reset_index(drop=True)

    # availability mask on feature columns (no fill)
    feat_cols = [c for c in clean.columns if c not in ID_COLUMNS]
    dropped_columns = []
    for c in feat_cols:
        avail = clean[c].notna().mean() if len(clean) else 0.0
        if avail < availability_min:
            dropped_columns.append(c)
    clean = clean.drop(columns=dropped_columns)

    # Drop rows that still contain NaN in any kept feature column after the column-availability
    # mask step above.  Such rows would cause Ridge.fit (and any other numpy-backed estimator)
    # to raise with "Input X contains NaN".  We record the count so callers can audit.
    kept_feat_cols = [c for c in clean.columns if c not in ID_COLUMNS]
    if kept_feat_cols:
        nan_row_mask = clean[kept_feat_cols].isna().any(axis=1)
        drop_reasons["nan_feature_row"] = int(nan_row_mask.sum())
        clean = clean.loc[~nan_row_mask].reset_index(drop=True)

    n_out = n_in - len(clean)
    rep = QCReport(
        pass_rate=(len(clean) / n_in) if n_in else 0.0,
        n_in=n_in, n_out=n_out, drop_reasons=drop_reasons,
        dropped_columns=dropped_columns,
        per_dataset={d: {"rows": int((clean["dataset"] == d).sum())}
                     for d in clean["dataset"].unique()},
    )
    return clean, rep
