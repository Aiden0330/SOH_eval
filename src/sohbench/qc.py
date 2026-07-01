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

    # non-positive: explicit garbage drop (before range check)
    soh = table["soh"]
    nonpositive = soh <= 0
    drop_reasons["soh_nonpositive"] = int(nonpositive.sum())
    after_np = table.loc[~nonpositive].reset_index(drop=True)

    # value-range: lower-inclusive, upper-exclusive [lower, upper)
    soh2 = after_np["soh"]
    in_range = (soh2 >= lower) & (soh2 < upper)
    drop_reasons["soh_out_of_range"] = int((~in_range).sum())
    clean = after_np.loc[in_range].reset_index(drop=True)

    # availability mask on feature columns (no fill)
    feat_cols = [c for c in clean.columns if c not in ID_COLUMNS]
    dropped_columns = []
    for c in feat_cols:
        avail = clean[c].notna().mean() if len(clean) else 0.0
        if avail < availability_min:
            dropped_columns.append(c)
    clean = clean.drop(columns=dropped_columns)

    # Do not drop rows only because a merged multi-dataset table has structural
    # feature gaps. Estimators fit train-fold imputers so this remains fair
    # under cross-dataset protocols without leaking held-out feature values.
    drop_reasons["nan_feature_row"] = 0

    n_out = n_in - len(clean)
    rep = QCReport(
        pass_rate=(len(clean) / n_in) if n_in else 0.0,
        n_in=n_in, n_out=n_out, drop_reasons=drop_reasons,
        dropped_columns=dropped_columns,
        per_dataset={d: {"rows": int((clean["dataset"] == d).sum())}
                     for d in clean["dataset"].unique()},
    )
    return clean, rep
