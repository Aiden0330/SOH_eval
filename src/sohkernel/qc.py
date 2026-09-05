from __future__ import annotations
import numpy as np, pandas as pd
from .contracts import QCReport, ID_COLUMNS

def qc_gate(table: pd.DataFrame, meta: pd.DataFrame, *, lower: float = 0.0,
            upper: float = 1.3, availability_min: float = 0.2,
            reject_bms: bool = True, keep_unlabeled: bool = False) -> tuple[pd.DataFrame, QCReport]:
    """Row and column QC.

    Rows without a label (``soh`` NaN — field segments before a unit's first capacity test, or an
    ``unlabeled_field`` dataset, Spec 12 §3–§4) are dropped and counted as ``soh_unlabeled`` unless
    ``keep_unlabeled`` is set (inference, audit and drift paths); the value checks apply to the
    labeled rows only.
    """
    n_in = len(table)
    drop_reasons: dict[str, int] = {}

    if reject_bms:
        bad = meta.loc[meta["label_source"] == "bms_reported", "dataset"].unique().tolist()
        if bad:
            raise ValueError(f"datasets with bms_reported labels rejected: {bad}")

    soh = pd.to_numeric(table["soh"], errors="coerce")
    unlabeled = soh.isna().to_numpy()
    drop_reasons["soh_unlabeled"] = 0 if keep_unlabeled else int(unlabeled.sum())

    # non-positive: explicit garbage drop (before range check)
    nonpositive = (soh <= 0).to_numpy()
    drop_reasons["soh_nonpositive"] = int(nonpositive.sum())

    # value-range: lower-inclusive, upper-exclusive [lower, upper)
    in_range = ((soh >= lower) & (soh < upper)).to_numpy()
    drop_reasons["soh_out_of_range"] = int((~in_range & ~unlabeled & ~nonpositive).sum())
    keep = (in_range & ~nonpositive) | (unlabeled if keep_unlabeled else np.zeros(n_in, dtype=bool))
    clean = table.loc[keep].reset_index(drop=True)

    # Ambiguous cycle identities: a (dataset, cell_id, cycle_index) seen more than once
    # cannot be ordered in a trajectory nor matched to one curve, so every row of such
    # an identity is rejected and counted (Spec 01 §4; extraction plan Task 7f).
    identity = [c for c in ("dataset", "cell_id", "cycle_index") if c in clean.columns]
    duplicated = clean.duplicated(identity, keep=False) if len(identity) == 3 else pd.Series(False, index=clean.index)
    drop_reasons["duplicate_cycle_identity"] = int(duplicated.sum())
    clean = clean.loc[~duplicated].reset_index(drop=True)

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
