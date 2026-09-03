from __future__ import annotations
import pandas as pd
from .contracts import feature_columns

def unified_features(table: pd.DataFrame):
    """Default 'unified' feature view: every estimator sees identical columns (D8 fairness)."""
    cols = feature_columns(table)
    return table[cols].copy(), cols
