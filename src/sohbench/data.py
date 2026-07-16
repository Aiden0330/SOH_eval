from __future__ import annotations
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
import pandas as pd
from .metadata import build_meta
from .qc import qc_gate

INFERENCE_REQUIRED_COLUMNS = {"dataset", "cell_id", "cycle_index"}
REQUIRED_COLUMNS = INFERENCE_REQUIRED_COLUMNS | {"soh"}
ALL_DATASETS = {"all", "*"}


def normalize_cycle_index(table: pd.DataFrame, *, context: str = "dataset") -> pd.DataFrame:
    if "cycle_index" not in table.columns:
        raise ValueError(f"{context}: missing required column cycle_index")

    source = table["cycle_index"]
    int64_max = int(np.iinfo(np.int64).max)
    if pd.api.types.is_bool_dtype(source.dtype):
        valid = pd.Series(False, index=source.index)
        numeric = pd.Series(0, index=source.index, dtype="int64")
    elif pd.api.types.is_integer_dtype(source.dtype):
        valid = (
            source.notna()
            & source.ge(0)
            & source.le(int64_max)
        ).fillna(False)
        numeric = source
    elif pd.api.types.is_float_dtype(source.dtype):
        values = source.to_numpy(dtype="float64", na_value=np.nan)
        valid = pd.Series(
            np.isfinite(values)
            & (values >= 0)
            & (values < float(2**63))
            & (values == np.floor(values)),
            index=source.index,
        )
        numeric = pd.Series(values, index=source.index)
    else:
        converted = []
        validity = []
        for value in source.tolist():
            candidate = 0
            is_valid = not isinstance(value, (bool, np.bool_))
            try:
                decimal_value = Decimal(str(value).strip())
                is_valid = (
                    is_valid
                    and decimal_value.is_finite()
                    and decimal_value == decimal_value.to_integral_value()
                )
                if is_valid:
                    candidate = int(decimal_value)
                    is_valid = 0 <= candidate <= int64_max
            except (InvalidOperation, TypeError, ValueError):
                is_valid = False
            converted.append(candidate)
            validity.append(is_valid)
        valid = pd.Series(validity, index=source.index, dtype=bool)
        numeric = pd.Series(converted, index=source.index, dtype="object")
    if not bool(valid.all()):
        invalid = source.loc[~valid].head(3).tolist()
        raise ValueError(
            f"{context}: cycle_index must contain finite integers in the "
            f"range 0..{int64_max}; "
            f"invalid values: {invalid}"
        )

    normalized = table.copy()
    normalized["cycle_index"] = numeric.astype("int64")
    return normalized

class DatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def discover_datasets(self, *, require_labels: bool = True) -> list[str]:
        processed = self.data_dir / "processed"
        if not processed.exists():
            return []

        names = []
        for path in sorted(processed.glob("*_features.pkl")):
            name = path.name.removesuffix("_features.pkl")
            try:
                df = pd.read_pickle(path)
            except Exception:
                continue
            required = REQUIRED_COLUMNS if require_labels else INFERENCE_REQUIRED_COLUMNS
            if required <= set(df.columns):
                names.append(name)
        return names

    def resolve_datasets(self, names, *, require_labels: bool = True) -> list[str]:
        if names is None:
            names = ["all"]
        if isinstance(names, str):
            names = [names]
        names = list(names)
        if not names or any(str(name).lower() in ALL_DATASETS for name in names):
            names = self.discover_datasets(require_labels=require_labels)
        if not names:
            raise ValueError(f"no valid *_features.pkl datasets found under {self.data_dir / 'processed'}")
        return names

    def _read_one(self, name: str, *, require_labels: bool = True) -> pd.DataFrame:
        path = self.data_dir / "processed" / f"{name}_features.pkl"
        if not path.exists():
            raise FileNotFoundError(f"missing dataset file: {path}")
        df = pd.read_pickle(path)
        required = REQUIRED_COLUMNS if require_labels else INFERENCE_REQUIRED_COLUMNS
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing required columns {missing}")
        # Normalize 'dataset' column to the canonical loader name so that
        # build_meta's DATASET_CHEMISTRY lookup (lowercase keys) works regardless
        # of how the pkl stores the dataset identifier (e.g. "NASA" vs "nasa").
        df = df.copy()
        df["dataset"] = name
        return normalize_cycle_index(df, context=name)

    def load(self, names, *, qc: bool = True, require_labels: bool = True, **qc_kwargs):
        names = self.resolve_datasets(names, require_labels=require_labels)
        table = pd.concat(
            [self._read_one(n, require_labels=require_labels) for n in names],
            ignore_index=True,
        )
        meta = build_meta(table)
        if qc:
            if "soh" not in table.columns:
                raise ValueError("label QC requires the soh column")
            table, report = qc_gate(table, meta, **qc_kwargs)
            meta = build_meta(table)  # rebuild after row drops
        else:
            from .contracts import QCReport
            report = QCReport(pass_rate=1.0, n_in=len(table), n_out=0)
        return table, meta, report
