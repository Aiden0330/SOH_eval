from __future__ import annotations
from pathlib import Path
import pandas as pd
from .metadata import build_meta
from .qc import qc_gate

REQUIRED_COLUMNS = {"dataset", "cell_id", "cycle_index", "soh"}
ALL_DATASETS = {"all", "*"}

class DatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def discover_datasets(self) -> list[str]:
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
            if REQUIRED_COLUMNS <= set(df.columns):
                names.append(name)
        return names

    def resolve_datasets(self, names) -> list[str]:
        if names is None:
            names = ["all"]
        if isinstance(names, str):
            names = [names]
        names = list(names)
        if not names or any(str(name).lower() in ALL_DATASETS for name in names):
            names = self.discover_datasets()
        if not names:
            raise ValueError(f"no valid *_features.pkl datasets found under {self.data_dir / 'processed'}")
        return names

    def _read_one(self, name: str) -> pd.DataFrame:
        path = self.data_dir / "processed" / f"{name}_features.pkl"
        if not path.exists():
            raise FileNotFoundError(f"missing dataset file: {path}")
        df = pd.read_pickle(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing required columns {missing}")
        # Normalize 'dataset' column to the canonical loader name so that
        # build_meta's DATASET_CHEMISTRY lookup (lowercase keys) works regardless
        # of how the pkl stores the dataset identifier (e.g. "NASA" vs "nasa").
        df = df.copy()
        df["dataset"] = name
        return df

    def load(self, names, *, qc: bool = True, **qc_kwargs):
        names = self.resolve_datasets(names)
        table = pd.concat([self._read_one(n) for n in names], ignore_index=True)
        meta = build_meta(table)
        if qc:
            table, report = qc_gate(table, meta, **qc_kwargs)
            meta = build_meta(table)  # rebuild after row drops
        else:
            from .contracts import QCReport
            report = QCReport(pass_rate=1.0, n_in=len(table), n_out=0)
        return table, meta, report
