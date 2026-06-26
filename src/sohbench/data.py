from __future__ import annotations
from pathlib import Path
import pandas as pd
from .metadata import build_meta
from .qc import qc_gate

class DatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

    def _read_one(self, name: str) -> pd.DataFrame:
        path = self.data_dir / "processed" / f"{name}_features.pkl"
        if not path.exists():
            raise FileNotFoundError(f"missing dataset file: {path}")
        df = pd.read_pickle(path)
        required = {"dataset", "cell_id", "cycle_index", "soh"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing required columns {missing}")
        return df

    def load(self, names, *, qc: bool = True, **qc_kwargs):
        table = pd.concat([self._read_one(n) for n in names], ignore_index=True)
        meta = build_meta(table)
        if qc:
            table, report = qc_gate(table, meta, **qc_kwargs)
            meta = build_meta(table)  # rebuild after row drops
        else:
            from .contracts import QCReport
            report = QCReport(pass_rate=1.0, n_in=len(table), n_out=0)
        return table, meta, report
