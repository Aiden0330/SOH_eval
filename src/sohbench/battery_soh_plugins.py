from __future__ import annotations

from pathlib import Path

from .trusted_plugins import TrustedPluginStore


DEFAULT_BATTERY_SOH_ROOT = Path(r"D:\battery_SOH")
ESTIMATOR_NAME = "battery_soh_tabular_ensemble"
FEATURESET_NAME = "battery_soh_operational"


def register_battery_soh_plugins(
    store: TrustedPluginStore,
    *,
    battery_root: str | Path = DEFAULT_BATTERY_SOH_ROOT,
    replace: bool = False,
) -> dict:
    root = validate_battery_soh_root(battery_root)
    estimator = store.upload_estimator_plugin(
        ESTIMATOR_NAME,
        f"{ESTIMATOR_NAME}.py",
        build_battery_soh_estimator_plugin_source(root).encode("utf-8"),
        replace=replace,
    )
    featureset = store.upload_featureset_plugin(
        FEATURESET_NAME,
        f"{FEATURESET_NAME}.py",
        build_battery_soh_featureset_plugin_source().encode("utf-8"),
        replace=replace,
    )
    return {
        "battery_soh_root": str(root),
        "estimator": estimator,
        "featureset": featureset,
    }


def validate_battery_soh_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    model_path = path / "soh_core" / "models" / "tabular_ensemble.py"
    if not model_path.exists():
        raise FileNotFoundError(f"battery_SOH TabularSOHEnsemble not found: {model_path}")
    return path


def build_battery_soh_estimator_plugin_source(root: str | Path) -> str:
    root_literal = repr(str(Path(root)))
    return f'''from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sohbench.contracts import ID_COLUMNS, feature_columns


BATTERY_SOH_ROOT = Path({root_literal})
if str(BATTERY_SOH_ROOT) not in sys.path:
    sys.path.insert(0, str(BATTERY_SOH_ROOT))

from soh_core.models.tabular_ensemble import TabularSOHEnsemble


class Estimator:
    name = "{ESTIMATOR_NAME}"
    feature_mode = "unified"
    backend = "battery_soh"

    def __init__(self, random_state: int = 2026):
        self.random_state = random_state
        self._model = TabularSOHEnsemble(random_state=random_state)
        self._cols = None

    def fit(self, X, y, meta=None):
        self._cols = [col for col in feature_columns(X) if col not in ID_COLUMNS]
        if not self._cols:
            raise ValueError("battery_SOH estimator requires at least one feature column")
        self._model.fit(self._to_array(X), np.asarray(y, dtype=float))
        return self

    def predict(self, X, meta=None):
        if self._cols is None:
            raise RuntimeError("model must be fitted before prediction")
        return np.asarray(self._model.predict(self._to_array(X)), dtype=float)

    def _to_array(self, X):
        frame = pd.DataFrame(X).copy()
        for col in self._cols or []:
            if col not in frame.columns:
                frame[col] = np.nan
        numeric = frame[self._cols].apply(pd.to_numeric, errors="coerce")
        return numeric.to_numpy(dtype=float)
'''


def build_battery_soh_featureset_plugin_source() -> str:
    return f'''from __future__ import annotations

import numpy as np
import pandas as pd


FEATURESET_NAME = "{FEATURESET_NAME}"
FEATURE_COLUMNS = [
    "battery_soh_voltage_sag",
]


def transform(table, meta=None):
    out = pd.DataFrame(table).copy()
    out["battery_soh_voltage_sag"] = 0.0

    group_cols = [col for col in ("dataset", "cell_id") if col in out.columns]
    groups = out.groupby(group_cols, sort=False) if group_cols else [(None, out)]
    for _, group in groups:
        idx = group.index
        voltage = _numeric(group, "voltage_mean", default=np.zeros(len(group), dtype=float))
        first_position = group["cycle_index"].astype(float).to_numpy().argmin() if "cycle_index" in group else 0
        out.loc[idx, "battery_soh_voltage_sag"] = float(voltage.iloc[first_position]) - voltage

    return out


def _numeric(frame, column, default):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
'''
