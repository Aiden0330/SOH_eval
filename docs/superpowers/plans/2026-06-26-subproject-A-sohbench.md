# Sub-project A — `sohbench` Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `sohbench` benchmark/evaluation harness — a unified algorithm interface plus a fair cross-condition comparison engine for SOH estimators — per [sub-project A spec](../specs/2026-06-26-subproject-A-benchmark-harness-spec.md).

**Architecture:** Python package `src/sohbench`, single-responsibility acyclic modules. A unified per-cycle feature table + per-cell meta feed pluggable `Estimator`s; leave-one-domain protocols generate folds with no cell leakage; a guaranteed-CI runner aggregates a D9 metric set across ≥5 seeds; results land as a frozen JSON/Parquet artifact schema the frontend (C) will consume. Clean-room — no dependency on the old `D:\battery_SOH` code/skills (its `data/processed/*.pkl` artifacts are read as data only).

**Tech Stack:** Python 3.11+, numpy, pandas, scikit-learn, scipy, pyarrow (Parquet), jsonschema, pytest.

**Conventions (locked here for cross-task consistency):**
- `ID_COLUMNS = ["dataset", "cell_id", "cycle_index", "soh"]`. A feature table `X` has these plus feature columns. `feature_columns(X)` returns columns not in `ID_COLUMNS`.
- Per-cell metadata travels in a separate `meta` DataFrame: `["dataset","cell_id","chemistry","c_rated_ah","label_source", ...]`.
- `Estimator.fit(X, y, meta)` / `predict(X, meta)`; `y` is the `soh` array for `X`'s rows. Baselines may ignore `meta`.
- SOH value bounds: `lower` default `0.0`, `upper` default `1.3` (SOH > 1.0 legal; D2).
- Tests live under `tests/` mirroring module names. Run all with `pytest -q`.

---

### Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/sohbench/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`, `README.md`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "sohbench"
version = "0.1.0"
description = "SOH benchmark & evaluation harness (sub-project A)"
requires-python = ">=3.11"
dependencies = ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "jsonschema"]

[project.optional-dependencies]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty `src/sohbench/__init__.py` and `tests/__init__.py`**

Both files: empty (0 bytes is fine).

- [ ] **Step 3: Write smoke test** — `tests/test_smoke.py`

```python
import sohbench

def test_package_imports():
    assert sohbench is not None
```

- [ ] **Step 4: Install editable + run smoke test**

Run: `python -m pip install -e ".[dev]" && pytest tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/sohbench/__init__.py tests/__init__.py tests/test_smoke.py README.md
git commit -m "chore(A): scaffold sohbench package"
```

---

### Task 1: `contracts.py` — data structures & Estimator protocol

**Files:**
- Create: `src/sohbench/contracts.py`
- Test: `tests/test_contracts.py`

- [ ] **Step 1: Write failing test** — `tests/test_contracts.py`

```python
import numpy as np
import pandas as pd
from sohbench.contracts import (
    ID_COLUMNS, feature_columns, CellMeta, Fold, QCReport, Verdict, Estimator,
)

def test_feature_columns_excludes_ids():
    df = pd.DataFrame({"dataset": ["a"], "cell_id": ["c1"], "cycle_index": [0],
                       "soh": [1.0], "f1": [0.5], "f2": [0.3]})
    assert feature_columns(df) == ["f1", "f2"]

def test_cellmeta_fields():
    m = CellMeta(dataset="a", cell_id="c1", chemistry="LFP", c_rated_ah=1.1,
                 label_source="calibration_capacity")
    assert m.chemistry == "LFP" and m.c_rated_ah == 1.1

def test_estimator_protocol_runtime_checkable():
    class Dummy:
        name = "d"; feature_mode = "unified"; backend = "sklearn"
        def fit(self, X, y, meta): ...
        def predict(self, X, meta): return np.zeros(len(X))
    assert isinstance(Dummy(), Estimator)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_contracts.py -v`
Expected: FAIL (ModuleNotFoundError: sohbench.contracts).

- [ ] **Step 3: Implement** — `src/sohbench/contracts.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Literal, Optional
import numpy as np
import pandas as pd

ID_COLUMNS = ["dataset", "cell_id", "cycle_index", "soh"]

def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ID_COLUMNS]

@dataclass(frozen=True)
class CellMeta:
    dataset: str
    cell_id: str
    chemistry: str
    c_rated_ah: float
    label_source: str  # "calibration_capacity" | "bms_reported" | "unknown"
    cathode: Optional[str] = None
    anode: Optional[str] = None
    cell_format: Optional[str] = None
    station: Optional[str] = None

@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    held_out_key: str

@dataclass
class QCReport:
    pass_rate: float
    n_in: int
    n_out: int
    drop_reasons: dict[str, int] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    per_dataset: dict[str, dict] = field(default_factory=dict)

@dataclass
class Verdict:
    name: str  # "generalization" | "calibration" | "physical_consistency"
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

@runtime_checkable
class Estimator(Protocol):
    name: str
    feature_mode: Literal["unified", "byo"]
    backend: Literal["sklearn", "pytorch", "external"]
    def fit(self, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame) -> None: ...
    def predict(self, X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray: ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_contracts.py -v` → Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/contracts.py tests/test_contracts.py
git commit -m "feat(A): contracts — ID columns, dataclasses, Estimator protocol"
```

---

### Task 2: `metrics.py` — the D9 metric set (pure functions)

**Files:**
- Create: `src/sohbench/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test** — `tests/test_metrics.py`

```python
import numpy as np
import pytest
from sohbench import metrics as M

yt = np.array([1.00, 0.90, 0.80, 0.70])
yp = np.array([0.98, 0.92, 0.78, 0.69])

def test_rmse_mae_max():
    assert M.mae(yt, yp) == pytest.approx(0.0175)
    assert M.rmse(yt, yp) == pytest.approx(np.sqrt(np.mean((yt-yp)**2)))
    assert M.max_error(yt, yp) == pytest.approx(0.02)

def test_mape():
    assert M.mape(yt, yp) == pytest.approx(np.mean(np.abs((yt-yp)/yt)))

def test_mean_bias_error_sign():
    # pred systematically higher than true => positive bias (over-estimation)
    assert M.mean_bias_error(np.array([0.8, 0.8]), np.array([0.9, 0.9])) == pytest.approx(0.1)
    assert M.mean_bias_error(np.array([0.8, 0.8]), np.array([0.7, 0.7])) == pytest.approx(-0.1)

def test_overestimation_rate():
    assert M.overestimation_rate(np.array([0.8, 0.8, 0.8]),
                                 np.array([0.9, 0.7, 0.9])) == pytest.approx(2/3)

def test_alpha_accuracy():
    # rel errors: 0.02,0.0222,0.025,0.0143 all < 0.15 => 1.0
    assert M.alpha_accuracy(yt, yp, alpha=0.15) == pytest.approx(1.0)

def test_eol_region_error_filters():
    # only samples with yt<=0.8 count: indices 2,3
    exp = np.sqrt(np.mean((yt[2:]-yp[2:])**2))
    assert M.eol_region_error(yt, yp, eol_threshold=0.8) == pytest.approx(exp)

def test_picp():
    lo = np.array([0.95, 0.85, 0.75, 0.65]); hi = np.array([1.05, 0.95, 0.85, 0.75])
    assert M.picp(yt, lo, hi) == pytest.approx(1.0)
    lo2 = np.array([0.99, 0.85, 0.75, 0.65])  # 1.00 still inside [0.99,1.05]
    assert M.picp(yt, lo2, hi) == pytest.approx(1.0)

def test_monotonicity_violation_rate():
    # SOH should be non-increasing; a rise > tol is a violation
    ordered = np.array([1.0, 0.99, 1.02, 0.98])  # one rise 0.99->1.02 (+0.03 > 0.005)
    assert M.monotonicity_violation_rate(ordered, rebound_tol=0.005) == pytest.approx(1/3)

def test_value_range_violation():
    preds = np.array([0.8, 1.25, 1.35, -0.1])  # 1.35 and -0.1 out of [0,1.3)
    assert M.value_range_violation(preds, lower=0.0, upper=1.3) == pytest.approx(0.5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_metrics.py -v` → Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** — `src/sohbench/metrics.py`

```python
from __future__ import annotations
import numpy as np

def _arr(*xs): return [np.asarray(x, dtype=float) for x in xs]

def rmse(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred); return float(np.sqrt(np.mean((yt - yp) ** 2)))

def mae(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred); return float(np.mean(np.abs(yt - yp)))

def max_error(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred); return float(np.max(np.abs(yt - yp)))

def mape(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred)
    mask = yt != 0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))

def mean_bias_error(y_true, y_pred) -> float:
    """Signed mean (pred - true). Positive => systematic over-estimation."""
    yt, yp = _arr(y_true, y_pred); return float(np.mean(yp - yt))

def overestimation_rate(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred); return float(np.mean(yp > yt))

def alpha_accuracy(y_true, y_pred, alpha: float = 0.15) -> float:
    yt, yp = _arr(y_true, y_pred)
    mask = yt != 0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask]) <= alpha))

def eol_region_error(y_true, y_pred, eol_threshold: float = 0.8) -> float:
    yt, yp = _arr(y_true, y_pred)
    sel = yt <= eol_threshold
    if not np.any(sel): return float("nan")
    return float(np.sqrt(np.mean((yt[sel] - yp[sel]) ** 2)))

def picp(y_true, lower, upper) -> float:
    yt, lo, hi = _arr(y_true, lower, upper)
    return float(np.mean((yt >= lo) & (yt <= hi)))

def monotonicity_violation_rate(y_pred_ordered, rebound_tol: float = 0.005) -> float:
    yp = np.asarray(y_pred_ordered, dtype=float)
    if len(yp) < 2: return 0.0
    diffs = np.diff(yp)
    return float(np.mean(diffs > rebound_tol))

def value_range_violation(y_pred, lower: float = 0.0, upper: float = 1.3) -> float:
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yp < lower) | (yp >= upper)))
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_metrics.py -v` → Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/metrics.py tests/test_metrics.py
git commit -m "feat(A): metrics — D9 set incl. signed MBE and overestimation-rate"
```

---

### Task 3: `registry.py` — name registries

**Files:**
- Create: `src/sohbench/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write failing test**

```python
import pytest
from sohbench import registry as R

def test_register_and_get():
    R.clear()
    @R.register_estimator
    class Foo:
        name = "foo"; feature_mode = "unified"; backend = "sklearn"
        def fit(self, X, y, meta): ...
        def predict(self, X, meta): ...
    assert R.get_estimator("foo") is Foo

def test_duplicate_raises():
    R.clear()
    @R.register_estimator
    class A:
        name = "dup"; feature_mode = "unified"; backend = "sklearn"
        def fit(self, X, y, meta): ...
        def predict(self, X, meta): ...
    with pytest.raises(ValueError):
        @R.register_estimator
        class B:
            name = "dup"; feature_mode = "unified"; backend = "sklearn"
            def fit(self, X, y, meta): ...
            def predict(self, X, meta): ...
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_registry.py -v` → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/registry.py`

```python
from __future__ import annotations

_ESTIMATORS: dict[str, type] = {}
_DATASETS: dict[str, object] = {}
_TASKS: dict[str, type] = {}

def _register(store, key, value):
    if key in store:
        raise ValueError(f"duplicate registration: {key!r}")
    store[key] = value
    return value

def register_estimator(cls):
    return _register(_ESTIMATORS, cls.name, cls)

def get_estimator(name): return _ESTIMATORS[name]

def register_dataset(name):
    def deco(obj): return _register(_DATASETS, name, obj)
    return deco

def get_dataset(name): return _DATASETS[name]

def register_task(cls):
    return _register(_TASKS, cls.name, cls)

def get_task(name): return _TASKS[name]

def clear():
    _ESTIMATORS.clear(); _DATASETS.clear(); _TASKS.clear()
```

- [ ] **Step 4: Run to verify it passes** — `pytest tests/test_registry.py -v` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/registry.py tests/test_registry.py
git commit -m "feat(A): registry for estimators/datasets/tasks with duplicate guard"
```

---

### Task 4: `metadata.py` — curated per-dataset metadata

**Files:**
- Create: `src/sohbench/metadata.py`
- Test: `tests/test_metadata.py`

> Chemistry/C_rated/label_source below are best-known values; each carries a `# VERIFY` note. Verify against
> each dataset's source documentation before trusting chemistry-out conclusions (D6).

- [ ] **Step 1: Write failing test**

```python
import pandas as pd
from sohbench.metadata import build_meta, DATASET_CHEMISTRY

def test_known_datasets_have_chemistry():
    for ds in ["nasa", "calce_cs2", "hust", "oxford"]:
        assert ds in DATASET_CHEMISTRY

def test_build_meta_one_row_per_cell():
    table = pd.DataFrame({
        "dataset": ["nasa","nasa","hust"], "cell_id": ["B0005","B0005","H1"],
        "cycle_index": [0,1,0], "soh": [1.0,0.99,1.0],
    })
    meta = build_meta(table)
    assert set(meta["cell_id"]) == {"B0005", "H1"}
    assert meta.loc[meta.cell_id=="H1","chemistry"].iloc[0] == "LFP"
    assert {"dataset","cell_id","chemistry","c_rated_ah","label_source"} <= set(meta.columns)
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/metadata.py`

```python
from __future__ import annotations
import pandas as pd

# VERIFY each against source docs before trusting chemistry-out results.
DATASET_CHEMISTRY = {
    "nasa": "LCO",       # NASA PCoE 18650 (LiCoO2)  # VERIFY
    "calce_cs2": "LCO",  # CALCE CS2 prismatic LCO    # VERIFY
    "hust": "LFP",       # HUST A123 18650 LFP        # VERIFY
    "oxford": "NMC",     # Oxford Kokam pouch NMC      # VERIFY
}
DATASET_C_RATED_AH = {"nasa": 2.0, "calce_cs2": 1.1, "hust": 1.1, "oxford": 0.74}  # VERIFY
DATASET_LABEL_SOURCE = {  # all reference-discharge calibrated capacity
    "nasa": "calibration_capacity", "calce_cs2": "calibration_capacity",
    "hust": "calibration_capacity", "oxford": "calibration_capacity",
}

def build_meta(table: pd.DataFrame) -> pd.DataFrame:
    cells = table[["dataset", "cell_id"]].drop_duplicates().reset_index(drop=True)
    cells["chemistry"] = cells["dataset"].map(DATASET_CHEMISTRY)
    cells["c_rated_ah"] = cells["dataset"].map(DATASET_C_RATED_AH)
    cells["label_source"] = cells["dataset"].map(DATASET_LABEL_SOURCE).fillna("unknown")
    return cells
```

- [ ] **Step 4: Run to verify it passes** → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/metadata.py tests/test_metadata.py
git commit -m "feat(A): curated per-dataset metadata (chemistry/C_rated/label_source)"
```

---

### Task 5: `qc.py` — pre-score QC gate

**Files:**
- Create: `src/sohbench/qc.py`
- Test: `tests/test_qc.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np, pandas as pd
from sohbench.qc import qc_gate
from sohbench.metadata import build_meta

def _table():
    return pd.DataFrame({
        "dataset": ["nasa"]*5,
        "cell_id": ["B0005"]*5,
        "cycle_index": [0,1,2,3,4],
        "soh": [1.25, 0.95, 1.35, 0.0, 0.80],   # 1.35 and 0.0 are out of [0,1.3)
        "f1": [1,2,3,4,5],
        "f_sparse": [np.nan, np.nan, np.nan, np.nan, 1.0],  # 20% available -> dropped (<20% strictly? keep boundary)
    })

def test_value_range_drop():
    t = _table(); meta = build_meta(t)
    clean, rep = qc_gate(t, meta, lower=0.0, upper=1.3)
    assert set(clean["soh"]) <= {1.25, 0.95, 0.80}      # 1.35 and 0.0 dropped
    assert rep.drop_reasons["soh_out_of_range"] == 2
    assert rep.n_out == 2

def test_soh_above_one_is_kept():
    t = _table(); meta = build_meta(t)
    clean, _ = qc_gate(t, meta, lower=0.0, upper=1.3)
    assert (clean["soh"] > 1.0).any()                    # 1.25 retained (D2)

def test_availability_drop_no_fill():
    t = _table(); meta = build_meta(t)
    clean, rep = qc_gate(t, meta, availability_min=0.5)
    assert "f_sparse" in rep.dropped_columns
    assert "f_sparse" not in clean.columns
    assert not clean["f1"].isna().any()                  # no synthetic fill introduced

def test_reject_bms_label_source():
    t = _table()
    meta = build_meta(t); meta.loc[:, "label_source"] = "bms_reported"
    import pytest
    with pytest.raises(ValueError):
        qc_gate(t, meta, reject_bms=True)
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/qc.py`

```python
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
    in_range = (table["soh"] >= lower) & (table["soh"] < upper)
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

    n_out = n_in - len(clean)
    rep = QCReport(
        pass_rate=(len(clean) / n_in) if n_in else 0.0,
        n_in=n_in, n_out=n_out, drop_reasons=drop_reasons,
        dropped_columns=dropped_columns,
        per_dataset={d: {"rows": int((clean["dataset"] == d).sum())}
                     for d in clean["dataset"].unique()},
    )
    return clean, rep
```

- [ ] **Step 4: Run to verify it passes** → 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/qc.py tests/test_qc.py
git commit -m "feat(A): QC gate — soh<1.3 range, availability mask (no fill), reject BMS labels"
```

---

### Task 6: `protocols.py` — leave-one-domain splits (leakage-safe)

**Files:**
- Create: `src/sohbench/protocols.py`
- Test: `tests/test_protocols.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np, pandas as pd
from sohbench.protocols import LeaveOneCellOut, LeaveOneDatasetOut, LeaveOneChemistryOut
from sohbench.metadata import build_meta

def _table():
    rows = []
    for ds, cell, chem in [("nasa","c1","LCO"),("nasa","c2","LCO"),
                           ("hust","h1","LFP"),("oxford","o1","NMC")]:
        for k in range(3):
            rows.append((ds, cell, k, 1.0 - 0.01*k))
    return pd.DataFrame(rows, columns=["dataset","cell_id","cycle_index","soh"])

def test_leave_one_cell_no_leakage():
    t = _table(); meta = build_meta(t)
    for fold in LeaveOneCellOut().generate(t, meta):
        train_cells = set(t.loc[fold.train_idx, "cell_id"])
        test_cells = set(t.loc[fold.test_idx, "cell_id"])
        assert train_cells.isdisjoint(test_cells)           # no cell in both
        assert len(test_cells) == 1

def test_leave_one_dataset_out():
    t = _table(); meta = build_meta(t)
    keys = [f.held_out_key for f in LeaveOneDatasetOut().generate(t, meta)]
    assert set(keys) == {"nasa", "hust", "oxford"}

def test_chemistry_out_downgrades_when_single_dataset(recwarn):
    t = _table(); meta = build_meta(t)
    # LCO has 1 dataset (nasa), LFP 1 (hust), NMC 1 (oxford) -> all single -> downgrade
    folds = list(LeaveOneChemistryOut().generate(t, meta))
    assert any("downgrade" in str(w.message).lower() for w in recwarn.list)
    # downgraded to dataset-out keys
    assert {f.held_out_key for f in folds} == {"nasa", "hust", "oxford"}
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/protocols.py`

```python
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from typing import Iterator
from .contracts import Fold

def _folds_by_group(table: pd.DataFrame, group_vals: pd.Series) -> Iterator[Fold]:
    idx = np.arange(len(table))
    for key in pd.unique(group_vals):
        test_mask = (group_vals == key).to_numpy()
        yield Fold(train_idx=idx[~test_mask], test_idx=idx[test_mask], held_out_key=str(key))

class LeaveOneCellOut:
    def generate(self, table, meta) -> Iterator[Fold]:
        keyed = table["dataset"].astype(str) + "/" + table["cell_id"].astype(str)
        yield from _folds_by_group(table, keyed)

class LeaveOneDatasetOut:
    def generate(self, table, meta) -> Iterator[Fold]:
        yield from _folds_by_group(table, table["dataset"].astype(str))

class LeaveOneChemistryOut:
    def generate(self, table, meta) -> Iterator[Fold]:
        chem = table.merge(meta[["dataset","cell_id","chemistry"]],
                           on=["dataset","cell_id"], how="left")["chemistry"]
        datasets_per_chem = (meta.groupby("chemistry")["dataset"].nunique())
        if (datasets_per_chem < 2).any():
            warnings.warn("chemistry-out downgrade: <2 datasets per chemistry; "
                          "using dataset-out to avoid lab/chemistry confound")
            yield from _folds_by_group(table, table["dataset"].astype(str))
        else:
            yield from _folds_by_group(table, chem)
```

- [ ] **Step 4: Run to verify it passes** → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/protocols.py tests/test_protocols.py
git commit -m "feat(A): leave-one-domain protocols with leakage guard + chemistry downgrade"
```

---

### Task 7: `conformal.py` — split-conformal CI wrapper (leakage-safe calibration)

**Files:**
- Create: `src/sohbench/conformal.py`
- Test: `tests/test_conformal.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np, pandas as pd
from sohbench.conformal import ConformalWrapper

class ConstResidEstimator:
    name = "const"; feature_mode = "unified"; backend = "sklearn"
    def fit(self, X, y, meta): self._mean = float(np.mean(y))
    def predict(self, X, meta): return np.full(len(X), self._mean)

def _xy(n):
    X = pd.DataFrame({"dataset":["a"]*n,"cell_id":["c"]*n,
                      "cycle_index":range(n),"soh":np.linspace(1.0,0.8,n),"f":np.arange(n)})
    return X, X["soh"].to_numpy()

def test_calibration_uses_train_tail_only():
    X, y = _xy(10)
    w = ConformalWrapper(ConstResidEstimator(), alpha=0.2, calib_frac=0.3)
    w.fit(X, y, meta=None)
    # last 30% (3 rows) are calibration; base must have been fit on first 7
    assert w._n_calib == 3 and w._n_fit == 7

def test_interval_covers_and_orders():
    X, y = _xy(50)
    w = ConformalWrapper(ConstResidEstimator(), alpha=0.1, calib_frac=0.3)
    w.fit(X, y, meta=None)
    lo, hi = w.predict_interval(X, meta=None)
    assert np.all(hi >= lo)
    assert w.ci_source == "conformal"
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/conformal.py`

```python
from __future__ import annotations
import numpy as np

class ConformalWrapper:
    """Split-conformal. Calibration carved from the TRAIN TAIL — never the test fold."""
    ci_source = "conformal"

    def __init__(self, base, alpha: float = 0.1, calib_frac: float = 0.2):
        self.base = base; self.alpha = alpha; self.calib_frac = calib_frac
        self._q = None; self._n_calib = 0; self._n_fit = 0

    def fit(self, X, y, meta):
        n = len(X); n_calib = max(1, int(round(n * self.calib_frac)))
        fit_slice = slice(0, n - n_calib); cal_slice = slice(n - n_calib, n)
        self._n_fit = n - n_calib; self._n_calib = n_calib
        Xf = X.iloc[fit_slice] if hasattr(X, "iloc") else X[fit_slice]
        yf = np.asarray(y)[fit_slice]
        self.base.fit(Xf, yf, meta)
        Xc = X.iloc[cal_slice] if hasattr(X, "iloc") else X[cal_slice]
        yc = np.asarray(y)[cal_slice]
        resid = np.abs(yc - self.base.predict(Xc, meta))
        # finite-sample conformal quantile
        level = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
        self._q = float(np.quantile(resid, min(level, 1.0), method="higher"))
        return self

    def predict(self, X, meta):
        return self.base.predict(X, meta)

    def predict_interval(self, X, meta, alpha: float | None = None):
        center = self.base.predict(X, meta)
        return center - self._q, center + self._q
```

- [ ] **Step 4: Run to verify it passes** → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/conformal.py tests/test_conformal.py
git commit -m "feat(A): split-conformal CI wrapper with train-tail calibration"
```

---

### Task 8: `data.py` — DatasetLoader (reads processed pkls + QC)

**Files:**
- Create: `src/sohbench/data.py`
- Test: `tests/test_data.py`

> Reads the external data artifacts; the test writes a tiny fixture pkl so it does not depend on the real data dir.

- [ ] **Step 1: Write failing test**

```python
import pandas as pd, numpy as np
from pathlib import Path
from sohbench.data import DatasetLoader

def _write_fixture(tmp_path):
    proc = tmp_path / "processed"; proc.mkdir()
    df = pd.DataFrame({
        "dataset": ["hust"]*4, "cell_id": ["h1"]*4, "cycle_index":[0,1,2,3],
        "soh":[1.02, 0.99, 1.35, 0.90],  # 1.35 dropped by QC
        "voltage_mean":[3.3,3.29,3.28,3.27],
    })
    df.to_pickle(proc / "hust_features.pkl")
    return tmp_path

def test_load_builds_table_meta_and_qc(tmp_path):
    root = _write_fixture(tmp_path)
    loader = DatasetLoader(data_dir=root)
    table, meta, rep = loader.load(["hust"])
    assert (table["soh"] < 1.3).all()                # 1.35 dropped
    assert (table["soh"] > 1.0).any()                # 1.02 retained
    assert meta.loc[meta.cell_id=="h1","chemistry"].iloc[0] == "LFP"
    assert rep.n_out == 1
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/data.py`

```python
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
```

- [ ] **Step 4: Run to verify it passes** → 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/data.py tests/test_data.py
git commit -m "feat(A): DatasetLoader reads processed pkls, builds meta, runs QC"
```

---

### Task 9: `features.py` — unified feature provider

**Files:**
- Create: `src/sohbench/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing test**

```python
import pandas as pd
from sohbench.features import unified_features

def test_unified_features_excludes_ids():
    df = pd.DataFrame({"dataset":["a"],"cell_id":["c"],"cycle_index":[0],
                       "soh":[1.0],"f1":[0.5],"f2":[0.3]})
    X, names = unified_features(df)
    assert names == ["f1","f2"]
    assert list(X.columns) == ["f1","f2"]
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/features.py`

```python
from __future__ import annotations
import pandas as pd
from .contracts import feature_columns

def unified_features(table: pd.DataFrame):
    """Default 'unified' feature view: every estimator sees identical columns (D8 fairness)."""
    cols = feature_columns(table)
    return table[cols].copy(), cols
```

- [ ] **Step 4: Run to verify it passes** → 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/features.py tests/test_features.py
git commit -m "feat(A): unified feature provider for fair comparison"
```

---

### Task 10: `estimators/baselines.py` — Ridge baseline (CI-capable)

**Files:**
- Create: `src/sohbench/estimators/__init__.py`, `src/sohbench/estimators/baselines.py`
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np, pandas as pd
from sohbench.estimators.baselines import RidgeEstimator
from sohbench.contracts import Estimator

def _xy(n=30):
    rng = np.random.default_rng(0)
    f = np.linspace(0,1,n)
    X = pd.DataFrame({"dataset":["a"]*n,"cell_id":["c"]*n,"cycle_index":range(n),
                      "soh":1.0-0.2*f,"f1":f,"f2":f**2})
    return X, X["soh"].to_numpy()

def test_ridge_is_estimator_and_predicts():
    X, y = _xy()
    est = RidgeEstimator()
    assert isinstance(est, Estimator)
    est.fit(X, y, meta=None)
    p = est.predict(X, meta=None)
    assert p.shape == (len(X),)
    assert np.mean(np.abs(p - y)) < 0.05
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement**

`src/sohbench/estimators/__init__.py`: empty.
`src/sohbench/estimators/baselines.py`:

```python
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from ..contracts import feature_columns
from ..registry import register_estimator

@register_estimator
class RidgeEstimator:
    name = "ridge"; feature_mode = "unified"; backend = "sklearn"

    def __init__(self, alpha: float = 1.0):
        self._scaler = StandardScaler(); self._model = Ridge(alpha=alpha)
        self._cols = None

    def fit(self, X, y, meta=None):
        self._cols = feature_columns(X)
        Z = self._scaler.fit_transform(X[self._cols].to_numpy())
        self._model.fit(Z, np.asarray(y)); return self

    def predict(self, X, meta=None):
        Z = self._scaler.transform(X[self._cols].to_numpy())
        return self._model.predict(Z)

    def export_cpp(self, path: str) -> None:
        coef = self._model.coef_; intercept = float(self._model.intercept_)
        mean = self._scaler.mean_; scale = self._scaler.scale_
        with open(path, "w") as f:
            f.write("// auto-generated ridge inference\n")
            f.write(f"// features={self._cols}\n")
            f.write("inline float soh_infer(const float* x){float z;float acc=%r;" % intercept)
            for i in range(len(coef)):
                f.write(f"z=(x[{i}]-{mean[i]!r}f)/{scale[i]!r}f;acc+={coef[i]!r}f*z;")
            f.write("return acc;}\n")
```

- [ ] **Step 4: Run to verify it passes** → 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/estimators tests/test_baselines.py
git commit -m "feat(A): Ridge baseline estimator with C++ export hook"
```

---

### Task 11: `tasks.py` — SOHEstimationTask

**Files:**
- Create: `src/sohbench/tasks.py`
- Test: `tests/test_tasks.py`

- [ ] **Step 1: Write failing test**

```python
from sohbench.tasks import SOHEstimationTask

def test_task_defines_target_and_metrics():
    t = SOHEstimationTask()
    assert t.target == "soh"
    assert "mae" in t.point_metrics and "mean_bias_error" in t.point_metrics
    assert t.acceptance["generalization"]["mae_max"] > 0
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/tasks.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from .registry import register_task

@register_task
@dataclass
class SOHEstimationTask:
    name: str = "soh_estimation"
    target: str = "soh"
    point_metrics: tuple = ("rmse","mae","mape","mean_bias_error","overestimation_rate",
                            "max_error","alpha_accuracy","eol_region_error")
    trajectory_metrics: tuple = ("monotonicity_violation_rate","value_range_violation")
    interval_metrics: tuple = ("picp",)
    acceptance: dict = field(default_factory=lambda: {
        "generalization": {"mae_max": 0.02},
        "calibration": {"picp_min": 0.90, "nominal": 0.90},
        "physical_consistency": {"monotonicity_violation_rate_max": 0.001,
                                 "value_range_violation_max": 0.0},
    })
```

- [ ] **Step 4: Run to verify it passes** → 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/tasks.py tests/test_tasks.py
git commit -m "feat(A): SOHEstimationTask — target, metric set, acceptance thresholds"
```

---

### Task 12: `acceptance.py` — verdicts from aggregated metrics

**Files:**
- Create: `src/sohbench/acceptance.py`
- Test: `tests/test_acceptance.py`

- [ ] **Step 1: Write failing test**

```python
from sohbench.acceptance import evaluate_verdicts
from sohbench.tasks import SOHEstimationTask

def test_pass_and_fail():
    task = SOHEstimationTask()
    good = {"mae":0.01,"picp":0.92,"monotonicity_violation_rate":0.0,"value_range_violation":0.0}
    bad  = {"mae":0.05,"picp":0.70,"monotonicity_violation_rate":0.10,"value_range_violation":0.2}
    vg = {v.name: v.passed for v in evaluate_verdicts(good, task)}
    vb = {v.name: v.passed for v in evaluate_verdicts(bad, task)}
    assert vg == {"generalization":True,"calibration":True,"physical_consistency":True}
    assert vb == {"generalization":False,"calibration":False,"physical_consistency":False}
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/acceptance.py`

```python
from __future__ import annotations
from .contracts import Verdict

def evaluate_verdicts(metrics: dict, task) -> list[Verdict]:
    a = task.acceptance; out = []

    g_ok = metrics.get("mae", float("inf")) <= a["generalization"]["mae_max"]
    out.append(Verdict("generalization", g_ok,
                       {"mae": metrics.get("mae")},
                       [] if g_ok else [f"mae {metrics.get('mae')} > {a['generalization']['mae_max']}"]))

    c_ok = metrics.get("picp", 0.0) >= a["calibration"]["picp_min"]
    out.append(Verdict("calibration", c_ok, {"picp": metrics.get("picp")},
                       [] if c_ok else [f"picp {metrics.get('picp')} < {a['calibration']['picp_min']}"]))

    p = a["physical_consistency"]
    p_ok = (metrics.get("monotonicity_violation_rate", 1.0) <= p["monotonicity_violation_rate_max"]
            and metrics.get("value_range_violation", 1.0) <= p["value_range_violation_max"])
    out.append(Verdict("physical_consistency", p_ok,
                       {k: metrics.get(k) for k in ("monotonicity_violation_rate","value_range_violation")},
                       [] if p_ok else ["physical-consistency thresholds exceeded"]))
    return out
```

- [ ] **Step 4: Run to verify it passes** → 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/acceptance.py tests/test_acceptance.py
git commit -m "feat(A): acceptance verdicts (generalization/calibration/physical)"
```

---

### Task 13: `runner.py` — BenchmarkRunner (matrix, multi-seed, guaranteed CI)

**Files:**
- Create: `src/sohbench/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np, pandas as pd
from sohbench.runner import BenchmarkRunner
from sohbench.protocols import LeaveOneCellOut
from sohbench.tasks import SOHEstimationTask
from sohbench.estimators.baselines import RidgeEstimator

def _table():
    rows = []
    for cell in ["c1","c2","c3"]:
        for k in range(15):
            rows.append(("ds", cell, k, 1.0-0.01*k, float(k), float(k*k)))
    return pd.DataFrame(rows, columns=["dataset","cell_id","cycle_index","soh","f1","f2"])

def test_runner_produces_results_with_ci_and_seeds():
    from sohbench.metadata import build_meta
    t = _table(); meta = build_meta(t)
    runner = BenchmarkRunner(seeds=[0,1,2,3,4])
    rs = runner.run(SOHEstimationTask(), table=t, meta=meta,
                    protocols={"loco": LeaveOneCellOut()},
                    estimators=[RidgeEstimator])
    assert set(rs.columns) >= {"estimator","protocol","held_out_key","fold","seed","metric","value"}
    assert (rs["metric"] == "picp").any()                 # CI guaranteed (conformal)
    assert rs["seed"].nunique() == 5

def test_failed_estimator_isolated():
    from sohbench.metadata import build_meta
    class Boom:
        name="boom"; feature_mode="unified"; backend="sklearn"
        def fit(self,X,y,meta): raise RuntimeError("boom")
        def predict(self,X,meta): ...
    t=_table(); meta=build_meta(t)
    rs = BenchmarkRunner(seeds=[0]).run(SOHEstimationTask(), table=t, meta=meta,
                                        protocols={"loco":LeaveOneCellOut()},
                                        estimators=[Boom])
    assert rs["value"].isna().any()                       # failed fold recorded, run completed
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/runner.py`

```python
from __future__ import annotations
import numpy as np, pandas as pd
from . import metrics as M
from .conformal import ConformalWrapper
from .features import unified_features

class BenchmarkRunner:
    def __init__(self, seeds=(0,1,2,3,4)):
        self.seeds = list(seeds)

    def run(self, task, table, meta, protocols, estimators) -> pd.DataFrame:
        records = []
        for est_factory in estimators:
            est_name = getattr(est_factory, "name", est_factory.__name__)
            for pname, proto in protocols.items():
                for fi, fold in enumerate(proto.generate(table, meta)):
                    tr, te = table.iloc[fold.train_idx], table.iloc[fold.test_idx]
                    for seed in self.seeds:
                        np.random.seed(seed)
                        rec_base = dict(estimator=est_name, protocol=pname,
                                        held_out_key=fold.held_out_key, fold=fi, seed=seed)
                        try:
                            est = est_factory() if isinstance(est_factory, type) else est_factory
                            if not hasattr(est, "predict_interval"):
                                est = ConformalWrapper(est, alpha=1 - task.acceptance["calibration"]["nominal"])
                            ytr = tr[task.target].to_numpy()
                            est.fit(tr, ytr, meta)
                            yte = te[task.target].to_numpy()
                            yp = est.predict(te, meta)
                            row = self._point_metrics(task, yte, yp)
                            lo, hi = est.predict_interval(te, meta)
                            row["picp"] = M.picp(yte, lo, hi)
                            row.update(self._trajectory_metrics(task, te, yp))
                        except Exception as e:  # isolate failed fold
                            row = {m: float("nan") for m in
                                   (*task.point_metrics, *task.interval_metrics, *task.trajectory_metrics)}
                            rec_base["error"] = str(e)
                        for metric, value in row.items():
                            records.append({**rec_base, "metric": metric, "value": value})
        return pd.DataFrame.from_records(records)

    def _point_metrics(self, task, yt, yp):
        out = {}
        for m in task.point_metrics:
            out[m] = getattr(M, m)(yt, yp)
        return out

    def _trajectory_metrics(self, task, te, yp):
        order = np.argsort(te["cycle_index"].to_numpy())
        return {"monotonicity_violation_rate": M.monotonicity_violation_rate(yp[order]),
                "value_range_violation": M.value_range_violation(yp)}
```

- [ ] **Step 4: Run to verify it passes** → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/runner.py tests/test_runner.py
git commit -m "feat(A): BenchmarkRunner — matrix, multi-seed, guaranteed CI, isolated failures"
```

---

### Task 14: `report.py` — artifacts (leaderboard/qc/verdicts) + JSON Schema

**Files:**
- Create: `src/sohbench/report.py`, `src/sohbench/schemas/leaderboard.schema.json`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write failing test**

```python
import json, pandas as pd, numpy as np
from pathlib import Path
from sohbench.report import write_artifacts
from sohbench.tasks import SOHEstimationTask

def _results():
    rows=[]
    for seed in range(5):
        for metric,val in [("mae",0.01),("picp",0.92),
                           ("monotonicity_violation_rate",0.0),("value_range_violation",0.0)]:
            rows.append(dict(estimator="ridge",feature_mode="unified",backend="sklearn",
                             protocol="loco",dataset="ds",held_out_key="c1",fold=0,
                             seed=seed,metric=metric,value=val))
    return pd.DataFrame(rows)

def test_write_artifacts(tmp_path):
    out = write_artifacts(_results(), qc_report=None, task=SOHEstimationTask(),
                          config={"datasets":["ds"]}, out_dir=tmp_path)
    lb = json.loads((Path(out)/"leaderboard.json").read_text())
    entry = lb[0]
    assert entry["estimator"]=="ridge" and entry["track"]=="unified"
    assert entry["metrics"]["mae"]["mean"]==0.01
    assert entry["verdicts"]["generalization"] is True
    assert (Path(out)/"results.parquet").exists()
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement**

`src/sohbench/schemas/leaderboard.schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "required": ["estimator","track","protocol","metrics","verdicts"],
    "properties": {
      "estimator": {"type":"string"},
      "track": {"enum":["unified","byo"]},
      "protocol": {"type":"string"},
      "metrics": {"type":"object"},
      "verdicts": {"type":"object"}
    }
  }
}
```

`src/sohbench/report.py`:

```python
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
from .acceptance import evaluate_verdicts
from .tasks import SOHEstimationTask

def write_artifacts(results: pd.DataFrame, qc_report, task, config, out_dir):
    out = Path(out_dir) / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out / "results.parquet")
    results.to_csv(out / "results.csv", index=False)

    leaderboard = []
    group_cols = ["estimator","protocol"]
    feature_mode = results["feature_mode"].iloc[0] if "feature_mode" in results else "unified"
    for (est, proto), g in results.groupby(group_cols):
        agg = g.groupby("metric")["value"].agg(["mean","std"])
        metrics = {m: {"mean": float(r["mean"]), "std": float(r["std"] if r["std"]==r["std"] else 0.0)}
                   for m, r in agg.iterrows()}
        flat = {m: v["mean"] for m, v in metrics.items()}
        verdicts = {v.name: v.passed for v in evaluate_verdicts(flat, task)}
        leaderboard.append({"estimator": est, "track": feature_mode, "protocol": proto,
                            "metrics": metrics, "verdicts": verdicts, "ci_source": "conformal"})
    (out / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    if qc_report is not None:
        (out / "qc_report.json").write_text(json.dumps(qc_report.__dict__, indent=2, default=str))
    (out / "run_config.json").write_text(json.dumps(config, indent=2, default=str))
    return str(out)
```

- [ ] **Step 4: Run to verify it passes** → 1 passed.

- [ ] **Step 5: Validate schema (add assertion) + commit**

Append to `tests/test_report.py`:

```python
def test_leaderboard_matches_schema(tmp_path):
    import jsonschema, json
    out = write_artifacts(_results(), qc_report=None, task=SOHEstimationTask(),
                          config={}, out_dir=tmp_path)
    lb = json.loads((Path(out)/"leaderboard.json").read_text())
    schema = json.loads((Path(__file__).parents[1]/"src/sohbench/schemas/leaderboard.schema.json").read_text())
    jsonschema.validate(lb, schema)
```

Run: `pytest tests/test_report.py -v` → 2 passed.

```bash
git add src/sohbench/report.py src/sohbench/schemas tests/test_report.py
git commit -m "feat(A): result artifacts (leaderboard/qc/config) + JSON schema for C"
```

---

### Task 15: `cli.py` — end-to-end run + smoke test

**Files:**
- Create: `src/sohbench/cli.py`, `src/sohbench/__main__.py`
- Test: `tests/test_cli_e2e.py`

- [ ] **Step 1: Write failing test**

```python
import pandas as pd
from pathlib import Path
from sohbench.cli import run_benchmark

def _fixture(tmp_path):
    proc = tmp_path/"processed"; proc.mkdir()
    rows=[]
    for cell in ["c1","c2","c3"]:
        for k in range(15):
            rows.append(("hust",cell,k,1.0-0.01*k,float(k),float(k*k)))
    pd.DataFrame(rows, columns=["dataset","cell_id","cycle_index","soh","f1","f2"]
                ).to_pickle(proc/"hust_features.pkl")
    return tmp_path

def test_end_to_end(tmp_path):
    root=_fixture(tmp_path)
    out = run_benchmark(data_dir=root, datasets=["hust"], estimators=["ridge"],
                        protocols=["loco"], seeds=[0,1,2,3,4], out_dir=tmp_path/"runs")
    assert (Path(out)/"leaderboard.json").exists()
    assert (Path(out)/"results.parquet").exists()
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** — `src/sohbench/cli.py`

```python
from __future__ import annotations
import argparse
from .data import DatasetLoader
from .registry import get_estimator
from .protocols import LeaveOneCellOut, LeaveOneDatasetOut, LeaveOneChemistryOut
from .tasks import SOHEstimationTask
from .runner import BenchmarkRunner
from .report import write_artifacts
import sohbench.estimators.baselines  # noqa: F401  (register baselines)

_PROTOCOLS = {"loco": LeaveOneCellOut, "lodo": LeaveOneDatasetOut, "lcho": LeaveOneChemistryOut}

def run_benchmark(data_dir, datasets, estimators, protocols, seeds, out_dir):
    table, meta, qc = DatasetLoader(data_dir).load(datasets)
    ests = [get_estimator(n) for n in estimators]
    protos = {p: _PROTOCOLS[p]() for p in protocols}
    rs = BenchmarkRunner(seeds=seeds).run(SOHEstimationTask(), table=table, meta=meta,
                                          protocols=protos, estimators=ests)
    rs["feature_mode"] = "unified"; rs["backend"] = "sklearn"; rs["dataset"] = ",".join(datasets)
    return write_artifacts(rs, qc_report=qc, task=SOHEstimationTask(),
                           config={"datasets":datasets,"estimators":estimators,
                                   "protocols":protocols,"seeds":seeds}, out_dir=out_dir)

def main(argv=None):
    p = argparse.ArgumentParser("sohbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--datasets", nargs="+", required=True)
    r.add_argument("--estimators", nargs="+", default=["ridge"])
    r.add_argument("--protocols", nargs="+", default=["loco"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    r.add_argument("--out-dir", default="runs")
    a = p.parse_args(argv)
    out = run_benchmark(a.data_dir, a.datasets, a.estimators, a.protocols, a.seeds, a.out_dir)
    print(out)

if __name__ == "__main__":
    main()
```

`src/sohbench/__main__.py`:

```python
from .cli import main
main()
```

- [ ] **Step 4: Run to verify it passes** — `pytest tests/test_cli_e2e.py -v` → 1 passed. Then full suite: `pytest -q` → all green.

- [ ] **Step 5: Commit**

```bash
git add src/sohbench/cli.py src/sohbench/__main__.py tests/test_cli_e2e.py
git commit -m "feat(A): CLI + end-to-end benchmark run over processed datasets"
```

---

### Task 16: Real-data smoke (NASA/CALCE/HUST/Oxford) + chemistry-out

**Files:**
- Create: `tests/test_real_data_smoke.py` (skipped if data dir absent)

- [ ] **Step 1: Write the test**

```python
import os, pytest
from pathlib import Path
from sohbench.cli import run_benchmark

DATA = Path(r"D:\battery_SOH\data")

@pytest.mark.skipif(not (DATA/"processed").exists(), reason="real data dir absent")
def test_four_datasets_run(tmp_path):
    out = run_benchmark(data_dir=DATA, datasets=["nasa","calce_cs2","hust","oxford"],
                        estimators=["ridge"], protocols=["loco","lodo","lcho"],
                        seeds=[0,1,2,3,4], out_dir=tmp_path)
    assert (Path(out)/"leaderboard.json").exists()
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_real_data_smoke.py -v -s`
Expected: PASS (or SKIP if data dir absent). Inspect printed leaderboard for sane MAE and the chemistry-out downgrade warning.

- [ ] **Step 3: Commit**

```bash
git add tests/test_real_data_smoke.py
git commit -m "test(A): real-data smoke across 4 datasets + leave-one-domain protocols"
```

---

## Self-Review

**Spec coverage** (each spec section → task):
- §2 modules: contracts(T1), registry(T3), metadata(T4), representation→folded into data/contracts (RawChannel stub deferred — see note), qc(T5), data(T8), features(T9), protocols(T6), metrics(T2), conformal(T7), tasks(T11), runner(T13), acceptance(T12), report(T14), cli(T15). ✅
- §3 dual feature mode: unified(T9) built; **BYO track is out of v1 scope here** — runner tags `feature_mode="unified"`; BYO wiring deferred (added as explicit follow-up below). ⚠️
- §4 Estimator contract: T1 + T10. ✅  §5 QC: T5/T8. ✅  §6 protocols: T6. ✅
- §7 metrics incl. MBE/overestimation: T2. ✅  §8 conformal CI: T7/T13. ✅
- §9 runner+acceptance: T12/T13. ✅  §10 artifacts+schema: T14. ✅  §11 errors: T13 (failed fold), T5/T8 (QC). ✅
- §12 tests: each task is TDD. ✅  §13 seams: Task abstraction(T11); RawChannel + BYO deferred. ✅(noted)

**Deferred to a follow-up plan (documented, not silent):** RawChannel raw loading, BYO feature-mode track end-to-end, weighted/adaptive conformal for cross-domain, forward-chaining & leave-one-station protocols, `export_cpp` golden-vector parity check, sudden-drop/non-degrading QC rules. All are spec §13 "designed-for" items or explicit v1-optional; none block a working harness.

**Placeholder scan:** no TBD/TODO; every code step has runnable code. ✅
**Type consistency:** `feature_columns`, `CellMeta`, `Fold(train_idx/test_idx/held_out_key)`, `QCReport`, `Verdict(name/passed/metrics/reasons)`, `Estimator.fit(X,y,meta)/predict(X,meta)`, `ConformalWrapper.predict_interval`, metric names match `SOHEstimationTask.point_metrics` and `acceptance.py` lookups. ✅

> **Note on `representation.py`:** the spec lists it as a module; this plan folds its v1 responsibilities (ID columns, validation) into `contracts.py`/`data.py` and defers `RawChannel` to the BYO/raw follow-up, to avoid an empty module. If you prefer a literal module-per-spec mapping, add a thin `representation.py` re-exporting `ID_COLUMNS`/`feature_columns` and the `RawChannel` stub.
