# Training Inference Model Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persisted pickle model packages, a model library, and an inference-only evaluation tab that measures transfer accuracy and prediction speed on selected datasets.

**Architecture:** Keep the existing training benchmark path as `fit + predict`, but let it save fitted estimators through a new model-store layer. Add a separate inference job path that loads model packages, validates feature compatibility, runs `predict` only, writes inference artifacts, and publishes them to the static frontend data directory.

**Tech Stack:** Python stdlib HTTP server, pytest, pandas, numpy, pickle, existing sohbench registries/runners/reports, browser ES modules, Node test runner.

## Global Constraints

- Data root remains `D:\battery_SOH\data`.
- Pickle model loading is trusted local mode; it is not a sandbox.
- `/api/upload/algorithm` remains source-code estimator upload for training jobs.
- New model upload uses pickle packages and must not replace the source-plugin upload contract.
- Inference evaluation must never call `fit`.
- Prediction timing records pure `predict` time separately from model load time.
- Existing static benchmark artifacts must keep working for old runs.
- Frontend copy is Chinese-facing.
- Manual code edits use `apply_patch`.
- New behavior follows TDD: every production change starts from a failing test.

---

## File Structure

- Create `src/sohbench/model_store.py`
  - Owns trusted pickle model package save/load/upload/catalog behavior.
  - Provides `ModelStore`, `ModelStoreError`, `LoadedModel`, and feature compatibility helpers.
- Create `src/sohbench/inference.py`
  - Owns inference-only benchmark execution and metric/timing record generation.
  - Provides `run_inference_benchmark(...)`.
- Modify `src/sohbench/runner.py`
  - Adds an optional fitted-model callback after successful `fit`.
  - Keeps the existing returned metric DataFrame compatible.
- Modify `src/sohbench/report.py`
  - Adds `write_inference_artifacts(...)`.
  - Optionally writes `model_catalog.json` for training runs.
- Modify `src/sohbench/evaluation_service.py`
  - Instantiates `ModelStore`.
  - Includes models in catalog.
  - Saves training-produced model packages.
  - Adds inference job lifecycle.
  - Syncs `inference_benchmark.json` and `model_catalog.json` when present.
- Modify `src/sohbench/api_server.py`
  - Adds `/api/upload/model`, `/api/models`, and `/api/inference-jobs` endpoints.
- Modify `scripts/build_frontend_artifacts.py`
  - Copies inference/model artifacts when they exist.
- Modify `frontend/app/dashboard-core.mjs`
  - Adds model/inference data normalization and ranking helpers.
- Modify `frontend/app/app.js`
  - Loads optional inference/model artifacts.
  - Adds model upload and inference job actions.
- Modify `frontend/app/index.html`
  - Adds Chinese tabs/sections for training evaluation, inference evaluation, and model library.
- Modify `frontend/app/styles.css`
  - Adds restrained tab, model table, and inference table styles.
- Add/modify tests:
  - `tests/test_model_store.py`
  - `tests/test_inference.py`
  - `tests/test_evaluation_service.py`
  - `tests/test_api_server.py`
  - `tests/test_report.py`
  - `tests/test_build_frontend_artifacts.py`
  - `frontend/app/dashboard-core.test.mjs`
  - `frontend/app/chinese-ui.test.mjs`

---

### Task 1: Model Store

**Files:**
- Create: `src/sohbench/model_store.py`
- Test: `tests/test_model_store.py`

**Interfaces:**
- Produces: `class ModelStore(base_dir: str | Path)`
- Produces: `ModelStore.save_training_model(name: str, estimator: object, manifest: dict, replace: bool = False) -> dict`
- Produces: `ModelStore.upload_model(name: str, model_filename: str, model_payload: bytes, manifest_filename: str, manifest_payload: bytes, replace: bool = False) -> dict`
- Produces: `ModelStore.catalog() -> list[dict]`
- Produces: `ModelStore.load_model(name: str) -> LoadedModel`
- Produces: `LoadedModel.predict(table: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray`
- Produces: `LoadedModel.predict_interval(table: pd.DataFrame, meta: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray] | None`
- Produces: `LoadedModel.compatibility(table: pd.DataFrame) -> dict`

- [ ] **Step 1: Write failing tests for saving, uploading, loading, and feature compatibility**

Add `tests/test_model_store.py`:

```python
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from sohbench.model_store import ModelStore, ModelStoreError


class PredictOnlyModel:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        raise AssertionError("inference path must not call fit")

    def predict(self, X, meta=None):
        return np.full(len(X), 0.91)


def _table():
    return pd.DataFrame({
        "dataset": ["demo"] * 4,
        "cell_id": ["c1"] * 4,
        "cycle_index": [0, 1, 2, 3],
        "soh": [1.0, 0.98, 0.96, 0.94],
        "voltage_mean": [3.5, 3.48, 3.46, 3.44],
        "resistance_proxy": [0.01, 0.02, 0.03, 0.04],
    })


def _manifest(name="demo_model"):
    return {
        "model_name": name,
        "model_kind": "pickle",
        "estimator": "predict_only",
        "featureset": "full",
        "feature_columns": ["voltage_mean", "resistance_proxy"],
        "target": "soh",
        "trained_on": ["train_demo"],
        "created_by_run": "unit",
        "predict_contract": "predict(X, meta=None)",
    }


def test_save_training_model_writes_pickle_manifest_and_catalog(tmp_path):
    store = ModelStore(tmp_path / "models")

    resource = store.save_training_model("demo_model", PredictOnlyModel(), _manifest())

    model_dir = tmp_path / "models" / "demo_model"
    assert resource["name"] == "demo_model"
    assert resource["source"] == "training"
    assert (model_dir / "model.pkl").exists()
    assert (model_dir / "model_manifest.json").exists()
    assert store.catalog()[0]["name"] == "demo_model"


def test_upload_model_loads_and_predicts_without_fit(tmp_path):
    store = ModelStore(tmp_path / "models")
    model_payload = pickle.dumps(PredictOnlyModel())
    manifest_payload = json.dumps(_manifest("uploaded_model")).encode("utf-8")

    store.upload_model(
        "uploaded_model",
        "uploaded_model.pkl",
        model_payload,
        "model_manifest.json",
        manifest_payload,
    )

    loaded = store.load_model("uploaded_model")
    pred = loaded.predict(_table(), pd.DataFrame())

    assert pred.tolist() == [0.91, 0.91, 0.91, 0.91]
    assert loaded.compatibility(_table())["ok"] is True


def test_model_store_rejects_missing_manifest_features(tmp_path):
    store = ModelStore(tmp_path / "models")
    model_payload = pickle.dumps(PredictOnlyModel())
    manifest = _manifest("bad_model")
    manifest["feature_columns"] = ["not_present"]

    with pytest.raises(ModelStoreError, match="missing feature columns"):
        store.upload_model(
            "bad_model",
            "bad_model.pkl",
            model_payload,
            "model_manifest.json",
            json.dumps(manifest).encode("utf-8"),
        )
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_model_store.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sohbench.model_store'`.

- [ ] **Step 3: Implement minimal model store**

Create `src/sohbench/model_store.py` with:

```python
from __future__ import annotations

import inspect
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ID_COLUMNS

SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


class ModelStoreError(ValueError):
    """Raised for invalid trusted local model packages."""


@dataclass
class LoadedModel:
    name: str
    model: Any
    manifest: dict
    path: Path
    load_ms: float = 0.0

    def feature_columns(self) -> list[str]:
        return [str(col) for col in self.manifest.get("feature_columns", [])]

    def input_columns(self, table: pd.DataFrame) -> list[str]:
        configured = self.manifest.get("input_columns")
        if configured:
            return [str(col) for col in configured]
        return [col for col in [*ID_COLUMNS, *self.feature_columns()] if col in table.columns]

    def compatibility(self, table: pd.DataFrame) -> dict:
        missing = [col for col in self.feature_columns() if col not in table.columns]
        return {"ok": not missing, "missing_feature_columns": missing}

    def predict(self, table: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        self._ensure_compatible(table)
        X = table[self.input_columns(table)].copy()
        predict = getattr(self.model, "predict", None)
        if not callable(predict):
            raise ModelStoreError(f"model {self.name!r} does not expose predict()")
        values = _call_predict(predict, X, meta)
        pred = np.asarray(values, dtype=float)
        if pred.shape[0] != len(table):
            raise ModelStoreError(f"model {self.name!r} returned {pred.shape[0]} predictions for {len(table)} rows")
        if not np.isfinite(pred).all():
            raise ModelStoreError(f"model {self.name!r} returned non-finite predictions")
        return pred

    def predict_interval(self, table: pd.DataFrame, meta: pd.DataFrame | None = None):
        fn = getattr(self.model, "predict_interval", None)
        if not callable(fn):
            return None
        self._ensure_compatible(table)
        X = table[self.input_columns(table)].copy()
        lo, hi = _call_predict(fn, X, meta)
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def _ensure_compatible(self, table: pd.DataFrame) -> None:
        missing = self.compatibility(table)["missing_feature_columns"]
        if missing:
            raise ModelStoreError(f"missing feature columns for {self.name}: {missing}")


class ModelStore:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.catalog_path = self.base_dir / "model_catalog.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._catalog = self._read_catalog()

    def save_training_model(self, name: str, estimator: object, manifest: dict, replace: bool = False) -> dict:
        return self._write_package(name, estimator, manifest, source="training", replace=replace)

    def upload_model(
        self,
        name: str,
        model_filename: str,
        model_payload: bytes,
        manifest_filename: str,
        manifest_payload: bytes,
        replace: bool = False,
    ) -> dict:
        if Path(model_filename).suffix.lower() not in {".pkl", ".pickle"}:
            raise ModelStoreError("model upload must be .pkl or .pickle")
        if Path(manifest_filename).name != "model_manifest.json":
            raise ModelStoreError("manifest upload must be model_manifest.json")
        try:
            model = pickle.loads(model_payload)
        except Exception as exc:
            raise ModelStoreError(f"model pickle could not be loaded: {exc}") from exc
        try:
            manifest = json.loads(manifest_payload.decode("utf-8"))
        except Exception as exc:
            raise ModelStoreError(f"model manifest could not be parsed: {exc}") from exc
        return self._write_package(name, model, manifest, source="upload", replace=replace)

    def load_model(self, name: str) -> LoadedModel:
        item = self._find(name)
        start = time.perf_counter()
        with open(item["model_path"], "rb") as fh:
            model = pickle.load(fh)
        load_ms = (time.perf_counter() - start) * 1000.0
        manifest = json.loads(Path(item["manifest_path"]).read_text(encoding="utf-8"))
        return LoadedModel(name=item["name"], model=model, manifest=manifest, path=Path(item["path"]), load_ms=load_ms)

    def catalog(self) -> list[dict]:
        return list(self._catalog.get("models", []))

    def _write_package(self, name: str, model: object, manifest: dict, source: str, replace: bool) -> dict:
        name = _safe_name(name)
        manifest = _validate_manifest(name, manifest)
        if not callable(getattr(model, "predict", None)):
            raise ModelStoreError("model must expose predict()")
        model_dir = self.base_dir / name
        if model_dir.exists() and not replace:
            raise ModelStoreError(f"model already exists: {name}")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        manifest_path = model_dir / "model_manifest.json"
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        resource = {
            "name": name,
            "kind": "model",
            "source": source,
            "path": str(model_dir),
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "estimator": manifest.get("estimator"),
            "featureset": manifest.get("featureset"),
            "feature_columns": list(manifest.get("feature_columns", [])),
            "trained_on": list(manifest.get("trained_on", [])),
            "created_by_run": manifest.get("created_by_run"),
        }
        self._upsert(resource)
        self.load_model(name).predict(_smoke_table(resource["feature_columns"]), pd.DataFrame())
        return resource

    def _read_catalog(self) -> dict:
        if not self.catalog_path.exists():
            return {"models": []}
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def _write_catalog(self) -> None:
        self.catalog_path.write_text(json.dumps(self._catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    def _upsert(self, resource: dict) -> None:
        items = [item for item in self._catalog.get("models", []) if item.get("name") != resource["name"]]
        items.append(resource)
        self._catalog["models"] = sorted(items, key=lambda item: item["name"])
        self._write_catalog()

    def _find(self, name: str) -> dict:
        for item in self._catalog.get("models", []):
            if item.get("name") == name:
                return item
        raise ModelStoreError(f"unknown model: {name}")


def _safe_name(name: str) -> str:
    value = str(name or "").strip()
    if not SAFE_NAME.match(value):
        raise ModelStoreError(f"invalid model name: {name!r}")
    return value


def _validate_manifest(name: str, manifest: dict) -> dict:
    required = ["model_name", "model_kind", "feature_columns", "target", "predict_contract"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ModelStoreError(f"model manifest missing keys: {missing}")
    if manifest["model_kind"] != "pickle":
        raise ModelStoreError("model manifest model_kind must be pickle")
    if not manifest.get("feature_columns"):
        raise ModelStoreError("model manifest must define feature_columns")
    manifest = dict(manifest)
    manifest["model_name"] = name
    return manifest


def _call_predict(fn, X: pd.DataFrame, meta: pd.DataFrame | None):
    params = inspect.signature(fn).parameters
    if len(params) >= 2:
        return fn(X, meta)
    return fn(X)


def _smoke_table(columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({
        "dataset": ["smoke", "smoke", "smoke"],
        "cell_id": ["c1", "c1", "c1"],
        "cycle_index": [0, 1, 2],
        "soh": [1.0, 0.99, 0.98],
    })
    for col in columns:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame
```

- [ ] **Step 4: Run model store tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_model_store.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/sohbench/model_store.py tests/test_model_store.py
git commit -m "feat: add trusted pickle model store"
```

---

### Task 2: Training Model Persistence

**Files:**
- Modify: `src/sohbench/runner.py`
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `src/sohbench/report.py`
- Test: `tests/test_evaluation_service.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `ModelStore.save_training_model(...)`
- Produces: `BenchmarkRunner.run(..., on_model_fitted: Callable | None = None) -> pd.DataFrame`
- Produces callback payload keys: `estimator`, `featureset`, `protocol`, `held_out_key`, `fold`, `seed`, `track`, `feature_mode`, `backend`
- Produces training run artifacts: `models/<model_name>/model.pkl`, `models/<model_name>/model_manifest.json`, `model_catalog.json`

- [ ] **Step 1: Write failing tests for callback and training-produced model catalog**

Append to `tests/test_runner.py`:

```python
import pandas as pd
import numpy as np

from sohbench.runner import BenchmarkRunner
from sohbench.tasks import SOHEstimationTask
from sohbench.contracts import Fold


class TinyProtocol:
    def generate(self, table, meta):
        yield Fold(train_idx=np.array([0, 1, 2]), test_idx=np.array([3, 4]), held_out_key="demo")


class TinyEstimator:
    name = "tiny_estimator"
    feature_mode = "unified"
    backend = "external"

    def fit(self, X, y, meta=None):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X, meta=None):
        return np.full(len(X), self.mean_)


def test_runner_calls_on_model_fitted_after_successful_fit():
    table = pd.DataFrame({
        "dataset": ["demo"] * 5,
        "cell_id": ["c1"] * 5,
        "cycle_index": [0, 1, 2, 3, 4],
        "soh": [1.0, 0.99, 0.98, 0.97, 0.96],
        "voltage_mean": [3.5, 3.49, 3.48, 3.47, 3.46],
    })
    seen = []

    BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(),
        table=table,
        meta=pd.DataFrame(),
        protocols={"demo": TinyProtocol()},
        estimators=[TinyEstimator],
        featuresets=["full"],
        on_model_fitted=lambda **kwargs: seen.append(kwargs),
    )

    assert len(seen) == 1
    assert seen[0]["record"]["estimator"] == "tiny_estimator"
    assert seen[0]["estimator"].mean_ == 0.99
    assert list(seen[0]["train_table"]["cycle_index"]) == [0, 1, 2]
```

Append to `tests/test_evaluation_service.py`:

```python
def test_training_job_publishes_model_catalog(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    job = service.create_job(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    finished = service.run_job_inline(job["id"])

    assert finished["status"] == "succeeded"
    assert (Path(finished["run_dir"]) / "model_catalog.json").exists()
    frontend_catalog = tmp_path / "frontend_data" / "model_catalog.json"
    assert frontend_catalog.exists()
    models = json.loads(frontend_catalog.read_text(encoding="utf-8"))["models"]
    assert any(item["estimator"] == "ridge" for item in models)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_runner.py::test_runner_calls_on_model_fitted_after_successful_fit tests/test_evaluation_service.py::test_training_job_publishes_model_catalog -v
```

Expected: FAIL because `BenchmarkRunner.run()` does not accept `on_model_fitted`.

- [ ] **Step 3: Add runner callback after successful fit**

In `src/sohbench/runner.py`, change the `run` signature to:

```python
def run(self, task, table, meta, protocols, estimators, featuresets=None, on_model_fitted=None) -> pd.DataFrame:
```

Immediately after `est.fit(tr, ytr, meta)`, add:

```python
if on_model_fitted is not None:
    on_model_fitted(
        estimator=est,
        record=dict(rec_base),
        train_table=tr.copy(),
        test_table=te.copy(),
        meta=meta,
    )
```

- [ ] **Step 4: Add model persistence in evaluation service**

In `src/sohbench/evaluation_service.py`:

```python
from .contracts import feature_columns
from .model_store import ModelStore, ModelStoreError
```

In `EvaluationService.__init__`:

```python
self.model_store = ModelStore(self.work_dir / "models")
```

In `catalog()`:

```python
catalog["models"] = self.model_store.catalog()
```

Inside `run_job_inline`, before calling the runner:

```python
saved_models = []
model_warnings = []

def save_fitted_model(**kwargs):
    record = kwargs["record"]
    fitted = kwargs["estimator"]
    train_table = kwargs["train_table"]
    fs_name = record.get("featureset", "full")
    model_name = "_".join([
        str(record["estimator"]),
        str(fs_name).replace("(", "").replace(")", "").replace("/", "_"),
        str(record["protocol"]),
        str(record["held_out_key"]).replace("/", "_"),
        f"seed{record['seed']}",
    ])
    manifest = {
        "model_name": model_name,
        "model_kind": "pickle",
        "estimator": record["estimator"],
        "featureset": fs_name,
        "feature_columns": feature_columns(train_table),
        "input_columns": list(train_table.columns),
        "target": "soh",
        "trained_on": sorted(map(str, train_table["dataset"].dropna().unique())),
        "held_out_key": str(record["held_out_key"]),
        "protocol": record["protocol"],
        "seed": int(record["seed"]),
        "created_by_run": job_id,
        "predict_contract": "predict(X, meta=None)",
    }
    try:
        saved_models.append(self.model_store.save_training_model(model_name, fitted, manifest, replace=True))
    except Exception as exc:
        model_warnings.append(f"{model_name}: {exc}")
```

Pass the callback:

```python
results = BenchmarkRunner(seeds=request["seeds"]).run(
    SOHEstimationTask(),
    table=table,
    meta=meta,
    protocols=protocols,
    estimators=estimators,
    featuresets=request["featuresets"],
    on_model_fitted=save_fitted_model,
)
```

Add model information to the report config:

```python
"saved_models": [item["name"] for item in saved_models],
"model_warnings": model_warnings,
```

After `write_artifacts(...)`, write the run model catalog:

```python
(raw_run_dir / "model_catalog.json").write_text(
    json.dumps({"models": self.model_store.catalog()}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
```

In the succeeded job update, include:

```python
"saved_models": [item["name"] for item in saved_models],
"model_warnings": model_warnings,
```

- [ ] **Step 5: Sync model catalog to frontend**

In `_sync_frontend_files(...)`, include optional file copy:

```python
for filename in ("leaderboard.json", "dataset_leaderboard.json", "qc_report.json", "run_config.json"):
    shutil.copy2(run_dir / filename, frontend_data_dir / filename)
for filename in ("model_catalog.json", "inference_benchmark.json"):
    source = run_dir / filename
    if source.exists():
        shutil.copy2(source, frontend_data_dir / filename)
```

- [ ] **Step 6: Run tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_runner.py::test_runner_calls_on_model_fitted_after_successful_fit tests/test_evaluation_service.py::test_training_job_publishes_model_catalog -v
```

Expected: all selected tests PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src/sohbench/runner.py src/sohbench/evaluation_service.py tests/test_runner.py tests/test_evaluation_service.py
git commit -m "feat: persist trained benchmark models"
```

---

### Task 3: Inference-Only Runner and Artifacts

**Files:**
- Create: `src/sohbench/inference.py`
- Modify: `src/sohbench/report.py`
- Test: `tests/test_inference.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `ModelStore.load_model(name)`
- Produces: `run_inference_benchmark(model_store, table, meta, model_names) -> list[dict]`
- Produces: `write_inference_artifacts(records: list[dict], config: dict, out_dir: str | Path) -> str`
- Produces artifact fields: `model`, `dataset`, `n_samples`, `mae`, `rmse`, `mean_bias_error`, `predict_ms`, `load_ms`, `ms_per_1000_samples`, `samples_per_second`, `status`, `error`

- [ ] **Step 1: Write failing inference tests**

Add `tests/test_inference.py`:

```python
import json
import pickle

import numpy as np
import pandas as pd

from sohbench.inference import run_inference_benchmark
from sohbench.model_store import ModelStore


class PredictOnlyModel:
    def fit(self, X, y=None, meta=None):
        raise AssertionError("fit must not be called")

    def predict(self, X, meta=None):
        return np.asarray(X["voltage_mean"], dtype=float) * 0.0 + 0.95


def _manifest():
    return {
        "model_name": "predict_only",
        "model_kind": "pickle",
        "estimator": "predict_only",
        "featureset": "full",
        "feature_columns": ["voltage_mean"],
        "target": "soh",
        "trained_on": ["train"],
        "created_by_run": "unit",
        "predict_contract": "predict(X, meta=None)",
    }


def _table():
    return pd.DataFrame({
        "dataset": ["new_a", "new_a", "new_b", "new_b"],
        "cell_id": ["c1", "c1", "c2", "c2"],
        "cycle_index": [0, 1, 0, 1],
        "soh": [0.96, 0.94, 0.90, 0.88],
        "voltage_mean": [3.5, 3.4, 3.3, 3.2],
    })


def test_inference_benchmark_predicts_by_dataset_without_fit(tmp_path):
    store = ModelStore(tmp_path / "models")
    store.upload_model(
        "predict_only",
        "predict_only.pkl",
        pickle.dumps(PredictOnlyModel()),
        "model_manifest.json",
        json.dumps(_manifest()).encode("utf-8"),
    )

    records = run_inference_benchmark(store, _table(), pd.DataFrame(), ["predict_only"])

    assert {row["dataset"] for row in records} == {"new_a", "new_b"}
    assert all(row["status"] == "succeeded" for row in records)
    assert all(row["predict_ms"] >= 0 for row in records)
    assert all(row["ms_per_1000_samples"] >= 0 for row in records)
    assert records[0]["mae"] is not None


def test_inference_benchmark_records_feature_mismatch(tmp_path):
    store = ModelStore(tmp_path / "models")
    manifest = _manifest()
    manifest["feature_columns"] = ["missing_signal"]
    store.save_training_model("bad_model", PredictOnlyModel(), manifest, replace=True)

    records = run_inference_benchmark(store, _table(), pd.DataFrame(), ["bad_model"])

    assert records[0]["status"] == "failed"
    assert "missing feature columns" in records[0]["error"]
```

Append to `tests/test_report.py`:

```python
from sohbench.report import write_inference_artifacts


def test_write_inference_artifacts(tmp_path):
    out = write_inference_artifacts(
        [
            {
                "model": "predict_only",
                "dataset": "new_a",
                "n_samples": 2,
                "mae": 0.01,
                "rmse": 0.02,
                "mean_bias_error": 0.0,
                "predict_ms": 1.2,
                "load_ms": 0.4,
                "ms_per_1000_samples": 600.0,
                "samples_per_second": 1666.7,
                "status": "succeeded",
                "error": "",
            }
        ],
        {"datasets": ["new_a"], "models": ["predict_only"]},
        tmp_path,
    )
    run_dir = Path(out)
    assert (run_dir / "inference_results.csv").exists()
    assert (run_dir / "inference_benchmark.json").exists()
    payload = json.loads((run_dir / "inference_benchmark.json").read_text(encoding="utf-8"))
    assert payload["records"][0]["model"] == "predict_only"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_inference.py tests/test_report.py::test_write_inference_artifacts -v
```

Expected: FAIL because `sohbench.inference` and `write_inference_artifacts` do not exist.

- [ ] **Step 3: Implement inference runner**

Create `src/sohbench/inference.py`:

```python
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import metrics as M


def run_inference_benchmark(model_store, table: pd.DataFrame, meta: pd.DataFrame | None, model_names: list[str]) -> list[dict]:
    records: list[dict] = []
    for model_name in model_names:
        try:
            loaded = model_store.load_model(model_name)
        except Exception as exc:
            records.append(_failed_record(model_name, "all", 0, 0.0, str(exc)))
            continue

        for dataset, group in table.groupby("dataset", sort=True):
            base = {
                "model": model_name,
                "dataset": str(dataset),
                "featureset": loaded.manifest.get("featureset"),
                "estimator": loaded.manifest.get("estimator"),
                "n_samples": int(len(group)),
                "load_ms": float(loaded.load_ms),
            }
            try:
                start = time.perf_counter()
                pred = loaded.predict(group, meta)
                predict_ms = (time.perf_counter() - start) * 1000.0
                metrics = _metrics(group, pred)
                records.append({
                    **base,
                    **metrics,
                    "predict_ms": float(predict_ms),
                    "ms_per_1000_samples": float(predict_ms / max(len(group), 1) * 1000.0),
                    "samples_per_second": float(len(group) / max(predict_ms / 1000.0, 1e-12)),
                    "status": "succeeded",
                    "error": "",
                })
            except Exception as exc:
                records.append({
                    **base,
                    "mae": None,
                    "rmse": None,
                    "mean_bias_error": None,
                    "monotonicity_violation_rate": None,
                    "value_range_violation": None,
                    "predict_ms": None,
                    "ms_per_1000_samples": None,
                    "samples_per_second": None,
                    "status": "failed",
                    "error": str(exc),
                })
    return records


def _metrics(group: pd.DataFrame, pred: np.ndarray) -> dict:
    out = {
        "monotonicity_violation_rate": M.monotonicity_violation_rate(pred[np.argsort(group["cycle_index"].to_numpy())]),
        "value_range_violation": M.value_range_violation(pred),
    }
    if "soh" in group.columns:
        y = group["soh"].to_numpy(dtype=float)
        out.update({
            "mae": M.mae(y, pred),
            "rmse": M.rmse(y, pred),
            "mean_bias_error": M.mean_bias_error(y, pred),
        })
    else:
        out.update({"mae": None, "rmse": None, "mean_bias_error": None})
    return out


def _failed_record(model_name: str, dataset: str, n_samples: int, load_ms: float, error: str) -> dict:
    return {
        "model": model_name,
        "dataset": dataset,
        "featureset": None,
        "estimator": None,
        "n_samples": n_samples,
        "load_ms": load_ms,
        "mae": None,
        "rmse": None,
        "mean_bias_error": None,
        "monotonicity_violation_rate": None,
        "value_range_violation": None,
        "predict_ms": None,
        "ms_per_1000_samples": None,
        "samples_per_second": None,
        "status": "failed",
        "error": error,
    }
```

- [ ] **Step 4: Implement inference artifact writer**

In `src/sohbench/report.py`, add:

```python
def write_inference_artifacts(records: list[dict], config: dict, out_dir):
    out = Path(out_dir) / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    frame.to_parquet(out / "inference_results.parquet")
    frame.to_csv(out / "inference_results.csv", index=False)
    payload = {
        "records": records,
        "summary": _inference_summary(records),
    }
    _dump(payload, out / "inference_benchmark.json")
    _dump(config, out / "inference_run_config.json")
    return str(out)


def _inference_summary(records: list[dict]) -> dict:
    succeeded = [row for row in records if row.get("status") == "succeeded"]
    fastest = min(succeeded, key=lambda row: row.get("ms_per_1000_samples") or float("inf"), default=None)
    best = min(
        [row for row in succeeded if row.get("mae") is not None],
        key=lambda row: row.get("mae"),
        default=None,
    )
    return {
        "n_records": len(records),
        "n_succeeded": len(succeeded),
        "fastest_model": fastest.get("model") if fastest else None,
        "fastest_ms_per_1000_samples": fastest.get("ms_per_1000_samples") if fastest else None,
        "best_transfer_model": best.get("model") if best else None,
        "best_transfer_mae": best.get("mae") if best else None,
    }
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_inference.py tests/test_report.py::test_write_inference_artifacts -v
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/sohbench/inference.py src/sohbench/report.py tests/test_inference.py tests/test_report.py
git commit -m "feat: add inference-only benchmark artifacts"
```

---

### Task 4: Backend API for Models and Inference Jobs

**Files:**
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `src/sohbench/api_server.py`
- Test: `tests/test_evaluation_service.py`
- Test: `tests/test_api_server.py`

**Interfaces:**
- Consumes: `run_inference_benchmark(...)`
- Consumes: `write_inference_artifacts(...)`
- Produces: `EvaluationService.create_inference_job(payload: dict, start: bool = True) -> dict`
- Produces: `EvaluationService.run_inference_job_inline(job_id: str) -> dict`
- Produces: `EvaluationService.get_inference_job(job_id: str) -> dict`
- Produces: `EvaluationService.list_inference_jobs() -> list[dict]`
- Produces endpoint `POST /api/upload/model`
- Produces endpoint `GET /api/models`
- Produces endpoint `POST /api/inference-jobs`
- Produces endpoint `GET /api/inference-jobs`
- Produces endpoint `GET /api/inference-jobs/{id}`

- [ ] **Step 1: Write failing service/API tests**

Append to `tests/test_evaluation_service.py`:

```python
import pickle
import numpy as np


class PickledPredictOnly:
    def fit(self, X, y=None, meta=None):
        raise AssertionError("fit must not be called")

    def predict(self, X, meta=None):
        return np.full(len(X), 0.95)


def _model_manifest(name="api_model"):
    return {
        "model_name": name,
        "model_kind": "pickle",
        "estimator": "pickled_predict_only",
        "featureset": "full",
        "feature_columns": ["voltage_mean"],
        "target": "soh",
        "trained_on": ["train"],
        "created_by_run": "unit",
        "predict_contract": "predict(X, meta=None)",
    }


def test_inference_job_runs_uploaded_pickle_model(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
    )

    job = service.create_inference_job(
        {"datasets": ["plugin_eval_demo"], "models": ["api_model"]},
        start=False,
    )
    finished = service.run_inference_job_inline(job["id"])

    assert finished["status"] == "succeeded"
    assert (Path(finished["run_dir"]) / "inference_benchmark.json").exists()
    assert (tmp_path / "frontend_data" / "inference_benchmark.json").exists()
```

Append to `tests/test_api_server.py`:

```python
def test_api_upload_model_and_start_inference_job(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        _post_json(
            base_url,
            "/api/upload/dataset",
            {
                "name": "api_demo",
                "filename": "api_demo.csv",
                "contentBase64": base64.b64encode(_dataset_bytes()).decode("ascii"),
            },
        )
        manifest = _model_manifest("api_model")
        uploaded = _post_json(
            base_url,
            "/api/upload/model",
            {
                "name": "api_model",
                "modelFilename": "api_model.pkl",
                "modelContentBase64": base64.b64encode(pickle.dumps(PickledPredictOnly())).decode("ascii"),
                "manifestFilename": "model_manifest.json",
                "manifestContentBase64": base64.b64encode(json.dumps(manifest).encode("utf-8")).decode("ascii"),
            },
        )
        assert uploaded["resource"]["name"] == "api_model"

        models = _get_json(base_url, "/api/models")
        assert any(item["name"] == "api_model" for item in models["models"])

        job = _post_json(
            base_url,
            "/api/inference-jobs",
            {"datasets": ["api_demo"], "models": ["api_model"]},
        )
        deadline = time.time() + 20
        status = job
        while time.time() < deadline and status["status"] not in {"succeeded", "failed"}:
            time.sleep(0.2)
            status = _get_json(base_url, f"/api/inference-jobs/{job['id']}")
        assert status["status"] == "succeeded"
    finally:
        server.shutdown()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_evaluation_service.py::test_inference_job_runs_uploaded_pickle_model tests/test_api_server.py::test_api_upload_model_and_start_inference_job -v
```

Expected: FAIL because `create_inference_job` and API endpoints do not exist.

- [ ] **Step 3: Implement inference job lifecycle**

In `src/sohbench/evaluation_service.py`, import:

```python
from .inference import run_inference_benchmark
from .report import write_artifacts, write_inference_artifacts
```

In `__init__`:

```python
self.inference_jobs_dir = self.work_dir / "inference_jobs"
self.inference_jobs_dir.mkdir(parents=True, exist_ok=True)
```

Add methods:

```python
def create_inference_job(self, payload: dict, *, start: bool = True) -> dict:
    request = self._normalize_inference_request(payload)
    job_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    job = {"id": job_id, "status": "queued", "created_at": _now(), "request": request, "kind": "inference"}
    self._save_inference_job(job)
    if start:
        thread = threading.Thread(target=self.run_inference_job_inline, args=(job_id,), daemon=True)
        thread.start()
    return job

def run_inference_job_inline(self, job_id: str) -> dict:
    job = self.get_inference_job(job_id)
    job.update({"status": "running", "started_at": _now()})
    self._save_inference_job(job)
    try:
        request = job["request"]
        loader = DatasetLoader(self.data_dir)
        datasets = loader.resolve_datasets(request["datasets"])
        table, meta, qc = loader.load(datasets)
        records = run_inference_benchmark(self.model_store, table, meta, request["models"])
        raw_run_dir = Path(write_inference_artifacts(records, {"datasets": datasets, **request}, self.runs_dir))
        (raw_run_dir / "model_catalog.json").write_text(
            json.dumps({"models": self.model_store.catalog()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        published_run_dir = self._publish_frontend_artifacts(raw_run_dir)
        job.update({"status": "succeeded", "finished_at": _now(), "raw_run_dir": str(raw_run_dir), "run_dir": str(published_run_dir)})
    except Exception as exc:
        job.update({"status": "failed", "finished_at": _now(), "error": str(exc), "traceback": traceback.format_exc(limit=12)})
    self._save_inference_job(job)
    return job

def get_inference_job(self, job_id: str) -> dict:
    path = self.inference_jobs_dir / f"{job_id}.json"
    if not path.exists():
        raise KeyError(job_id)
    return json.loads(path.read_text(encoding="utf-8"))

def list_inference_jobs(self) -> list[dict]:
    jobs = []
    for path in sorted(self.inference_jobs_dir.glob("*.json"), reverse=True):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return jobs

def _normalize_inference_request(self, payload: dict) -> dict:
    datasets = _string_list(payload.get("datasets") or ["all"])
    models = _string_list(payload.get("models") or [])
    if not models:
        raise ValueError("at least one model is required")
    known = {item["name"] for item in self.model_store.catalog()}
    missing = [name for name in models if name not in known]
    if missing:
        raise ValueError(f"unknown models: {missing}")
    return {"datasets": datasets, "models": models}

def _save_inference_job(self, job: dict) -> None:
    self.inference_jobs_dir.mkdir(parents=True, exist_ok=True)
    (self.inference_jobs_dir / f"{job['id']}.json").write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
```

Add to `catalog()`:

```python
catalog["inference_jobs"] = self.list_inference_jobs()
```

- [ ] **Step 4: Implement API endpoints**

In `src/sohbench/api_server.py`, import `ModelStoreError`:

```python
from .model_store import ModelStoreError
```

In `do_GET`, add:

```python
elif path == "/api/models":
    self._send_json({"models": service.model_store.catalog()})
elif path == "/api/inference-jobs":
    self._send_json({"jobs": service.list_inference_jobs()})
elif path.startswith("/api/inference-jobs/"):
    job_id = path.rsplit("/", 1)[-1]
    self._send_json(service.get_inference_job(job_id))
```

In `do_POST`, add:

```python
elif path == "/api/upload/model":
    resource = service.model_store.upload_model(
        payload.get("name"),
        payload.get("modelFilename", ""),
        _decode_named_content(payload, "modelContentBase64"),
        payload.get("manifestFilename", ""),
        _decode_named_content(payload, "manifestContentBase64"),
        replace=bool(payload.get("replace", False)),
    )
    self._send_json({"ok": True, "resource": resource})
elif path == "/api/inference-jobs":
    self._send_json(service.create_inference_job(payload, start=True), status=HTTPStatus.ACCEPTED)
```

Change the bad-request exception tuple to:

```python
except (TrustedPluginError, ModelStoreError, ValueError, KeyError) as exc:
```

Add helper:

```python
def _decode_named_content(payload: dict, key: str) -> bytes:
    encoded = payload.get(key)
    if not encoded:
        raise ValueError(f"{key} is required")
    return base64.b64decode(encoded)
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_evaluation_service.py::test_inference_job_runs_uploaded_pickle_model tests/test_api_server.py::test_api_upload_model_and_start_inference_job -v
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/sohbench/evaluation_service.py src/sohbench/api_server.py tests/test_evaluation_service.py tests/test_api_server.py
git commit -m "feat: add inference job API"
```

---

### Task 5: Frontend Inference Tab and Model Library

**Files:**
- Modify: `frontend/app/dashboard-core.mjs`
- Modify: `frontend/app/dashboard-core.test.mjs`
- Modify: `frontend/app/app.js`
- Modify: `frontend/app/index.html`
- Modify: `frontend/app/styles.css`
- Modify: `frontend/app/chinese-ui.test.mjs`

**Interfaces:**
- Consumes backend catalog fields: `models`, `inference_jobs`
- Consumes `frontend/app/data/inference_benchmark.json`
- Consumes `frontend/app/data/model_catalog.json`
- Produces `modelOptions(catalogOrArtifact) -> list`
- Produces `inferenceRows(payload, selectedModels, selectedDatasets) -> list`
- Produces `inferenceSummary(payload) -> dict`
- Produces UI sections with IDs: `mode-tabs`, `training-panel`, `inference-panel`, `model-library`, `inference-models`, `inference-datasets`, `start-inference`, `inference-results`

- [ ] **Step 1: Write failing frontend core tests**

Append to `frontend/app/dashboard-core.test.mjs`:

```javascript
import {
  inferenceRows,
  inferenceSummary,
  modelOptions,
  buildInferencePayload,
} from "./dashboard-core.mjs";

test("modelOptions merges catalog and artifact models", () => {
  const rows = modelOptions({
    models: [
      { name: "ridge_full", source: "training", estimator: "ridge", featureset: "full" },
      { name: "external_mlp", source: "upload", estimator: "mlp", featureset: "ic_dva" },
    ],
  });
  assert.deepEqual(rows.map((row) => row.name), ["external_mlp", "ridge_full"]);
  assert.equal(rows[0].badge, "上传模型");
});

test("inferenceRows filters and sorts by latency", () => {
  const payload = {
    records: [
      { model: "slow", dataset: "nasa", status: "succeeded", ms_per_1000_samples: 20, mae: 0.2 },
      { model: "fast", dataset: "nasa", status: "succeeded", ms_per_1000_samples: 5, mae: 0.3 },
      { model: "fast", dataset: "hust", status: "failed", ms_per_1000_samples: null, mae: null },
    ],
  };
  const rows = inferenceRows(payload, ["fast", "slow"], ["nasa"]);
  assert.deepEqual(rows.map((row) => row.model), ["fast", "slow"]);
});

test("inferenceSummary finds fastest and best transfer model", () => {
  const summary = inferenceSummary({
    records: [
      { model: "a", status: "succeeded", ms_per_1000_samples: 10, samples_per_second: 100000, mae: 0.2 },
      { model: "b", status: "succeeded", ms_per_1000_samples: 5, samples_per_second: 200000, mae: 0.3 },
    ],
  });
  assert.equal(summary.fastest.model, "b");
  assert.equal(summary.bestTransfer.model, "a");
});

test("buildInferencePayload uses checked models and datasets", () => {
  const payload = buildInferencePayload({
    models: ["m1", "m2"],
    datasets: ["nasa"],
  });
  assert.deepEqual(payload.models, ["m1", "m2"]);
  assert.deepEqual(payload.datasets, ["nasa"]);
});
```

- [ ] **Step 2: Run frontend tests to verify RED**

Run:

```powershell
node --test frontend/app/dashboard-core.test.mjs
```

Expected: FAIL because the new exports do not exist.

- [ ] **Step 3: Implement frontend core helpers**

In `frontend/app/dashboard-core.mjs`, export:

```javascript
export function modelOptions(source = {}) {
  const models = Array.isArray(source.models) ? source.models : [];
  return models
    .map((item) => ({
      name: String(item.name || ""),
      source: item.source || "training",
      estimator: item.estimator || "",
      featureset: item.featureset || "",
      featureColumns: item.feature_columns || item.featureColumns || [],
      trainedOn: item.trained_on || item.trainedOn || [],
      badge: item.source === "upload" ? "上传模型" : "训练产物",
    }))
    .filter((item) => item.name)
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function inferenceRows(payload = {}, selectedModels = [], selectedDatasets = []) {
  const modelSet = new Set(selectedModels || []);
  const datasetSet = new Set(selectedDatasets || []);
  return (payload.records || [])
    .filter((row) => !modelSet.size || modelSet.has(row.model))
    .filter((row) => !datasetSet.size || datasetSet.has(row.dataset))
    .sort((a, b) => {
      const av = a.ms_per_1000_samples ?? Number.POSITIVE_INFINITY;
      const bv = b.ms_per_1000_samples ?? Number.POSITIVE_INFINITY;
      return av - bv || String(a.model).localeCompare(String(b.model));
    });
}

export function inferenceSummary(payload = {}) {
  const ok = (payload.records || []).filter((row) => row.status === "succeeded");
  const fastest = [...ok].sort((a, b) => (a.ms_per_1000_samples ?? Infinity) - (b.ms_per_1000_samples ?? Infinity))[0] || null;
  const bestTransfer = ok.filter((row) => row.mae != null).sort((a, b) => a.mae - b.mae)[0] || null;
  const highestThroughput = [...ok].sort((a, b) => (b.samples_per_second ?? 0) - (a.samples_per_second ?? 0))[0] || null;
  return { count: ok.length, fastest, bestTransfer, highestThroughput };
}

export function buildInferencePayload({ models = [], datasets = [] } = {}) {
  return {
    models: [...models].map(String).filter(Boolean),
    datasets: [...datasets].map(String).filter(Boolean),
  };
}
```

- [ ] **Step 4: Add optional artifact loading**

In `frontend/app/app.js`, extend `FALLBACK`:

```javascript
inference: { records: [], summary: {} },
modelCatalog: { models: [] },
```

In `load()`, fetch optional files:

```javascript
const [lb, qc, cfg, byDataset, inference, modelCatalog] = await Promise.all([
  fetch("./data/leaderboard.json" + suffix).then((r) => { if (!r.ok) throw 0; return r.json(); }),
  fetch("./data/qc_report.json" + suffix).then((r) => (r.ok ? r.json() : null)).catch(() => null),
  fetch("./data/run_config.json" + suffix).then((r) => (r.ok ? r.json() : null)).catch(() => null),
  fetch("./data/dataset_leaderboard.json" + suffix).then((r) => (r.ok ? r.json() : [])).catch(() => []),
  fetch("./data/inference_benchmark.json" + suffix).then((r) => (r.ok ? r.json() : { records: [] })).catch(() => ({ records: [] })),
  fetch("./data/model_catalog.json" + suffix).then((r) => (r.ok ? r.json() : { models: [] })).catch(() => ({ models: [] })),
]);
DATA = { leaderboard: lb, datasetLeaderboard: byDataset || [], qc, config: cfg || {}, inference, modelCatalog };
```

Add model selection state:

```javascript
let activeMode = "training";
let inferenceSelection = {
  models: new Set(),
  datasets: new Set(),
};
```

- [ ] **Step 5: Add model upload and inference job actions**

In `uploadResource(kind)`, add a separate branch for `kind === "model"`:

```javascript
if (kind === "model") {
  const modelFile = $("upload-model-file")?.files?.[0];
  const manifestFile = $("upload-model-manifest")?.files?.[0];
  if (!modelFile || !manifestFile) {
    setJobMessage("请选择模型 Pickle 文件和 model_manifest.json。");
    return;
  }
  const name = ($("upload-model-name").value || modelFile.name.replace(/\.[^.]+$/, "")).trim();
  const response = await postJson("/api/upload/model", {
    name,
    modelFilename: modelFile.name,
    modelContentBase64: await fileToBase64(modelFile),
    manifestFilename: manifestFile.name,
    manifestContentBase64: await fileToBase64(manifestFile),
  });
  inferenceSelection.models.add(response.resource.name);
  await loadBackendCatalog();
  render();
  setJobMessage(`已上传模型 ${response.resource.name}`);
  return;
}
```

Add:

```javascript
async function startInferenceEvaluation() {
  const payload = buildInferencePayload({
    models: selectedValues("inference-models"),
    datasets: selectedValues("inference-datasets"),
  });
  if (!payload.models.length || !payload.datasets.length) {
    setJobMessage("请至少选择一个模型和一个数据集。");
    return;
  }
  try {
    CURRENT_JOB = await postJson("/api/inference-jobs", payload);
    pollInferenceJob(CURRENT_JOB.id);
  } catch (err) {
    setJobMessage(`推理任务启动失败：${err.message}`);
  }
}
```

Add `pollInferenceJob(jobId)` mirroring `pollJob`, using `/api/inference-jobs/{id}`.

- [ ] **Step 6: Add DOM sections**

In `frontend/app/index.html`, add a top-level tab strip near the workbench:

```html
<div class="mode-tabs" id="mode-tabs" role="tablist" aria-label="评测模式">
  <button class="mode-tab" data-mode="training" aria-selected="true">训练测评</button>
  <button class="mode-tab" data-mode="inference" aria-selected="false">推理测评</button>
  <button class="mode-tab" data-mode="models" aria-selected="false">模型库</button>
</div>
```

Add an inference panel:

```html
<section class="panel" id="inference-panel" data-mode-panel="inference">
  <div class="section-head">
    <div>
      <h2>推理测评</h2>
      <p>选择已有 Pickle 模型和数据集，只执行 predict，比较迁移精度与推理速度。</p>
    </div>
    <button id="start-inference" class="primary">启动推理测评</button>
  </div>
  <div class="upload-grid">
    <label>模型名称<input id="upload-model-name" type="text" /></label>
    <label>Pickle 文件<input id="upload-model-file" type="file" accept=".pkl,.pickle" /></label>
    <label>Manifest<input id="upload-model-manifest" type="file" accept=".json" /></label>
    <button data-upload-kind="model">上传模型</button>
  </div>
  <div class="resource-columns">
    <div><h3>模型</h3><div id="inference-models" class="resource-list"></div></div>
    <div><h3>数据集</h3><div id="inference-datasets" class="resource-list"></div></div>
  </div>
  <div id="inference-results"></div>
</section>
```

Add a model library panel:

```html
<section class="panel" id="model-library" data-mode-panel="models">
  <div class="section-head">
    <h2>模型库</h2>
  </div>
  <div id="model-library-table"></div>
</section>
```

- [ ] **Step 7: Render inference and model library**

In `frontend/app/app.js`, import helpers:

```javascript
  buildInferencePayload,
  inferenceRows,
  inferenceSummary,
  modelOptions,
```

Add render calls:

```javascript
renderModeTabs();
renderInferencePanel(dsOptions);
renderModelLibrary();
```

Add:

```javascript
function renderModeTabs() {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.mode === activeMode));
    button.onclick = () => {
      activeMode = button.dataset.mode;
      render();
    };
  });
  document.querySelectorAll("[data-mode-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.modePanel !== activeMode;
  });
}

function renderInferencePanel(dsOptions) {
  const models = modelOptions(CATALOG || DATA.modelCatalog || {});
  if (!inferenceSelection.models.size && models.length) inferenceSelection.models.add(models[0].name);
  if (!inferenceSelection.datasets.size) dsOptions.forEach((item) => inferenceSelection.datasets.add(item.name));
  renderResourceList("inference-models", models, inferenceSelection.models, (item) => `${item.estimator || "模型"} / ${item.featureset || "特征"}`);
  renderResourceList("inference-datasets", dsOptions, inferenceSelection.datasets, (item) => `${item.rows.toLocaleString()} 行`);
  const rows = inferenceRows(DATA.inference || {}, [...inferenceSelection.models], [...inferenceSelection.datasets]);
  const summary = inferenceSummary({ records: rows });
  $("inference-results").innerHTML = renderInferenceSummary(summary) + renderInferenceTable(rows);
  const start = $("start-inference");
  if (start) start.onclick = startInferenceEvaluation;
}
```

Implement `renderInferenceSummary(summary)` and `renderInferenceTable(rows)` with Chinese labels and existing `fmt` helper.

- [ ] **Step 8: Add styles**

In `frontend/app/styles.css`, add:

```css
.mode-tabs {
  display: flex;
  gap: 8px;
  margin: 16px 0;
}
.mode-tab {
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
}
.mode-tab[aria-selected="true"] {
  background: var(--primary);
  color: white;
}
.resource-columns {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.inference-kpis {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0;
}
@media (max-width: 760px) {
  .resource-columns,
  .inference-kpis {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 9: Run frontend tests to verify GREEN**

Run:

```powershell
node --test frontend/app/dashboard-core.test.mjs frontend/app/chinese-ui.test.mjs
```

Expected: all frontend tests PASS.

- [ ] **Step 10: Commit**

Run:

```powershell
git add frontend/app/dashboard-core.mjs frontend/app/dashboard-core.test.mjs frontend/app/app.js frontend/app/index.html frontend/app/styles.css frontend/app/chinese-ui.test.mjs
git commit -m "feat: add inference evaluation frontend"
```

---

### Task 6: Build Artifact Script, End-to-End Smoke, and Browser Verification

**Files:**
- Modify: `scripts/build_frontend_artifacts.py`
- Test: `tests/test_build_frontend_artifacts.py`

**Interfaces:**
- Consumes optional run files: `inference_benchmark.json`, `model_catalog.json`, `inference_run_config.json`
- Produces frontend files when optional artifacts exist.

- [ ] **Step 1: Write failing script test**

Append to `tests/test_build_frontend_artifacts.py`:

```python
def test_build_frontend_artifacts_copies_inference_and_model_files(tmp_path):
    run_dir = tmp_path / "runs" / "20260101T000000"
    run_dir.mkdir(parents=True)
    for name, payload in {
        "leaderboard.json": [],
        "dataset_leaderboard.json": [],
        "qc_report.json": {},
        "run_config.json": {},
        "inference_benchmark.json": {"records": []},
        "model_catalog.json": {"models": []},
    }.items():
        (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "frontend_data"
    build_frontend_artifacts(run_dir, out_dir)

    assert (out_dir / "inference_benchmark.json").exists()
    assert (out_dir / "model_catalog.json").exists()
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_build_frontend_artifacts.py::test_build_frontend_artifacts_copies_inference_and_model_files -v
```

Expected: FAIL because optional files are not copied or helper name differs.

- [ ] **Step 3: Update script optional file list**

In `scripts/build_frontend_artifacts.py`, add these optional frontend files to the copy list:

```python
OPTIONAL_FRONTEND_FILES = (
    "inference_benchmark.json",
    "inference_run_config.json",
    "model_catalog.json",
)
```

Copy each optional file only when the source exists.

- [ ] **Step 4: Run backend and frontend targeted tests**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_model_store.py tests/test_inference.py tests/test_report.py tests/test_runner.py::test_runner_calls_on_model_fitted_after_successful_fit tests/test_evaluation_service.py tests/test_api_server.py tests/test_build_frontend_artifacts.py -v
node --test frontend/app/dashboard-core.test.mjs frontend/app/chinese-ui.test.mjs
```

Expected: all selected tests PASS.

- [ ] **Step 5: Run a tiny local end-to-end inference job**

Run a short Python script through the venv:

```powershell
$env:PYTHONPATH='src'; @'
import json, pickle, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sohbench.evaluation_service import EvaluationService

class Model:
    def predict(self, X, meta=None):
        return np.full(len(X), 0.95)

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    service = EvaluationService(
        data_dir=root / "data",
        work_dir=root / "work",
        runs_dir=root / "runs",
        frontend_data_dir=root / "frontend",
    )
    table = pd.DataFrame({
        "dataset": ["new"] * 12,
        "cell_id": ["c1"] * 12,
        "cycle_index": list(range(12)),
        "soh": [1.0 - 0.01 * i for i in range(12)],
        "voltage_mean": [3.5 - 0.01 * i for i in range(12)],
    })
    service.store.upload_dataset("new", "new.csv", table.to_csv(index=False).encode("utf-8"))
    manifest = {
        "model_name": "smoke_model",
        "model_kind": "pickle",
        "estimator": "constant",
        "featureset": "full",
        "feature_columns": ["voltage_mean"],
        "target": "soh",
        "trained_on": ["old"],
        "created_by_run": "manual",
        "predict_contract": "predict(X, meta=None)",
    }
    service.model_store.upload_model(
        "smoke_model",
        "smoke_model.pkl",
        pickle.dumps(Model()),
        "model_manifest.json",
        json.dumps(manifest).encode("utf-8"),
    )
    job = service.create_inference_job({"datasets": ["new"], "models": ["smoke_model"]}, start=False)
    finished = service.run_inference_job_inline(job["id"])
    assert finished["status"] == "succeeded", finished
    assert (root / "frontend" / "inference_benchmark.json").exists()
    print(finished["run_dir"])
'@ | .\.venv\Scripts\python.exe -
```

Expected: prints the inference run directory and exits 0.

- [ ] **Step 6: Start or refresh local services**

If no server is running, start:

```powershell
Start-Process -WindowStyle Hidden -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "http.server", "8000", "--directory", "frontend/app" -WorkingDirectory "D:\SOH_eval"
Start-Process -WindowStyle Hidden -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "sohbench.api_server", "--host", "127.0.0.1", "--port", "8001", "--data-dir", "D:\battery_SOH\data" -WorkingDirectory "D:\SOH_eval"
```

Expected: `http://localhost:8000/` shows the frontend and `http://localhost:8001/api/health` returns `{"ok": true, "mode": "trusted_local"}`.

- [ ] **Step 7: Browser verification**

Open `http://localhost:8000/?v=inference-models` and verify:

- tabs show `训练测评`, `推理测评`, `模型库`
- old accuracy panels still render
- inference tab shows model selector or empty state
- model upload controls accept `.pkl` and `model_manifest.json`
- no obvious text overlap on desktop viewport

- [ ] **Step 8: Final verification**

Run:

```powershell
git diff --check
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/test_model_store.py tests/test_inference.py tests/test_report.py tests/test_evaluation_service.py tests/test_api_server.py tests/test_build_frontend_artifacts.py -v
node --test frontend/app/dashboard-core.test.mjs frontend/app/chinese-ui.test.mjs
```

Expected: no whitespace errors and all tests PASS.

- [ ] **Step 9: Commit**

Run:

```powershell
git add scripts/build_frontend_artifacts.py tests/test_build_frontend_artifacts.py
git commit -m "feat: publish inference frontend artifacts"
```
