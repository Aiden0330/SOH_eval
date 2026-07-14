# SOH Fabric Contract & Evidence Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不移除旧 API 和旧前端产物的前提下，建立无目标泄漏、结构化失败、稳定 comparison key 和可校验 EvidencePack 的测评基础。

**Architecture:** 新增 `sohbench.fabric` 小型内核，现有 `BenchmarkRunner` 和 `EvaluationService` 通过兼容适配器调用。Phase 0 仍在当前进程执行，但算法只接收 target-stripped 视图，报告同时写旧 JSON 和新 EvidencePack。

**Tech Stack:** Python 3.11+、dataclasses、hashlib、jsonschema、pandas、NumPy、SciPy、pytest。

## Global Constraints

- 不新增数据库或后台进程；它们属于下一子计划。
- 不删除 `write_artifacts`、`BenchmarkRunner.run` 或 v1 API。
- 所有新增公开类型从 `sohbench.fabric` 导出。
- 所有 DataFrame 摘要必须与行顺序规则一起定义。
- 测试视图不得包含 `soh`。
- 旧 leaderboard 仍可读取，但必须新增 coverage、失败和证据引用。

---

### Task 1: Canonical IDs And Digests

**Files:**
- Create: `src/sohbench/fabric/__init__.py`
- Create: `src/sohbench/fabric/ids.py`
- Test: `tests/fabric/test_ids.py`
- Create: `tests/fabric/__init__.py`

**Interfaces:**
- Produces: `canonical_json(value) -> bytes`
- Produces: `sha256_json(value) -> str`
- Produces: `sha256_file(path: Path) -> str`
- Produces: `new_id(prefix: str) -> str`

- [ ] **Step 1: Write the failing digest tests**

```python
from pathlib import Path

from sohbench.fabric.ids import canonical_json, new_id, sha256_file, sha256_json


def test_sha256_json_is_key_order_independent():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_canonical_json_rejects_nan():
    import pytest
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"value": float("nan")})


def test_sha256_file_and_new_id(tmp_path: Path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"evidence")
    assert len(sha256_file(path)) == 64
    assert new_id("RUN").startswith("RUN-")
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `python -m pytest tests/fabric/test_ids.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'sohbench.fabric'`.

- [ ] **Step 3: Implement canonical encoding and IDs**

```python
# src/sohbench/fabric/ids.py
from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON contains a non-finite float")
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        _finite(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_id(prefix: str) -> str:
    clean = str(prefix).strip().upper()
    if not clean or not clean.replace("_", "").isalnum():
        raise ValueError(f"invalid id prefix: {prefix!r}")
    return f"{clean}-{uuid.uuid4().hex}"
```

Export the four functions from `src/sohbench/fabric/__init__.py`.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/fabric/test_ids.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/__init__.py src/sohbench/fabric/ids.py tests/fabric/__init__.py tests/fabric/test_ids.py
git commit -m "feat: add canonical evidence identifiers"
```

### Task 2: Immutable RunSpec And Comparison Key

**Files:**
- Create: `src/sohbench/fabric/specs.py`
- Create: `src/sohbench/fabric/schemas/run-spec.schema.json`
- Test: `tests/fabric/test_specs.py`
- Modify: `src/sohbench/fabric/__init__.py`

**Interfaces:**
- Consumes: `sha256_json`
- Produces: `SpecRef(kind, spec_id, version, digest)`
- Produces: `RunSpec(...).to_dict() -> dict`
- Produces: `RunSpec.comparison_key -> str`
- Produces: `validate_run_spec_document(document: dict) -> None`

- [ ] **Step 1: Write failing spec tests**

```python
import pytest

from sohbench.fabric.specs import RunSpec, SpecRef, validate_run_spec_document


def ref(kind: str, name: str) -> SpecRef:
    return SpecRef(kind=kind, spec_id=name, version="1.0.0", digest=(name[0] * 64))


def run_spec(candidate: str = "ridge") -> RunSpec:
    return RunSpec(
        run_type="training",
        task=ref("task", "soh"),
        dataset=ref("dataset_snapshot", "seven"),
        protocol=ref("protocol", "loco"),
        metric=ref("metric", "soh-core"),
        policy=ref("policy", "fair-v1"),
        candidates=(ref("candidate", candidate),),
        seeds=(0, 1, 2),
    )


def test_candidate_change_does_not_change_comparison_key():
    assert run_spec("ridge").comparison_key == run_spec("gbdt").comparison_key


def test_policy_change_changes_comparison_key():
    first = run_spec()
    second = RunSpec(**{**first.__dict__, "policy": ref("policy", "other")})
    assert first.comparison_key != second.comparison_key


def test_schema_rejects_unknown_run_type():
    document = run_spec().to_dict()
    document["run_type"] = "magic"
    with pytest.raises(ValueError, match="run_type"):
        validate_run_spec_document(document)
```

- [ ] **Step 2: Verify tests fail before implementation**

Run: `python -m pytest tests/fabric/test_specs.py -q`

Expected: FAIL because `sohbench.fabric.specs` does not exist.

- [ ] **Step 3: Implement the immutable contract types**

```python
# src/sohbench/fabric/specs.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Literal

import jsonschema

from .ids import sha256_json

RunType = Literal["training", "adaptation", "inference"]


@dataclass(frozen=True)
class SpecRef:
    kind: str
    spec_id: str
    version: str
    digest: str


@dataclass(frozen=True)
class RunSpec:
    run_type: RunType
    task: SpecRef
    dataset: SpecRef
    protocol: SpecRef
    metric: SpecRef
    policy: SpecRef
    candidates: tuple[SpecRef, ...]
    seeds: tuple[int, ...] = (0,)
    idempotency_key: str = ""

    def to_dict(self) -> dict:
        value = asdict(self)
        value["candidates"] = [asdict(item) for item in self.candidates]
        value["seeds"] = list(self.seeds)
        return value

    @property
    def comparison_key(self) -> str:
        return sha256_json({
            "task": asdict(self.task),
            "dataset": asdict(self.dataset),
            "protocol": asdict(self.protocol),
            "metric": asdict(self.metric),
            "policy": asdict(self.policy),
        })


def validate_run_spec_document(document: dict) -> None:
    schema_path = files("sohbench.fabric.schemas").joinpath("run-spec.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError as exc:
        location = ".".join(map(str, exc.absolute_path)) or "run_spec"
        raise ValueError(f"{location}: {exc.message}") from exc
```

Create `src/sohbench/fabric/schemas/__init__.py` and a JSON schema that requires the fields above, limits `run_type` to the three values, requires 64-character lowercase hex digests, and sets `additionalProperties: false` for RunSpec and SpecRef.

- [ ] **Step 4: Run spec tests and existing contract tests**

Run: `python -m pytest tests/fabric/test_specs.py tests/test_contracts.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/specs.py src/sohbench/fabric/schemas tests/fabric/test_specs.py src/sohbench/fabric/__init__.py
git commit -m "feat: define immutable evaluation run specs"
```

### Task 3: DatasetSnapshot And Target-stripped Fold Views

**Files:**
- Create: `src/sohbench/fabric/snapshots.py`
- Create: `src/sohbench/fabric/views.py`
- Test: `tests/fabric/test_snapshots.py`
- Test: `tests/fabric/test_views.py`
- Modify: `src/sohbench/fabric/__init__.py`

**Interfaces:**
- Produces: `frame_digest(frame: pd.DataFrame) -> str`
- Produces: `DatasetSnapshot.from_frame(frame, datasets, label_digest, qc) -> DatasetSnapshot`
- Produces: `FoldViews(train_x, train_y, test_x, test_y)`
- Produces: `make_fold_views(table, fold, target="soh") -> FoldViews`
- Produces: `assert_fold_integrity(table, fold) -> None`

- [ ] **Step 1: Write snapshot and leakage tests**

```python
import numpy as np
import pandas as pd
import pytest

from sohbench.contracts import Fold
from sohbench.fabric.snapshots import DatasetSnapshot, frame_digest
from sohbench.fabric.views import assert_fold_integrity, make_fold_views


def table():
    return pd.DataFrame({
        "dataset": ["d"] * 4,
        "cell_id": ["a", "a", "b", "b"],
        "cycle_index": [0, 1, 0, 1],
        "soh": [1.0, .99, 1.0, .98],
        "voltage_mean": [3.5, 3.4, 3.6, 3.3],
    })


def test_frame_digest_is_stable_after_row_reordering():
    data = table()
    assert frame_digest(data) == frame_digest(data.sample(frac=1, random_state=4))


def test_fold_views_strip_target_from_both_algorithm_views():
    fold = Fold(np.array([0, 1]), np.array([2, 3]), "d/b")
    views = make_fold_views(table(), fold)
    assert "soh" not in views.train_x
    assert "soh" not in views.test_x
    assert views.train_y.tolist() == [1.0, .99]
    assert views.test_y.tolist() == [1.0, .98]


def test_fold_integrity_rejects_asset_overlap():
    fold = Fold(np.array([0]), np.array([1]), "bad")
    with pytest.raises(ValueError, match="asset overlap"):
        assert_fold_integrity(table(), fold)
```

- [ ] **Step 2: Verify both test modules fail**

Run: `python -m pytest tests/fabric/test_snapshots.py tests/fabric/test_views.py -q`

Expected: FAIL with missing module errors.

- [ ] **Step 3: Implement deterministic snapshots and views**

Use this digest rule in `snapshots.py`: sort by present keys from `dataset`, `cell_id`, `cycle_index`; include ordered column names and dtype strings; append `pd.util.hash_pandas_object(..., index=False).values.tobytes()` to SHA-256.

```python
@dataclass(frozen=True)
class DatasetSnapshot:
    snapshot_id: str
    digest: str
    datasets: tuple[str, ...]
    rows: int
    assets: int
    label_digest: str
    qc: dict

    @classmethod
    def from_frame(cls, frame, datasets, label_digest, qc):
        digest = frame_digest(frame)
        return cls(
            snapshot_id=f"snapshot-{digest[:16]}",
            digest=digest,
            datasets=tuple(sorted(map(str, datasets))),
            rows=len(frame),
            assets=frame[["dataset", "cell_id"]].drop_duplicates().shape[0],
            label_digest=str(label_digest),
            qc=dict(qc),
        )
```

```python
# src/sohbench/fabric/views.py
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sohbench.contracts import Fold


@dataclass(frozen=True)
class FoldViews:
    train_x: pd.DataFrame
    train_y: np.ndarray
    test_x: pd.DataFrame
    test_y: np.ndarray


def assert_fold_integrity(table: pd.DataFrame, fold: Fold) -> None:
    train_idx = set(map(int, fold.train_idx))
    test_idx = set(map(int, fold.test_idx))
    if not train_idx or not test_idx:
        raise ValueError("fold contains an empty train or test partition")
    if train_idx & test_idx:
        raise ValueError("fold index overlap")
    train_assets = set(map(tuple, table.iloc[list(train_idx)][["dataset", "cell_id"]].astype(str).to_numpy()))
    test_assets = set(map(tuple, table.iloc[list(test_idx)][["dataset", "cell_id"]].astype(str).to_numpy()))
    if train_assets & test_assets:
        raise ValueError("fold asset overlap")


def make_fold_views(table: pd.DataFrame, fold: Fold, target: str = "soh") -> FoldViews:
    assert_fold_integrity(table, fold)
    train = table.iloc[fold.train_idx].copy()
    test = table.iloc[fold.test_idx].copy()
    return FoldViews(
        train_x=train.drop(columns=[target]),
        train_y=train[target].to_numpy(copy=True),
        test_x=test.drop(columns=[target]),
        test_y=test[target].to_numpy(copy=True),
    )
```

- [ ] **Step 4: Run focused tests and data/protocol regression tests**

Run: `python -m pytest tests/fabric/test_snapshots.py tests/fabric/test_views.py tests/test_data.py tests/test_protocols.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/snapshots.py src/sohbench/fabric/views.py src/sohbench/fabric/__init__.py tests/fabric/test_snapshots.py tests/fabric/test_views.py
git commit -m "feat: add immutable dataset snapshots and safe fold views"
```

### Task 4: Make BenchmarkRunner Target-safe And Seed-aware

**Files:**
- Modify: `src/sohbench/runner.py`
- Modify: `tests/test_runner.py`
- Create: `tests/fabric/test_runner_leakage.py`

**Interfaces:**
- Consumes: `make_fold_views`
- Produces: `_instantiate_estimator(factory, seed)`
- Extends: `BenchmarkRunner.run(..., on_predictions=None)`
- `on_predictions` receives `record`, `sample_frame`, `y_pred`, `interval_low`, `interval_high`

- [ ] **Step 1: Add failing target and seed tests**

```python
class SpyEstimator:
    name = "spy"
    feature_mode = "unified"
    backend = "external"
    seen_random_states = []

    def __init__(self, random_state=0):
        self.random_state = random_state
        self.seen_random_states.append(random_state)

    def fit(self, X, y, meta=None):
        assert "soh" not in X.columns
        self.mean = float(np.mean(y))
        return self

    def predict(self, X, meta=None):
        assert "soh" not in X.columns
        return np.full(len(X), self.mean)

    def predict_interval(self, X, meta=None):
        center = self.predict(X, meta)
        return center, center


def test_runner_strips_target_and_propagates_seed(simple_table, simple_meta):
    SpyEstimator.seen_random_states.clear()
    BenchmarkRunner(seeds=[3, 9]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
    )
    assert set(SpyEstimator.seen_random_states) == {3, 9}


def test_prediction_callback_has_stable_sample_ids_but_algorithm_never_sees_truth(simple_table, simple_meta):
    captured = []
    BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), simple_table, simple_meta,
        {"loco": LeaveOneCellOut()}, [SpyEstimator],
        on_predictions=lambda **value: captured.append(value),
    )
    assert captured
    assert {"dataset", "cell_id", "cycle_index", "soh"} <= set(captured[0]["sample_frame"].columns)
    assert len(captured[0]["y_pred"]) == len(captured[0]["sample_frame"])
```

Define `simple_table` and `simple_meta` fixtures locally using three cells and five cycles per cell.

- [ ] **Step 2: Run the two new tests and observe failure**

Run: `python -m pytest tests/fabric/test_runner_leakage.py -q`

Expected: FAIL because the current runner passes `soh` into `fit/predict` and has no callback.

- [ ] **Step 3: Refactor runner around FoldViews**

Add `_instantiate_estimator` using constructor signature inspection and `copy.deepcopy` for instance factories:

```python
def _instantiate_estimator(factory, seed):
    if isinstance(factory, type):
        params = inspect.signature(factory).parameters
        return factory(random_state=seed) if "random_state" in params else factory()
    estimator = copy.deepcopy(factory)
    setter = getattr(estimator, "set_seed", None)
    if callable(setter):
        setter(seed)
    elif hasattr(estimator, "random_state"):
        estimator.random_state = seed
    return estimator
```

Within every fold call `make_fold_views(data_table, fold, task.target)`. Pass `views.train_x` and `views.test_x` to estimators, and `views.train_y`/`views.test_y` only to metric/control code. Build `sample_frame` from the original test rows with `dataset`, `cell_id`, `cycle_index`, and the target. Call `on_predictions` after intervals are available.

- [ ] **Step 4: Run runner, estimator and conformal regressions**

Run: `python -m pytest tests/fabric/test_runner_leakage.py tests/test_runner.py tests/test_baselines.py tests/test_byo.py tests/test_conformal.py -q`

Expected: all tests PASS. Update legacy tests only where they asserted that estimator input contained `soh`; do not weaken the new target-stripping assertions.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/runner.py tests/test_runner.py tests/fabric/test_runner_leakage.py
git commit -m "fix: isolate targets and propagate evaluation seeds"
```

### Task 5: Move Feature Transforms Inside Fold Boundaries

**Files:**
- Create: `src/sohbench/fabric/feature_pipeline.py`
- Modify: `src/sohbench/trusted_plugins.py`
- Modify: `src/sohbench/runner.py`
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `src/sohbench/battery_soh_plugins.py`
- Test: `tests/fabric/test_feature_pipeline.py`
- Modify: `tests/test_trusted_plugins.py`
- Modify: `tests/test_battery_soh_plugins.py`

**Interfaces:**
- Produces: `FeatureState(payload: dict, digest: str)`
- Produces: `FeaturePipeline.fit(train_x, train_y, meta) -> FeatureState`
- Produces: `FeaturePipeline.transform(frame, state, meta) -> pd.DataFrame`
- Produces: `assert_prefix_invariant(pipeline, frame, meta) -> None`
- Extends: `TrustedPluginStore.feature_pipeline(name) -> FeaturePipeline | None`
- Extends: `BenchmarkRunner.run(..., feature_pipelines=None)`

- [ ] **Step 1: Write failing fold-scope and causality tests**

```python
def test_pipeline_fit_never_receives_test_rows():
    seen = {}

    class MeanPipeline:
        def fit(self, train_x, train_y, meta):
            seen["cycles"] = train_x["cycle_index"].tolist()
            return FeatureState.create({"mean": float(train_y.mean())})

        def transform(self, frame, state, meta):
            out = frame.copy()
            out["train_mean_feature"] = state.payload["mean"]
            return out

    # Use a TinyProtocol with train cycles [0,1,2] and test cycles [3,4].
    # Run the benchmark with feature_pipelines={"mean": MeanPipeline()}.
    assert seen["cycles"] == [0, 1, 2]


def test_prefix_invariance_rejects_cycle_fraction_transform():
    class FuturePipeline:
        def fit(self, train_x, train_y, meta):
            return FeatureState.create({})
        def transform(self, frame, state, meta):
            out = frame.copy()
            out["fraction"] = out["cycle_index"] / out["cycle_index"].max()
            return out

    with pytest.raises(ValueError, match="prefix invariant"):
        assert_prefix_invariant(FuturePipeline(), trajectory_table(), pd.DataFrame())
```

- [ ] **Step 2: Verify tests fail against current global transform behavior**

Run: `python -m pytest tests/fabric/test_feature_pipeline.py tests/test_battery_soh_plugins.py -q`

Expected: FAIL; current service invokes feature transforms before protocol split and battery plugin uses full-trajectory maximum.

- [ ] **Step 3: Implement fold-scoped feature contracts**

```python
@dataclass(frozen=True)
class FeatureState:
    payload: dict
    digest: str

    @classmethod
    def create(cls, payload: dict):
        clean = dict(payload)
        return cls(clean, sha256_json(clean))


class IdentityFeaturePipeline:
    def fit(self, train_x, train_y, meta):
        return FeatureState.create({"kind": "identity"})
    def transform(self, frame, state, meta):
        return frame.copy()
```

For legacy transform-only plugins, implement `LegacyStatelessPipeline` that calls the transform on a copy without target columns and uses a state containing plugin digest. Add `assert_prefix_invariant` that transforms the full trajectory and a prefix, compares common output feature columns for the prefix rows, and raises when values differ beyond `rtol=1e-9, atol=1e-12`.

Change runner loop order to protocol -> fold -> featureset -> seed. Fit the selected pipeline only on `train_x`, transform train and test separately, then call the estimator. Remove `self.store.apply_feature_transforms(...)` from both service run methods and pass `feature_pipelines={name: self.store.feature_pipeline(name) ...}` to the runner.

In `build_battery_soh_featureset_plugin_source`, remove `battery_soh_cycle_fraction` and `battery_soh_resistance_growth` from candidates and remove their full-trajectory calculations. Keep capacity/time/voltage/current/temperature features that use row-local or prefix-invariant inputs.

- [ ] **Step 4: Run leakage, plugin and service tests**

Run: `python -m pytest tests/fabric/test_feature_pipeline.py tests/fabric/test_runner_leakage.py tests/test_trusted_plugins.py tests/test_battery_soh_plugins.py tests/test_evaluation_service.py -q`

Expected: all tests PASS; no service call applies feature transforms to the unsplit table.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/feature_pipeline.py src/sohbench/trusted_plugins.py src/sohbench/runner.py src/sohbench/evaluation_service.py src/sohbench/battery_soh_plugins.py tests/fabric/test_feature_pipeline.py tests/test_trusted_plugins.py tests/test_battery_soh_plugins.py
git commit -m "fix: fit feature pipelines inside evaluation folds"
```

### Task 6: Structured Failures And RunOutcome

**Files:**
- Create: `src/sohbench/fabric/failures.py`
- Create: `src/sohbench/fabric/outcomes.py`
- Test: `tests/fabric/test_outcomes.py`
- Modify: `src/sohbench/runner.py`
- Modify: `src/sohbench/report.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Produces: `FailureCategory` enum
- Produces: `classify_exception(exc) -> FailureCategory`
- Produces: `RunOutcome.from_results(results) -> RunOutcome`
- Runner result columns: `unit_status`, `error_category`, `error_type`, `error_message`

- [ ] **Step 1: Write failing all-failed and partial tests**

```python
def result_rows(statuses):
    rows = []
    for index, status in enumerate(statuses):
        rows.append({
            "estimator": "demo", "featureset": "full", "protocol": "loco",
            "held_out_key": f"cell-{index}", "fold": index, "seed": 0,
            "metric": "mae", "value": 0.1 if status == "succeeded" else np.nan,
            "unit_status": status,
        })
    return pd.DataFrame(rows)


def test_all_failed_is_failed():
    assert RunOutcome.from_results(result_rows(["failed", "failed"])).outcome == "failed"


def test_mixed_units_are_partial_and_keep_attempted_denominator():
    outcome = RunOutcome.from_results(result_rows(["succeeded", "failed"]))
    assert outcome.outcome == "partial"
    assert (outcome.attempted, outcome.succeeded, outcome.failed) == (2, 1, 1)
```

Add a report test asserting leaderboard entries contain `coverage` and `execution` counts and that an all-NaN group has no normal verdict.

- [ ] **Step 2: Run tests and observe missing outcome semantics**

Run: `python -m pytest tests/fabric/test_outcomes.py tests/test_report.py -q`

Expected: FAIL because `RunOutcome` and coverage fields do not exist.

- [ ] **Step 3: Implement categories, outcomes and result columns**

`FailureCategory` values must exactly match `INPUT_CONTRACT`, `DATA_QC`, `LEAKAGE`, `ALGORITHM`, `RESOURCE`, `INFRASTRUCTURE`, `METRIC`, and `POLICY`. Default unknown estimator exceptions to `ALGORITHM`.

```python
@dataclass(frozen=True)
class RunOutcome:
    outcome: str
    attempted: int
    succeeded: int
    failed: int

    @classmethod
    def from_results(cls, results):
        units = results.drop_duplicates(["estimator", "featureset", "protocol", "held_out_key", "fold", "seed"])
        attempted = len(units)
        succeeded = int((units["unit_status"] == "succeeded").sum())
        failed = attempted - succeeded
        outcome = "complete" if attempted and failed == 0 else "partial" if succeeded else "failed"
        return cls(outcome, attempted, succeeded, failed)
```

On successful runner records set empty error fields and `unit_status="succeeded"`. On exception, classify and record type/message with `unit_status="failed"`. Update report aggregation to compute counts from unique execution-unit keys, expose `coverage=succeeded/attempted`, and return `verdicts={}` when succeeded is zero.

- [ ] **Step 4: Run focused and regression tests**

Run: `python -m pytest tests/fabric/test_outcomes.py tests/test_runner.py tests/test_report.py tests/test_acceptance.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/failures.py src/sohbench/fabric/outcomes.py src/sohbench/runner.py src/sohbench/report.py tests/fabric/test_outcomes.py tests/test_runner.py tests/test_report.py
git commit -m "feat: make failures and partial outcomes explicit"
```

### Task 7: Compute Trajectory Metrics Per Asset

**Files:**
- Modify: `src/sohbench/runner.py`
- Modify: `tests/test_runner.py`
- Test: `tests/fabric/test_trajectory_metrics.py`

**Interfaces:**
- Produces: `_trajectory_metrics(task, test_x, y_pred) -> dict[str, float]`
- Groups by: `dataset`, `cell_id`

- [ ] **Step 1: Write a failing cross-cell ordering test**

```python
def test_trajectory_metrics_do_not_join_independent_cells():
    test_x = pd.DataFrame({
        "dataset": ["d"] * 4,
        "cell_id": ["a", "a", "b", "b"],
        "cycle_index": [0, 1, 0, 1],
    })
    predictions = np.array([1.0, .9, 1.0, .9])
    values = BenchmarkRunner(seeds=[0])._trajectory_metrics(SOHEstimationTask(), test_x, predictions)
    assert values["monotonicity_violation_rate"] == 0.0
```

The current global cycle sort produces the sequence `[1.0, 1.0, .9, .9]` or another cross-cell ordering and can hide or create violations depending on row order; add a second shuffled-row assertion to prove invariance.

- [ ] **Step 2: Run and verify the shuffled-row case fails**

Run: `python -m pytest tests/fabric/test_trajectory_metrics.py -q`

Expected: at least one assertion FAILS under current global sorting.

- [ ] **Step 3: Implement per-asset metric calculation**

Create a temporary frame with predictions, group by present `dataset/cell_id`, sort each group by `cycle_index`, calculate every trajectory metric per group, and return the unweighted mean across assets. Raise a metric failure if no asset contains two ordered points.

```python
def _trajectory_metrics(self, task, test_x, y_pred):
    frame = test_x[["dataset", "cell_id", "cycle_index"]].copy()
    frame["prediction"] = np.asarray(y_pred, dtype=float)
    per_asset = []
    for _, asset in frame.groupby(["dataset", "cell_id"], sort=True):
        asset = asset.sort_values("cycle_index", kind="stable")
        if len(asset) < 2:
            continue
        per_asset.append(task.trajectory_metrics(
            asset["prediction"].to_numpy(),
            asset["cycle_index"].to_numpy(),
        ))
    if not per_asset:
        raise ValueError("trajectory metrics require an asset with at least two cycles")
    return {key: float(np.mean([row[key] for row in per_asset])) for key in per_asset[0]}
```

- [ ] **Step 4: Run runner and metric tests**

Run: `python -m pytest tests/fabric/test_trajectory_metrics.py tests/test_runner.py tests/test_metrics.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/runner.py tests/test_runner.py tests/fabric/test_trajectory_metrics.py
git commit -m "fix: aggregate trajectory metrics per asset"
```

### Task 8: EvidencePack Dual-write And Unique Run Directories

**Files:**
- Create: `src/sohbench/fabric/evidence.py`
- Create: `src/sohbench/fabric/schemas/evidence-pack.schema.json`
- Create: `src/sohbench/fabric/schemas/prediction-record.schema.json`
- Create: `src/sohbench/fabric/schemas/metric-record.schema.json`
- Test: `tests/fabric/test_evidence.py`
- Modify: `src/sohbench/report.py`
- Modify: `tests/test_report.py`

**Interfaces:**
- Produces: `write_evidence_pack(run_dir, manifest, artifact_paths) -> Path`
- Produces: `verify_evidence_pack(path) -> dict`
- Produces: `validate_prediction_records(frame) -> None`
- Produces: `validate_metric_records(frame) -> None`
- Extends: `write_artifacts(..., run_id=None, evidence_context=None, predictions=None)`

- [ ] **Step 1: Write failing evidence integrity tests**

```python
def test_evidence_pack_verifies_and_detects_tamper(tmp_path):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    result = run_dir / "results.json"
    result.write_text('{"ok":true}', encoding="utf-8")
    path = write_evidence_pack(run_dir, {"run_id": "RUN-one"}, [result])
    assert verify_evidence_pack(path)["ok"] is True
    result.write_text('{"ok":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_evidence_pack(path)


def test_write_artifacts_uses_run_id_not_second_timestamp(tmp_path):
    first = Path(write_artifacts(results(), None, task(), {}, tmp_path, run_id="RUN-a"))
    second = Path(write_artifacts(results(), None, task(), {}, tmp_path, run_id="RUN-b"))
    assert first != second
    assert first.name == "RUN-a"
    assert second.name == "RUN-b"


def test_prediction_records_require_stable_sample_and_execution_keys():
    records = prediction_frame().drop(columns=["sample_id"])
    with pytest.raises(ValueError, match="sample_id"):
        validate_prediction_records(records)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/fabric/test_evidence.py tests/test_report.py -q`

Expected: FAIL because evidence functions and `run_id` parameter are absent.

- [ ] **Step 3: Implement atomic evidence writing**

`write_evidence_pack` must build an artifact list of relative path, byte size and SHA-256, set `schema_version="1.0.0"`, write `evidence-pack.json.tmp`, fsync the file, then `Path.replace()` it to `evidence-pack.json`. `verify_evidence_pack` validates the JSON schema and every referenced artifact digest. PredictionRecord requires run/candidate/feature/protocol/fold/seed, dataset/asset/cycle/sample keys, prediction, optional truth/interval and abstention status. MetricRecord requires metric/version/value/unit/direction, aggregation scope and the contributing sample/asset refs. Validators reject duplicate execution/sample keys and non-finite required values.

```python
def write_evidence_pack(run_dir: Path, manifest: dict, artifact_paths: list[Path]) -> Path:
    artifacts = [{
        "path": path.relative_to(run_dir).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in sorted(artifact_paths, key=lambda item: item.relative_to(run_dir).as_posix())]
    document = {"schema_version": "1.0.0", **manifest, "artifacts": artifacts}
    validate_evidence_pack_document(document)
    target = run_dir / "evidence-pack.json"
    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    return target


def validate_prediction_records(frame: pd.DataFrame) -> None:
    required = ["run_id", "candidate_id", "featureset", "protocol", "fold", "seed",
                "dataset", "cell_id", "cycle_index", "sample_id", "y_pred"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"missing prediction fields: {', '.join(missing)}")
    key = required[:-1]
    if frame.duplicated(key).any():
        raise ValueError("duplicate prediction execution/sample key")
```

Change `write_artifacts` and `write_inference_artifacts` to use `run_id or new_id("RUN")`. Refuse to write when the target directory already contains files. Preserve all existing artifact filenames, add optional `predictions.parquet`, and write EvidencePack after every artifact is closed.

- [ ] **Step 4: Run evidence, report and standalone packaging tests**

Run: `python -m pytest tests/fabric/test_evidence.py tests/test_report.py tests/test_standalone_frontend.py -q`

Expected: all tests PASS and old JSON files still exist.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/evidence.py src/sohbench/fabric/schemas/evidence-pack.schema.json src/sohbench/fabric/schemas/prediction-record.schema.json src/sohbench/fabric/schemas/metric-record.schema.json src/sohbench/report.py tests/fabric/test_evidence.py tests/test_report.py
git commit -m "feat: dual-write verifiable evidence packs"
```

### Task 9: Asset Bootstrap Statistics And Comparable Leaderboards

**Files:**
- Create: `src/sohbench/fabric/statistics.py`
- Test: `tests/fabric/test_statistics.py`
- Modify: `src/sohbench/report.py`
- Modify: `src/sohbench/schemas/leaderboard.schema.json`
- Modify: `tests/test_report.py`

**Interfaces:**
- Produces: `asset_metric_table(predictions, metric="mae") -> pd.DataFrame`
- Produces: `bootstrap_mean_ci(values, confidence=.95, n_resamples=2000, seed=0) -> dict`
- Produces: `paired_delta(left, right, key_columns, value_column) -> dict`
- Leaderboard entries gain: `comparison_key`, `coverage`, `execution`, `confidence_intervals`

- [ ] **Step 1: Write failing statistics tests**

```python
def test_asset_bootstrap_uses_assets_not_rows():
    predictions = pd.DataFrame({
        "dataset": ["d"] * 6,
        "cell_id": ["a", "a", "a", "b", "b", "b"],
        "y_true": [1, 1, 1, 1, 1, 1],
        "y_pred": [1, 1, 1, 0, 0, 0],
    })
    assets = asset_metric_table(predictions, "mae")
    assert assets["value"].tolist() == [0.0, 1.0]


def test_bootstrap_is_reproducible():
    first = bootstrap_mean_ci(np.array([0.0, .1, .2, .3]), n_resamples=500, seed=7)
    second = bootstrap_mean_ci(np.array([0.0, .1, .2, .3]), n_resamples=500, seed=7)
    assert first == second
    assert first["low"] <= first["mean"] <= first["high"]
```

- [ ] **Step 2: Run tests and verify missing module failure**

Run: `python -m pytest tests/fabric/test_statistics.py -q`

Expected: FAIL because statistics module does not exist.

- [ ] **Step 3: Implement deterministic asset-level statistics**

Use `numpy.random.default_rng(seed).choice(values, size=(n_resamples, len(values)), replace=True)` and percentile bounds. `asset_metric_table` groups by `dataset/cell_id`, calculates MAE or RMSE per asset, and rejects empty/non-finite groups. `paired_delta` inner-joins identical fold/seed/held-out keys and returns mean delta, CI and paired count.

```python
def bootstrap_mean_ci(values, confidence=.95, n_resamples=2000, seed=0):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be non-empty and finite")
    draws = np.random.default_rng(seed).choice(
        values, size=(n_resamples, values.size), replace=True
    ).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(draws, alpha)),
        "high": float(np.quantile(draws, 1.0 - alpha)),
        "confidence": confidence,
        "n_assets": int(values.size),
    }


def asset_metric_table(predictions, metric="mae"):
    def score(group):
        error = group["y_pred"].to_numpy() - group["y_true"].to_numpy()
        value = np.mean(np.abs(error)) if metric == "mae" else np.sqrt(np.mean(error ** 2))
        return pd.Series({"value": float(value)})
    return predictions.groupby(["dataset", "cell_id"], sort=True).apply(score).reset_index()
```

Update report to write `statistics.json`, place CI and execution counts on leaderboard entries, and include the RunSpec comparison key from `evidence_context`. Keep metric mean/std for backward compatibility.

- [ ] **Step 4: Validate statistics and leaderboard schema**

Run: `python -m pytest tests/fabric/test_statistics.py tests/test_report.py -q`

Expected: all tests PASS; `jsonschema.validate` accepts the extended leaderboard.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/fabric/statistics.py src/sohbench/report.py src/sohbench/schemas/leaderboard.schema.json tests/fabric/test_statistics.py tests/test_report.py
git commit -m "feat: add asset-level confidence intervals"
```

### Task 10: Integrate RunSpec And Evidence Into EvaluationService

**Files:**
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `tests/test_evaluation_service.py`
- Modify: `tests/test_api_server.py`
- Create: `tests/fabric/test_phase0_golden.py`
- Create: `tests/fixtures/fabric/expected-evidence-summary.json`

**Interfaces:**
- Consumes: RunSpec, DatasetSnapshot, FoldViews, RunOutcome, EvidencePack
- Produces: job fields `run_id`, `execution_state`, `outcome`, `verdict`, `evidence_path`
- Preserves: legacy `status` field mapped from new fields

- [ ] **Step 1: Write failing service and golden tests**

Add a service test that starts a one-dataset, one-estimator, one-seed inline job and asserts:

```python
assert job["run_id"].startswith("RUN-")
assert job["execution_state"] == "REVIEW"
assert job["outcome"] in {"complete", "partial", "failed"}
assert Path(job["evidence_path"]).name == "evidence-pack.json"
assert verify_evidence_pack(job["evidence_path"])["ok"] is True
```

Add an all-failing estimator test asserting `status="failed"`, `outcome="failed"`, and no normal leaderboard verdict. The golden test compares stable fields only: schema versions, comparison key, counts, artifact names and failure categories. Exclude UUIDs, paths and timestamps.

- [ ] **Step 2: Run the new tests and verify missing fields**

Run: `python -m pytest tests/fabric/test_phase0_golden.py tests/test_evaluation_service.py -q`

Expected: FAIL because jobs lack new contract/evidence fields.

- [ ] **Step 3: Build and attach compatibility RunSpec**

In `_normalize_request`, retain current arrays and add an internal builder that creates SpecRefs from canonical request fragments and DatasetSnapshot after load. Collect prediction callback records into a DataFrame with stable `sample_id`. Pass `run_id`, `evidence_context`, and predictions to `write_artifacts`.

Map state after report creation:

```python
outcome = RunOutcome.from_results(results)
job.update({
    "run_id": run_id,
    "execution_state": "REVIEW" if outcome.outcome != "failed" else "FAILED",
    "outcome": outcome.outcome,
    "verdict": "not_reviewed" if outcome.outcome != "failed" else "fail",
    "status": "succeeded" if outcome.outcome == "complete" else "partial" if outcome.outcome == "partial" else "failed",
    "evidence_path": str(raw_run_dir / "evidence-pack.json"),
})
```

Apply equivalent fields to inference jobs. Keep API response shape backward compatible and expose the new fields unchanged.

- [ ] **Step 4: Run Phase 0 verification**

Run:

```powershell
python -m pytest tests/fabric tests/test_runner.py tests/test_report.py tests/test_evaluation_service.py tests/test_api_server.py tests/test_model_store.py tests/test_standalone_frontend.py -q
```

Expected: all tests PASS. Then run `python -m pytest -q`; expected zero failures.

- [ ] **Step 5: Commit**

```powershell
git add src/sohbench/evaluation_service.py tests/test_evaluation_service.py tests/test_api_server.py tests/fabric/test_phase0_golden.py tests/fixtures/fabric/expected-evidence-summary.json
git commit -m "feat: integrate evidence contracts with legacy evaluation"
```

## Phase 0 Review Gate

Before starting the next sub-plan, a reviewer must verify:

- [ ] Search confirms `est.predict(te` and `est.fit(tr` no longer exist in `runner.py`.
- [ ] Search confirms `apply_feature_transforms(table` is absent from service training and inference paths.
- [ ] A malicious estimator cannot read `soh` from its X input in unit tests.
- [ ] All-failed and partial matrices produce different outcomes.
- [ ] Two runs created in the same second use different directories.
- [ ] Tampering with any referenced artifact fails EvidencePack verification.
- [ ] Existing frontend JSON and standalone HTML tests remain green.
