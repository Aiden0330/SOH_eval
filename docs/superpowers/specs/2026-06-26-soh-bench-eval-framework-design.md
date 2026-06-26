# SOH Benchmark Framework (`soh_bench`) — Design Spec

> **Date**: 2026-06-26
> **Status**: Approved design (brainstorming complete) — ready for implementation plan
> **Scope**: This spec covers ONLY the benchmark/evaluation framework — the first subsystem of the
> larger SOH project (estimation + adaptation + frontend). Algorithm layers (L1–L6 of the layered
> design) and the frontend tool are separate specs.
> **Parent design**: [`SOH泛化性增强分层方案.md`](../../../SOH泛化性增强分层方案.md)

---

## 0. Purpose & Context

The benchmark framework is the **measurement foundation** of the project. The parent layered-design
doc's core thesis is that SOH generalization is won at the feature / domain-adaptation / **validation-
protocol** layers — and explicitly warns: *"if the validation protocol is wrong, none of the upstream
effort can be correctly measured; never use random K-fold CV — cycles of the same cell are highly
correlated and random splits leak."*

Therefore the first deliverable is a framework that can honestly measure **cross-domain** generalization
of whole SOH estimator pipelines across heterogeneous working-condition datasets.

This is a **complete rewrite** (new project `D:\SOH_Opus`). The legacy repo `D:\battery_SOH` and its
`soh_core/evaluation/{harness,metrics,splits}.py` are **reference only** — not migrated. Legacy data
**artifacts** under `D:\battery_SOH\data` ARE reused (data, not code).

## 1. Locked Decisions (from brainstorming)

| # | Decision | Value |
|---|---|---|
| Benchmark subject | Whole estimator pipeline | `fit` / `predict` contract |
| Input contract | Feature table by default, raw optional | Default = unified 17-feature per-cycle table |
| Validation protocols (v1) | Leave-one-domain | Leave-one-**cell**-out + Leave-one-**chemistry**-out (station / forward-chain designed-for, not built) |
| Metrics | Point estimate required, interval optional | RMSE / MAE / EOL-region error / max error + conditional PICP |
| Run output | Structured results + leaderboard | per-fold/metric table (parquet/csv) + summary leaderboard (markdown/json) |
| Architecture | Thin protocol + registry (Approach A) | Interfaces shaped so a config-driven matrix runner (Approach B) is an additive layer, not a rewrite |

## 2. Module Boundaries

Package `soh_bench` under `src/`. Single-responsibility, independently testable modules. Dependency
direction is one-way and acyclic.

```
src/soh_bench/
  contracts.py    # Estimator Protocol + Fold / Result dataclasses (definitions only, no logic)
  registry.py     # decorator registry: estimators & datasets registered/retrieved by name
  metadata.py     # dataset -> chemistry mapping table (curated per dataset, NOT guessed)
  data.py         # DatasetLoader: reads processed/*.pkl feature tables, joins chemistry, raw-optional hook
  protocols.py    # LeaveOneCellOut / LeaveOneChemistryOut -> Fold iterators
  metrics.py      # rmse / mae / max_error / eol_region_error / picp
  runner.py       # BenchmarkRunner: expands (datasets x protocols x estimators) -> ResultSet
  report.py       # ResultSet -> leaderboard markdown/json + detail parquet/csv
  estimators/     # v1 built-in baselines (Ridge, etc.) — minimal, also serve as test fixtures
    __init__.py
    baselines.py
  cli.py          # python -m soh_bench run --config ...
```

**Dependency graph (acyclic):**
`contracts <- {registry, data, protocols, metrics} <- runner <- {report, cli}`
`data <- metadata`

The four leaf modules `metrics / protocols / registry / data` are mutually independent → suitable for
parallel subagent implementation during the build phase.

## 3. Data Flow

```
DatasetLoader(data_dir = D:\battery_SOH\data)        # path configurable
  reads processed/{name}_features.pkl
    -> unified long table: [dataset, cell_id, cycle_index, soh, <17 capacity-side features>]
  joins metadata chemistry label
    -> adds column chemistry in {LFP, NMC/NCA, LCO, ...}
        |
        v
Protocol.generate(table) -> Fold(train_idx, test_idx, held_out_key)
  LeaveOneCellOut:      group by cell_id,  hold out one cell per fold
  LeaveOneChemistryOut: group by chemistry, hold out one chemistry per fold
        |
        v
BenchmarkRunner: for each (estimator x protocol x fold)
  est.fit(train_df, y_train)  ->  est.predict(test_df)  [+ predict_interval if available]
  metrics.compute(y_true, y_pred, intervals?) -> record per-fold per-metric
        |
        v
ResultSet  (tidy long table: estimator, protocol, held_out_key, metric, value)
        |
        +-- report -> leaderboard (aggregate mean +/- std per estimator x protocol) markdown/json
        +-- detail table parquet/csv written to runs/<timestamp>/
```

### Input data contract

The unified per-cycle feature table (verified present in `data/processed/*.pkl` for NASA / CALCE CS2 /
HUST / Oxford) has columns:
`dataset, cell_id, cycle_index, soh, time_s_max, voltage_{mean,std,min,max}, current_{mean,std},
abs_current_mean, temperature_mean, charge_capacity_max, q_win_ah, t_win_s, ic_peak_ah_per_v,
ic_peak_voltage_v, cycle_index_feature, cycle_fraction`.

ID/label columns: `dataset, cell_id, cycle_index, soh`. Everything else = feature columns (exposed via
`feature_columns()` helper). Column set varies slightly across datasets (e.g., NASA has extra
`ambient_temp_c`); the loader normalizes to the **intersection** of feature columns present across the
datasets selected for a given run, and records which columns were dropped.

### Chemistry labels (curated, not guessed)

`metadata.py` holds a `DATASET_CHEMISTRY` map. Values MUST be verified against each dataset's source
documentation before being trusted — chemistry-out generalization claims depend on them. Initial entries
are written with a `# VERIFY` marker and confirmed during implementation, not asserted from memory.
Datasets in scope to label: NASA, CALCE CS2, HUST, Oxford (extensible).

### Raw-optional channel

Estimators set `requires_raw = True` to opt into a raw per-cycle channel (voltage/current/temperature
curves). v1 ships the **hook and contract** only; the raw loader path is stubbed to raise a clear
`NotImplementedError` until L1 feature benchmarking is in scope. Default estimators use the feature table.

## 4. Estimator Contract

```python
@runtime_checkable
class Estimator(Protocol):
    name: str
    requires_raw: bool                       # default False

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> None: ...
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...           # required, point estimate

    # optional capability (detected via hasattr):
    def predict_interval(
        self, df: pd.DataFrame, alpha: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]: ...                          # (lower, upper)
```

- The framework passes the **whole feature table** (including id columns) to `fit`/`predict`; the
  estimator selects its own columns via the `feature_columns()` helper. This keeps the door open for
  pipelines that need metadata (e.g., working-condition gating / clustering) without changing the contract.
- `predict_interval` is detected with `hasattr`. Present → PICP computed; absent → PICP recorded as `NA`.
- Registration via decorators: `@register_estimator` / `@register_dataset`. Duplicate name → error.

## 5. Metrics

| Metric | Definition | Notes |
|---|---|---|
| `rmse` | root mean squared error | point |
| `mae` | mean absolute error | point |
| `max_error` | worst-case absolute error | safety-relevant |
| `eol_region_error` | RMSE restricted to samples with SOH <= `eol_threshold` (default 0.8) | end-of-life region, safety-critical; threshold configurable |
| `picp` | Prediction Interval Coverage Probability | only for estimators exposing `predict_interval`; else `NA` |

Each metric is a pure function `(y_true, y_pred[, intervals]) -> float`, unit-testable in isolation.

## 6. Error Handling

| Situation | Handling |
|---|---|
| estimator `fit`/`predict` raises inside a fold | record failed fold (metrics = NaN + error summary), **continue remaining folds**, list failures in report |
| protocol yields < 2 groups (e.g., only 1 chemistry present) | skip that protocol with a warning, never silently |
| PICP requested but estimator lacks `predict_interval` | metric = `NA`, annotated in report |
| feature table schema mismatch / NaN / inf | loader **structural validation fails fast** (eval_harness Gate 1) with a clear message |
| missing data path / dataset file | clear load-time error naming the missing dataset |

### Alignment with eval_harness "three gates"

- **Gate 1 (structural contract)**: loader schema validation + ResultSet shape.
- **Gate 2 (numerical sanity)**: post-run checks on metric ranges and cross-domain degradation direction
  (mount point implemented v1; specific checks extensible).
- **Gate 3 (acceptance thresholds)**: EOL-region error and PICP thresholds — values via config; v1
  implements the mounting mechanism, thresholds are configurable inputs.

## 7. Testing Strategy (TDD)

Tests written first, per module. Critical tests encode the parent design's hard rules directly.

- **metrics**: known-array assertions for rmse/mae/max; `eol_region_error` counts only SOH <= threshold
  samples; `picp` coverage computed correctly.
- **protocols (leakage gate)**: synthetic tiny dataset → assert **no cell appears in both train and test**
  (directly guards the "never leak" red line); chemistry-out asserts an entire chemistry is held out.
- **registry**: register/retrieve; duplicate-name raises.
- **runner**: synthetic data + dummy estimator → end-to-end ResultSet shape; inject a raising estimator
  → failed fold is isolated, run completes.
- **data loader**: small fixture pkl → loads and passes schema validation; malformed fixture → fails.
- **report**: fixed ResultSet → deterministic, reproducible leaderboard output.

## 8. Result Artifacts

A run writes to `runs/<timestamp>/`:
- `results.parquet` (+ `results.csv`) — tidy long detail table.
- `leaderboard.md` and `leaderboard.json` — aggregated mean ± std per estimator × protocol, plus a
  failed-fold section.
- `config.json` — the resolved run configuration (datasets, protocols, estimators, seed, thresholds) for
  reproducibility.

## 9. Out of Scope (v1)

- Leave-one-station-out and forward-chaining time splits (designed-for via the `Protocol` abstraction;
  not implemented).
- Raw per-cycle loading / L1 feature extraction benchmarking (hook + contract only).
- Config-driven YAML matrix runner with caching/resume (interfaces shaped for it; the additive layer is
  a later spec).
- The actual L1–L6 algorithm implementations (separate spec) — v1 ships only minimal baseline estimators
  needed to exercise and test the framework.
- The frontend tool and its three mockups (separate spec; will consume `results.parquet` / `leaderboard.json`).

## 10. Build Decomposition (for the implementation plan)

Leaf modules buildable in parallel: `contracts`, `metrics`, `protocols`, `registry`, `metadata`.
Then `data` (depends on metadata/contracts), then `runner` (depends on all leaves + contracts), then
`report` and `cli`. Each module ships with its tests (TDD red→green) before integration.
