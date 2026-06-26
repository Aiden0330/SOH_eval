# SOH Benchmark Framework (`soh_bench`) — Design Spec (v2)

> **Date**: 2026-06-26
> **Status**: v2 — revised after base-contract re-evaluation; ready for implementation plan
> **Supersedes**: v1 (brainstorming-approved). Changes driven by
> [`2026-06-26-soh-reeval-and-redesign.md`](2026-06-26-soh-reeval-and-redesign.md) (routed via
> phm-router → soh-adapt-v2; audited against eval_harness / data_unify / features / retrain / cpp_export).
> **Scope**: ONLY the benchmark/evaluation framework — the first subsystem of the larger SOH project
> (estimation + adaptation + frontend). Algorithm layers (L1–L6) and the frontend are separate specs.
> **Parent design**: [`SOH泛化性增强分层方案.md`](../../../SOH泛化性增强分层方案.md)
> **Base-contract authority**: this framework IS the `phm-eval-harness` base layer (per phm-router skill
> map). Its gate semantics MUST align with soh-adapt 1.0 `eval_harness`, not invent a parallel standard.

---

## 0. Purpose & Context

The benchmark framework is the **measurement foundation** of the project. Core thesis (parent doc):
generalization is won at the feature / domain-adaptation / **validation-protocol** layers — *"if the
validation protocol is wrong, none of the upstream effort can be correctly measured; never use random
K-fold CV — cycles of the same cell are highly correlated and random splits leak."*

The base contract adds a second hard truth the v1 spec missed: **the data itself is not trustworthy
until QC'd.** Measured defects in `data/processed/*.pkl`: NASA `soh ∈ [0.000, 1.222]`, CALCE CS2
`soh ∈ [0.000, 1.018]` (physically impossible — base physical bound is `[0.6, 1.05]`), with 10–13% of
cycles showing >0.5% capacity rises. A benchmark that trusts these labels ranks algorithms on garbage.

Therefore the deliverable is a framework that (a) **QC-gates data before scoring**, (b) honestly measures
**cross-domain** generalization, and (c) applies the base **eval_harness three-gate** model.

This is a **complete rewrite** (`D:\SOH_Opus`). Legacy `soh_core/evaluation/*` is reference only. Legacy
data **artifacts** under `D:\battery_SOH\data` ARE reused (data, not code).

## 1. Locked Decisions

Brainstorming decisions, as revised by the re-evaluation (revisions marked **[v2]**):

| # | Decision | Value |
|---|---|---|
| Benchmark subject | Whole estimator pipeline | `fit` / `predict` contract |
| Input contract | Feature table by default, raw optional | Default = unified per-cycle feature table |
| Validation protocols (v1) | Leave-one-domain | Leave-one-**cell**-out + Leave-one-**dataset**-out; chemistry-out only when ≥2 datasets/chemistry **[v2]** |
| Uncertainty / CI | **Framework guarantees CI: native `predict_interval` or auto split-conformal fallback; report labels `ci_source`** **[v2]** | aligns base red line R7 (CI mandatory for G2 coverage) while keeping `predict_interval` optional at the contract level |
| Metrics | Point + interval + **physical-consistency** **[v2]** | RMSE / MAE / EOL-region / max / PICP / monotonicity-violation-rate / value-range-violation |
| Gate model | **eval_harness canonical three gates** **[v2]** | G1 anti-forgetting (deferred) · G2 generalization/holdout (v1) · G3 physical-consistency (v1) |
| Data QC | **Pre-score QC gate in loader** **[v2]** | value-range / label-source / availability-mask; QC report per run |
| Run output | Structured results + leaderboard + QC report | parquet/csv + markdown/json, with gate verdicts & warnings |
| C++ parity | **Mount `--check-cpp-parity`; Ridge wired through** **[v2]** | golden-vector parity for estimators exposing `export_cpp` |
| Architecture | Thin protocol + registry (Approach A) | shaped so a config-driven matrix runner (Approach B) is additive |

### Decisions taken on the user's behalf (user answered "no preference" on all four)

1. CI → **synthesis option** (framework guarantees a CI for every estimator: native intervals if
   provided, else auto split-conformal; all estimators are G2-eligible; the report labels CI provenance
   `ci_source ∈ {native, conformal}` so conformal-derived coverage is never silently equated with native).
2. v1 scope → **include all three** (physical-consistency gate + data/label QC gate + chemistry
   separability check) — the minimum to keep the numbers trustworthy.
3. G1 anti-forgetting → **deferred** (no defined "home scenario" yet); keep `forgetting_delta` metric and
   a baseline-registration mount point for when it is defined.
4. C++ parity → **mount + Ridge wired through**; other estimators marked `cpp_exportable=unknown`.

## 2. Module Boundaries

Package `soh_bench` under `src/`. Single-responsibility, acyclic dependencies.

```
src/soh_bench/
  contracts.py    # Estimator Protocol + Fold / Result / QCReport / GateVerdict dataclasses (defs only)
  registry.py     # decorator registry: estimators & datasets by name
  metadata.py     # per-dataset map: chemistry, label_source, C_rated, home-scenario flag (curated, NOT guessed)
  qc.py           # data QC gate: value-range / label-source / availability-mask  [v2]
  data.py         # DatasetLoader: reads processed/*.pkl, runs qc, joins metadata, raw-optional hook
  protocols.py    # LeaveOneCellOut / LeaveOneDatasetOut / LeaveOneChemistryOut(+separability assert)
  metrics.py      # rmse / mae / max_error / eol_region_error / picp / monotonicity_violation_rate / value_range_violation / forgetting_delta
  conformal.py    # split-conformal wrapper: turns a point estimator into an interval estimator  [v2]
  gates.py        # G1/G2/G3 gate evaluation: dataset axis x check-ladder -> GateVerdict  [v2]
  runner.py       # BenchmarkRunner: (datasets x protocols x estimators) -> ResultSet + GateVerdicts
  report.py       # ResultSet -> leaderboard md/json + QC report + gate PASS/FAIL + warnings
  cpp_parity.py   # optional golden-vector Python<->C++ parity check for estimators exposing export_cpp  [v2]
  estimators/     # v1 built-in baselines (Ridge wired through C++; +1-2 others as fixtures)
  cli.py          # python -m soh_bench run --config ... [--check-cpp-parity]
```

**Dependency graph (acyclic):**
`contracts <- {registry, metadata, qc, metrics, conformal}`
`metadata <- qc <- data`
`{metrics, conformal} <- gates`
`{data, protocols, gates, estimators} <- runner <- {report, cpp_parity, cli}`

Leaf modules buildable in parallel: `contracts, metrics, qc, metadata, registry, conformal`.

## 3. Data Flow

```
DatasetLoader(data_dir = D:\battery_SOH\data)
  reads processed/{name}_features.pkl  -> per-cycle table [dataset, cell_id, cycle_index, soh, <features>]
  joins metadata: chemistry, label_source, C_rated
        |
        v
QC GATE (qc.py) — BEFORE any scoring                                   [v2, base R2/R3]
  value-range:  drop rows with soh outside [0.6, 1.05] (configurable); count in QC report, never silent
  label-source: reject datasets with label_source=bms_reported; warn on unknown          (base R4)
  availability: per-feature availability mask across selected datasets;
                drop columns with availability < 20%; NO global-mean fill                 (base R3)
  -> clean table + QCReport (pass-rate, drop reasons); source pass-rate <70% flagged red
        |
        v
Protocol.generate(clean_table) -> Fold(train_idx, test_idx, held_out_key)
  LeaveOneCellOut      : group by cell_id
  LeaveOneDatasetOut   : group by dataset
  LeaveOneChemistryOut : group by chemistry; ASSERT >=2 datasets/chemistry else downgrade to dataset-out + warn
        |
        v
BenchmarkRunner: for each (estimator x protocol x fold)
  est.fit(train_df, y_train)
  if no predict_interval: wrap with conformal (calibration split from train tail)         [v2, base R4]
  est.predict(test_df) [+ interval]
  metrics.compute(...) -> per-fold per-metric
        |
        v
gates.py: assemble GateVerdict per axis
  G2 (generalization): MAE<=2%, PICP>=90% (CI guaranteed via native or conformal; ci_source labeled)
  G3 (physical): monotonicity_violation_rate<=0.1%, value_range_violation==0
  G1 (anti-forgetting): SKIPPED in v1 (mount point present)
        |
        v
ResultSet (tidy long) + GateVerdicts + QCReport
        |
        +-- report -> leaderboard (mean+/-std per estimator x protocol) + gate PASS/FAIL + QC + warnings
        +-- detail parquet/csv + config.json -> runs/<timestamp>/
        +-- (optional) cpp_parity -> per-estimator golden-vector parity result
```

### Input data contract & curated metadata

Per-cycle feature table (verified for NASA / CALCE CS2 / HUST / Oxford). ID/label columns:
`dataset, cell_id, cycle_index, soh`; the rest are features (via `feature_columns()`).

`metadata.py` holds per-dataset records — **values verified against source docs, not guessed** (`# VERIFY`
markers resolved during implementation):
`chemistry` (LFP / NMC / NCA / LCO ...), `label_source` (calibration_capacity / bms_reported / unknown),
`C_rated_Ah`, `is_home_scenario` (for future G1).

### Raw-optional channel

Estimators set `requires_raw = True` to opt into a raw per-cycle channel. v1 ships the hook + contract
only; the raw path raises a clear `NotImplementedError` until L1 feature benchmarking is in scope.

## 4. Estimator Contract

```python
@runtime_checkable
class Estimator(Protocol):
    name: str
    requires_raw: bool                        # default False

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> None: ...
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...            # required, point estimate

    # optional — strongly recommended; absence => G2 gate_eligible=False (auto-conformal still applied):
    def predict_interval(self, df, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]: ...
    # optional — enables C++ parity check:
    def export_cpp(self, path: str) -> None: ...
```

- Framework passes the **whole feature table**; estimator selects columns via `feature_columns()`.
- **Uncertainty policy [v2]**: the framework guarantees every estimator has a CI. If `predict_interval`
  is absent, the runner wraps the estimator with `conformal.py` (split-conformal, calibration set carved
  from the train tail — never the test fold). All estimators are therefore G2-eligible; the report labels
  `ci_source ∈ {native, conformal}` so conformal-derived PICP is never silently equated with native CI.
  (Cross-domain folds may violate conformal exchangeability → under-coverage will surface in PICP itself;
  weighted/adaptive conformal is a v2 upgrade per parent-doc L4.)
- `fit(train)` then `predict(test)` as separate calls structurally prevents test-fold leakage into fit.
  Standardization-on-train-only and conformal-calibration-split are estimator/wrapper obligations,
  guarded by a leakage smoke-test (§7).
- Registration via `@register_estimator` / `@register_dataset`; duplicate name → error.

## 5. Metrics

| Metric | Definition | Axis |
|---|---|---|
| `rmse` / `mae` / `max_error` | standard error metrics | G2 |
| `eol_region_error` | RMSE over samples with SOH <= `eol_threshold` (default 0.8) | G2 |
| `picp` | Prediction Interval Coverage Probability (native or conformal-derived, labeled) | G2 |
| `monotonicity_violation_rate` | fraction of consecutive predicted steps rising > `rebound_tol` (default 0.5%), per cell trajectory | **G3** |
| `value_range_violation` | fraction of predictions outside `[0.6, 1.05]` | **G3** |
| `forgetting_delta` | MAE increase on a registered home scenario vs its baseline | **G1 (deferred; metric defined, unused in v1)** |

Each metric is a pure function, unit-tested in isolation. Monotonicity is evaluated on **predicted SOH
vs throughput/cycle order** (per parent-doc refinement: throughput axis, allowing calibration rebound),
not calendar time.

## 6. Gate Model (eval_harness canonical)

Three **dataset axes**; "structural → numerical → acceptance" is the **check-ladder within each axis**
(not a separate set of gates — this resolves the v1 naming collision):

| Gate | Dataset axis | Acceptance (configurable) | v1 |
|---|---|---|---|
| **G1 anti-forgetting** | registered home scenario | home MAE degradation <= 10% | **deferred** (mount only) |
| **G2 generalization** | leave-one-domain folds | MAE <= 2% SOH, PICP >= 90% | **yes** |
| **G3 physical-consistency** | per-cell predicted trajectory | monotonicity_violation_rate <= 0.1%, value_range_violation == 0 | **yes** |

Check-ladder per gate: structural (schema/shape) → numerical (range/cross-domain degradation direction)
→ acceptance (thresholds). A gate yields `GateVerdict{pass, eligible, metrics, reasons}`.

## 7. Error Handling

| Situation | Handling |
|---|---|
| estimator raises inside a fold | record failed fold (metrics NaN + error), continue, list in report |
| protocol < 2 groups | skip with warning |
| chemistry-out with <2 datasets/chemistry | downgrade to dataset-out + explicit confound warning |
| estimator lacks `predict_interval` | auto-conformal wrap (train-tail calibration); `ci_source=conformal` labeled |
| QC: soh out of `[0.6,1.05]` | drop rows, count in QCReport; if a fold loses >X% rows, flag |
| QC: `label_source=bms_reported` | reject dataset (base R4), clear error |
| QC: feature availability < 20% | drop column, no fill, record in QCReport |
| source QC pass-rate < 70% | flag red, do not silently include |
| schema mismatch / NaN / inf | structural validation fails fast |
| `export_cpp` parity mismatch > tol | mark `cpp_exportable=False` with delta, do not fail the run |

## 8. Result Artifacts

`runs/<timestamp>/`: `results.parquet` (+`.csv`), `gates.json` (G1/G2/G3 verdicts), `qc_report.json`,
`leaderboard.md` / `leaderboard.json` (aggregated mean±std + gate PASS/FAIL + `gate_eligible` flags +
QC/confound warnings + failed-fold section), `config.json` (resolved run config for reproducibility),
optional `cpp_parity.json`.

## 9. Testing Strategy (TDD)

Tests first, per module. Critical tests encode base red lines directly:

- **metrics**: known-array rmse/mae/max; `eol_region_error` filters by threshold; `picp` coverage;
  `monotonicity_violation_rate` counts only >tol rises; `value_range_violation` flags out-of-range.
- **qc [v2]**: fixture with soh=1.22/0.0 → dropped + counted; `bms_reported` dataset → rejected;
  low-availability column → dropped without mean-fill (assert no synthetic values introduced).
- **protocols (leakage gate)**: synthetic data → **no cell in both train and test**; dataset-out holds
  out a whole dataset; chemistry-out with 1 dataset/chemistry → downgrades + warns.
- **conformal [v2]**: calibration set carved from train tail only (assert no test indices used);
  wrapped point estimator yields ~nominal coverage on synthetic data.
- **gates [v2]**: point-only estimator gets `ci_source=conformal` and still receives a G2 verdict;
  G3 fails on a non-monotone / out-of-range prediction series; thresholds configurable.
- **registry / runner / report**: register-retrieve + duplicate error; end-to-end ResultSet shape;
  injected raising estimator isolated; deterministic leaderboard.
- **cpp_parity [v2]**: built-in Ridge `export_cpp` → Python↔C++ golden-vector parity within tol
  (reuses the train_export_demo pattern from soh-adapt-v2).

## 10. Out of Scope (v1)

- G1 anti-forgetting evaluation (metric + mount only; no home scenario registered yet).
- Leave-one-station-out & forward-chaining time splits (Protocol abstraction designed for them).
- Raw per-cycle loading / L1 feature-extraction benchmarking (hook + contract only).
- Config-driven YAML matrix runner with caching/resume (interfaces shaped for it).
- The L1–L6 algorithm implementations (separate spec); v1 ships minimal baselines (Ridge wired through
  C++, +1–2 fixtures) to exercise the framework.
- The frontend tool & three mockups (separate spec; consumes `results.parquet`/`leaderboard.json`/`gates.json`).

## 11. Build Decomposition (for the implementation plan)

Parallel leaves: `contracts, metrics, qc, metadata, registry, conformal`. Then `data` (needs
metadata/qc), `protocols`, `gates` (needs metrics/conformal); then `runner` (integrates), then
`report`, `cpp_parity`, `cli`. Each module ships with its tests (TDD red→green) before integration.
The leaf set is well-suited to parallel subagent implementation.
