# Sub-project A — SOH Benchmark & Evaluation Harness (`sohbench`) — Spec

> **Date**: 2026-06-26
> **Status**: design spec for sub-project A (the spine); ready for user review → writing-plans.
> **Clean-room**: derived from this directory's md + `D:\battery_SOH\data` data + external research only. Does
> NOT reference the old `D:\battery_SOH` skills / `phm-router` / `soh-adapt` contracts. Supersedes the earlier
> `soh-bench` v1/v2 specs (which were pre-research and anchored to those old skills).
> **Grounding**: [research-grounding & decisions D1–D12](2026-06-26-research-grounding-and-scheme-decisions.md).
> **Parent design**: [`SOH泛化性增强分层方案.md`](../../../SOH泛化性增强分层方案.md).
> **Build order**: A (this) → C-for-A (frontend of A's features) → B (our SOH algorithms) + C-for-B. So A's
> **result-artifact schema (§10) is a first-class contract** the frontend will consume.

---

## 0. Purpose

A is the **spine serving both project goals**:
1. a **unified algorithm interface** that both our own SOH algorithms (sub-project B) and **other teams'
   existing algorithms** implement; and
2. a **benchmark/evaluation harness** that runs them fairly across working-condition datasets and reports
   comparable results.

Because every algorithm — ours and external — implements one interface, "evaluating our framework" and
"comparing different teams' algorithms" are the **same machinery**. Prior art validates this pattern
(BatteryML's modular, config-driven harness with a pluggable model interface; BatteryLife's cross-condition
scale; SOHbenchmark's reproducible-pipeline motivation). A adopts that pattern, clean-room.

A is Python. External non-Python algorithms are wrapped behind a thin Python adapter implementing the
interface. A itself imposes **no** non-DNN/C++ constraint — those constraints belong to our own entries (B);
external DL/pytorch algorithms can be compared freely.

## 1. Locked Decisions (A-relevant subset of D1–D12)

| Ref | Decision in A |
|---|---|
| D1 | SOH = `C_now / C_rated` (capacity-side), recorded explicitly |
| D2 | QC value bound `[lower, 1.3)`, `lower` configurable; SOH > 1.0 is legal |
| D3 | EOL threshold 80% (configurable; 90% for industrial large-capacity) |
| D4 | CI guaranteed for every estimator (native or auto split-conformal); cross-domain weighted/adaptive conformal; `ci_source` labeled — grounded in EU Battery Regulation 2023/1542 auditability + conformal-as-field-standard |
| D5 | Leave-one-cell-out (anti-leakage baseline) + leave-one-dataset-out + leave-one-chemistry-out (separability-asserted); forward-chaining designed-for |
| D6 | Pre-score QC gate: value-range, sudden-drop fault exclusion, non-degrading/short-life exclusion, feature-availability mask (no mean-fill), label-source |
| D8 | Fairness: unified preprocessing + **multi-seed averaging (≥5)** + **dual feature mode** (unified-feature main leaderboard + bring-your-own-features track, labeled) |
| D9 | Metrics: RMSE / MAE / MAPE / EOL-region error / max error / 15%-accuracy / PICP / monotonicity-violation-rate / value-range-violation |
| D10 | Physical-consistency: monotonicity on throughput axis with rebound tolerance (capacity regeneration is real) + value-range |
| D12 | Algorithm-agnostic **Task** abstraction so safety-warning evaluation can be added later without reworking the harness |

## 2. Module Boundaries

Package `sohbench` under `src/`. Single-responsibility, acyclic. (B will later import the contract from here.)

```
src/sohbench/
  contracts.py    # Task, Estimator Protocol, CellMeta, Fold, Result, QCReport, Verdict dataclasses (defs only)
  registry.py     # decorator registry: estimators, datasets, tasks by name
  metadata.py     # per-dataset curated map: chemistry, cathode/anode, C_rated_Ah, label_source, format (NOT guessed; # VERIFY)
  representation.py # unified data model: FeatureTable + CellMeta (+ RawChannel stub) — BatteryData-like
  qc.py           # QC gate: value-range / sudden-drop / non-degrading / availability-mask / label-source -> clean data + QCReport
  data.py         # DatasetLoader: reads processed/*.pkl, builds representation, runs qc, joins metadata
  features.py     # unified feature view + feature_columns(); the "unified" feature-mode provider
  protocols.py    # LeaveOneCellOut / LeaveOneDatasetOut / LeaveOneChemistryOut(+separability) / ForwardChaining(designed-for)
  metrics.py      # the D9 metric set, each a pure function
  conformal.py    # split-conformal wrapper (+ weighted/adaptive variant hook for cross-domain)
  tasks.py        # SOHEstimationTask (v1): target=soh, metric set, acceptance config; Task base for future safety task
  runner.py       # BenchmarkRunner: (task x dataset x protocol x estimator x seed) -> ResultSet + Verdicts
  acceptance.py   # configurable acceptance checks -> Verdict (generalization / calibration / physical-consistency)
  report.py       # ResultSet -> leaderboard + QCReport + Verdicts + run_config (the C-facing artifacts, §10)
  cli.py          # python -m sohbench run --config ...
```

**Dependency graph (acyclic):**
`contracts <- {registry, metadata, metrics, conformal}`
`metadata <- qc <- data ; representation <- data ; features <- data`
`{metrics, conformal} <- {tasks, acceptance}`
`{data, features, protocols, tasks, acceptance} <- runner <- {report, cli}`

Parallel-buildable leaves: `contracts, metrics, qc, metadata, registry, conformal, representation`.

## 3. Unified Data Representation & Dual Feature Mode

### 3.1 Representation (BatteryData-like)

- `FeatureTable` — per-cycle rows: `[dataset, cell_id, cycle_index, soh, <unified features…>]` (default input;
  the processed `*.pkl` already supply this for NASA/CALCE/HUST/Oxford).
- `CellMeta` — per-cell: `{dataset, cell_id, chemistry, cathode, anode, C_rated_Ah, format, label_source, station?}`.
- `RawChannel` (stub) — per-cycle raw curves (V/I/T/Q). v1 ships the hook + contract; the path raises a clear
  `NotImplementedError` until L1 raw-feature work (sub-project B) needs it. Internal-resistance fields, if
  present in a source, are **not** ingested for our entries (B is capacity-side only); external algorithms may
  request raw and use what they want.

### 3.2 Dual feature mode (D8)

- **unified** (default, main leaderboard): the harness supplies identical `features.py` columns to every
  estimator → maximally fair and comparable. Directly answers the field's "unfair comparison from
  inconsistent preprocessing" problem.
- **byo** (bring-your-own-features): an estimator declares `feature_mode="byo"`; the harness hands it the
  `RawChannel`/`CellMeta` and the estimator extracts its own features (e.g., end-to-end or custom HI). Results
  run on a **separate, clearly-labeled track**; never mixed into the unified leaderboard ranking.

## 4. Task & Estimator Contract

```python
@runtime_checkable
class Estimator(Protocol):
    name: str
    feature_mode: Literal["unified", "byo"]     # default "unified"
    backend: Literal["sklearn", "pytorch", "external"]

    def fit(self, X: pd.DataFrame, y: np.ndarray, meta: CellMeta) -> None: ...
    def predict(self, X: pd.DataFrame, meta: CellMeta) -> np.ndarray: ...      # point estimate (required)

    # optional capabilities (detected via hasattr):
    def predict_interval(self, X, meta, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]: ...
    def export_cpp(self, path: str) -> None: ...     # deployable entries (B); enables optional C++ parity check
```

- `meta` is passed so pipelines can normalize/gate by `C_rated`, chemistry, station (e.g., working-condition
  gating, per-capacity normalization) without contract changes.
- `fit(train)` then `predict(test)` as separate calls structurally prevents test-fold leakage into fit.
- **Task** (`tasks.py`) parameterizes target column, metric set, and acceptance config. v1 = `SOHEstimationTask`
  (target `soh`). A future `SafetyWarningTask` plugs in with different target/metrics, reusing data/protocols/
  runner unchanged — the D12 seam.
- Registration via `@register_estimator` / `@register_dataset` / `@register_task`; duplicate name → error.

## 5. Data Loading + QC Gate (D2/D6)

`DatasetLoader(data_dir=D:\battery_SOH\data)` → representation → **QC gate before any scoring**:

1. **value-range**: drop rows with `soh` outside `[lower, 1.3)` (configurable); counted, never silent.
2. **sudden-drop fault**: flag/exclude cycles with non-physical SOH step drops suspected as equipment faults
   (BatteryLife practice).
3. **non-degrading / short-life**: optionally exclude cells that never reach EOL band or have `< min_cycles`
   (configurable; relevant to per-cell trajectory metrics).
4. **availability mask**: across selected datasets, drop feature columns with availability `< 20%`; **no
   global-mean fill** (fill values become fake tree split points).
5. **label-source**: `metadata.label_source`; `bms_reported` rejected by default (self-reported SOH is not a
   trustworthy label), `unknown` warned.

Emits `QCReport` (per-dataset pass-rate, drop reasons, dropped columns); source pass-rate `< 70%` flagged.

## 6. Protocols (D5)

`Protocol.generate(clean_table, meta) -> Iterator[Fold(train_idx, test_idx, held_out_key)]`:

- `LeaveOneCellOut` — group by `cell_id` (anti-leakage baseline; the established battery-wise protocol).
- `LeaveOneDatasetOut` — group by `dataset`.
- `LeaveOneChemistryOut` — group by `chemistry`; **assert ≥ 2 datasets per chemistry**, else downgrade to
  dataset-out + explicit confound warning (avoids overclaiming chemistry generalization).
- `ForwardChaining` — early-life train / late-life test (extrapolation); abstraction present, v1 optional.

Random K-fold over cycles is **disallowed by construction** (no such protocol exists in the module).

## 7. Metrics (D9)

Pure functions `(y_true, y_pred[, intervals][, order]) -> float`:

| Metric | Note |
|---|---|
| `rmse`, `mae`, `max_error` | standard error |
| `mape` | field-standard relative error |
| `alpha_accuracy` (15%) | fraction within 15% relative error (BatteryLife) |
| `eol_region_error` | RMSE over SOH ≤ `eol_threshold` (default 0.8) |
| `picp` | interval coverage; native or conformal-derived (labeled) |
| `monotonicity_violation_rate` | fraction of consecutive predicted steps rising > `rebound_tol` (default 0.5%) along throughput/cycle order — **not** calendar time |
| `value_range_violation` | fraction of predictions outside `[lower, 1.3)` |

## 8. Uncertainty (D4)

The harness **guarantees a CI for every estimator**: if `predict_interval` is absent, `conformal.py` wraps it
(split-conformal, calibration carved from the **train tail**, never the test fold). Cross-domain folds violate
exchangeability → a **weighted/adaptive conformal** variant is used there (standard split-conformal for
within-domain). Every PICP is reported with `ci_source ∈ {native, conformal, weighted_conformal}` so
conformal-derived coverage is never silently equated with native. Grounded in EU Battery Regulation
auditability and conformal prediction being the field-standard SOH UQ method.

## 9. Runner & Acceptance (D8)

`BenchmarkRunner` expands `(task × dataset × protocol × estimator × seed)`:
- **seeds ≥ 5**, results aggregated mean ± std (eliminates initialization/seed bias; BatteryML practice).
  Deterministic estimators report zero seed-variance (documented), still run for uniformity.
- per fold: `fit(train)` → guarantee-CI → `predict(test)` → metrics.
- failures isolated per fold (NaN + error), run continues.

`acceptance.py` applies **configurable acceptance checks** → `Verdict{pass, metrics, reasons}` along three
dimensions (named from first principles + field practice, not from any external skill's gate taxonomy):
- **generalization** — error thresholds per cross-domain protocol (e.g., MAE ≤ target).
- **calibration** — PICP within tolerance of nominal coverage.
- **physical-consistency** — `monotonicity_violation_rate ≤ tol`, `value_range_violation == 0`.

Thresholds live in the run config; the harness ships sensible defaults and the mounting mechanism.

## 10. Result Artifacts — the C-facing contract

A run writes `runs/<timestamp>/` with a **stable, documented schema** (frozen as the contract the frontend
C-for-A consumes):

- `results.parquet` (+`.csv`) — tidy long: `[task, estimator, feature_mode, backend, protocol, dataset,
  held_out_key, fold, seed, metric, value]`.
- `leaderboard.json` — aggregated per `(estimator, feature_mode, protocol)`: `metric → {mean, std}`, plus
  `verdicts`, `ci_source`, and `track ∈ {unified, byo}`. Unified-track ranking is separate from byo-track.
- `qc_report.json` — per-dataset pass-rate, drop reasons, dropped columns.
- `verdicts.json` — acceptance results (generalization/calibration/physical-consistency) with reasons.
- `run_config.json` — resolved config: datasets, protocols, estimators, seeds, thresholds, `sohbench` version,
  data fingerprints (reproducibility).
- `cpp_parity.json` (optional) — for estimators exposing `export_cpp`, golden-vector Python↔C++ deltas.

A JSON Schema for `leaderboard.json` / `qc_report.json` / `verdicts.json` is shipped so C-for-A can validate.

## 11. Error Handling

| Situation | Handling |
|---|---|
| estimator raises in a fold | failed fold (NaN + error), continue, list in report |
| protocol < 2 groups | skip with warning |
| chemistry-out < 2 datasets/chemistry | downgrade to dataset-out + confound warning |
| no `predict_interval` | auto-conformal wrap; `ci_source` labeled |
| QC: soh out of `[lower,1.3)` | drop rows, count in QCReport; fold losing > X% rows flagged |
| QC: `label_source=bms_reported` | reject dataset, clear error |
| QC: feature availability < 20% | drop column, no fill, record |
| source pass-rate < 70% | flag, do not silently include |
| schema/NaN/inf structural defect | fail fast with clear message |
| `export_cpp` parity mismatch > tol | mark `cpp_exportable=False` + delta; do not fail run |

## 12. Testing Strategy (TDD)

Tests first, per module; critical tests encode the hard rules:

- **metrics**: known-array rmse/mae/mape/max/alpha; `eol_region_error` filters by threshold; `picp` coverage;
  `monotonicity_violation_rate` counts only > tol rises along order; `value_range_violation` flags out-of-range.
- **qc**: soh=1.25 kept (legal), soh=1.35 / 0.0 dropped + counted; `bms_reported` rejected; sudden-drop excluded;
  low-availability column dropped with **no synthetic fill** (assert no new values introduced).
- **protocols (leakage gate)**: synthetic data → **no cell in both train and test**; dataset-out holds a whole
  dataset; chemistry-out with 1 dataset/chemistry → downgrades + warns; assert no random-cycle-split path exists.
- **conformal**: calibration carved from train tail only (assert no test indices); wrapped point estimator →
  ~nominal coverage on synthetic data; cross-domain uses weighted variant.
- **runner/acceptance**: synthetic data + dummy estimator → ResultSet shape; ≥5 seeds aggregated; injected
  raising estimator isolated; verdicts pass/fail on crafted predictions; dual feature-mode kept on separate tracks.
- **representation/data**: fixture pkl → representation + meta built; QC integrated; raw channel raises clearly.
- **report**: fixed ResultSet → deterministic leaderboard; artifacts validate against shipped JSON Schema.
- **registry**: register/retrieve estimators/datasets/tasks; duplicate → error.

## 13. Out of Scope (v1) / Seams

- Sub-project B's actual SOH algorithms (this spec ships only minimal baseline estimators — e.g., Ridge,
  optionally one tree + one GPR — to exercise the harness and tests; B implements the real L1–L6 entries).
- The frontend (C-for-A) — separate spec; consumes §10 artifacts.
- Safety-warning evaluation — `Task` seam present; not built.
- Raw-channel loading / L1 raw-feature benchmarking — hook + contract only.
- Config-driven YAML matrix runner with caching/resume — interfaces shaped for it; additive later.
- Forward-chaining protocol & leave-one-station-out — abstractions present; v1 optional/deferred.

## 14. Build Decomposition (for writing-plans + multi-agent)

Parallel leaves (independent, TDD red→green): `contracts`, `metrics`, `qc`, `metadata`, `registry`,
`conformal`, `representation`. Then `data` (needs metadata/qc/representation), `features`, `protocols`,
`tasks`/`acceptance` (need metrics/conformal). Then `runner` (integrates), then `report`, `cli`. Each module
ships with its tests before integration. The leaf set maps cleanly onto parallel subagent assignments.
