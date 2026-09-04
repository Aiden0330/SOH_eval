# Spec 01 — SOH kernel invariants

- **Status:** **v1.0 — signed 2026-09-03.** Review delegated by the owner (2026-09-03) and executed by Claude Fable 5.1: every threshold below was checked against `src/sohkernel` at the `kernel-extraction` HEAD of 2026-09-03; corrections and reviewer notes are marked ▸. Changing any pinned number still requires a dated review note in the test that pins it.
- **Applies to:** `sohkernel` (extracted from `sohbench`), the frozen workbench, and any tool — including `sohagent` — that trains, searches, adapts or claims SOH estimators on this data.
- **Sources:** sub-project A spec and research-grounding decisions (2026-06-26); label-safety v1.1 and its §6 revisions (2026-08-20/21/25/27); agent-guided search design (2026-08-26); `src/sohbench/{contracts,qc,feature_safety,protocols,metrics,tasks,acceptance,conformal,adaptation,runner}.py` at HEAD 732923a; v1.0 review against `src/sohkernel/{contracts,qc,feature_safety,protocols,metrics,tasks,acceptance,conformal,adaptation,runner,api,datasheets,statistics,report}.py` (kernel 0.1.0).

These are the rules that must hold regardless of which tool produces a number. They are enforced by kernel tests; a tool may add rules, never relax these. Changing a threshold requires a dated review note in the test that pins it (the pattern of `tests/test_label_leak_inventory.py`).

## 1. Data model

- A **feature table** has `ID_COLUMNS = ["dataset", "cell_id", "cycle_index", "soh"]` plus feature columns; `feature_columns(X)` is everything not in `ID_COLUMNS`.
- **Per-cell metadata** travels separately: `dataset, cell_id, chemistry, c_rated_ah, label_source` (+ optional cathode, anode, cell_format, station). Chemistry and rated capacity come from verified datasheets; entries still marked `VERIFY` cannot support a chemistry-level claim.
- **SOH definition:** `soh := calibration_capacity / rated_capacity` from a reference test (`label_source = "calibration_capacity"`). BMS-reported labels are rejected by default (`reject_bms=True`).
- **Bounds:** `0.0 ≤ soh < 1.3` (lower-inclusive, upper-exclusive; `qc_gate(lower=0.0, upper=1.3)`); values above 1.0 are legal (fresh-cell overshoot) and must be kept. ▸ QC drops in this order and counts each reason: `soh_nonpositive` (soh ≤ 0), `soh_out_of_range`, then `duplicate_cycle_identity` (a `(dataset, cell_id, cycle_index)` seen more than once is rejected whole, Task 7f).
- **Availability:** a feature column with < 20 % non-null values is dropped (`availability_min=0.2`); nothing is ever imputed by the kernel.
- **Every dataset has a datasheet** declaring the label anchor (how `soh` is computed and from which measured quantities), chemistry, rated capacity source, integration segments used for windowed features, window bounds, license and provenance digests (Spec 06). ▸ Implemented 2026-09-03: `data/datasheets/<name>.yaml`, read by `sohkernel.datasheets` (`chemistry_verified` = no `VERIFY` under `chemistry`; `is_signed`); `preflight` warns on every `lcho` fold whose held-out chemistry is unverified. The chemistry labels were checked against primary sources on 2026-09-03 (D4): `rwth_isea_public` was mislabelled LFP and is NMC/graphite (Sanyo/Panasonic UR18650E); `nasa` stays unverified (no primary source names the chemistry).

## 2. Label safety (four layers, all mandatory)

1. **Structural ban (static).** Label-source columns and everything affinely derived from them are never model inputs on any dataset: `charge_capacity_max`, `capacity_ah`, `rated_capacity`, `rated_capacity_ah`, `charge_energy_proxy_wh`, `cumulative_capacity_proxy_ah` (`feature_safety.LABEL_SOURCE_FEATURE_COLUMNS`). Enforced inside the feature selector itself, so "all columns" selectors and bring-your-own estimators cannot see them.
2. **Value probes (dynamic).** Per dataset group with n ≥ 64: exact copy/complement checks; **affine reconstruction** `label ~ a·x + b` with R² ≥ 0.995 is a violation (legitimate physics tops out near 0.98); **integral coverage** for accumulation features: median `q_win / rated capacity` (datasheet nameplate from the metadata, cell-prefix overrides included; the table's capacity column is the fallback only for datasets without metadata — changed 2026-09-03, D5) ≥ 90 % blocks, > 60 % warns.
3. **Enforcement at every boundary.** Ingestion scan (findings recorded on the dataset, the dataset is not rejected); training admission (the *combination* is rejected with column, group, statistic and remedy in the message); transform audit behind a label-isolation boundary (a transform never receives the label column); trusted inference (a model's manifest columns are audited before `predict`). Fitted feature pipelines are fitted inside each evaluation fold on the training rows only and their output is audited before an estimator sees it (`sohkernel.pipelines`, 2026-09-03).
4. **Standing inventory.** A pinned per-dataset expectation table runs against the real data and fails in both directions: a new finding, or a known finding that disappears.

**Search interaction rule.** Any optimizer or agent that selects features, windows, or programs runs the audit as an **admission oracle before scoring**. It is never a penalty term; excluded candidates never execute and their reasons are reported.

## 3. Protocols

- **No random cycle split anywhere.** Cycles of one cell are never split between train and test.
- **Leave-one-domain family:** `loco` (cell), `lodo` (dataset), `lcho` (chemistry; when a held-out chemistry spans a single dataset the fold is flagged as lab-confounded and cannot support a chemistry claim).
- **Forward-chaining** (`forward`): per cell, the earliest `max(min_train_cycles, round(train_frac · n))` cycles are its early life (defaults 20 cycles, 0.6); one fold per eligible cell tests its later cycles and trains on the pooled early life of all cells (`pooled=False` restricts training to the cell's own early cycles). No future cycle of any cell reaches training. Required for any temporal or adaptation claim (implemented 2026-09-03, extraction plan Task 7a). ▸ **Pooled early-life training confirmed as the default on 2026-09-03** (owner decision, delegated): it is the deployment reading — a fleet model fitted on every unit's history so far and asked to extrapolate each unit forward — and it gives estimators enough rows to fit; `pooled=False` is the stricter per-cell tracker with 20–60 training rows per cell and answers a different question, so it stays an explicit option, never the default. Cells shorter than their early life feed the pooled early life only and are reported with a warning.
- **Nested evaluation for search.** Inner datasets are search-visible and scored under `loco` per dataset; ranking is by the **worst-case** inner MAE, not the mean. The outer dataset is invisible during search and is scored **once** under `lodo`/`forward` for the finalists (top-k), with every finalist's result disclosed. The report carries an `integrity` block stating this.

## 4. Metrics and verdicts

- **Point:** `rmse, mae, mape, mean_bias_error` (signed, pred − true), `overestimation_rate, max_error, alpha_accuracy (α = 0.15), eol_region_error (soh ≤ 0.8)`.
- **Trajectory (per cell, ordered by cycle):** `monotonicity_violation_rate` (rebound tolerance 0.005), `value_range_violation` (outside `[0, 1.3)`).
- **Interval:** `picp` plus mean interval width.
- **Extended (inference records):** MAE and bias per SOH band, per-cell worst and p90, EOL (80 %) cycle error with undetected-cell count.
- **Verdict thresholds:** generalization `mae ≤ 0.02`; calibration `picp ≥ 0.90` at nominal 0.90; physical consistency `monotonicity_violation_rate ≤ 0.001` and `value_range_violation = 0`.

## 5. Uncertainty

- Every estimator reports an interval. Estimators without native intervals are wrapped in split conformal with calibration carved from the **train tail** (the last `calib_frac = 0.2` of the training rows in row order), never from the test fold; the quantile uses the finite-sample correction `ceil((n+1)(1−α))/n` at α = 0.1.
- Cross-domain folds use weighted/adaptive conformal, because exchangeability fails across labs. ▸ The weighted wrapper falls back to the plain quantile when its density-ratio classifier is degenerate and then says so in `ci_source`; a row labelled `weighted_conformal` is never silently unweighted.
- Conformal recalibration at inference fits on the same K calibration residuals and declares it in the channel string.

## 6. Seeds and statistics

- Any leaderboard entry or claim uses **≥ 5 seeds**; the seed must reach the estimator (`set_seed`/`random_state`), not only NumPy's global state. Implemented 2026-09-03 (extraction plan Task 7b): the runner seeds every estimator instance and the weighted conformal wrapper. ▸ The kernel records the seeds it ran and does not refuse fewer (`api.score` accepts any list; the CLI defaults to `0 1 2 3 4`); the five-seed floor is enforced where a number becomes a claim — the claim flow of Spec 04 (P15) and the benchmark suite (Spec 09). Search may run one seed (§10).
- Leaderboards report mean and std across folds and seeds separately; claims add cell-level bootstrap intervals and paired comparisons (Spec 04). ▸ Every leaderboard entry now also carries `units` (attempted / succeeded / failed, Task 7e) and `confidence_intervals` (seeded percentile bootstrap over per-cell errors for `loco`/`forward`, 90 %, B = 1000; `unavailable` with a reason for pooled folds, Task 7h); `eol_region_error` is omitted, not zeroed, when a fold has no EOL sample (Task 7g).

## 7. Adaptation legality

- The trained artifact is frozen at inference; adaptation composes operators around it.
- Every operator declares its **information channel** (`OPERATOR_CHANNELS`); the channel string is provenance.
- Calibration rows are excluded from evaluation. `earliest` (first K cycles per cell) is the default and the only split allowed for a forward claim; `spread` is permitted only as an explicitly labelled periodic-checkup backtest.
- Label-free operators (CORAL on target feature statistics, per-cell monotone projection) may be applied without calibration labels; residual and recalibration operators require declared K.

## 8. Provenance

- A run writes `results.parquet/csv`, `leaderboard.json` (validated against `schemas/leaderboard.schema.json`), `dataset_leaderboard.json`, `qc_report.json`, `run_config.json` (`api.write_artifacts`).
- A model manifest carries feature columns, transforms, window bounds, `trained_on` datasets, seeds, and is bound immutably by SHA-256 (`model@sha256`). ▸ Binding and invalidation live in the workbench store (`sohbench.model_store`); the kernel side is `sohkernel.inference_gate` (manifest columns audited before `predict`) and `sohkernel.records` (prediction records, `protocol@sha256`). Any tool that stores models must keep the same binding.
- Changing feature semantics (window rebuild, transform change) triggers an **invalidation sweep**: artifacts trained on the old semantics are marked `invalidated` with the reason, leaderboard entries included.
- Datasets record `label_source_columns` from the ingestion scan.

## 9. Kernel API surface (for any tool)

`sohkernel.api`: `load(datasets) → (table, meta, qc_report)`, `normalize_request(payload) → request`, `preflight(request) → Preflight(ready, reasons, warnings, combinations, protocols, …)`, `transform(table, meta, featuresets) → table` (label-isolated), `score(task, table, meta, protocols, estimators, featuresets, seeds) → results`, `verdicts(metrics, task) → [Verdict]`, `write_artifacts(...) → path`; `sohkernel.protocols.PROTOCOLS = {loco, lodo, lcho, forward}`. ▸ Verified 2026-09-03; the platform (`sohagent.evaluator`) uses exactly this surface and reproduces the frozen workbench to 1e-17 (M1 parity).

Guarantees: pure with respect to jobs and cases; deterministic given seeds and data digests; **never imports** `sohbench` or `sohagent`.

## 10. Known limitations to close (from the 2026-09-02 report)

▸ Chemistry coverage after the 2026-09-03 datasheet review: LFP (hust, snl_lfp, uconn), LCO (calce_cs2; nasa carries the label but is unverified), LCO/NCO (oxford), NMC (rwth_isea_public, 8 of the 48 Sanyo/Panasonic UR18650E cells). Every non-LFP chemistry spans at most two datasets and one lab, so `lcho` folds stay lab-confounded and cannot support a chemistry claim until a second dataset per chemistry is onboarded (D1). `forward` implemented 2026-09-03 (Task 7a); search defaults to one seed (claims need five, §6); the window-coverage audit used `charge_capacity_max` as its denominator, which was distorted on calce (fixed 2026-09-03, D5: rated capacity first). Each is a tracked task, not a relaxation of this spec.
