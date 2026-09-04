# Kernel extraction — `sohbench` → `sohkernel` (+ frozen workbench)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or superpowers:executing-plans, task by task. Steps use `- [ ]` checkboxes. Every task ends with the full suite green (`pytest -q`, 414+ tests) and the direction test green.

> **Status (2026-09-03, Wave 3):** Tasks 0–11 done, K7h included; 517 backend tests + 5 node suites green; every task is one commit on `kernel-extraction` (unpushed). Wave 3 also added dataset datasheets with `VERIFY` markers (`data/datasheets/`, `sohkernel.datasheets`, lcho preflight warning) and the M1 parity proof in soh-agent (`bench/parity/2026-09-03-m1-lfp-transfer-hust.md`: the platform reproduces the frozen workbench to 1e-17). Owner decisions pending: the pooled default of `forward` (Task 7a), D4 NASA chemistry (datasheet `VERIFY`), datasheet signatures (D6), S01 sign-off of Spec 01.

> **Execution note (2026-09-02, Wave 1):** the superpowers plugin is not installed in the executing environment, so Tasks 0–1 were run directly by one agent following the executing-plans discipline (one task at a time, full suite + direction test as the gate, one commit per task). Subagent-driven dispatch is reserved for tasks that are independent enough to parallelise; the Wave 2 moves (Tasks 2–5) are sequential by construction and stay with one agent.

**Goal:** carve the durable kernel out of `src/sohbench` into `src/sohkernel` without changing behaviour, so that the workbench keeps working and the new agent platform (`projects/soh-agent`) can depend on the kernel alone. See ADR-001 (`docs/adr/2026-09-02-ADR-001-kernel-extraction-and-agent-platform.md`).

**Rules (locked):**
- Behaviour-preserving moves only. Any behaviour change (forward-chaining, seed injection) is its own task with its own tests.
- `sohkernel` never imports `sohbench`. Enforced by `tests/kernel/test_import_direction.py` from Task 1 onward.
- Every moved module leaves a shim in `sohbench/<name>.py` (`from sohkernel.<name> import *  # noqa` plus explicit names used by the workbench) so no workbench file changes in the same task.
- Registries (`registry.py`) move to the kernel; registration stays explicit (`sohkernel.estimators.register_all()`), not an import side effect.
- Data root only via `SOHBENCH_DATA_DIR` (Task 0); no `D:\` default path remains in code, scripts, tests or the frontend (Windows-era fixtures that exercise legacy-path relocation are the only permitted occurrences).

---

### Task 0: Housekeeping and baseline

**Files:** `.gitignore`, `src/sohbench/paths.py`, `.github/workflows/*.yml`, `README.md`

- [x] Commit or discard the uncommitted `frontend/app/data/*.json` changes — done 2026-09-02: `recommendation_summary.json` tracked (the standalone build test needs it); the July docs/specs/exports and `.superpowers/sdd` reports committed; `frontend/app/battery-dashboard/` (79 MB synced asset), `frontend/mockups.zip` and the 11 MB standalone export ignored. Still untracked, owner's call: `frontend/app/data/invalidated_training_results.json` and the phase/quickstart/diagnostics profile directories.
- [x] Decide the base branch (`feat/phase2-dl` vs `master`) and tag the baseline: `git tag v0-workbench` (owner ran this 2026-09-02; branch `kernel-extraction`).
- [x] `paths.py`: single `data_root()` that reads `SOHBENCH_DATA_DIR`, falls back to `<repo>/data` (`default_data_dir()` kept as alias); `battery_soh_root()` reads `BATTERY_SOH_ROOT`, falls back to the sibling checkout; every `D:\\` default removed from `src/`, `scripts/`, the frontend and the API schema (empty `battery_soh_root` now means the server default); `tests/test_real_data_smoke.py` and `tests/test_label_leak_inventory.py` use `data_root()` and skip when absent; `tests/test_paths.py` pins it.
- [x] CI: `regression` matrix job on `ubuntu-latest` + `macos-latest` (kernel direction step, full `pytest -q`, `node --test frontend/app/`); the Windows phase 3.1.1 gate job kept. Tests that inspect the untracked synced dashboard or the unpublished phase 2.2 golden profile skip when those are absent, so a clean checkout is green (verified in a scratch worktree: 447 passed / 9 skipped, node 4 suites).
- [x] README: English-first, cross-platform quick start, environment table, extraction status; `docs/architecture/README.md` marks the phase notes historical.

Acceptance: suite green on macOS and in a clean checkout (CI run pending push); no `D:\\` *default* remains — the grep now only hits Windows-era fixtures that exercise legacy-path relocation (`tests/test_model_store.py`, `tests/test_battery_dashboard_native_sync_package.py`) and the negative assertion in `tests/test_standalone_frontend.py`.

---

### Task 1: Package skeleton and the direction test

**Files:** `src/sohkernel/__init__.py`, `pyproject.toml`, `tests/kernel/__init__.py`, `tests/kernel/test_import_direction.py`

- [x] `pyproject.toml`: `[tool.setuptools.packages.find] where = ["src"]` already finds both packages; `torch = ["torch"]` extra added (alias of `dl`); version `sohkernel` independently later (Task 11).
- [x] `src/sohkernel/__init__.py`: `__version__ = "0.1.0.dev0"`, no imports.
- [x] Direction test: walks `src/sohkernel/**/*.py` with `ast`, fails on any `import sohbench` / `from sohbench` / `import sohagent` and on `importlib.import_module("sohbench…")` string imports (`tests/kernel/test_import_direction.py`, 3 tests).

Acceptance: `pytest tests/kernel -q` passes (3 tests) — done 2026-09-02.

---

### Task 2: Move leaf modules (no kernel-internal deps)

Modules: `contracts`, `metrics`, `metadata`, `registry`, `feature_safety`, `raw_windows`, `method_contracts`, `preprocessing`.

- [ ] `git mv src/sohbench/<m>.py src/sohkernel/<m>.py` for each; fix relative imports inside them (there are none except `feature_safety`→ none).
- [ ] Add shims in `src/sohbench/<m>.py` re-exporting the public names (use `__all__` where present; otherwise explicit names used by workbench call sites, found with `grep -rn "sohbench\.<m>\|from \.<m> import\|from \. import <m>" src/sohbench`).
- [ ] Move their tests: `git mv tests/test_<m>.py tests/kernel/test_<m>.py`, switch imports to `sohkernel`.
- [ ] Keep `tests/test_label_leak_inventory.py` in place but importing `sohkernel.feature_safety`; it is the audit trail and must keep its history.

Acceptance: full suite green; direction test green.

---

### Task 3: Move dependent kernel modules

Modules in order: `qc`, `data`, `protocols`, `tasks`, `acceptance`, `conformal`, `features`, `featuresets`, `window_bank`, `report`, `runner`.

- [ ] Move each with its test, updating `from .x import` to kernel-relative imports; `featuresets` and `runner` currently do `from . import registry as R` — point at `sohkernel.registry`.
- [ ] `report.py`: keep artifact writing in the kernel (Parquet + JSON + schema validation); nothing frontend-specific lives here (check `frontend_publish` is not referenced).
- [ ] Shims as in Task 2.

Acceptance: suite green; `python -m sohbench run --data-dir $SOHBENCH_DATA_DIR --datasets hust --estimators ridge --protocols loco` still produces `leaderboard.json`.

---

### Task 4: Move estimators and make registration explicit

**Files:** `src/sohkernel/estimators/*`, `src/sohkernel/estimators/__init__.py`

- [ ] `git mv src/sohbench/estimators src/sohkernel/estimators`; keep `preprocessing` import pointing at the kernel.
- [ ] `estimators/__init__.py`: `register_all(include_torch: bool = torch_available())` that imports `baselines, physics, mechanism, byo` and, when torch is present, `deep, pinn, neural, torch_base`. Decorators keep registering on import; `register_all` is the only sanctioned entry point.
- [ ] Shim package `src/sohbench/estimators/__init__.py` calling `sohkernel.estimators.register_all()` and re-exporting the class names the workbench references (`cli.py`, `evaluation_service.py`).

Acceptance: suite green including torch-dependent tests when torch is installed; `python -c "import sohkernel.estimators as E; E.register_all(); from sohkernel.registry import get_estimator; get_estimator('gbdt')"`.

---

### Task 5: Break the inverted dependency and move `adaptation`

- [ ] Identify what `adaptation.py` imports from `evaluation_service` (`grep -n evaluation_service src/sohbench/adaptation.py`) — expected: model-binding / transform helpers used by `InferenceAdapter`.
- [ ] Move those helpers into the kernel (`sohkernel/binding.py` for immutable model binding, `sohkernel/transforms.py` for `_inference_transform_featuresets`-style logic) **or** pass them in as callables/parameters to `InferenceAdapter`; choose the move if the helper is pure, the parameter if it touches the model store.
- [ ] `git mv adaptation.py` + test; shim.

Acceptance: direction test green with `adaptation` in the kernel; adaptation tests green; the audited case run path in the workbench still applies CORAL/recalibration (run `tests/test_inference*.py`).

---

### Task 6: Kernel API facade (the surface the platform will call)

**Files:** `src/sohkernel/api.py`, `tests/kernel/test_api.py`

Extract the *pure* parts of `evaluation_service` into six functions; leave job/case orchestration behind.

- [x] `load(datasets, *, data_dir=None, qc=True) -> (table, meta, QCReport)` (wraps `DatasetLoader.resolve_datasets` + `load`; data root from the argument or `SOHBENCH_DATA_DIR`).
- [x] `preflight(request, *, data_dir | table/meta/qc_report, transform=None) -> Preflight` — the label-safety, window-DOD, coverage and protocol-feasibility checks moved out of `EvaluationService.training_preflight` (which now delegates, injecting its plugin-aware store transform); returns `ready`, per-combination `reasons`/`warnings`, profiles, summary and the transformed table. `normalize_request` moved too (the service's `_normalize_request` delegates).
- [x] `transform(table, meta, featuresets) -> table` — registered featureset transforms behind the label-isolation boundary (`feature_safety.apply_transform_without_label`); row count and required columns enforced, undeliverable candidates materialised as NaN.
- [x] `score(task, table, meta, protocols, estimators, featuresets, seeds) -> DataFrame` — thin wrapper over `BenchmarkRunner.run` (protocols/estimators by name or object).
- [x] `verdicts(metrics, task) -> list[Verdict]` — `acceptance.evaluate_verdicts`.
- [x] `write_artifacts(results, qc_report, task, config, out_dir) -> path` — `report.write_artifacts`.
- [x] `PROTOCOLS = {"loco": LeaveOneCellOut, "lodo": LeaveOneDatasetOut, "lcho": LeaveOneChemistryOut}` exported from `sohkernel.protocols`; the service imports it instead of keeping its own menu.
- [x] Tests (`tests/kernel/test_api.py`, 10 tests): `preflight` never selects `charge_capacity_max` and blocks a featureset built on it; a transform that reconstructs the label blocks its combinations; a ≥ 90 % window blocks, > 60 % warns, 45 % is ready; `transform` never receives the label column (spy featureset); score → verdicts → artifacts round trip validates against the leaderboard schema.

Acceptance: `evaluation_service.training_preflight` delegates to `sohkernel.api.preflight` (behaviour identical; existing service tests green) — done 2026-09-03.

---

### Task 7: Kernel-only behaviour additions (P0 from the report)

Separate commits, each with tests.

- [x] `protocols.ForwardChaining(train_frac=0.6, min_train_cycles=20, pooled=True)`: per cell, the earliest fraction is its early life; one fold per eligible cell tests its later cycles and trains on the pooled early life of all cells (`pooled=False` = the cell's own early cycles only); `held_out_key = "<dataset>/<cell>@forward"`; registered as `"forward"` in `PROTOCOLS`; the preflight reports eligible cells (done 2026-09-03, 6 tests). **Decision to confirm with the owner:** pooled early-life training as the default reading of "per cell, train on the earliest fraction".
- [x] Seed injection (2026-09-03): `runner._instantiate_estimator(factory, seed)` passes `random_state=` / `seed=` to class constructors that accept it, deep-copies instances per fit and seeds them through `set_seed()` or a `random_state`/`seed` attribute; `WeightedConformalWrapper` gets `random_state=seed`; `set_seed` added to `gbdt`, `extra_trees`, `stress_resid`, `mlp`, the torch `_DeepBase` family and `pinn`. `tests/kernel/test_seeds.py`: mlp differs across seeds, ridge is identical, leaderboard std over seeds > 0 for mlp.
- [x] `adaptation.calibration_split(..., strategy="earliest")` is the default everywhere (verified 2026-09-03: the function, `InferenceAdapter`, `AdaptationSearch` and the workbench payload all default to `earliest`); provenance now carries `calibration_split`, `calibration_scenario` (`forward_deployment` | `periodic_checkup_backtest`) and `supports_forward_claim`, so a `spread` run is labelled as a periodic-checkup backtest (3 kernel tests).

Ports from the archived Codex line (`archive/verification-fabric-phase0`; read the reference with `git show archive/verification-fabric-phase0:<path>`). Each is a behaviour change with its own tests; none is a cherry-pick (all conflict — see ADR-001).

- [x] **K7b reference:** seed propagation as in `d6b1efa` / `f86b753` — ported 2026-09-03 (see the seed-injection bullet); the per-fold `np.random.seed(seed)` legacy scope is kept.
- [x] **K7d — fit feature pipelines inside folds** (2026-09-03). Verified mainline: `store.apply_feature_transforms` runs once on the whole table before `BenchmarkRunner.run`, and the built-in transforms are row-local or use nameplate metadata, so nothing fitted leaks today — the risk is agent-written programs. `sohkernel.pipelines` defines the `FeaturePipeline` contract (`fit(train_x, train_y, meta) -> state`, `transform(frame, state, meta)`), `StatelessPipeline` for legacy callables and `prepare_fold`, which fits on the fold's training rows only, applies to both partitions, enforces the output contract (rows, index, identity columns, no label) and audits the created columns; `BenchmarkRunner.run(feature_pipelines=...)` fits once per fold (shared across estimators, featuresets and seeds), projects per fold and fails every unit of a fold whose pipeline breaks. `tests/kernel/test_pipelines.py`: the same transform applied globally halves the MAE (peeking) while the fold-local version cannot beat not having the feature; audited output; adapter; contract violations.
- [x] **K7e — explicit failure and partial outcomes** (2026-09-03). Every result row carries `status` (`ok | failed`) and failed units add `error_type` + `error`; `report._unit_outcomes` keeps failed units out of the aggregates (even when they carry finite values) and each leaderboard / dataset-leaderboard entry states `units = {attempted, succeeded, failed, outcome: complete | partial | failed}` (schema updated). Verdicts keep their previous semantics; the Codex failure taxonomy (`fabric/failures.py`) is not ported — the exception class name is recorded instead.
- [x] **K7f — trajectory metrics per asset** (2026-09-03). Parity confirmed: `runner._trajectory_metrics` already scores monotonicity per cell (`metrics.per_cell_monotonicity_violation_rate`, transition-weighted — mainline's deliberate 2026-08-30 choice, kept; Codex averaged assets unweighted) and the value-range check is a per-prediction fraction unaffected by grouping. New: `qc_gate` rejects every row of an ambiguous `(dataset, cell_id, cycle_index)` identity and counts them as `duplicate_cycle_identity` (the real processed data carries 64 such rows in 32 duplicated identities, all in uconn, now dropped at QC); the runner refuses to score a test fold with duplicate identities (unit fails with the reason).
- [x] **K7g — EOL fold metric omitted when unavailable** (2026-09-03): `metrics.EOL_THRESHOLD = 0.8`; `runner._point_metrics` skips `eol_region_error` for a fold whose test truth never reaches the EOL region, so the row is absent (leaderboard aggregates over the folds that have it and omits the key when none does) instead of a NaN.
- [x] **K7h — asset-level confidence intervals in `report.py`** (2026-09-03). `sohkernel.statistics.bootstrap_mean_ci` (seeded percentile bootstrap of the mean, 1000 resamples, 90 % by default) and `asset_confidence_intervals` resample the per-asset (per-cell) errors of `loco` / `forward` evidence; every leaderboard entry carries `confidence_intervals` (`status: available` with `intervals[mae|rmse|...] = {lower, upper, point}`, or `status: unavailable` with a reason for pooled protocols such as `lodo` / `lcho`). Schema updated; feeds Spec 04 §4. Reference: `f0e8d4d`.

Acceptance: new tests green; leaderboard `std` > 0 for stochastic models under 5 seeds; the leakage test of K7d passes; failed units visibly excluded — done 2026-09-03 for K7a–K7h.

---

### Task 8: Trusted-inference gate and prediction records into the kernel

- [x] `sohkernel/inference_gate.py::assert_manifest_label_safe(manifest_columns, table, *, target, model_name)` raises `FeatureSafetyError` (structural ban first, then the value probes) and returns the audit; `model_store.LoadedModel._ensure_label_safe` delegates and maps the error to `ModelStoreError` with the identical message (2026-09-03).
- [x] `sohkernel/records.py` owns the PredictionRecord shape (`PREDICTION_RECORD_FIELDS`, `validate_prediction_record`), the row builder `prediction_evidence_rows`, the identity derivations (`candidate_id`, `id_token`, `split_assignment_id`), `inference_metrics` / `extended_metrics` / `per_cell_monotonicity_violation` and `failed_record`; `sohbench.inference` imports them under its old private names, so records stay byte-identical. The evaluation-case schema (`curveRef` etc.) is workbench evidence and stays there. Tests: `tests/kernel/test_inference_gate.py`, `test_records.py`, and `test_inference_metrics.py` moved to the kernel (2026-09-03).

Acceptance: workbench inference tests green; direction test green — done 2026-09-03.

---

### Task 9: Convert phase scripts into tests, then retire them

- [x] `tests/workbench/test_release_gates.py` (2026-09-03): the measured chain reaches a signed pack, reload stability, tamper-evident pack + audit events, canonical digest, incompatible-scope and missing-evidence blocking — built on the workbench's measured-case pipeline with a fixture dataset. `tests/kernel/test_golden_invariants.py`: the nasa/B0018 LOCO fold scores exactly the cell's QC-passed cycles, the label never reaches the estimator, predictions stay in range and the per-cycle contributions reconstruct the fold MAE (skips without the real data).
- [x] `phase30_release_gate.py`, `scripts/run_phase30_release_gate.py` and its test deleted; the Windows CI job runs the suites directly instead of the phase 3.1.1 script (the script stays for local one-shot reports). **Deviation:** `phase22_golden.py` and `phase24_pipeline.py` are kept — reading them showed they are the workbench's measured-case release pipeline (model re-inference → case → evidence → retest → reconciliation → signed pack), exercised by fixture tests and by `scripts/build_phase22_golden_case.py` / `run_phase24_pipeline_acceptance.py`, not gate scripts; deleting them would remove frozen-workbench functionality rather than convert invariants.

Acceptance (revised 2026-09-03): the phase 3.0 gate module is gone and its invariants are tests; the two remaining `phase*` modules are the workbench release pipeline behind the golden-case scripts; CI green run pending push.

---

### Task 10: Freeze the workbench

- [x] `README.md` states that `sohbench` is frozen at `v0-workbench` (bug fixes and extraction shims only) and links ADR-001 (done with G5, 2026-09-02).
- [x] `docs/architecture/README.md` indexes the phase notes as historical (done with G5; phase 3.0 note updated in Task 9).

---

### Task 11: Kernel packaging

- [x] `sohkernel.__version__ = "0.1.0"`, extras `torch`, `[project.scripts] sohkernel = "sohkernel.cli:main"` (`load`, `preflight`, `score`; JSON output; blocked matrices exit 2), schemas shipped as package data. Both packages still ship in the one `sohbench` distribution — a separate `sohkernel` wheel follows once the API has been stable for a milestone (ADR-001) (2026-09-03).
- [x] `soh-agent/.venv` installs the kernel editable from the sibling checkout (`uv pip install -e ../SOH_eval`); `import sohkernel.api` and the `sohkernel` console script work from that venv (2026-09-03).

Acceptance: `python -c "import sohkernel, sohkernel.api"` works from the platform's virtualenv — verified 2026-09-03.

---

## Self-review checklist

- Direction test present from Task 1 and never skipped.
- Every task leaves the workbench behaviour identical (Tasks 2–6, 8) or is explicitly a behaviour change with tests (Task 7).
- No `D:\` literals; data root only via environment.
- Registries have one explicit entry point.
- Phase scripts are gone only after their checks exist as tests.
