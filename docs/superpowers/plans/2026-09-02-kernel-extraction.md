# Kernel extraction — `sohbench` → `sohkernel` (+ frozen workbench)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or superpowers:executing-plans, task by task. Steps use `- [ ]` checkboxes. Every task ends with the full suite green (`pytest -q`, 414+ tests) and the direction test green.

**Goal:** carve the durable kernel out of `src/sohbench` into `src/sohkernel` without changing behaviour, so that the workbench keeps working and the new agent platform (`projects/soh-agent`) can depend on the kernel alone. See ADR-001 (`docs/adr/2026-09-02-ADR-001-kernel-extraction-and-agent-platform.md`).

**Rules (locked):**
- Behaviour-preserving moves only. Any behaviour change (forward-chaining, seed injection) is its own task with its own tests.
- `sohkernel` never imports `sohbench`. Enforced by `tests/kernel/test_import_direction.py` from Task 1 onward.
- Every moved module leaves a shim in `sohbench/<name>.py` (`from sohkernel.<name> import *  # noqa` plus explicit names used by the workbench) so no workbench file changes in the same task.
- Registries (`registry.py`) move to the kernel; registration stays explicit (`sohkernel.estimators.register_all()`), not an import side effect.
- Data root only via `SOHBENCH_DATA_DIR` (Task 0); no `D:\` literals remain in code or tests.

---

### Task 0: Housekeeping and baseline

**Files:** `.gitignore`, `src/sohbench/paths.py`, `.github/workflows/*.yml`, `README.md`

- [ ] Commit or discard the uncommitted `frontend/app/data/*.json` changes (owner decision; they are generated artifacts, so `git checkout --` is acceptable if the last published run is already in `runs/`).
- [ ] Decide the base branch (`feat/phase2-dl` vs `master`) and tag the baseline: `git tag v0-workbench` (owner runs this).
- [ ] `paths.py`: single `data_root()` that reads `SOHBENCH_DATA_DIR`, falls back to `<repo>/data`; grep and replace every `D:\\` / `D:/` literal in `src/`, `scripts/`, `tests/` (`tests/test_real_data_smoke.py` uses `data_root()` and skips when absent).
- [ ] CI: add a job that runs the full `pytest -q` on Linux and macOS, not only `scripts/run_phase311_production_stability.py`.
- [ ] README: replace the Windows quick start with `python -m venv .venv && .venv/bin/pip install -e ".[dev]" && pytest -q`.

Acceptance: suite green on macOS and in CI; `grep -rn "D:\\\\" src tests scripts` returns nothing.

---

### Task 1: Package skeleton and the direction test

**Files:** `src/sohkernel/__init__.py`, `pyproject.toml`, `tests/kernel/__init__.py`, `tests/kernel/test_import_direction.py`

- [ ] `pyproject.toml`: `[tool.setuptools.packages.find] where = ["src"]` already finds both packages; add `[project.optional-dependencies] torch = ["torch"]`; version `sohkernel` independently later (Task 11).
- [ ] `src/sohkernel/__init__.py`: `__version__ = "0.1.0.dev0"`, no imports.
- [ ] Direction test: walk `src/sohkernel/**/*.py` with `ast`, fail on any `import sohbench` / `from sohbench` / `import sohagent`.

Acceptance: `pytest tests/kernel -q` passes (1 test).

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

- [ ] `load(datasets, *, data_dir=None, qc=True) -> (table, meta, QCReport)` (wraps `DatasetLoader.resolve_datasets` + `load`).
- [ ] `preflight(request) -> Preflight` — the label-safety, window-DOD, coverage and protocol-feasibility checks now inside `EvaluationService.training_preflight`, without any store/job access; returns `ready`, per-combination `reasons` and `warnings`.
- [ ] `transform(table, meta, featuresets) -> table` — `store.apply_feature_transforms` behind the label-isolation boundary (`feature_safety.apply_transform_without_label`).
- [ ] `score(task, table, meta, protocols, estimators, featuresets, seeds) -> DataFrame` — thin wrapper over `BenchmarkRunner.run`.
- [ ] `verdicts(metrics, task) -> list[Verdict]` — `acceptance.evaluate_verdicts`.
- [ ] `write_artifacts(results, qc_report, task, config, out_dir) -> path` — `report.write_artifacts`.
- [ ] `PROTOCOLS = {"loco": LeaveOneCellOut, "lodo": LeaveOneDatasetOut, "lcho": LeaveOneChemistryOut}` exported from `sohkernel.protocols`.
- [ ] Tests: `preflight` rejects a request containing `charge_capacity_max`; blocks a ≥ 90 % window; warns at > 60 %; `transform` never receives the label column (assert via a spy featureset).

Acceptance: `evaluation_service.training_preflight` delegates to `sohkernel.api.preflight` (behaviour identical; existing service tests green).

---

### Task 7: Kernel-only behaviour additions (P0 from the report)

Separate commits, each with tests.

- [ ] `protocols.ForwardChaining(train_frac=0.6, min_train_cycles=20)`: per cell, train on the earliest fraction, test on the rest; `held_out_key = "<dataset>/<cell>@forward"`; registered as `"forward"` in `PROTOCOLS`.
- [ ] Seed injection: `BenchmarkRunner` passes `seed` to estimators exposing `set_seed(seed)` or a `random_state` constructor argument; add `set_seed` to `gbdt`, `extra_trees`, `mlp`, torch estimators; test that two seeds give different predictions for `mlp` and identical for `ridge`.
- [ ] `adaptation.calibration_split(..., strategy="earliest")` becomes the default in kernel API calls; `"spread"` stays available but the returned provenance carries `calibration_split="spread"` explicitly (already recorded — verify).

Ports from the archived Codex line (`archive/verification-fabric-phase0`; read the reference with `git show archive/verification-fabric-phase0:<path>`). Each is a behaviour change with its own tests; none is a cherry-pick (all conflict — see ADR-001).

- [ ] **K7b reference:** seed propagation as in `d6b1efa` / `f86b753` — `_instantiate_estimator(factory, seed)` passes `random_state` when the constructor accepts it, else calls `set_seed`, else sets the attribute; a legacy random scope seeds `random` and `numpy` per fold.
- [ ] **K7d — fit feature pipelines inside folds.** Any fitted transform (scalers, CORAL statistics, window bounds chosen from data) is fitted on the training fold only and applied to the test fold; verify current mainline behaviour first (`store.apply_feature_transforms` runs before `BenchmarkRunner.run`). Reference: `1fdf6ed`, `320b250`, `src/sohbench/fabric/feature_pipeline.py`, `tests/fabric/test_runner_leakage.py`. Test: a transform that peeks at test rows changes the score; the isolated version does not.
- [ ] **K7e — explicit failure and partial outcomes.** A failed fold/seed produces an `outcome` record (`ok | failed | partial`) with the error class; failed units never enter leaderboard aggregates or evidence, and the leaderboard entry states how many units failed. Reference: `45ed3a3`, `32d6884`, `fabric/outcomes.py`, `fabric/failures.py`.
- [ ] **K7f — trajectory metrics per asset.** Monotonicity and value-range metrics are aggregated per cell/asset (mainline already sorts per cell since 2026-08-30 — confirm parity) and ambiguous cycle identities (duplicate `cycle_index` within a cell) are rejected at QC. Reference: `4a29255`, `b65a429`.
- [ ] **K7g — EOL fold metric omitted when unavailable** (no cell reaches the EOL threshold in the fold) instead of NaN-polluting aggregates. Reference: `8ecb62a`.
- [ ] **K7h — asset-level confidence intervals in `report.py`** (cell-level bootstrap; feeds Spec 04 §4). Reference: `f0e8d4d`.

Acceptance: new tests green; leaderboard `std` > 0 for stochastic models under 5 seeds; the leakage test of K7d passes; failed units visibly excluded.

---

### Task 8: Trusted-inference gate and prediction records into the kernel

- [ ] Move `model_store.LoadedModel._ensure_label_safe` logic into `sohkernel/inference_gate.py` as `assert_manifest_label_safe(manifest_columns, table)`; `model_store` calls it.
- [ ] Move the prediction-record dataclasses/schema (`inference.py`, `schemas/evaluation-case.schema.json` parts that describe predictions) into `sohkernel/records.py`; `inference.py` imports them.

Acceptance: workbench inference tests green; direction test green.

---

### Task 9: Convert phase scripts into tests, then retire them

- [ ] For each check in `phase30_release_gate.py` and `phase22_golden.py`, write a test: reload stability, audit digests, incompatible-scope blocking, missing-evidence blocking → `tests/workbench/test_release_gates.py`; golden numeric invariants → `tests/kernel/test_golden_invariants.py` (skip when data absent).
- [ ] Delete `phase22_golden.py`, `phase24_pipeline.py`, `phase30_release_gate.py` and their `scripts/run_phase*.py` wrappers; update `.github/workflows/phase311-production-stability.yml` to run the tests instead.

Acceptance: no `phase*` module in `src/`; CI green.

---

### Task 10: Freeze the workbench

- [ ] `README.md`: state that `sohbench` (workbench) is frozen at `v0-workbench` and imports `sohkernel`; link ADR-001.
- [ ] `docs/architecture/README.md` index: mark phase docs as historical.

---

### Task 11: Kernel packaging

- [ ] `pyproject.toml`: expose `sohkernel` with its own version (`0.1.0`), extras `torch`, and `[project.scripts] sohkernel = "sohkernel.cli:main"` (a minimal CLI: `load`, `preflight`, `score`).
- [ ] `pip install -e .` from `projects/soh-agent` via `[tool.uv.sources]`/path dependency (see the platform repo's `pyproject.toml`).

Acceptance: `python -c "import sohkernel, sohkernel.api"` works from the platform's virtualenv.

---

## Self-review checklist

- Direction test present from Task 1 and never skipped.
- Every task leaves the workbench behaviour identical (Tasks 2–6, 8) or is explicitly a behaviour change with tests (Task 7).
- No `D:\` literals; data root only via environment.
- Registries have one explicit entry point.
- Phase scripts are gone only after their checks exist as tests.
