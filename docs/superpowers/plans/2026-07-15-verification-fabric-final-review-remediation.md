# Verification Fabric Final Review Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:subagent-driven-development. Every production change must be preceded by a focused failing test whose failure is recorded in the report.

**Goal:** Close every Critical/Important finding from the Phase 0 whole-branch review without weakening legacy compatibility, evidence identity, or the existing Chinese workbench.

**Architecture:** Preserve the current Phase 0 contracts and add four boundary controls: asset-local inference validation, execution and artifact isolation, transactional publication, and mode-local frontend state. The historical dirty workspace at `D:\SOH_eval` and source dashboard at `D:\battery_SOH` are read-only references; implementation belongs only in the isolated feature worktree.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn-compatible estimators, pytest, vanilla JavaScript ES modules, Node assert tests, static HTML/CSS.

## Global Constraints

- Do not modify, reset, stage, or clean the original dirty worktree at `D:\SOH_eval`.
- Keep training and inference terminal states compatible with `succeeded`, `partial`, and `failed`.
- Preserve legacy Pickle and trusted-plugin workflows while making evidence identity content-addressed.
- Published artifacts must either switch as one validated snapshot or leave the last-known-good snapshot untouched.
- Chinese UI text must be UTF-8 and must not use `?` placeholders for labels.
- Use no new runtime dependency unless the repository already declares it.
- All fixes follow RED -> GREEN -> focused regression -> full-suite verification.

---

### Task 11: Asset-Local Inference Evidence

**Files:**
- Modify: `src/sohbench/inference.py`
- Modify only if boundary validation requires it: `src/sohbench/trusted_plugins.py`
- Test: `tests/test_inference.py`
- Test: `tests/test_trusted_plugins.py`

**Interfaces:**
- `run_inference_benchmark(...)` continues to return one result record per model-dataset combination.
- Prediction arrays and optional interval arrays must be finite one-dimensional arrays of exactly `n_samples`.
- Trajectory metrics group by `(dataset, cell_id)`, sort by normalized cycle, reject duplicate asset-cycle identities, and average eligible asset rates equally.

- [ ] Add a failing regression proving two independently monotone cells produce a zero violation rate and duplicate `(dataset, cell_id, cycle)` rows are rejected.
- [ ] Run the focused test and record the expected RED failure caused by cross-cell concatenation.
- [ ] Add failing regressions for `(n, 1)`, wrong-length, NaN, and infinity predictions/intervals; assert the combination fails instead of publishing metrics.
- [ ] Run the focused tests and record the expected RED failures caused by broadcasting or non-finite success metrics.
- [ ] Implement the smallest shared array validator and asset-local trajectory calculation.
- [ ] Run `pytest tests/test_inference.py tests/test_trusted_plugins.py -q` and record GREEN output.

### Task 12: Reproducible Execution and Content-Bound Assets

**Files:**
- Modify: `src/sohbench/runner.py`
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `src/sohbench/model_store.py`
- Modify as required: `src/sohbench/fabric/specs.py`
- Modify as required: `src/sohbench/fabric/evidence.py`
- Test: `tests/test_runner.py`
- Test: `tests/test_evaluation_service.py`
- Test: `tests/test_model_store.py`
- Test as required: `tests/fabric/test_specs.py`, `tests/fabric/test_evidence.py`

**Interfaces:**
- A candidate identity must bind the exact model/plugin/manifest/feature bytes used by the run through canonical SHA-256 references.
- Replacing bytes behind the same display name must change `CandidateSpec.digest` and the enclosing `RunSpec` identity.
- Concurrent jobs with different seeds must not share process-global NumPy or Python random state during candidate execution.
- Training model publication must use run-scoped immutable names or directories; a later identical configuration must not overwrite a prior run's catalog entry or bytes.
- Artifact publication writes to a temporary sibling and atomically renames only after Pickle, manifest, hashes, and catalog entry are complete.

- [ ] Add a controlled interleaving test proving seeds `111` and `222` retain their own expected streams.
- [ ] Add a failing identity test proving different artifact SHA values under one display name produce different candidate and run digests, and evidence verification rejects tampering.
- [ ] Add a failing two-run test proving identical training configurations retain two immutable model sets with their original `created_by_run` values.
- [ ] Record RED output for all three defects before production edits.
- [ ] Implement private RNG propagation where supported and serialize the complete legacy global-RNG execution unit where isolation cannot be injected.
- [ ] Bind canonical artifact and feature digests into candidate/evidence semantics without breaking legacy comparison-key-only packs.
- [ ] Publish run-scoped models atomically and guard catalog updates with the existing service synchronization boundary or a dedicated lock.
- [ ] Run the focused Python tests above and record GREEN output.

### Task 13: Transactional Dashboard and Portable Publication

**Files:**
- Modify: `scripts/sync_battery_dashboard_native.py`
- Modify: `scripts/package_portable_frontend.py`
- Modify: `src/sohbench/frontend_publish.py`
- Test: `tests/test_api_server.py`
- Test: `tests/test_standalone_frontend.py`
- Create or modify a focused publication test module if separation improves clarity.

**Interfaces:**
- Native dashboard sync validates every manifest reference before changing the live target.
- Portable package construction happens in a sibling temporary directory and atomically replaces the destination only after complete validation.
- Frontend JSON publication uses one lock and a prepared snapshot so readers never observe cross-run mixtures.
- On any copy, validation, or rename failure, the prior target remains byte-for-byte readable and no temporary publication is exposed as live.

- [ ] Add failing last-known-good tests for missing source manifest assets, injected copy failure, and concurrent profile publication.
- [ ] Record RED output showing the prior target is currently changed or mixed.
- [ ] Implement validate -> stage -> fsync/close -> atomic directory switch with cleanup of only owned temporary siblings.
- [ ] Run focused publication, API, and standalone tests and record GREEN output.

### Task 14: Frontend Mode Isolation and Complete Delivery Surface

**Files:**
- Modify: `frontend/app/dashboard-core.mjs`
- Modify: `frontend/app/dashboard-core.test.mjs`
- Modify: `frontend/app/app.js`
- Modify: `frontend/app/chinese-ui.test.mjs`
- Modify: `frontend/app/index.html`
- Modify: `frontend/app/styles.css`
- Create: `frontend/app/native-dashboard.test.mjs`
- Create/update generated delivery assets under `frontend/app/battery-dashboard/` only from validated source input.

**Interfaces:**
- Training and inference keep separate active job records and poll tokens; stale responses cannot replace the current mode's state.
- A second launch for the same mode is disabled while its active job is non-terminal, while the other mode remains independently observable.
- `compatible_only` produces an explicit model-dataset pair matrix (or split requests) so one incompatible model never removes another model's valid pair.
- Frontend inference summary uses the backend summary when present; local fallback filters non-finite values and applies deterministic tie keys `(metric, model, dataset, featureset, estimator)`.
- Add a Chinese `SOH 完整看板` navigation item that opens the native dashboard and a Chinese delivery operations/report surface for sync, portable package, and standalone package APIs.
- Real `frontend/app` packaging tests must pass with the native dashboard present; no source label may be `?`.

- [ ] Add failing Node tests for stale poll rejection, same-mode duplicate launch prevention, pairwise compatibility, order-independent ties, native tab markup, Chinese labels, and real-app portable packaging.
- [ ] Run both Node suites and the focused Python delivery tests; record expected RED failures.
- [ ] Implement mode-local state/tokens, pairwise launch payloads or split orchestration, stable summaries, and the native dashboard/delivery UI.
- [ ] Generate/copy native dashboard assets transactionally from `D:\battery_SOH\dashboard` or the read-only reference at `D:\SOH_eval\frontend\app\battery-dashboard`; never edit either reference.
- [ ] Run `node frontend/app/dashboard-core.test.mjs`, `node frontend/app/chinese-ui.test.mjs`, and `node frontend/app/native-dashboard.test.mjs`; record GREEN output.

### Task 15: Integrated Verification and Re-review

**Files:**
- Update: `.superpowers/sdd/progress.md` (ignored ledger)
- Update: `.superpowers/sdd/final-remediation-report.md` (ignored report)

- [ ] Run focused Phase 0, inference, service, publication, and frontend suites.
- [ ] Run the complete Python suite and all frontend Node suites.
- [ ] Run `git diff --check` and confirm the isolated worktree is clean after commit.
- [ ] Generate a new merge-base-to-HEAD review package.
- [ ] Dispatch two independent read-only reviewers across the full branch.
- [ ] Repeat one TDD fix wave for any remaining Critical/Important finding; finish only after both reviewers approve.

