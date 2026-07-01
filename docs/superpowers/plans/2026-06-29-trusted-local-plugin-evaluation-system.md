# Trusted Local Plugin Evaluation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trusted local plugin mode that lets the Chinese dashboard upload datasets, estimator plugins, and feature-set plugins, then trigger a benchmark run that refreshes frontend artifacts.

**Architecture:** Add a stdlib JSON HTTP backend over focused service modules, keeping the existing benchmark runner and report/artifact pipeline unchanged. The frontend gains a compact workbench panel that talks to the backend and reloads its existing dashboard JSON after a successful job.

**Tech Stack:** Python 3.11+, stdlib `http.server`, pandas, numpy, scikit-learn, existing sohbench registry/runner/report code, vanilla JavaScript modules.

## Global Constraints

- Uploaded Python plugins are trusted local code and run in-process.
- Dataset uploads must validate `dataset`, `cell_id`, `cycle_index`, and `soh`.
- Backend must work without FastAPI/uvicorn or new runtime dependencies.
- Existing dashboard JSON shape must remain compatible.
- Do not revert unrelated dirty worktree changes.

---

### Task 1: Resource Registry And Plugin Loader

**Files:**
- Create: `src/sohbench/trusted_plugins.py`
- Test: `tests/test_trusted_plugins.py`

**Interfaces:**
- Produces: `TrustedPluginStore(base_dir, data_dir)`, `upload_dataset(name, filename, payload, replace=False)`, `upload_estimator_plugin(name, filename, payload, replace=False)`, `upload_featureset_plugin(name, filename, payload, replace=False)`, `apply_feature_transforms(table, meta, featuresets)`, `catalog()`

- [ ] Write failing tests for dataset upload, estimator plugin loading, and feature-set transform registration.
- [ ] Run `pytest tests/test_trusted_plugins.py -q` and confirm failures are due to missing module.
- [ ] Implement minimal plugin store and smoke tests.
- [ ] Run `pytest tests/test_trusted_plugins.py -q` and confirm pass.

### Task 2: Evaluation Job Service

**Files:**
- Create: `src/sohbench/evaluation_service.py`
- Test: `tests/test_evaluation_service.py`

**Interfaces:**
- Consumes: `TrustedPluginStore`
- Produces: `EvaluationService.create_job(payload)`, `EvaluationService.run_job_inline(job_id)`, `EvaluationService.get_job(job_id)`, `EvaluationService.catalog()`

- [ ] Write failing tests for inline benchmark execution and frontend artifact sync.
- [ ] Run `pytest tests/test_evaluation_service.py -q` and confirm missing service failure.
- [ ] Implement JSON job store, inline runner, and background thread launcher.
- [ ] Run `pytest tests/test_evaluation_service.py -q` and confirm pass.

### Task 3: JSON HTTP API

**Files:**
- Create: `src/sohbench/api_server.py`
- Test: `tests/test_api_server.py`

**Interfaces:**
- Consumes: `EvaluationService`
- Produces: `create_handler(service)`, `serve(host, port, service)`, routes `/api/health`, `/api/catalog`, `/api/upload/dataset`, `/api/upload/algorithm`, `/api/upload/featureset`, `/api/jobs`, `/api/jobs/{id}`

- [ ] Write failing tests using `ThreadingHTTPServer` and `urllib.request`.
- [ ] Run `pytest tests/test_api_server.py -q` and confirm missing API failure.
- [ ] Implement CORS, JSON parsing, route handling, and error responses.
- [ ] Run `pytest tests/test_api_server.py -q` and confirm pass.

### Task 4: Frontend Workbench

**Files:**
- Modify: `frontend/app/index.html`
- Modify: `frontend/app/app.js`
- Modify: `frontend/app/styles.css`
- Modify: `frontend/app/dashboard-core.mjs`
- Modify: `frontend/app/dashboard-core.test.mjs`

**Interfaces:**
- Consumes: backend routes from Task 3
- Produces: Chinese upload controls, catalog checkboxes, job launch, polling, dashboard refresh

- [ ] Write failing JS tests for catalog normalization and job payload construction.
- [ ] Run `node frontend/app/dashboard-core.test.mjs` and confirm expected failure.
- [ ] Implement workbench helpers and UI wiring.
- [ ] Run `node frontend/app/dashboard-core.test.mjs` and `node frontend/app/chinese-ui.test.mjs`.

### Task 5: Public Dataset And Plugin Validation

**Files:**
- Create: `scripts/validate_trusted_plugin_workflow.py`
- Create under work/data as needed: downloaded public data and generated plugin files

**Interfaces:**
- Consumes: HTTP API or `EvaluationService`
- Produces: a new benchmark run and synced frontend artifacts containing the new dataset, estimator, and featureset

- [ ] Search for a public battery degradation dataset not in `calce_cs2`, `hust`, `nasa`, `oxford`, `snl_lfp`, or `uconn`.
- [ ] Download a small public file or archive and convert it into sohbench rows.
- [ ] Generate a new estimator plugin and feature-set plugin.
- [ ] Trigger the evaluation through the new backend path.
- [ ] Verify `frontend/app/data/run_config.json` and `leaderboard.json` include the new resources.

### Task 6: Local Server Startup And Smoke Verification

**Files:**
- No required source file changes unless smoke reveals a bug.

**Interfaces:**
- Consumes: `python -m sohbench.api_server` and existing static frontend server
- Produces: running frontend/backend URLs and browser smoke evidence

- [ ] Start backend on `http://localhost:8001`.
- [ ] Start or reuse frontend on `http://localhost:8000`.
- [ ] Use browser automation to assert Chinese workbench controls, catalog rows, and dashboard tables render.
- [ ] Run focused Python and JS test suites before final response.
