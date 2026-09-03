# Frontend Sync And Package Actions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backend APIs and a Chinese frontend operations panel for one-click SOH dashboard sync and offline package generation.

**Architecture:** Existing script functions remain the source of truth. `EvaluationService` exposes small service methods, `api_server` maps POST endpoints, and `frontend/app/app.js` renders/updates one diagnostics operations card.

**Tech Stack:** Python stdlib HTTP server, existing `EvaluationService`, vanilla JS frontend, Node static tests, pytest API tests.

## Global Constraints

- ESS detection assets stay excluded by default.
- No shell command execution from the backend; call Python functions directly.
- UI copy is Chinese.
- Default source is `D:\battery_SOH\dashboard`.
- Default outputs are `dist/soh-eval-portable` and `dist/soh-eval-standalone.html`.

---

### Task 1: Backend API Actions

**Files:**
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `src/sohbench/api_server.py`
- Test: `tests/test_api_server.py`

**Interfaces:**
- Produces: `EvaluationService.sync_native_dashboard(...) -> dict`
- Produces: `EvaluationService.package_portable_frontend(...) -> dict`
- Produces: `EvaluationService.package_standalone_frontend(...) -> dict`

- [ ] Write failing API tests for the three POST endpoints using temporary dashboard/app fixtures.
- [ ] Run the API tests and confirm the endpoints return 404 or missing method failures.
- [ ] Add service methods that import and call script functions directly.
- [ ] Add POST routes in `api_server.create_handler`.
- [ ] Run the API tests and confirm they pass.

### Task 2: Frontend Operations Panel

**Files:**
- Modify: `frontend/app/index.html`
- Modify: `frontend/app/app.js`
- Modify: `frontend/app/styles.css`
- Test: `frontend/app/chinese-ui.test.mjs`

**Interfaces:**
- Consumes: `POST /api/sync/native-dashboard`
- Consumes: `POST /api/package/portable`
- Consumes: `POST /api/package/standalone`

- [ ] Write failing static tests for the new diagnostics card and endpoints.
- [ ] Run Node tests and confirm they fail on missing UI/actions.
- [ ] Add the card markup to Global Diagnostics.
- [ ] Add `PACKAGE_STATUS`, render function, and three button handlers.
- [ ] Add compact CSS for the operations card.
- [ ] Run Node tests and confirm they pass.

### Task 3: Integrated Verification

**Files:**
- Modify only if tests reveal a gap.

- [ ] Run focused pytest API/script tests.
- [ ] Run focused Node frontend tests.
- [ ] Rebuild standalone/portable outputs if required.
- [ ] Optionally refresh the browser URL with a cache-busting version.
