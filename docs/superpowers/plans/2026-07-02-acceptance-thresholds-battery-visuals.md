# Acceptance Thresholds And Battery Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add preset-aware acceptance details and prepare optional battery_SOH algorithm/visualization data integration.

**Architecture:** Keep acceptance recomputation in `frontend/app/dashboard-core.mjs`; keep DOM rendering and drawer state in `frontend/app/app.js`; reuse existing drawer styling in `frontend/app/styles.css`. No new backend endpoint is required for this slice.

**Tech Stack:** Vanilla JavaScript ES modules, static JSON artifacts, Node test runner, Python pytest for standalone export.

## Global Constraints

- Preserve Chinese UI copy.
- Do not add runtime dependencies.
- Do not increase page height; use drawers and fixed-height scrolling panels.
- Keep training acceptance and inference success as separate concepts.
- Existing dirty worktree changes must not be reverted.

---

### Task 1: Acceptance Core

**Files:**
- Modify: `frontend/app/dashboard-core.mjs`
- Test: `frontend/app/dashboard-core.test.mjs`

**Interfaces:**
- Produces: `ACCEPTANCE_PRESETS`, `acceptanceDetailSummary({ rows, datasetRows, preset })`, and enhanced `acceptanceKpiSummary({ mode, rows, datasetRows, inference, preset })`.

- [x] **Step 1: Write failing tests** for strict/engineering/exploratory preset counts and failed-dimension details.
- [x] **Step 2: Run** `node --test frontend\app\dashboard-core.test.mjs` and confirm failure before implementation.
- [x] **Step 3: Implement core functions** in `dashboard-core.mjs`.
- [x] **Step 4: Re-run** `node --test frontend\app\dashboard-core.test.mjs`.

### Task 2: Frontend Controls And Drawer

**Files:**
- Modify: `frontend/app/app.js`
- Modify: `frontend/app/index.html`
- Modify: `frontend/app/styles.css`
- Test: `frontend/app/chinese-ui.test.mjs`

**Interfaces:**
- Consumes: `ACCEPTANCE_PRESETS`, `acceptanceDetailSummary`, `acceptanceKpiSummary`.
- Produces: threshold preset selector, `验收详情` button, and `acceptance-detail-drawer`.

- [x] **Step 1: Write failing static tests** for preset selector, drawer DOM ids, and script version.
- [x] **Step 2: Run** `node --test frontend\app\chinese-ui.test.mjs` and confirm failure.
- [x] **Step 3: Implement UI rendering and event wiring**.
- [x] **Step 4: Re-run** `node --test frontend\app\dashboard-core.test.mjs frontend\app\chinese-ui.test.mjs`.

### Task 3: battery_SOH Optional Data Hooks

**Files:**
- Modify: `frontend/app/app.js`
- Modify: `frontend/app/dashboard-core.mjs`
- Test: `frontend/app/dashboard-core.test.mjs`

**Interfaces:**
- Produces: optional summary for `algorithm_details`, `soh_curves`, and `feature_visuals` if present in active profile diagnostics.

- [x] **Step 1: Write failing tests** for optional battery visual asset summary.
- [x] **Step 2: Implement the summary without requiring files to exist**.
- [x] **Step 3: Surface availability in acceptance/details or diagnostics notes**.

### Task 4: Verification And Export

**Files:**
- Rebuild: `dist/soh-eval-standalone.html`
- Rebuild: `docs/exports/soh-eval-standalone.html`

- [x] **Step 1: Run Node tests.**
- [x] **Step 2: Run standalone pytest.**
- [x] **Step 3: Rebuild standalone HTML exports.**
- [x] **Step 4: Browser-check training and inference KPI behavior.**
