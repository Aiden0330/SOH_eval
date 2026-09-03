# battery_SOH Curve Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate compact curve preview data from `battery_SOH` curve JSON and render it in the global diagnostics frontend.

**Architecture:** Extend `battery_dashboard_import.py` to create `battery_curve_previews.json` from sampled curve channels. Extend `dashboard-core.mjs` with a chart-ready view helper and render it in the existing battery diagnostics asset detail section with inline SVG.

**Tech Stack:** Python stdlib JSON/pathlib, pytest, vanilla JavaScript, SVG, Node test runner.

## Global Constraints

- Do not copy full `*_curves.json` by default.
- Keep preview data compact: at most 3 curves per dataset and 80 sampled points per curve.
- Do not add frontend chart dependencies.
- Keep standalone HTML portable with embedded preview JSON.

---

### Task 1: Importer Curve Preview Artifact

**Files:**
- Modify: `src/sohbench/battery_dashboard_import.py`
- Test: `tests/test_battery_dashboard_import.py`

**Interfaces:**
- Produces: `battery_curve_previews.json`
- Adds manifest entry: `optional_artifacts` and `artifact_files` include `battery_curve_previews.json`

- [x] **Step 1: Write failing pytest** that publishes a dashboard with numeric `xjtu_curves.json` and asserts `battery_curve_previews.json` exists with sampled points.
- [x] **Step 2: Run** `pytest tests/test_battery_dashboard_import.py::test_publish_generates_curve_preview_artifact -q` and confirm it fails because the preview artifact is missing.
- [x] **Step 3: Implement preview generation helpers for normal charge curves and ESS detection curves.**
- [x] **Step 4: Re-run** `pytest tests/test_battery_dashboard_import.py::test_publish_generates_curve_preview_artifact -q`.

### Task 2: Frontend Core Preview View

**Files:**
- Modify: `frontend/app/dashboard-core.mjs`
- Test: `frontend/app/dashboard-core.test.mjs`

**Interfaces:**
- Consumes: `diagnostics.curvePreviews.datasets[]`
- Produces: `batteryCurvePreviewView(diagnostics, selectedDataset)`

- [x] **Step 1: Write failing Node test** for dataset options, selected dataset fallback, and chart-ready series labels.
- [x] **Step 2: Run** `node --test frontend\app\dashboard-core.test.mjs`.
- [x] **Step 3: Implement `batteryCurvePreviewView`.**
- [x] **Step 4: Re-run** `node --test frontend\app\dashboard-core.test.mjs`.

### Task 3: Frontend Rendering And Export

**Files:**
- Modify: `frontend/app/app.js`
- Modify: `frontend/app/styles.css`
- Modify: `frontend/app/chinese-ui.test.mjs`
- Generated: `frontend/app/data/profiles/battery-soh-diagnostics/battery_curve_previews.json`
- Generated: `dist/soh-eval-standalone.html`
- Generated: `docs/exports/soh-eval-standalone.html`

**Interfaces:**
- Consumes: `batteryCurvePreviewView`
- Produces: dataset-selectable inline SVG preview in global diagnostics.

- [x] **Step 1: Add failing static assertions** for `battery_curve_previews.json`, `batteryCurvePreviewView`, `renderBatteryCurvePreview`, and `.battery-curve-preview`.
- [x] **Step 2: Run** `node --test frontend\app\dashboard-core.test.mjs frontend\app\chinese-ui.test.mjs`.
- [x] **Step 3: Render the preview panel and SVG without external dependencies.**
- [x] **Step 4: Re-import `D:\battery_SOH\dashboard\data` and rebuild standalone HTML.**
- [x] **Step 5: Verify with Node/Python tests and browser visibility checks.**
