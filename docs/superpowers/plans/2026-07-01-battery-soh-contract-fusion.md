# battery_SOH Contract Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import `D:\battery_SOH` diagnostic dashboard artifacts into `SOH_eval` profiles and display their summary in the global diagnostics tab.

**Architecture:** Keep `SOH_eval` as the platform shell. Add a small import/publish layer for `battery_SOH/dashboard/data` JSON files and a lightweight optional frontend reader. Do not merge raw datasets or replace benchmark leaderboard artifacts in this slice.

**Tech Stack:** Python stdlib, existing `sohbench.frontend_publish`, vanilla JavaScript modules, existing Node tests.

## Global Constraints

- Use TDD: write failing tests before production changes.
- Do not move or copy raw data from `D:\battery_SOH\data`.
- Imported dashboard artifacts must use `battery_dashboard_` prefixes.
- Curve artifacts are optional and excluded by default.
- Existing `leaderboard.json`, `dataset_leaderboard.json`, `run_config.json`, `model_catalog.json`, and inference artifacts must remain unchanged.

---

### Task 1: Diagnostic Artifact Import Contract

**Files:**
- Create: `src/sohbench/battery_dashboard_import.py`
- Create: `tests/test_battery_dashboard_import.py`
- Modify: `src/sohbench/evaluation_service.py`
- Modify: `src/sohbench/api_server.py`
- Create: `scripts/import_battery_dashboard_profile.py`

**Interfaces:**
- Produces: `stage_battery_dashboard_artifacts(source_dir: Path, target_dir: Path, include_curves: bool = False) -> list[str]`
- Produces: `publish_battery_dashboard_profile(frontend_data_dir: Path, source_dir: Path, include_curves: bool = False, profile_name: str = "battery-soh-diagnostics") -> dict`
- Produces API endpoint: `POST /api/import/battery-dashboard`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Implement minimal importer and endpoint**
- [ ] **Step 4: Verify GREEN**

### Task 2: Frontend Diagnostic Summary Reader

**Files:**
- Modify: `frontend/app/dashboard-core.mjs`
- Modify: `frontend/app/dashboard-core.test.mjs`
- Modify: `frontend/app/app.js`
- Modify: `frontend/app/index.html`
- Modify: `frontend/app/styles.css`
- Modify: `frontend/app/chinese-ui.test.mjs`

**Interfaces:**
- Produces: `batteryDiagnosticSummary(diagnostics: object) -> object`
- Consumes optional artifacts on `DATA.batteryDiagnostics`

- [ ] **Step 1: Write failing frontend tests**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Implement summary helper and render section**
- [ ] **Step 4: Verify GREEN**

### Task 3: Verification

**Files:**
- No production files unless fixes are required.

- [ ] Run Python targeted tests.
- [ ] Run Node frontend tests.
- [ ] Import the real `D:\battery_SOH\dashboard\data` artifacts to a profile.
- [ ] Open/reload the frontend and verify the diagnostics profile is visible.
