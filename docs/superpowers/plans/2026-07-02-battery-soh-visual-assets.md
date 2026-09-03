# battery_SOH Visual Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate lightweight battery_SOH algorithm/detail assets and curve indexes for the frontend diagnostics profile.

**Architecture:** Extend `battery_dashboard_import.py` with pure helper functions that summarize optional source docs and reports. Keep frontend runtime changes minimal because optional JSON loading and summary display already exist.

**Tech Stack:** Python stdlib JSON/pathlib, pytest, vanilla JS frontend tests.

## Global Constraints

- Do not load or copy large curve JSON unless `include_curves=true`.
- Keep generated optional JSON small and schema-versioned.
- Preserve existing profile import behavior and file names.
- Do not add runtime dependencies.

---

### Task 1: Importer Optional Asset Generation

**Files:**
- Modify: `src/sohbench/battery_dashboard_import.py`
- Test: `tests/test_battery_dashboard_import.py`

**Interfaces:**
- Produces: `battery_algorithm_design.json`, `battery_model_comparisons.json`, `battery_feature_windows.json`, and `manifest["curve_artifacts"]`.

- [x] **Step 1: Write failing pytest** covering optional report/doc generation and curve index.
- [x] **Step 2: Run** `pytest tests/test_battery_dashboard_import.py -q` and confirm failure.
- [x] **Step 3: Implement pure helper functions and call them from `publish_battery_dashboard_profile()`.**
- [x] **Step 4: Re-run** `pytest tests/test_battery_dashboard_import.py -q`.

### Task 2: Frontend Recognition Regression

**Files:**
- Modify: `frontend/app/dashboard-core.test.mjs`
- Modify: `frontend/app/chinese-ui.test.mjs`

**Interfaces:**
- Consumes: generated optional JSON file names and `curve_artifacts`.
- Produces: stable tests proving the frontend recognizes algorithm/detail assets.

- [x] **Step 1: Add failing assertions for `curve_artifacts` support.**
- [x] **Step 2: Run** `node --test frontend\app\dashboard-core.test.mjs frontend\app\chinese-ui.test.mjs`.
- [x] **Step 3: Adjust frontend helper only if needed.**
- [x] **Step 4: Re-run Node tests.**

### Task 3: Re-import And Export

**Files:**
- Generated: `frontend/app/data/profiles/battery-soh-diagnostics/*.json`
- Generated: `dist/soh-eval-standalone.html`
- Generated: `docs/exports/soh-eval-standalone.html`

**Interfaces:**
- Consumes: `D:\battery_SOH\dashboard\data`, `D:\battery_SOH\docs\design`, and `D:\battery_SOH\experiments\reports`.
- Produces: updated frontend profile and standalone HTML.

- [x] **Step 1: Run importer against `D:\battery_SOH\dashboard\data`.**
- [x] **Step 2: Rebuild standalone HTML.**
- [x] **Step 3: Browser-check diagnostics asset summary.**
