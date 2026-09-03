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

**Global Constraints:**
- Use TDD: write failing tests before production changes.
- Imported dashboard artifacts must use `battery_dashboard_` prefixes.
- Existing training, inference, model library, and global diagnostics behavior must remain compatible.
- Do not edit backend/importer files owned by Task 1.

**Behavior Requirements:**
- `loadProfileArtifacts` should fetch these optional files with fallbacks:
  - `battery_dashboard_manifest.json`, fallback `{ scenarios: [] }`
  - `battery_dashboard_overview.json`, fallback `{ scenarios: [], transfers: [] }`
  - `battery_dashboard_ess_detection.json`, fallback `null`
- Store them as `DATA.batteryDiagnostics = { manifest, overview, essDetection }`.
- `fallbackData()` must include an empty `batteryDiagnostics` object.
- Add `batteryDiagnosticSummary(diagnostics)` to `dashboard-core.mjs`.
- Summary should return:
  - `available`: true when any diagnostic scenario or ESS detection payload exists.
  - `scenarioCount`: number of `overview.scenarios` if present, else `manifest.scenarios`.
  - `labScenarioCount`: count of scenarios whose `kind` is not `"field"` or `"detection"`.
  - `fieldScenarioCount`: count of scenarios whose `kind` is `"field"`.
  - `transferCount`: number of `overview.transfers`.
  - `passedCount`: count of scenarios with `overall_pass === true`.
  - `essAvailable`: true when ESS detection payload has `kind === "detection"`.
  - `essWorstLabel`: label like `battery_01 / real_world_02-03 / L3 / 175 mV` when `summary.worst` exists, else `"暂无"`.
  - `sourceLabel`: `"battery_SOH dashboard"` when available, else `"未导入"`.
- Add a card near the top of the existing global diagnostics body:
  - `<section class="card battery-diagnostics-card" id="battery-diagnostics-card">`
  - Include target node `<div id="battery-diagnostics-summary"></div>`.
- `render()` should call `renderBatteryDiagnosticsPanel()`.
- If no imported diagnostics are available, show a compact note saying no battery_SOH diagnostics have been imported.
- When available, show scenario/transfer/ESS summary cards and a short table of up to 6 scenarios from `overview.scenarios`.
- Update `chinese-ui.test.mjs` to assert the new HTML id/class and app helper references exist.

**Test Requirements:**
- First update `frontend/app/dashboard-core.test.mjs` to import and test `batteryDiagnosticSummary`.
- First update `frontend/app/chinese-ui.test.mjs` to expect the new card/id and app render function.
- Run Node tests and confirm RED before production code changes.
- Then implement minimal code and rerun:
  - `node --test frontend/app/dashboard-core.test.mjs frontend/app/chinese-ui.test.mjs`

**Report:**
- Write report to `.superpowers/sdd/task-2-report.md`.
- Include RED command/failure summary, GREEN command/pass summary, changed files, and concerns.
