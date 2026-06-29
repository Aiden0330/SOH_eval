# Figma UI/UX Dashboard Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Figma UI/UX direction in the local static frontend.

**Architecture:** Keep the existing artifact and API data flow, but reorganize each tab into a distinct workbench. Training becomes a workflow orchestrator, inference becomes a transfer analysis workbench, and model library becomes an asset governance view.

**Tech Stack:** Static HTML/CSS, browser-native JavaScript modules, Node test runner.

## Global Constraints

- Do not change backend API contracts in this UI pass.
- Keep all user-facing interface copy in Simplified Chinese.
- Preserve existing resource upload and job-start behavior.
- Keep cards at 8px border radius where possible.
- Avoid decorative landing-page or marketing-style layout.

---

### Task 1: Add UI Structure Tests

**Files:**
- Modify: `frontend/app/chinese-ui.test.mjs`
- Modify: `frontend/app/dashboard-core.test.mjs`

**Interfaces:**
- Consumes: existing static HTML and exported helpers from `frontend/app/dashboard-core.mjs`
- Produces: tests that fail until the new workbench shells and helper outputs exist

- [ ] **Step 1: Add failing assertions for tab-specific shells**

Add assertions that require:

```js
assert.match(html, /id="training-workflow"/);
assert.match(html, /id="training-artifacts"/);
assert.match(html, /id="inference-compatibility"/);
assert.match(html, /id="inference-insights"/);
assert.match(html, /id="model-library-summary"/);
assert.match(app, /renderTrainingWorkflow/);
assert.match(app, /renderInferenceInsights/);
assert.match(app, /renderModelLibrarySummary/);
```

- [ ] **Step 2: Add failing helper tests**

Import and test:

```js
inferenceInsightSummary,
modelAssetRows,
trainingArtifactSummary,
```

Use fixed records and model catalog input to assert Chinese labels and sorted rows.

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
node --test frontend/app/dashboard-core.test.mjs frontend/app/chinese-ui.test.mjs
```

Expected: FAIL because the new ids/functions/helpers do not exist yet.

### Task 2: Add Frontend Data Helpers

**Files:**
- Modify: `frontend/app/dashboard-core.mjs`
- Test: `frontend/app/dashboard-core.test.mjs`

**Interfaces:**
- Produces:
  - `trainingArtifactSummary(config, modelCatalog)`
  - `inferenceInsightSummary(records)`
  - `modelAssetRows(modelCatalog, inferencePayload)`

- [ ] **Step 1: Implement `trainingArtifactSummary`**

Return a compact object:

```js
{
  pickleCount,
  datasetCount,
  generatedCount,
  uploadedCount,
  label
}
```

- [ ] **Step 2: Implement `inferenceInsightSummary`**

Return:

```js
{
  succeededCount,
  failedCount,
  bestTransfer,
  fastest,
  recommendation,
  risk
}
```

- [ ] **Step 3: Implement `modelAssetRows`**

Return catalog models enriched with latest inference fields:

```js
{
  name,
  source,
  estimator,
  featureset,
  trainedOn,
  featureColumnCount,
  latestMae,
  latestSpeed,
  badge
}
```

- [ ] **Step 4: Run helper tests to verify GREEN**

Run:

```bash
node --test frontend/app/dashboard-core.test.mjs
```

Expected: PASS.

### Task 3: Rework HTML Tab Panels

**Files:**
- Modify: `frontend/app/index.html`
- Test: `frontend/app/chinese-ui.test.mjs`

**Interfaces:**
- Consumes: existing IDs used by JavaScript upload and job functions
- Produces: new containers for differentiated tab layouts

- [ ] **Step 1: Update training panel structure**

Add:

```html
<div class="training-layout">
  <aside id="training-workflow"></aside>
  <div class="training-config">...</div>
  <aside id="training-artifacts"></aside>
</div>
```

Keep existing upload IDs, run selector IDs, protocol controls, and `start-eval`.

- [ ] **Step 2: Update inference panel structure**

Add:

```html
<div class="inference-workbench">
  <aside id="inference-compatibility"></aside>
  <div class="inference-main">...</div>
  <aside id="inference-insights"></aside>
</div>
```

Keep existing model upload IDs, model list ID, dataset list ID, and `start-inference`.

- [ ] **Step 3: Update model library structure**

Add:

```html
<div id="model-library-summary"></div>
<div id="model-library-table"></div>
```

- [ ] **Step 4: Run structure test**

Run:

```bash
node --test frontend/app/chinese-ui.test.mjs
```

Expected: PASS after JavaScript function names also exist.

### Task 4: Render New Workbench Views

**Files:**
- Modify: `frontend/app/app.js`
- Test: `frontend/app/chinese-ui.test.mjs`

**Interfaces:**
- Consumes: new helper exports from `dashboard-core.mjs`
- Produces:
  - `renderTrainingWorkflow(plan)`
  - `renderTrainingArtifacts()`
  - `renderInferenceCompatibility(models, datasets)`
  - `renderInferenceInsights(summary)`
  - `renderModelLibrarySummary(models)`

- [ ] **Step 1: Import helper functions**

Import `trainingArtifactSummary`, `inferenceInsightSummary`, and `modelAssetRows`.

- [ ] **Step 2: Split training rendering**

Keep upload and run controls, then render workflow steps and artifact status from current catalog/job/model data.

- [ ] **Step 3: Split inference rendering**

Render compatibility summary, inference matrix, and insight cards separately.

- [ ] **Step 4: Split model library rendering**

Render summary cards above the existing model table and show latest inference metrics in rows.

- [ ] **Step 5: Run UI structure test**

Run:

```bash
node --test frontend/app/chinese-ui.test.mjs
```

Expected: PASS.

### Task 5: Apply UI/UX Styling

**Files:**
- Modify: `frontend/app/styles.css`

**Interfaces:**
- Consumes: new class names from HTML/JS
- Produces: compact, differentiated, responsive layouts

- [ ] **Step 1: Add workbench grid styles**

Create styles for `.training-layout`, `.inference-workbench`, `.workflow-panel`, `.artifact-panel`, `.insight-panel`, and `.model-summary-grid`.

- [ ] **Step 2: Add tab-specific accents**

Use blue for training, green for inference, amber for model library.

- [ ] **Step 3: Add responsive rules**

Collapse three-column workbenches to one column below 900px.

### Task 6: Verify And Commit

**Files:**
- All modified frontend files and docs

**Interfaces:**
- Produces: verified frontend UI implementation

- [ ] **Step 1: Run Node tests**

Run:

```bash
node --test frontend/app/dashboard-core.test.mjs frontend/app/chinese-ui.test.mjs
```

Expected: PASS.

- [ ] **Step 2: Run whitespace check**

Run:

```bash
git diff --check -- frontend/app/index.html frontend/app/app.js frontend/app/styles.css frontend/app/dashboard-core.mjs frontend/app/chinese-ui.test.mjs frontend/app/dashboard-core.test.mjs
```

Expected: no new whitespace errors.

- [ ] **Step 3: Browser verify**

Open:

```text
http://localhost:8000/?v=figma-uiux
```

Verify that the three tabs show distinct first screens.

- [ ] **Step 4: Commit**

Stage only files touched for this feature and commit with:

```bash
git commit -m "feat: apply figma uiux dashboard layout"
```
