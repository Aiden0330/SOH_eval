# SOH Fabric Frontend V2 & Evidence Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以测评决策为中心大幅重构当前静态前端，交付单一导航、九个职责明确的工作页面、可恢复运行状态、公平比较和可下钻到原始充电曲线/特征窗口的证据中心，同时保持在线 API、离线目录包和单 HTML 三种使用方式。

**Architecture:** 保留原生 ES Modules 和当前 Python 静态服务，新增 `frontend/app/v2` 壳层、纯函数状态模块与页面模块。DataSource 明确区分 live、offline 和 demo，页面只消费 API v2/EvidencePack 视图。Plotly.js 与 Lucide 作为本地 vendor 资产，不依赖 CDN。`?ui=v2` 先并行启用，验收后 V2 成为默认，旧界面保留 `?ui=legacy` 一个迁移周期。

**Tech Stack:** HTML5、CSS Custom Properties、原生 ES Modules、Plotly.js、Lucide、Node test runner、Playwright、Python 打包脚本。

## Global Constraints

- 全部界面文字为中文；数据/模型/指标 ID 保持原始标识并提供中文辅助说明。
- 只保留 224px 左侧顶层导航；不再出现重复顶部页签。
- 顶部上下文栏固定 56px，显示 workspace、数据来源、comparison scope 和系统菜单。
- 页面各自只有一个主决策，不在训练、推理、全局诊断之间复用同一组 KPI。
- 长列表固定高度、sticky header、内部滚动；内容变化不得撑高整体页面或改变控制尺寸。
- 卡片圆角不超过 8px，不嵌套卡片；页面区段使用无框布局或全宽带。
- 图标按钮使用 Lucide 并有可访问名称和 tooltip；颜色不是唯一状态表达。
- 375、768、1024、1440px 均不得出现重叠、边框溢出、横向页面滚动或文字裁切。
- live、offline 和 demo 数据不得静默混合；每页始终显示当前来源徽标。
- 不可比的 comparison key 不得绘制在同一统计图或排行榜。
- 缺少原始曲线必须显示原因与 coverage，禁止用插值生成假曲线。
- 单 HTML 默认保存全部聚合/逐数据集/逐样本结果与选定曲线；完整原始曲线须显式选择并显示体积。

---

### Task 1: Add Frontend Test Harness, Local Vendors And Design Tokens

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Create: `playwright.config.mjs`
- Create: `scripts/vendor_frontend_assets.py`
- Create: `frontend/app/v2/styles/tokens.css`
- Create: `frontend/app/v2/styles/base.css`
- Create: `frontend/app/v2/tests/vendor-assets.test.mjs`
- Modify: `.gitignore`

**Interfaces:**
- npm scripts: `test:ui-unit`, `test:ui-e2e`, `vendor:ui`
- Local assets: `frontend/app/vendor/plotly.min.js`, `frontend/app/vendor/lucide.min.js`, `frontend/app/vendor/manifest.json`
- Token groups: primitive, semantic, component; spacing uses 8px rhythm

- [ ] **Step 1: Write the failing vendor and token tests**

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("frontend vendors are local and checksummed", async () => {
  const manifest = JSON.parse(await readFile("frontend/app/vendor/manifest.json", "utf8"));
  assert.equal(manifest.assets.length, 2);
  for (const asset of manifest.assets) {
    assert.match(asset.sha256, /^[a-f0-9]{64}$/);
    assert.equal(asset.source.includes("http"), false);
  }
});
```

Add a token test requiring `--space-1` through `--space-8`, semantic surface/text/border/status colors, fixed sidebar/topbar/list dimensions, zero negative letter spacing and a maximum component radius of 8px.

- [ ] **Step 2: Run tests and confirm missing package/vendor files**

Run: `node --test frontend/app/v2/tests/vendor-assets.test.mjs`

Expected: FAIL because the V2 files do not exist.

- [ ] **Step 3: Add dependencies and deterministic vendor copying**

Use exact locked versions of `plotly.js-dist-min` and `lucide`; add `@playwright/test` as a dev dependency. The Python vendor script reads files from `node_modules`, copies them atomically, computes SHA-256 and writes a manifest. Tokens define a neutral white/gray work surface, cobalt action color, green success, amber warning and red failure; avoid a one-hue or purple-gradient theme.

```json
{
  "private": true,
  "scripts": {
    "vendor:ui": "python scripts/vendor_frontend_assets.py",
    "test:ui-unit": "node --test frontend/app/*.test.mjs frontend/app/v2/tests/*.test.mjs",
    "test:ui-e2e": "playwright test"
  },
  "dependencies": {
    "lucide": "1.24.0",
    "plotly.js-dist-min": "3.7.0"
  },
  "devDependencies": {
    "@playwright/test": "1.61.1"
  }
}
```

```python
VENDORS = {
    "plotly.min.js": Path("node_modules/plotly.js-dist-min/plotly.min.js"),
    "lucide.min.js": Path("node_modules/lucide/dist/umd/lucide.min.js"),
}


def vendor_assets(target=Path("frontend/app/vendor")):
    target.mkdir(parents=True, exist_ok=True)
    assets = []
    for name, source in VENDORS.items():
        destination = target / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        assets.append({"path": name, "source": source.as_posix(), "sha256": sha256_file(destination)})
    write_json_atomic(target / "manifest.json", {"assets": assets})
```

- [ ] **Step 4: Install, vendor and run the tests**

Run: `npm install`

Run: `npm run vendor:ui`

Run: `npm run test:ui-unit`

Expected: install exits `0`; vendor test and existing `frontend/app/*.test.mjs` tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add package.json package-lock.json playwright.config.mjs scripts/vendor_frontend_assets.py frontend/app/v2/styles frontend/app/v2/tests/vendor-assets.test.mjs frontend/app/vendor .gitignore
git commit -m "build: add frontend v2 test and vendor foundation"
```

### Task 2: Build The V2 Shell, Router And Recoverable URL State

**Files:**
- Create: `frontend/app/v2/app.mjs`
- Create: `frontend/app/v2/core/router.mjs`
- Create: `frontend/app/v2/core/store.mjs`
- Create: `frontend/app/v2/core/view.mjs`
- Create: `frontend/app/v2/components/app-shell.mjs`
- Create: `frontend/app/v2/styles/shell.css`
- Create: `frontend/app/v2/tests/router.test.mjs`
- Create: `frontend/app/v2/tests/store.test.mjs`
- Modify: `frontend/app/index.html`
- Modify: `frontend/app/app.js`

**Interfaces:**
- Routes: `overview`, `experiments`, `runs`, `compare`, `transfer`, `failures`, `data`, `models`, `evidence`
- Query state: `ui`, `workspace`, `source`, `comparison`, `run`, `evidence`, `sample`, `cycle`, `cycles`
- Produces: `createRouter(windowLike)`, `createStore(initialState)`, `mountV2(root, dependencies)`

- [ ] **Step 1: Write failing route round-trip and history tests**

```javascript
test("evidence sample URL round trips", () => {
  const router = createRouter(fakeWindow("?ui=v2&page=evidence&evidence=EV-1&sample=S-2&cycle=31"));
  assert.deepEqual(router.current().selection, {
    evidenceId: "EV-1", sampleId: "S-2", cycleId: "31", cycleIds: ["31"],
  });
  assert.match(router.href({ page: "runs", runId: "RUN-7" }), /page=runs.*run=RUN-7/);
});
```

Add tests that browser back/forward restores selections, unknown pages redirect to overview, and mutable store updates do not mutate prior snapshots.

- [ ] **Step 2: Run tests and confirm router/store are absent**

Run: `node --test frontend/app/v2/tests/router.test.mjs frontend/app/v2/tests/store.test.mjs`

Expected: FAIL because V2 core modules do not exist.

- [ ] **Step 3: Implement the single-sidebar shell**

Render nine navigation items with Lucide icons, a 56px context bar and a single `<main>` region. Use semantic buttons/links and `aria-current`. Desktop sidebar is 224px; below 1024px it collapses to an icon rail; below 768px it becomes an accessible drawer opened by a menu icon. `?ui=legacy` mounts the current application without V2 DOM. No V2 page content is implemented in this task beyond stable empty-state headings.

```javascript
export const ROUTES = Object.freeze([
  ["overview", "可信总览", "shield-check"],
  ["experiments", "实验设计", "flask-conical"],
  ["runs", "运行中心", "activity"],
  ["compare", "比较实验室", "chart-no-axes-combined"],
  ["transfer", "跨域泛化", "grid-3x3"],
  ["failures", "失败分析", "triangle-alert"],
  ["data", "数据与协议", "database"],
  ["models", "模型与推理", "cpu"],
  ["evidence", "证据中心", "file-check-2"],
]);

export function mountV2(root, dependencies) {
  const router = createRouter(window);
  const store = createStore({ route: router.current(), drawerOpen: false });
  root.replaceChildren(renderAppShell({ routes: ROUTES, store, router }));
  router.subscribe((route) => {
    store.update((state) => ({ ...state, route }));
    dependencies.renderPage(root.querySelector("main"), route, dependencies);
  });
  router.start();
  return { router, store };
}
```

- [ ] **Step 4: Run unit and legacy UI regressions**

Run: `npm run test:ui-unit`

Expected: all V2 router/store and existing Chinese/native dashboard tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add frontend/app/v2/app.mjs frontend/app/v2/core frontend/app/v2/components/app-shell.mjs frontend/app/v2/styles/shell.css frontend/app/v2/tests/router.test.mjs frontend/app/v2/tests/store.test.mjs frontend/app/index.html frontend/app/app.js
git commit -m "feat: add frontend v2 application shell"
```

### Task 3: Separate Live, Offline And Demo Data Sources

**Files:**
- Create: `frontend/app/v2/core/api-client.mjs`
- Create: `frontend/app/v2/core/data-source.mjs`
- Create: `frontend/app/v2/components/source-badge.mjs`
- Create: `frontend/app/v2/components/error-state.mjs`
- Create: `frontend/app/v2/tests/api-client.test.mjs`
- Create: `frontend/app/v2/tests/data-source.test.mjs`

**Interfaces:**
- Produces: `ApiClient(baseUrl, fetchImpl)` with v2 endpoint methods
- Produces: `LiveDataSource`, `OfflineEvidenceSource`, `DemoDataSource`
- Produces: `resolveDataSource({ query, globals, fetchImpl }) -> DataSource`
- Data source descriptor: mode, label, provenance, readOnly, lastUpdated, health

- [ ] **Step 1: Write failing source-priority and error-envelope tests**

```javascript
test("standalone evidence cannot silently fall back to demo", async () => {
  const source = resolveDataSource({
    query: new URLSearchParams("source=offline"),
    globals: { __SOH_EVIDENCE_BUNDLE__: { manifest: { evidence_id: "EV-1" } } },
    fetchImpl: () => { throw new Error("network must not be used"); },
  });
  assert.equal(source.descriptor.mode, "offline");
  assert.equal(source.descriptor.readOnly, true);
});
```

The same source suite must cover API correlation IDs, aborting stale requests, UTF-8 Chinese error text and explicit `demo` selection. A failed live source must render disconnected state, not old static data.

- [ ] **Step 2: Run tests and confirm source types are missing**

Run: `node --test frontend/app/v2/tests/api-client.test.mjs frontend/app/v2/tests/data-source.test.mjs`

Expected: FAIL because the source modules do not exist.

- [ ] **Step 3: Implement explicit source resolution and request cancellation**

Resolution order is explicit query selection, standalone global, configured live API, then a visible unavailable state. Demo data is used only with `source=demo`. Normalize API errors to `code/category/message/impact/correlationId/details`. Every page request owns an AbortController and ignores responses for a superseded route scope.

```javascript
export function resolveDataSource({ query, globals, fetchImpl }) {
  const requested = query.get("source");
  if (requested === "demo") return new DemoDataSource(globals.__SOH_DEMO_DATA__ ?? {});
  if (requested === "offline" || (!requested && globals.__SOH_EVIDENCE_BUNDLE__)) {
    if (!globals.__SOH_EVIDENCE_BUNDLE__) throw new DataSourceError("OFFLINE_BUNDLE_MISSING");
    return new OfflineEvidenceSource(globals.__SOH_EVIDENCE_BUNDLE__);
  }
  if (requested === "live" || (!requested && globals.__SOH_API_BASE__)) {
    return new LiveDataSource(new ApiClient(globals.__SOH_API_BASE__, fetchImpl));
  }
  return new UnavailableDataSource("未配置实时 API 或离线证据包");
}

export class ApiClient {
  constructor(baseUrl, fetchImpl = fetch) { this.baseUrl = baseUrl; this.fetchImpl = fetchImpl; }
  async request(path, { signal, ...options } = {}) {
    const response = await this.fetchImpl(new URL(path, this.baseUrl), { ...options, signal });
    const body = await response.json();
    if (!response.ok) throw normalizeApiError(body, response.status);
    return body;
  }
}
```

- [ ] **Step 4: Run source and core UI tests**

Run: `npm run test:ui-unit`

Expected: all tests PASS; no fallback path combines two modes.

- [ ] **Step 5: Commit**

```powershell
git add frontend/app/v2/core/api-client.mjs frontend/app/v2/core/data-source.mjs frontend/app/v2/components/source-badge.mjs frontend/app/v2/components/error-state.mjs frontend/app/v2/tests/api-client.test.mjs frontend/app/v2/tests/data-source.test.mjs
git commit -m "feat: separate live offline and demo frontend data"
```

### Task 4: Implement The Trusted Overview

**Files:**
- Create: `frontend/app/v2/pages/overview.mjs`
- Create: `frontend/app/v2/components/readiness-strip.mjs`
- Create: `frontend/app/v2/components/blocker-list.mjs`
- Create: `frontend/app/v2/components/run-health.mjs`
- Create: `frontend/app/v2/styles/overview.css`
- Create: `frontend/app/v2/tests/overview-model.test.mjs`
- Modify: `frontend/app/v2/app.mjs`

**Interfaces:**
- Displays only: release readiness, blocking issues, latest signed conclusions, run-system health
- Primary action: resolve highest-priority blocker or open review-ready evidence
- Does not display: training candidate picker, inference picker or model upload form

- [ ] **Step 1: Write failing readiness projection tests**

```javascript
test("partial run is not counted as an accepted signed conclusion", () => {
  const model = buildOverviewModel({
    runs: [{ execution_state: "SIGNED", outcome: "partial", verdict: "warn", coverage: 0.7 }],
    policies: [{ minimum_coverage: 1 }],
  });
  assert.equal(model.acceptedCount, 0);
  assert.equal(model.blockers[0].code, "INSUFFICIENT_COVERAGE");
});
```

Add tests that queued/running runs affect system health but not release readiness, and that `SIGNED` does not imply `pass`.

- [ ] **Step 2: Run model tests and verify the page is missing**

Run: `node --test frontend/app/v2/tests/overview-model.test.mjs`

Expected: FAIL because `buildOverviewModel` is absent.

- [ ] **Step 3: Build a dense decision-oriented overview**

Use one horizontal readiness strip, an unframed two-column body for blockers and latest signed conclusions, and a compact run-health band. Avoid hero copy and decorative cards. Each conclusion links to Comparison Lab or Evidence Center; each blocker links to the page that can resolve it. Dynamic counts occupy fixed-width slots.

```javascript
export function buildOverviewModel({ runs, policies }) {
  const minimumCoverage = Math.max(...policies.map((item) => item.minimum_coverage), 0);
  const blockers = [];
  const accepted = runs.filter((run) => run.execution_state === "SIGNED"
    && run.verdict === "pass" && run.coverage >= minimumCoverage);
  for (const run of runs.filter((item) => item.execution_state === "SIGNED")) {
    if (run.coverage < minimumCoverage) {
      blockers.push({ code: "INSUFFICIENT_COVERAGE", runId: run.run_id, page: "evidence" });
    }
  }
  return {
    acceptedCount: accepted.length,
    blockers,
    conclusions: accepted,
    health: summarizeRunHealth(runs),
  };
}

export function renderOverview(model) {
  return html`<section class="overview" aria-labelledby="overview-title">
    <h1 id="overview-title">可信总览</h1>
    ${renderReadinessStrip(model)}
    <div class="overview__body">${renderBlockers(model.blockers)}${renderConclusions(model.conclusions)}</div>
    ${renderRunHealth(model.health)}
  </section>`;
}
```

- [ ] **Step 4: Run UI units and render smoke test**

Run: `npm run test:ui-unit`

Expected: all tests PASS; the page contains one `<h1>` and one primary command.

- [ ] **Step 5: Commit**

```powershell
git add frontend/app/v2/pages/overview.mjs frontend/app/v2/components/readiness-strip.mjs frontend/app/v2/components/blocker-list.mjs frontend/app/v2/components/run-health.mjs frontend/app/v2/styles/overview.css frontend/app/v2/tests/overview-model.test.mjs frontend/app/v2/app.mjs
git commit -m "feat: add trusted evaluation overview"
```

### Task 5: Implement Experiment Studio And Preflight

**Files:**
- Create: `frontend/app/v2/pages/experiments.mjs`
- Create: `frontend/app/v2/components/spec-selector.mjs`
- Create: `frontend/app/v2/components/preflight-panel.mjs`
- Create: `frontend/app/v2/components/runspec-summary.mjs`
- Create: `frontend/app/v2/styles/experiments.css`
- Create: `frontend/app/v2/tests/experiment-state.test.mjs`
- Modify: `frontend/app/v2/app.mjs`

**Interfaces:**
- Workflow: task/track -> data snapshot -> candidates/features -> protocol/policy -> preflight -> freeze RunSpec -> start
- Primary action: `冻结并启动测评`
- Uses: `POST /api/v2/preflight`, `POST /api/v2/runs`
- Upload links route to Data & Protocol or Model & Inference registries instead of embedding three duplicate upload blocks

- [ ] **Step 1: Write failing selection and preflight-gating tests**

```javascript
test("run cannot start until the current spec passes preflight", () => {
  let state = initialExperimentState();
  state = reduceExperiment(state, { type: "selectDataset", id: "DS-seven" });
  state = reduceExperiment(state, { type: "preflightPassed", specDigest: "old" });
  state = reduceExperiment(state, { type: "selectCandidate", id: "ridge" });
  assert.equal(canStartRun(state), false);
  assert.equal(state.preflight.status, "stale");
});
```

The same experiment-state suite must cover incompatible feature policy, zero-shot versus adaptation separation, idempotency key reuse on double click and estimated combination/resource display.

- [ ] **Step 2: Run tests and confirm experiment state is missing**

Run: `node --test frontend/app/v2/tests/experiment-state.test.mjs`

Expected: FAIL because the page reducer does not exist.

- [ ] **Step 3: Build the experiment workbench**

Use a 260px left step rail and a flexible main editor on desktop, stacked steps on mobile. Candidate and dataset selectors have fixed 320px heights with search, select-all scoped to filters and internal scrolling. Preflight occupies the right summary rail only on wide screens and becomes a collapsible section below 1024px. Show comparison key preview and frozen spec digest before launch.

```javascript
export function reduceExperiment(state, event) {
  const next = structuredClone(state);
  if (event.type === "selectDataset") next.selection.datasetId = event.id;
  if (event.type === "selectCandidate") next.selection.candidateIds = toggle(next.selection.candidateIds, event.id);
  if (event.type === "preflightPassed") {
    next.preflight = { status: "passed", specDigest: event.specDigest, result: event.result };
    return next;
  }
  next.preflight = { status: "stale", specDigest: null, result: null };
  return next;
}

export function canStartRun(state) {
  return state.preflight.status === "passed"
    && state.preflight.specDigest === digestRunDraft(state.selection)
    && state.selection.candidateIds.length > 0;
}

export async function startExperiment(state, api) {
  if (!canStartRun(state) || state.launch.pending) return null;
  state.launch.pending = true;
  const key = state.launch.idempotencyKey ?? crypto.randomUUID();
  state.launch.idempotencyKey = key;
  return api.createRun(freezeRunSpec(state.selection), { idempotencyKey: key });
}
```

- [ ] **Step 4: Run unit tests and API mock smoke**

Run: `npm run test:ui-unit`

Expected: all tests PASS; rapid double click creates one request with one idempotency key.

- [ ] **Step 5: Commit**

```powershell
git add frontend/app/v2/pages/experiments.mjs frontend/app/v2/components/spec-selector.mjs frontend/app/v2/components/preflight-panel.mjs frontend/app/v2/components/runspec-summary.mjs frontend/app/v2/styles/experiments.css frontend/app/v2/tests/experiment-state.test.mjs frontend/app/v2/app.mjs
git commit -m "feat: add contract first experiment studio"
```

### Task 6: Implement Run Center With Event Recovery

**Files:**
- Create: `frontend/app/v2/pages/runs.mjs`
- Create: `frontend/app/v2/components/run-table.mjs`
- Create: `frontend/app/v2/components/event-timeline.mjs`
- Create: `frontend/app/v2/components/attempt-panel.mjs`
- Create: `frontend/app/v2/core/run-events.mjs`
- Create: `frontend/app/v2/styles/runs.css`
- Create: `frontend/app/v2/tests/run-events.test.mjs`
- Modify: `frontend/app/v2/app.mjs`

**Interfaces:**
- Primary action: inspect/cancel/retry one selected run
- Polls: `GET /api/v2/runs/{run_id}/events?after=<sequence>`
- Persists: selected run and last sequence in URL/session state
- Shows: execution state, outcome, verdict, attempt, progress checkpoint, elapsed time and failure category

- [ ] **Step 1: Write failing event merge and refresh-recovery tests**

```javascript
test("event reducer ignores duplicates and preserves attempt history", () => {
  const state = reduceRunEvents(emptyRunState(), [event(1, 1), event(2, 1)]);
  const restored = reduceRunEvents(state, [event(2, 1), event(3, 2)]);
  assert.deepEqual(restored.events.map((item) => item.sequence), [1, 2, 3]);
  assert.deepEqual([...restored.attempts.keys()], [1, 2]);
});
```

The same run-event suite must cover cancelled polling, network retry backoff, terminal-state stop, running duration that does not resize the table and retry creating a new attempt rather than rewriting the old row.

- [ ] **Step 2: Run tests and confirm event utilities are missing**

Run: `node --test frontend/app/v2/tests/run-events.test.mjs`

Expected: FAIL because run event modules do not exist.

- [ ] **Step 3: Build a master-detail run workspace**

Use a fixed-height run table (minimum 360px, maximum `calc(100vh - 240px)`) and a 360px detail rail with tabs for timeline, attempts and artifacts. On narrower layouts, detail becomes an accessible drawer. Cancel and retry require confirmation and display server correlation IDs on failure. Refresh reconstructs the selected run from URL and resumes after the stored sequence.

```javascript
export function reduceRunEvents(state, incoming) {
  const bySequence = new Map(state.events.map((item) => [item.sequence, item]));
  for (const event of incoming) bySequence.set(event.sequence, event);
  const events = [...bySequence.values()].sort((a, b) => a.sequence - b.sequence);
  const attempts = new Map();
  for (const event of events) {
    if (!attempts.has(event.attempt)) attempts.set(event.attempt, []);
    attempts.get(event.attempt).push(event);
  }
  const latest = events.at(-1);
  return {
    ...state, events, attempts,
    lastSequence: latest?.sequence ?? 0,
    executionState: latest?.execution_state ?? state.executionState,
    outcome: latest?.outcome ?? state.outcome,
    verdict: latest?.verdict ?? state.verdict,
  };
}

export async function followRunEvents(api, runId, initial, onState, signal) {
  let state = initial;
  while (!signal.aborted && !isTerminal(state.executionState)) {
    const page = await api.runEvents(runId, state.lastSequence, { signal });
    state = reduceRunEvents(state, page.items);
    onState(state);
    await abortableDelay(page.retry_after_ms ?? 1000, signal);
  }
  return state;
}
```

- [ ] **Step 4: Run unit tests and simulated refresh flow**

Run: `npm run test:ui-unit`

Expected: all tests PASS; a RUNNING fixture resumes from the next event without duplicate timeline entries.

- [ ] **Step 5: Commit**

```powershell
git add frontend/app/v2/pages/runs.mjs frontend/app/v2/components/run-table.mjs frontend/app/v2/components/event-timeline.mjs frontend/app/v2/components/attempt-panel.mjs frontend/app/v2/core/run-events.mjs frontend/app/v2/styles/runs.css frontend/app/v2/tests/run-events.test.mjs frontend/app/v2/app.mjs
git commit -m "feat: add recoverable run center"
```

### Task 7: Implement Compare Lab And Cross-domain Generalization

**Files:**
- Create: `frontend/app/v2/pages/compare.mjs`
- Create: `frontend/app/v2/pages/transfer.mjs`
- Create: `frontend/app/v2/components/comparison-scope.mjs`
- Create: `frontend/app/v2/components/leaderboard-table.mjs`
- Create: `frontend/app/v2/components/ci-plot.mjs`
- Create: `frontend/app/v2/components/pareto-plot.mjs`
- Create: `frontend/app/v2/components/transfer-matrix.mjs`
- Create: `frontend/app/v2/styles/compare.css`
- Create: `frontend/app/v2/tests/comparison-model.test.mjs`
- Create: `frontend/app/v2/tests/transfer-model.test.mjs`
- Modify: `frontend/app/v2/app.mjs`

**Interfaces:**
- Compare primary action: select a defensible candidate set and submit for review
- Transfer primary action: locate worst held-out domain and open its evidence
- Inputs require one comparison key
- Outputs: mean/median/std/95% CI, paired delta, coverage, failures, worst domain, Pareto accuracy/latency

- [ ] **Step 1: Write failing comparability and matrix tests**

```javascript
test("mixed comparison keys are blocked before charting", () => {
  const result = buildComparisonModel([
    row({ candidate: "ridge", comparison_key: "A" }),
    row({ candidate: "gbdt", comparison_key: "B" }),
  ]);
  assert.equal(result.comparable, false);
  assert.equal(result.charts.length, 0);
  assert.equal(result.blocker.code, "MIXED_COMPARISON_SCOPE");
});
```

The same analysis suite must cover CI ordering, partial coverage warning, no scalar score for accuracy/latency Pareto, zero-shot/few-shot/adaptation separation and a transfer matrix cell preserving run/evidence/sample links.

- [ ] **Step 2: Run model tests and verify pages are absent**

Run: `node --test frontend/app/v2/tests/comparison-model.test.mjs frontend/app/v2/tests/transfer-model.test.mjs`

Expected: FAIL because comparison models do not exist.

- [ ] **Step 3: Build two distinct analytical workspaces**

Compare Lab uses a sticky scope bar, leaderboard, CI/paired comparison and Pareto views. Cross-domain uses a protocol selector, transfer matrix, worst-domain ranking and adaptation-cost band. Clicking a candidate row routes to evidence lineage; clicking a matrix cell routes with run, held-out domain and evidence filters. Plotly charts include accessible tabular fallback and never truncate candidate IDs without a tooltip.

```javascript
export function buildComparisonModel(rows) {
  const keys = new Set(rows.map((row) => row.comparison_key));
  if (keys.size !== 1) {
    return { comparable: false, charts: [], blocker: { code: "MIXED_COMPARISON_SCOPE" } };
  }
  const ordered = [...rows].sort((a, b) => a.metrics.mae.mean - b.metrics.mae.mean);
  return {
    comparable: true,
    comparisonKey: ordered[0].comparison_key,
    leaderboard: ordered,
    charts: [ciTrace(ordered), paretoTrace(ordered)],
    coverageWarnings: ordered.filter((row) => row.coverage < 1),
  };
}

export function buildTransferModel(rows) {
  return {
    protocols: groupBy(rows, (row) => row.adaptation_mode),
    matrix: matrixBy(rows, "training_domain", "held_out_domain"),
    worstDomains: worstDomainRows(rows),
    cellHref: (row) => routeHref("evidence", {
      evidence: row.evidence_id, run: row.run_id, domain: row.held_out_domain,
    }),
  };
}
```

- [ ] **Step 4: Run comparison/transfer and vendor tests**

Run: `npm run test:ui-unit`

Expected: all tests PASS; charts are not instantiated for mixed scopes.

- [ ] **Step 5: Commit**

```powershell
git add frontend/app/v2/pages/compare.mjs frontend/app/v2/pages/transfer.mjs frontend/app/v2/components/comparison-scope.mjs frontend/app/v2/components/leaderboard-table.mjs frontend/app/v2/components/ci-plot.mjs frontend/app/v2/components/pareto-plot.mjs frontend/app/v2/components/transfer-matrix.mjs frontend/app/v2/styles/compare.css frontend/app/v2/tests/comparison-model.test.mjs frontend/app/v2/tests/transfer-model.test.mjs frontend/app/v2/app.mjs
git commit -m "feat: add fair comparison and transfer workspaces"
```

### Task 8: Implement Failure, Data And Model Asset Workspaces

**Files:**
- Create: `frontend/app/v2/pages/failures.mjs`
- Create: `frontend/app/v2/pages/data-assets.mjs`
- Create: `frontend/app/v2/pages/model-assets.mjs`
- Create: `frontend/app/v2/components/failure-facets.mjs`
- Create: `frontend/app/v2/components/asset-list.mjs`
- Create: `frontend/app/v2/components/asset-detail.mjs`
- Create: `frontend/app/v2/components/upload-dialog.mjs`
- Create: `frontend/app/v2/styles/assets.css`
- Create: `frontend/app/v2/tests/asset-models.test.mjs`
- Modify: `frontend/app/v2/app.mjs`

**Interfaces:**
- Failure page: category/impact facets, affected units, minimum reproduction and retry draft
- Data page: DatasetSpec, LabelSpec, snapshot, QC, license and split/fit-scope preview
- Model page: AlgorithmPackage/FeaturePackage upload, ModelArtifact registry, lineage/schema and Frozen Inference launch
- Lists use fixed height and internal scroll; uploads occur in modal dialogs

- [ ] **Step 1: Write failing page-projection and fixed-list contract tests**

```javascript
test("model registry distinguishes training output and external upload", () => {
  const model = buildModelAssetView([
    asset({ origin: "training_run", trust_tier: "T1" }),
    asset({ origin: "external_upload", trust_tier: "T2" }),
  ]);
  assert.equal(model.groups.training.length, 1);
  assert.equal(model.groups.uploaded.length, 1);
  assert.equal(model.groups.uploaded[0].warningCode, "PICKLE_TRUSTED_LOCAL_ONLY");
});
```

Add tests that a model without manifest cannot launch formal inference, field datasets show label-readiness, failure retry carries a new attempt reason, and list component exposes a stable `--asset-list-height` rather than growing per row.

- [ ] **Step 2: Run tests and confirm page models are missing**

Run: `node --test frontend/app/v2/tests/asset-models.test.mjs`

Expected: FAIL because asset/failure modules do not exist.

- [ ] **Step 3: Build three purpose-specific workspaces**

Failure Analysis uses a category histogram/facet rail and a unit table, not global system KPIs. Data & Protocol uses a fixed-height dataset list with one detail pane and separate tabs for data card, QC and protocol preview. Model & Inference uses a fixed-height model list (desktop 440px, mobile 360px), detail pane and a Frozen Inference form with taller model/dataset selectors than the old UI; bottom result summary is capped at 220px and scrolls internally. Replace native file-input text with a styled button and selected filename while keeping the accessible input.

```javascript
export function buildModelAssetView(assets) {
  const groups = { training: [], uploaded: [] };
  for (const asset of assets) {
    const item = {
      ...asset,
      launchable: Boolean(asset.manifest_ref && asset.feature_schema),
      warningCode: asset.origin === "external_upload" ? "PICKLE_TRUSTED_LOCAL_ONLY" : null,
    };
    groups[asset.origin === "training_run" ? "training" : "uploaded"].push(item);
  }
  return { groups };
}

export function renderAssetList(items, selectedId, onSelect) {
  const root = document.createElement("div");
  root.className = "asset-list";
  root.setAttribute("role", "listbox");
  root.setAttribute("aria-label", "资产列表");
  const scroll = document.createElement("div");
  scroll.className = "asset-list__scroll";
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(item.asset_id === selectedId));
    button.title = item.asset_id;
    button.textContent = item.display_name;
    button.addEventListener("click", () => onSelect(item.asset_id));
    scroll.append(button);
  }
  root.append(scroll);
  return root;
}
```

- [ ] **Step 4: Run units and Chinese text regression**

Run: `npm run test:ui-unit`

Expected: all tests PASS; no Unicode replacement character `U+FFFD` or known corrupted Chinese fragments appear in rendered fixtures.

- [ ] **Step 5: Commit**

```powershell
git add frontend/app/v2/pages/failures.mjs frontend/app/v2/pages/data-assets.mjs frontend/app/v2/pages/model-assets.mjs frontend/app/v2/components/failure-facets.mjs frontend/app/v2/components/asset-list.mjs frontend/app/v2/components/asset-detail.mjs frontend/app/v2/components/upload-dialog.mjs frontend/app/v2/styles/assets.css frontend/app/v2/tests/asset-models.test.mjs frontend/app/v2/app.mjs
git commit -m "feat: add failure data and model workspaces"
```

### Task 9: Implement Evidence Center Review And Lineage

**Files:**
- Create: `frontend/app/v2/pages/evidence.mjs`
- Create: `frontend/app/v2/components/evidence-list.mjs`
- Create: `frontend/app/v2/components/lineage-graph.mjs`
- Create: `frontend/app/v2/components/review-panel.mjs`
- Create: `frontend/app/v2/components/integrity-panel.mjs`
- Create: `frontend/app/v2/styles/evidence.css`
- Create: `frontend/app/v2/tests/evidence-model.test.mjs`
- Modify: `frontend/app/v2/app.mjs`

**Interfaces:**
- Evidence subsections: `结论与谱系`, `样本与曲线`, `审核与签署`, `离线导出`
- Primary action changes by subsection: inspect conclusion, inspect sample, submit/sign review or export
- Uses: evidence read, review and sign API v2 endpoints

- [ ] **Step 1: Write failing integrity and signing-state tests**

```javascript
test("tampered evidence cannot expose sign action", () => {
  const model = buildEvidenceModel({
    integrity: { ok: false, mismatches: ["predictions.parquet"] },
    run: { execution_state: "REVIEW", outcome: "complete", verdict: "pass" },
  });
  assert.equal(model.actions.sign.enabled, false);
  assert.equal(model.blockers[0].code, "EVIDENCE_DIGEST_MISMATCH");
});
```

Add tests that a signed `warn` result remains signed but visually distinct from pass, reviewer identity/reason is required, lineage nodes retain digest links and offline mode hides write actions.

- [ ] **Step 2: Run tests and confirm Evidence Center is absent**

Run: `node --test frontend/app/v2/tests/evidence-model.test.mjs`

Expected: FAIL because Evidence Center models do not exist.

- [ ] **Step 3: Build evidence list, lineage and review surfaces**

Use an evidence list/detail layout with integrity status always visible. Lineage graph renders Task/RunSpec, DatasetSnapshot, FeatureState, ModelArtifact, PredictionRecord, metrics and review/signature as compact linked nodes; provide a table fallback. Signing requires confirmation showing comparison key, outcome, verdict, coverage and digest. Offline export is a separate subsection, not a global dashboard card.

```javascript
export function buildEvidenceModel({ integrity, run, source = { readOnly: false } }) {
  const blockers = [];
  if (!integrity.ok) blockers.push({
    code: "EVIDENCE_DIGEST_MISMATCH", files: integrity.mismatches,
  });
  const canReview = !source.readOnly && integrity.ok && run.execution_state === "REVIEW";
  return {
    run,
    integrity,
    blockers,
    statusTone: run.verdict === "pass" ? "success" : run.verdict === "warn" ? "warning" : "danger",
    actions: {
      review: { enabled: canReview },
      sign: { enabled: canReview && Boolean(run.review?.actor && run.review?.reason) },
      export: { enabled: integrity.ok },
    },
    lineage: buildLineageNodes(run.evidence_manifest),
  };
}

export async function signEvidence(model, api, input) {
  if (!model.actions.sign.enabled) throw new Error("EVIDENCE_NOT_SIGNABLE");
  return api.signEvidence(model.run.evidence_id, {
    actor: requiredText(input.actor), verdict: input.verdict, reason: requiredText(input.reason),
  });
}
```

- [ ] **Step 4: Run Evidence Center and source-mode tests**

Run: `npm run test:ui-unit`

Expected: all tests PASS; offline mode is read-only and corrupted evidence has no sign command.

- [ ] **Step 5: Commit**

```powershell
git add frontend/app/v2/pages/evidence.mjs frontend/app/v2/components/evidence-list.mjs frontend/app/v2/components/lineage-graph.mjs frontend/app/v2/components/review-panel.mjs frontend/app/v2/components/integrity-panel.mjs frontend/app/v2/styles/evidence.css frontend/app/v2/tests/evidence-model.test.mjs frontend/app/v2/app.mjs
git commit -m "feat: add evidence review and lineage center"
```

### Task 10: Add The SOH Sample And Curve Explorer

**Files:**
- Create: `frontend/app/v2/pages/evidence-samples.mjs`
- Create: `frontend/app/v2/components/soh-trajectory.mjs`
- Create: `frontend/app/v2/components/cycle-curve-stack.mjs`
- Create: `frontend/app/v2/components/feature-highlight-table.mjs`
- Create: `frontend/app/v2/components/sample-lineage.mjs`
- Create: `frontend/app/v2/core/curve-selection.mjs`
- Create: `frontend/app/v2/styles/evidence-samples.css`
- Create: `frontend/app/v2/tests/curve-selection.test.mjs`
- Create: `frontend/app/v2/tests/evidence-samples-model.test.mjs`
- Modify: `frontend/app/v2/pages/evidence.mjs`

**Interfaces:**
- URL: `page=evidence&section=samples&evidence=<id>&sample=<id>&cycle=<id>&cycles=<id,id>`
- Click SOH point: show truth/prediction/CI/error, V/I/T/Q, IC/DVA/DCIR when available, feature highlights, QC and lineage
- Shift-select: at most three cycles
- Modes: raw/processed, synchronized zoom/brush, table fallback

- [ ] **Step 1: Write failing point-selection and availability tests**

```javascript
test("shift selection keeps at most three cycles in chronological order", () => {
  let state = createCurveSelection();
  state = selectCycle(state, "10", { append: false });
  state = selectCycle(state, "20", { append: true });
  state = selectCycle(state, "30", { append: true });
  state = selectCycle(state, "40", { append: true });
  assert.deepEqual(state.cycleIds, ["20", "30", "40"]);
});


test("missing raw data returns a reason instead of a fabricated series", () => {
  const model = buildSampleEvidenceModel(sampleWithoutRawCurves());
  assert.equal(model.curves.available, false);
  assert.equal(model.curves.reason, "raw_curves_not_present");
  assert.equal(model.curves.series.length, 0);
});
```

Add tests that point selection updates the URL, stale curve requests are aborted, highlights reference raw segment IDs, raw/processed parameters are visible and SOH/curve axes remain linked to the selected cycle.

- [ ] **Step 2: Run tests and confirm explorer modules are missing**

Run: `node --test frontend/app/v2/tests/curve-selection.test.mjs frontend/app/v2/tests/evidence-samples-model.test.mjs`

Expected: FAIL because curve selection and sample models do not exist.

- [ ] **Step 3: Build the linked evidence layout**

Desktop layout uses a 320px sample/asset rail and a flexible analysis canvas. The SOH trajectory occupies the upper 34%; V/I/T/Q and IC/DVA occupy the middle 44%; feature/QC/lineage uses a compact bottom 22% with internal scroll. On mobile, use tabs for trajectory, curves and evidence details. Plotly click emits stable cycle IDs; Shift-click adds a comparison, with the oldest selection removed after three. Highlight windows use translucent overlays and labels tied to FeatureProvenance, not freehand annotations.

```javascript
export function selectCycle(state, cycleId, { append = false } = {}) {
  const current = append ? state.cycleIds.filter((id) => id !== cycleId) : [];
  const cycleIds = [...current, String(cycleId)]
    .sort((a, b) => Number(a) - Number(b))
    .slice(-3);
  return { ...state, cycleIds, activeCycleId: String(cycleId) };
}

export function bindTrajectorySelection(plot, state, router, loadCurves) {
  plot.on("plotly_click", async (event) => {
    const point = event.points[0];
    const next = selectCycle(state.value, point.customdata.cycle_id, {
      append: Boolean(event.event.shiftKey),
    });
    state.value = next;
    router.replace({ cycle: next.activeCycleId, cycles: next.cycleIds.join(",") });
    await loadCurves(next.activeCycleId);
  });
}

export function highlightShapes(provenance) {
  return provenance.flatMap((item) => item.windows.map((window) => ({
    type: "rect", xref: "x", yref: "paper", x0: window.voltage?.[0] ?? window.time?.[0],
    x1: window.voltage?.[1] ?? window.time?.[1], y0: 0, y1: 1,
    fillcolor: "rgba(55, 95, 230, 0.16)", line: { width: 1, color: "#375fe6" },
    label: { text: item.feature_id },
  })));
}
```

- [ ] **Step 4: Run unit tests and static fixture render**

Run: `npm run test:ui-unit`

Expected: all tests PASS; selecting a point fetches only that sample/cycle and updates the URL.

- [ ] **Step 5: Commit**

```powershell
git add frontend/app/v2/pages/evidence-samples.mjs frontend/app/v2/components/soh-trajectory.mjs frontend/app/v2/components/cycle-curve-stack.mjs frontend/app/v2/components/feature-highlight-table.mjs frontend/app/v2/components/sample-lineage.mjs frontend/app/v2/core/curve-selection.mjs frontend/app/v2/styles/evidence-samples.css frontend/app/v2/tests/curve-selection.test.mjs frontend/app/v2/tests/evidence-samples-model.test.mjs frontend/app/v2/pages/evidence.mjs
git commit -m "feat: add linked soh sample curve evidence explorer"
```

### Task 11: Package V2 As Portable Directory And Single HTML

**Files:**
- Modify: `scripts/package_portable_frontend.py`
- Modify: `scripts/build_standalone_frontend.py`
- Create: `src/sohbench/fabric/offline_bundle.py`
- Create: `src/sohbench/fabric/schemas/offline-bundle.schema.json`
- Create: `tests/fabric/test_offline_evidence_bundle.py`
- Modify: `tests/test_standalone_frontend.py`
- Modify: `tests/test_battery_dashboard_native_sync_package.py`

**Interfaces:**
- Bundle includes: all aggregate/per-dataset results, PredictionRecord index, failure records, manifests, lineage and selected curve payloads
- Curve policies: `representative`, `pinned`, `all`
- Produces: bundle manifest with evidence IDs, file/section digests, coverage, omitted curve reasons and byte estimates
- Standalone globals: `__SOH_STANDALONE__`, `__SOH_EVIDENCE_BUNDLE__`

- [ ] **Step 1: Write failing completeness and offline-navigation tests**

```python
def test_default_bundle_keeps_all_results_and_selected_curves(tmp_path, evidence_fixture):
    bundle = build_offline_bundle(evidence_fixture, tmp_path, curve_policy="representative")
    assert set(bundle.manifest["datasets"]) == set(evidence_fixture.dataset_ids)
    assert bundle.manifest["results"]["aggregate_complete"] is True
    assert bundle.manifest["results"]["sample_index_complete"] is True
    assert bundle.manifest["curves"]["complete"] is False
    assert bundle.manifest["curves"]["omitted_reason"] == "representative_policy"
```

Add a standalone test loading the generated HTML without API access, opening Compare Lab, then Evidence Center sample route for an embedded pinned sample. Assert local Plotly/Lucide assets are inlined and no `http://`/`https://` script URL exists.

- [ ] **Step 2: Run packaging tests and expose missing V2 modules/assets**

Run: `python -m pytest tests/fabric/test_offline_evidence_bundle.py tests/test_standalone_frontend.py tests/test_battery_dashboard_native_sync_package.py -q`

Expected: FAIL because current packaging only understands legacy assets/data.

- [ ] **Step 3: Implement manifest-driven recursive bundling**

Portable directory copies all V2 modules, CSS and vendor assets with checksums. Single HTML resolves the ES module graph, rewrites local imports to inlined module URLs or a deterministic concatenated bootstrap, embeds CSS/vendor scripts and serializes evidence with `<` escaped to avoid script termination. Estimate size before `curve_policy="all"`; reject when over configured limit unless `--allow-large` is passed. Remove the deleted anomaly-detection asset from all manifests.

```python
def build_offline_bundle(evidence_paths, output, curve_policy="representative", allow_large=False):
    packs = [verify_evidence_pack(path) for path in evidence_paths]
    sections = collect_complete_result_sections(packs)
    sections["curves"] = select_curve_payloads(packs, curve_policy)
    estimated = estimate_bundle_bytes(sections)
    if curve_policy == "all" and estimated > MAX_DEFAULT_BUNDLE_BYTES and not allow_large:
        raise BundleTooLarge(estimated, MAX_DEFAULT_BUNDLE_BYTES)
    manifest = build_bundle_manifest(packs, sections, curve_policy, estimated)
    validate_offline_bundle(manifest)
    write_sections_atomic(output, sections, manifest)
    return OfflineBundle(output, manifest)


def safe_script_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def standalone_evidence_script(bundle):
    payload = safe_script_json(bundle.to_dict())
    return f"<script>window.__SOH_STANDALONE__=true;window.__SOH_EVIDENCE_BUNDLE__={payload};</script>"
```

- [ ] **Step 4: Build and verify both packages**

Run: `python scripts/package_portable_frontend.py --output dist/soh-eval-portable`

Run: `python scripts/build_standalone_frontend.py --output dist/soh-eval-standalone.html`

Run: `python -m pytest tests/fabric/test_offline_evidence_bundle.py tests/test_standalone_frontend.py tests/test_battery_dashboard_native_sync_package.py -q`

Expected: commands exit `0`; packages open without local data/API and every embedded section digest verifies.

- [ ] **Step 5: Commit**

```powershell
git add scripts/package_portable_frontend.py scripts/build_standalone_frontend.py src/sohbench/fabric/offline_bundle.py src/sohbench/fabric/schemas/offline-bundle.schema.json tests/fabric/test_offline_evidence_bundle.py tests/test_standalone_frontend.py tests/test_battery_dashboard_native_sync_package.py
git commit -m "feat: package frontend v2 with offline evidence"
```

### Task 12: Run Responsive, Accessibility And Cutover Acceptance

**Files:**
- Create: `frontend/e2e/v2-navigation.spec.mjs`
- Create: `frontend/e2e/v2-experiment-run.spec.mjs`
- Create: `frontend/e2e/v2-evidence-curves.spec.mjs`
- Create: `frontend/e2e/v2-responsive.spec.mjs`
- Create: `frontend/e2e/v2-accessibility.spec.mjs`
- Create: `frontend/e2e/fixtures/fabric-api.mjs`
- Create: `docs/verification/frontend-v2-report.md`
- Modify: `frontend/app/index.html`
- Modify: `frontend/app/README.md`
- Modify: `README.md`

**Interfaces:**
- Viewports: 375x812, 768x1024, 1024x768, 1440x900
- Critical flow: design run -> preflight -> start -> refresh/recover -> compare -> evidence -> click SOH point -> curve/highlight
- Cutover: V2 default; `?ui=legacy` explicitly mounts legacy

- [ ] **Step 1: Write the end-to-end tests against mocked deterministic API v2**

Assert nine unique sidebar entries, no duplicate top-level tab bar, one primary action per page, event recovery after `page.reload()`, mixed comparison rejection, sample URL restoration, three-cycle selection, visible raw-unavailable reason and local-only assets. Accessibility tests cover keyboard navigation, focus visibility, landmarks, names, contrast and table fallbacks.

```javascript
for (const viewport of [
  { width: 375, height: 812 }, { width: 768, height: 1024 },
  { width: 1024, height: 768 }, { width: 1440, height: 900 },
]) {
  test(`v2 has no overlap at ${viewport.width}px`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await installFabricApiFixture(page);
    await page.goto("/?ui=v2&page=evidence&section=samples&evidence=EV-1&sample=S-1&cycle=10");
    await expect(page.locator("[data-top-level-nav] a")).toHaveCount(9);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.getByTestId("soh-point-20").click({ modifiers: ["Shift"] });
    await expect(page).toHaveURL(/cycles=10%2C20/);
    await expect(page.getByTestId("feature-highlight-table")).toBeVisible();
  });
}
```

- [ ] **Step 2: Install Chromium and run E2E before cutover**

Run: `npx playwright install chromium`

Run: `npm run test:ui-e2e`

Expected before final fixes: tests may FAIL only on documented layout/accessibility assertions; no functional API mock failure remains.

- [ ] **Step 3: Fix all overflow/overlap states and switch the default**

Use Playwright screenshots and bounding-box assertions at all four viewports. Assert `document.documentElement.scrollWidth <= window.innerWidth`, every interactive element is inside its panel, fixed lists retain their height after filtering and no text node contains Unicode code point `U+FFFD`. Make V2 the default path and keep legacy only behind `?ui=legacy`. Do not delete legacy files in this phase.

```css
.app-shell { min-height: 100dvh; display: grid; grid-template-columns: 224px minmax(0, 1fr); }
.app-main { min-width: 0; overflow: clip; }
.asset-list { height: var(--asset-list-height, 440px); min-height: 0; overflow: hidden; }
.asset-list__scroll { height: calc(100% - 48px); overflow: auto; }
@media (max-width: 1023px) { .app-shell { grid-template-columns: 64px minmax(0, 1fr); } }
@media (max-width: 767px) {
  .app-shell { display: block; }
  .app-sidebar { position: fixed; inset: 0 auto 0 0; width: min(88vw, 320px); transform: translateX(-100%); }
  .app-sidebar[data-open="true"] { transform: translateX(0); }
  .asset-list { --asset-list-height: 360px; }
}
```

```javascript
const query = new URLSearchParams(location.search);
if (query.get("ui") === "legacy") mountLegacyApp();
else mountV2(document.querySelector("#app"), dependencies);
```

- [ ] **Step 4: Run complete frontend, package and Python regressions**

Run: `npm run test:ui-unit`

Run: `npm run test:ui-e2e`

Run: `python -m pytest tests/test_standalone_frontend.py tests/test_battery_dashboard_native_sync_package.py tests/fabric/test_api_v2.py tests/fabric/test_curve_api.py -q`

Run: `python -m pytest -q`

Expected: every command exits `0`; screenshots show no overlap at all target viewports.

- [ ] **Step 5: Write the visual acceptance report and commit**

Record page-purpose matrix, viewport screenshots, overflow measurements, keyboard/accessibility observations, live/offline parity, package sizes and remaining legacy removal date/condition. Include the exact browser URL used for final acceptance.

```powershell
git add frontend/e2e docs/verification/frontend-v2-report.md frontend/app/index.html frontend/app/README.md README.md
git commit -m "feat: make verification fabric ui the default"
```

## Frontend Review Gate

- [ ] Nine top-level pages have unique responsibilities and one sidebar; Evidence Center contains the Sample & Curves subsection.
- [ ] Training, frozen inference and evidence review show different controls and KPIs.
- [ ] Dataset/model selectors and registry lists use fixed heights with internal scrolling.
- [ ] Refresh restores active runs and URL-scoped sample/cycle selections.
- [ ] Comparison charts reject mixed comparison keys and display CI/coverage/failure context.
- [ ] Clicking an SOH point reveals available V/I/T/Q, IC/DVA/DCIR and FeatureProvenance highlights without fabricated data.
- [ ] Live/offline/demo modes are visibly distinct and never silently mixed.
- [ ] Portable directory and single HTML retain all result scopes and selected evidence without a local data directory.
- [ ] 375/768/1024/1440px pass overlap, overflow, Chinese encoding and keyboard checks.
- [ ] `?ui=legacy` remains available for one migration cycle after V2 becomes default.

## Plan Verification Commands

```powershell
npm install
npm run vendor:ui
npm run test:ui-unit
npx playwright install chromium
npm run test:ui-e2e
python scripts/package_portable_frontend.py --output dist/soh-eval-portable
python scripts/build_standalone_frontend.py --output dist/soh-eval-standalone.html
python -m pytest tests/test_standalone_frontend.py tests/test_battery_dashboard_native_sync_package.py tests/fabric/test_api_v2.py tests/fabric/test_curve_api.py -q
python -m pytest -q
```

Expected: every command exits `0`; V2 opens as the default, legacy remains available by query flag, and both offline packages render all required result/evidence views without API access.
