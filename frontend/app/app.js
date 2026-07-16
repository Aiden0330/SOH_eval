/* sohbench dashboard: data-driven render of the Lab Light design.
   Loads benchmark artifacts from ./data and falls back to embedded sample data. */

"use strict";

import {
  algorithmInsights,
  bestComparableEntry,
  buildInferencePayload,
  buildEvaluationPayload,
  catalogOptions,
  evaluationPlanSummary,
  evaluationIdentity,
  evaluationLabel,
  evaluationMatrixRows,
  datasetOptions,
  inferenceInsightSummary,
  inferenceDatasetOptions,
  inferenceCompatibilitySummary,
  inferenceLaunchPlan,
  inferenceRows,
  inferenceAuditSummary,
  inferenceActionState,
  inferenceSummary,
  leaderboardCoverage,
  modelAssetRows,
  metricMean,
  modelOptions,
  nextDatasetSelection,
  preferredInferenceModels,
  profileOptions,
  scopeSummary,
  stabilityRows,
  summarizeBySelection,
  taskHistoryRows,
  trainingArtifactSummary,
} from "./dashboard-core.mjs?v=task-history-20260630";

const FALLBACK = {
  config: { datasets: ["nasa", "calce_cs2", "hust", "oxford"], protocols: ["lodo"], seeds: [0, 1] },
  leaderboard: [
    e("ridge", "full", 0.4791, { rmse: 0.500, mbe: 0.000, picp: 0.262, ci: "conformal" }),
    e("ridge", "voltage", 0.4105, { ci: "conformal" }),
    e("ridge", "ic_dva", 0.1736, { ci: "conformal", picp: 0.541 }),
    e("gbdt", "full", 0.2042, { rmse: 0.238, mbe: 0.021, picp: 0.079, ci: "conformal" }),
    e("gbdt", "voltage", 0.1963, { ci: "conformal" }),
    e("gbdt", "ic_dva", 0.1168, { ci: "conformal", picp: 0.604 }),
    e("mlp", "full", 0.5687, { ci: "conformal" }),
    e("mlp", "voltage", 0.2476, { ci: "conformal" }),
    e("mlp", "ic_dva", 0.1762, { ci: "conformal", picp: 0.648 }),
    e("phys_resid", "full", 0.2789, { ci: "conformal" }),
  ],
  datasetLeaderboard: [],
  inference: { records: [], summary: {} },
  modelCatalog: { models: [] },
  qc: {
    pass_rate: 0.9981,
    n_in: 164775,
    n_out: 318,
    drop_reasons: { soh_out_of_range: 318 },
    dropped_columns: ["ambient_temp_c"],
    per_dataset: {
      nasa: { rows: 2529 },
      calce_cs2: { rows: 15287 },
      hust: { rows: 146122 },
      oxford: { rows: 519 },
    },
  },
};

function e(estimator, featureset, mae, o = {}) {
  return {
    estimator,
    featureset,
    protocol: "lodo",
    track: "unified",
    ci_source: o.ci || "conformal",
    metrics: {
      mae: { mean: mae, std: 0 },
      rmse: o.rmse != null ? { mean: o.rmse, std: 0 } : null,
      mean_bias_error: o.mbe != null ? { mean: o.mbe, std: 0 } : null,
      picp: o.picp != null ? { mean: o.picp, std: 0 } : null,
      monotonicity_violation_rate: { mean: o.mono ?? 0.2, std: 0 },
      value_range_violation: { mean: 0, std: 0 },
    },
    verdicts: o.verdicts || { generalization: false, calibration: false, physical_consistency: false },
  };
}

const $ = (id) => document.getElementById(id);
const URL_PARAMS = new URLSearchParams(location.search);
const API_BASE = URL_PARAMS.get("api") || "http://localhost:8001";
const DATA_VERSION = URL_PARAMS.get("v") || "live";
const INITIAL_PROFILE = safeProfileName(URL_PARAMS.get("profile") || localStorage.getItem("sohbench.profile") || "root");
const FS_ORDER = ["full", "voltage", "ic_dva", "partial_window", "capacity_time", "(byo)"];
const fmt = (v, d = 3, sign = false) =>
  v == null ? "无" : (sign && v >= 0 ? "+" : "") + v.toFixed(d);
const uniq = (a) => [...new Set(a)];
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
}[ch]));

function fsOrder(sets) {
  return [...sets].sort((a, b) => (FS_ORDER.indexOf(a) + 1 || 99) - (FS_ORDER.indexOf(b) + 1 || 99));
}

function safeProfileName(value) {
  const name = String(value || "root").trim();
  return /^[A-Za-z0-9_][A-Za-z0-9_-]*$/.test(name) ? name : "root";
}

function heat(t) {
  const h = 140 - 140 * Math.max(0, Math.min(1, t));
  return `hsl(${h} 65% 92%)`;
}

let DATA = null;
let activeFs = null;
let activeDatasets = null;
let activeMode = "training";
let scoped = null;
let CATALOG = null;
let WORKBENCH_ERROR = "";
let CURRENT_JOB = null;
let resourceSelection = {
  datasets: new Set(),
  estimators: new Set(),
  featuresets: new Set(),
};
let inferenceSelection = {
  models: new Set(),
  datasets: new Set(),
};
let inferenceLaunchStrategy = "compatible_only";
let taskHistoryOpen = false;
let taskHistoryFilters = {
  kind: "all",
  status: "all",
  strategy: "all",
  dataset: "",
};
let lastTaskHistoryRows = [];
let taskHistoryExpandedIds = new Set();
let taskHistoryBusyId = "";
let taskHistoryMessage = "";
let PROFILE_INDEX = { profiles: [] };
let activeProfile = INITIAL_PROFILE;
let activeProfileOption = { name: "root", label: "当前根目录", kind: "root", active: true };

async function load(options = {}) {
  const suffix = `?v=${encodeURIComponent(DATA_VERSION)}${options.cacheBust ? `&t=${Date.now()}` : ""}`;
  PROFILE_INDEX = await loadProfileIndex(suffix);
  activeProfileOption = activeProfileSelection();
  try {
    DATA = await loadProfileArtifacts(activeProfileOption.name, suffix);
  } catch (_) {
    if (activeProfileOption.name !== "root") {
      activeProfile = "root";
      activeProfileOption = activeProfileSelection();
      try {
        DATA = await loadProfileArtifacts("root", suffix);
      } catch (_) {
        DATA = fallbackData();
      }
    } else {
      DATA = fallbackData();
    }
  }
  await loadBackendCatalog();
  render();
}

async function loadProfileIndex(suffix = "") {
  try {
    const response = await fetch("./data/profiles/index.json" + suffix);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (_) {
    return { profiles: [] };
  }
}

function activeProfileSelection() {
  const options = profileOptions(PROFILE_INDEX, activeProfile);
  const selected = options.find((item) => item.active) || options[0];
  activeProfile = selected.name;
  return selected;
}

function profileDataPath(profile = "root") {
  return profile === "root" ? "./data" : `./data/profiles/${encodeURIComponent(profile)}`;
}

async function loadProfileArtifacts(profile, suffix) {
  const base = profileDataPath(profile);
  const [lb, qc, cfg, byDataset, inference, inferenceConfig, modelCatalog, manifest] = await Promise.all([
    fetchJson(`${base}/leaderboard.json${suffix}`),
    fetchJson(`${base}/qc_report.json${suffix}`, null),
    fetchJson(`${base}/run_config.json${suffix}`, {}),
    fetchJson(`${base}/dataset_leaderboard.json${suffix}`, []),
    fetchJson(`${base}/inference_benchmark.json${suffix}`, { records: [], summary: {} }),
    fetchJson(`${base}/inference_run_config.json${suffix}`, {}),
    fetchJson(`${base}/model_catalog.json${suffix}`, { models: [] }),
    fetchJson(`${base}/manifest.json${suffix}`, {}),
  ]);
  return {
    leaderboard: lb,
    datasetLeaderboard: byDataset || [],
    qc,
    config: cfg || {},
    inference: inference || { records: [], summary: {} },
    inferenceConfig: inferenceConfig || {},
    modelCatalog: modelCatalog || { models: [] },
    profileManifest: manifest || {},
  };
}

async function fetchJson(url, fallback) {
  const response = await fetch(url);
  if (!response.ok) {
    if (arguments.length > 1) return fallback;
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

function fallbackData() {
  return {
    leaderboard: FALLBACK.leaderboard,
    datasetLeaderboard: FALLBACK.datasetLeaderboard,
    qc: FALLBACK.qc,
    config: FALLBACK.config,
    inference: FALLBACK.inference,
    inferenceConfig: {},
    modelCatalog: FALLBACK.modelCatalog,
    profileManifest: {},
    fallback: true,
  };
}

async function loadBackendCatalog() {
  try {
    const response = await fetch(`${API_BASE}/api/catalog`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    CATALOG = await response.json();
    WORKBENCH_ERROR = "";
    seedResourceSelection();
  } catch (err) {
    CATALOG = null;
    WORKBENCH_ERROR = "后端未连接";
  }
}

function seedResourceSelection() {
  const options = catalogOptions(CATALOG || {});
  if (!resourceSelection.datasets.size) {
    (DATA?.config?.datasets || options.datasets.map((item) => item.name)).forEach((name) => {
      if (options.datasets.some((item) => item.name === name)) resourceSelection.datasets.add(name);
    });
  }
  if (!resourceSelection.estimators.size && options.estimators.length) {
    const current = DATA?.config?.estimators || [];
    current.forEach((name) => {
      if (options.estimators.some((item) => item.name === name)) resourceSelection.estimators.add(name);
    });
    if (!resourceSelection.estimators.size) {
      resourceSelection.estimators.add(options.estimators.some((item) => item.name === "ridge") ? "ridge" : options.estimators[0].name);
    }
  }
  if (!resourceSelection.featuresets.size && options.featuresets.length) {
    const current = DATA?.config?.featuresets || [];
    current.forEach((name) => {
      if (options.featuresets.some((item) => item.name === name)) resourceSelection.featuresets.add(name);
    });
    if (!resourceSelection.featuresets.size) {
      resourceSelection.featuresets.add(options.featuresets.some((item) => item.name === "full") ? "full" : options.featuresets[0].name);
    }
  }
}

function render() {
  const aggregate = DATA.leaderboard || [];
  if (!aggregate.length) {
    $("leaderboard").innerHTML = '<div class="state">暂无数据。</div>';
    return;
  }

  const dsOptions = currentDatasetOptions();
  const inferenceDsOptions = currentInferenceDatasetOptions(dsOptions);
  const allDatasets = dsOptions.map((item) => item.name);
  if (!activeDatasets || !activeDatasets.size) activeDatasets = new Set(allDatasets);
  scoped = summarizeBySelection(aggregate, DATA.datasetLeaderboard || [], [...activeDatasets], allDatasets);

  const lb = scoped.entries || [];
  const sets = fsOrder(uniq(lb.map((x) => x.featureset || "full")));
  if (!activeFs || !sets.includes(activeFs)) activeFs = sets.includes("ic_dva") ? "ic_dva" : sets[0];

  renderRunMeta();
  renderKpis(lb, dsOptions);
  renderScopeContext(dsOptions);
  renderModeTabs();
  wireProfileSelector(profileOptions(PROFILE_INDEX, activeProfile));
  renderWorkbench();
  renderInferencePanel(inferenceDsOptions);
  renderModelLibrary();
  renderTaskHistoryDrawer();
  renderDatasetPanel(dsOptions);
  renderStabilityPanel();
  renderMatrix(lb, sets);
  renderTabs(sets);
  renderInsights(lb);
  renderLeaderboard(lb);
  renderPicp(lb);
  renderQc();
  renderFoot();
}

function currentDatasetOptions() {
  return datasetOptions(DATA.config, DATA.qc, DATA.datasetLeaderboard || []);
}

function currentInferenceDatasetOptions(dsOptions = currentDatasetOptions()) {
  return inferenceDatasetOptions({
    baseOptions: dsOptions,
    catalog: CATALOG || {},
    inference: DATA?.inference || {},
  });
}

function currentInferenceAuditSummary() {
  return inferenceAuditSummary({
    inferenceConfig: DATA?.inferenceConfig || {},
    catalog: CATALOG || {},
    currentJob: CURRENT_JOB?.kind === "inference" ? CURRENT_JOB : null,
  });
}

function renderRunMeta() {
  const c = DATA.config || {};
  const parts = [];
  parts.push(`展示数据 <b>${esc(activeProfileOption.label || activeProfileOption.name || "当前根目录")}</b>`);
  if (c.datasets) parts.push(`数据集 <b>${c.datasets.length}</b>`);
  const protos = c.protocols || uniq(DATA.leaderboard.map((x) => x.protocol));
  if (protos?.length) parts.push(`协议 <b>${protos.join(",")}</b>`);
  if (c.seeds) parts.push(`种子 <b>${c.seeds.length}</b>`);
  parts.push(`来源 <b>${DATA.fallback ? "示例" : "实测"}</b>`);
  $("runmeta").innerHTML = parts.map((p) => `<span>${p}</span>`).join("");
}

function wireProfileSelector(options = []) {
  const select = $("profile-select");
  if (!select) return;
  const current = activeProfileOption?.name || activeProfile || "root";
  select.innerHTML = options.map((item) =>
    `<option value="${esc(item.name)}">${esc(item.label)}${item.kind && item.kind !== "root" ? ` · ${esc(item.kind)}` : ""}</option>`
  ).join("");
  select.value = options.some((item) => item.name === current) ? current : "root";
  select.onchange = () => {
    activeProfile = safeProfileName(select.value);
    localStorage.setItem("sohbench.profile", activeProfile);
    resetProfileScopedState();
    load({ cacheBust: true });
  };
}

function resetProfileScopedState() {
  activeFs = null;
  activeDatasets = null;
  scoped = null;
  resourceSelection = {
    datasets: new Set(),
    estimators: new Set(),
    featuresets: new Set(),
  };
  inferenceSelection = {
    models: new Set(),
    datasets: new Set(),
  };
}

function renderKpis(lb, dsOptions) {
  const withMae = lb.filter((x) => metricMean(x, "mae") != null);
  const best = bestComparableEntry(withMae, selectedDatasetNames(), DATA.datasetLeaderboard || []);
  const selected = dsOptions.filter((item) => activeDatasets.has(item.name));
  const selectedRows = selected.reduce((sum, item) => sum + item.rows, 0);
  const qc = DATA.qc;
  let pass = 0, total = 0;
  lb.forEach((x) => Object.values(x.verdicts || {}).forEach((v) => { total++; if (v) pass++; }));
  const kpi = (lab, val, hint, color) =>
    `<div class="kpi"><div class="lab">${lab}</div><div class="val"${color ? ` style="color:${color}"` : ""}>${val}</div><div class="hint">${hint}</div></div>`;
  $("kpis").innerHTML =
    kpi("可比最佳 MAE", best ? esc(evaluationLabel(best)) : "无",
      best ? `MAE ${fmt(metricMean(best, "mae"))} / ${esc(best.featureset)}` : "无指标",
      "var(--primary)") +
    kpi("数据集范围", `${selected.length}/${dsOptions.length}`,
      `${selectedRows.toLocaleString()} 条 QC 样本`) +
    kpi("验收通过", `${pass}/${total}`, "泛化 / 校准 / 物理一致", pass ? "var(--warn)" : "var(--fail)") +
    kpi("数据 QC 通过率", qc ? (100 * qc.pass_rate).toFixed(2) + "%" : "无",
      qc ? `剔除 ${qc.n_out} 行` : "无 QC", "var(--pass)");
}

function renderScopeContext(dsOptions) {
  const summary = scopeSummary({
    datasetOptions: dsOptions,
    selectedDatasets: selectedDatasetNames(),
    scoped,
    config: DATA.config || {},
    fallback: DATA.fallback,
  });
  const names = summary.selectedNames.length <= 4
    ? summary.selectedNames.join(" / ")
    : `${summary.selectedNames.slice(0, 4).join(" / ")} / +${summary.selectedNames.length - 4}`;
  const c = DATA.config || {};
  const chip = (text, kind = "") => `<span class="chip ${esc(kind)}" title="${esc(text)}">${esc(text)}</span>`;
  const chipList = (items, fallback, kind = "") => {
    const visible = (items || []).filter(Boolean).slice(0, 4);
    if (!visible.length) return chip(fallback, kind);
    const extra = (items || []).length - visible.length;
    return visible.map((item) => chip(item, kind)).join("") + (extra > 0 ? chip(`+${extra}`, kind) : "");
  };
  $("scope-context").innerHTML =
    `<div class="scope-group">` +
      `<span class="scope-label">数据集</span>` +
      `<div class="chips">${chipList(summary.selectedNames, names || "全部数据集")}</div>` +
    `</div>` +
    `<div class="scope-group">` +
      `<span class="scope-label">算法</span>` +
      `<div class="chips">${chipList(c.estimators || [], "全部算法", "green")}</div>` +
    `</div>` +
    `<div class="scope-group">` +
      `<span class="scope-label">特征</span>` +
      `<div class="chips">${chipList(c.featuresets || [], "全部特征", "amber")}</div>` +
    `</div>` +
    `<button class="ghost-button" type="button" title="${esc(summary.notice)}">管理范围</button>`;
}

function scopePill(label, value, hint, kind) {
  return `<div class="scope-pill ${esc(kind)}">` +
    `<span>${esc(label)}</span><b>${esc(value)}</b><small>${esc(hint)}</small>` +
  `</div>`;
}

function selectedDatasetNames() {
  return [...(activeDatasets || new Set())];
}

function renderWorkbench() {
  if (!$("workbench-card")) return;
  const state = $("api-state");
  const jobState = $("job-state");
  if (!CATALOG) {
    state.textContent = WORKBENCH_ERROR || "未连接";
    state.className = "api-state fail";
    renderResourceList("run-datasets", [], resourceSelection.datasets);
    renderResourceList("run-estimators", [], resourceSelection.estimators);
    renderResourceList("run-featuresets", [], resourceSelection.featuresets);
    renderRunPlan();
    renderTrainingWorkflow(evaluationPlanSummary(buildCurrentEvaluationPayload()));
    renderTrainingArtifacts();
    jobState.textContent = "后端未连接，静态结果仍可浏览。";
    wireWorkbenchActions(false);
    return;
  }

  const options = catalogOptions(CATALOG);
  state.textContent = "已连接";
  state.className = "api-state ok";
  renderResourceList("run-datasets", options.datasets, resourceSelection.datasets, (item) => `${item.rows.toLocaleString()} 行`);
  renderResourceList("run-estimators", options.estimators, resourceSelection.estimators, (item) => item.featureMode || item.backend);
  renderResourceList("run-featuresets", options.featuresets, resourceSelection.featuresets, (item) => item.columns?.length ? `${item.columns.length} 列` : "选择器");
  renderRunPlan();
  renderTrainingWorkflow(evaluationPlanSummary(buildCurrentEvaluationPayload()));
  renderTrainingArtifacts();
  renderJobState();
  wireWorkbenchActions(true);
}

function renderResourceList(id, items, selected, hint = () => "", onChange = renderRunPlan) {
  const node = $(id);
  if (!node) return;
  if (!items.length) {
    node.innerHTML = '<div class="note">暂无可用资源</div>';
    return;
  }
  node.innerHTML = items.map((item) => {
    const checked = selected.has(item.name);
    return `<label class="resource-item">` +
      `<input type="checkbox" value="${esc(item.name)}" ${checked ? "checked" : ""} />` +
      `<span class="name" title="${esc(item.name)}">${esc(item.name)}</span>` +
      `<span class="badge" title="${esc(hint(item))}">${esc(item.badge || "")}</span>` +
      `</label>`;
  }).join("");
  node.querySelectorAll("input[type='checkbox']").forEach((input) => {
    input.onchange = () => {
      if (input.checked) selected.add(input.value);
      else selected.delete(input.value);
      if (onChange) onChange();
    };
  });
}

function renderRunPlan() {
  const node = $("run-plan");
  if (!node) return;
  const payload = buildCurrentEvaluationPayload();
  const plan = evaluationPlanSummary(payload);
  const cls = plan.ready ? "ready" : "incomplete";
  node.innerHTML =
    `<div class="run-plan-main ${cls}">` +
      `<span>预计组合数</span>` +
      `<b>${esc(plan.comboLabel)}</b>` +
      `<small>${esc(plan.partsLabel)}</small>` +
    `</div>` +
    `<div class="run-plan-note">${plan.ready ? "评测完成后会自动发布到仪表盘数据。" : "请至少选择一个数据集、算法、特征、协议和 seed。"}</div>`;
  renderWorkbenchSteps(plan);
  renderTrainingWorkflow(plan);
  renderTrainingArtifacts();
}

function renderWorkbenchSteps(plan = null) {
  const node = $("workbench-steps");
  if (!node) return;
  const jobStatus = CURRENT_JOB?.status || "";
  const items = [
    ["上传", CATALOG ? "complete" : "pending", CATALOG ? "资源目录已载入" : "等待后端连接"],
    ["校验", plan?.ready ? "complete" : "pending", plan?.ready ? "选择完整" : "选择不完整"],
    ["评测", jobStatus === "running" || jobStatus === "queued" ? "active" : (jobStatus === "succeeded" ? "complete" : "pending"), jobStatus || "未启动"],
    ["发布", jobStatus === "succeeded" ? "complete" : "pending", jobStatus === "succeeded" ? "已发布到仪表盘" : "等待评测完成"],
  ];
  node.innerHTML = items.map(([label, state, hint], index) =>
    `<div class="workbench-step ${state}">` +
      `<span>${index + 1}</span>` +
      `<b>${esc(label)}</b>` +
      `<small>${esc(hint)}</small>` +
    `</div>`).join("");
}

function renderTrainingWorkflow(plan = null) {
  const node = $("training-workflow");
  if (!node) return;
  const jobStatus = CURRENT_JOB?.status || "";
  const steps = [
    ["资源接入", CATALOG ? "complete" : "pending", CATALOG ? "资源目录已加载" : "等待后端连接"],
    ["组合配置", plan?.ready ? "active" : "pending", plan?.partsLabel || "请选择数据、算法、特征"],
    ["训练执行", jobStatus === "running" || jobStatus === "queued" ? "active" : (jobStatus === "succeeded" ? "complete" : "pending"), jobStatus || "尚未启动"],
    ["模型发布", jobStatus === "succeeded" ? "complete" : "pending", "Pickle 模型进入模型资产库"],
  ];
  node.innerHTML =
    `<div class="panel-title"><span>训练流程</span><b>实验编排台</b></div>` +
    `<div class="panel-scroll">` +
    `<div class="workflow-list">` +
      steps.map(([label, state, hint], index) =>
        `<div class="workflow-step ${esc(state)}">` +
          `<span>${index + 1}</span>` +
          `<div><b>${esc(label)}</b><small>${esc(hint)}</small></div>` +
        `</div>`).join("") +
    `</div>` +
    `<div class="note">训练页负责上传源码算法和特征方案，完成后产出可复用 Pickle 模型。</div>` +
    `</div>`;
}

function renderTrainingArtifacts() {
  const node = $("training-artifacts");
  if (!node) return;
  const summary = trainingArtifactSummary(DATA?.config || {}, currentModelCatalog());
  const jobStatus = CURRENT_JOB?.status || "未启动";
  node.innerHTML =
    `<div class="panel-title"><span>训练产物</span><b>模型资产库</b></div>` +
    `<div class="panel-scroll">` +
    `<div class="artifact-stack">` +
      artifactMetric("Pickle 模型", summary.label, `${summary.generatedCount} 训练产出 / ${summary.uploadedCount} 外部上传`, "model") +
      artifactMetric("数据集范围", `${summary.datasetCount || 0} 个数据集`, "训练矩阵使用的数据源范围", "dataset") +
      artifactMetric("任务状态", jobStatus, CURRENT_JOB?.id || "等待新任务", "job") +
    `</div>` +
    `<button class="secondary-action" data-mode-shortcut="models">查看模型资产库</button>` +
    `</div>`;
  const button = node.querySelector("[data-mode-shortcut='models']");
  if (button) {
    button.onclick = () => {
      activeMode = "models";
      render();
    };
  }
}

function artifactMetric(label, value, hint, kind) {
  return `<div class="artifact-metric ${esc(kind)}">` +
    `<span>${esc(label)}</span>` +
    `<b>${esc(value)}</b>` +
    `<small>${esc(hint)}</small>` +
  `</div>`;
}

function buildCurrentEvaluationPayload() {
  return buildEvaluationPayload({
    datasets: selectedValues("run-datasets"),
    estimators: selectedValues("run-estimators"),
    featuresets: selectedValues("run-featuresets"),
    protocols: [...document.querySelectorAll("input[name='run-protocol']:checked")].map((item) => item.value),
    seedsText: $("job-seeds")?.value || "0",
  });
}

function renderJobState() {
  const node = $("job-state");
  if (!node) return;
  if (!CURRENT_JOB) {
    node.textContent = "上传资源通过合约校验后，可启动一轮新的本机评测。";
    return;
  }
  const bits = [`任务 ${CURRENT_JOB.id}`, `状态 ${CURRENT_JOB.status}`];
  if (CURRENT_JOB.run_dir) bits.push(`结果 ${CURRENT_JOB.run_dir}`);
  if (CURRENT_JOB.status === "succeeded") bits.push("已发布到仪表盘");
  if (CURRENT_JOB.error) bits.push(`错误 ${CURRENT_JOB.error}`);
  node.innerHTML = bits.map((bit, i) => i === 0 ? `<strong>${esc(bit)}</strong>` : esc(bit)).join(" / ");
}

function wireWorkbenchActions(enabled) {
  document.querySelectorAll("[data-upload-kind]").forEach((button) => {
    button.disabled = !enabled;
    button.onclick = enabled ? () => uploadResource(button.dataset.uploadKind) : null;
  });
  const start = $("start-eval");
  if (start) {
    start.disabled = !enabled;
    start.onclick = enabled ? startEvaluation : null;
  }
  document.querySelectorAll("input[name='run-protocol']").forEach((input) => {
    input.onchange = renderRunPlan;
  });
  const seeds = $("job-seeds");
  if (seeds) seeds.oninput = renderRunPlan;
}

function renderModeTabs() {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    const selected = button.dataset.mode === activeMode;
    button.setAttribute("aria-selected", String(selected));
    button.onclick = () => {
      activeMode = button.dataset.mode;
      render();
    };
  });
  document.querySelectorAll("[data-side-mode]").forEach((button) => {
    const selected = button.dataset.sideMode === activeMode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-current", selected ? "page" : "false");
    button.onclick = () => {
      activeMode = button.dataset.sideMode;
      render();
    };
  });
  document.querySelectorAll("[data-mode-shortcut]").forEach((button) => {
    button.onclick = () => {
      activeMode = button.dataset.modeShortcut;
      render();
    };
  });
  document.querySelectorAll("[data-mode-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.modePanel !== activeMode;
  });
}

function renderTaskHistoryDrawer() {
  const drawer = $("task-history-drawer");
  const backdrop = $("task-history-backdrop");
  if (!drawer || !backdrop) return;
  drawer.hidden = !taskHistoryOpen;
  backdrop.hidden = !taskHistoryOpen;
  drawer.setAttribute("aria-hidden", taskHistoryOpen ? "false" : "true");

  const history = taskHistoryRows(CATALOG || {}, taskHistoryFilters);
  lastTaskHistoryRows = history.rows;
  const status = $("task-history-status");
  if (status) {
    const current = taskHistoryFilters.status || "all";
    status.innerHTML = `<option value="all">全部</option>` +
      history.statusOptions.map((item) => `<option value="${esc(item)}">${esc(statusLabel(item))}</option>`).join("");
    status.value = history.statusOptions.includes(current) ? current : "all";
    if (status.value !== current) taskHistoryFilters.status = status.value;
  }
  setSelectValue("task-history-kind", taskHistoryFilters.kind);
  setSelectValue("task-history-strategy", taskHistoryFilters.strategy);
  const dataset = $("task-history-dataset");
  if (dataset) dataset.value = taskHistoryFilters.dataset || "";

  const summary = $("task-history-summary");
  if (summary) {
    const visibleCount = Math.min(history.filteredCount, 80);
    summary.textContent = CATALOG
      ? `显示 ${visibleCount} / ${history.totalCount} 个任务${history.filteredCount > 80 ? "，请继续筛选查看更多" : ""}`
      : "后端未连接，暂无任务历史。";
  }
  const message = $("task-history-message");
  if (message) {
    message.textContent = taskHistoryMessage;
    message.classList.toggle("active", Boolean(taskHistoryMessage));
  }
  const list = $("task-history-list");
  if (list) {
    list.innerHTML = history.rows.length
      ? history.rows.slice(0, 80).map(renderTaskHistoryRow).join("")
      : `<div class="task-history-empty">暂无匹配任务</div>`;
  }
  wireTaskHistoryDrawer();
}

function wireTaskHistoryDrawer() {
  const open = $("open-task-history");
  if (open) {
    open.onclick = () => {
      taskHistoryOpen = true;
      renderTaskHistoryDrawer();
    };
  }
  const close = $("close-task-history");
  if (close) close.onclick = closeTaskHistoryDrawer;
  const backdrop = $("task-history-backdrop");
  if (backdrop) backdrop.onclick = closeTaskHistoryDrawer;
  const bindSelect = (id, key) => {
    const node = $(id);
    if (!node) return;
    node.onchange = () => {
      taskHistoryFilters = { ...taskHistoryFilters, [key]: node.value };
      renderTaskHistoryDrawer();
    };
  };
  bindSelect("task-history-kind", "kind");
  bindSelect("task-history-status", "status");
  bindSelect("task-history-strategy", "strategy");
  const dataset = $("task-history-dataset");
  if (dataset) {
    dataset.oninput = () => {
      taskHistoryFilters = { ...taskHistoryFilters, dataset: dataset.value };
      renderTaskHistoryDrawer();
    };
  }
  document.querySelectorAll("[data-history-action='rerun']").forEach((button) => {
    button.onclick = () => rerunTaskFromHistory(button.dataset.taskId || "");
  });
  document.querySelectorAll("[data-history-action='export']").forEach((button) => {
    button.onclick = () => exportTaskAudit(button.dataset.taskId || "");
  });
  document.querySelectorAll(".task-history-detail").forEach((detail) => {
    detail.ontoggle = () => {
      const taskId = detail.dataset.taskId || "";
      if (!taskId) return;
      if (detail.open) taskHistoryExpandedIds.add(taskId);
      else taskHistoryExpandedIds.delete(taskId);
    };
  });
}

function closeTaskHistoryDrawer() {
  taskHistoryOpen = false;
  renderTaskHistoryDrawer();
}

function setSelectValue(id, value) {
  const node = $(id);
  if (node) node.value = value || "all";
}

function renderTaskHistoryRow(row) {
  const expanded = taskHistoryExpandedIds.has(row.id);
  const busy = taskHistoryBusyId === row.id;
  return `<article class="task-history-item ${esc(row.kind)} ${esc(row.status)}">` +
    `<div class="task-history-main">` +
      `<span>${esc(row.kindLabel)} · ${esc(row.statusLabel)}</span>` +
      `<b>${esc(row.id || "未知任务")}</b>` +
      `<small>${esc(row.detail)}</small>` +
    `</div>` +
    `<div class="task-history-meta">` +
      `<span>${esc(row.strategyLabel)}</span>` +
      `<span>${esc(row.datasetLabel)}</span>` +
      `<span>${esc(row.runDir || row.error || row.createdAt || "暂无 run 目录")}</span>` +
    `</div>` +
    `<div class="task-history-actions">` +
      `<button type="button" data-history-action="rerun" data-task-id="${esc(row.id)}" ${busy ? "disabled" : ""}>重新运行同配置</button>` +
      `<button type="button" data-history-action="export" data-task-id="${esc(row.id)}">导出审计 JSON</button>` +
    `</div>` +
    `<details class="task-history-detail" data-task-id="${esc(row.id)}" ${expanded ? "open" : ""}>` +
      `<summary>任务详情</summary>` +
      `<pre>${esc(row.detailJson || "{}")}</pre>` +
    `</details>` +
  `</article>`;
}

function statusLabel(status) {
  if (status === "succeeded") return "成功";
  if (status === "failed") return "失败";
  if (status === "running") return "运行中";
  if (status === "queued") return "排队中";
  return status || "未知";
}

async function rerunTaskFromHistory(taskId) {
  const row = lastTaskHistoryRows.find((item) => item.id === taskId);
  if (!row) return;
  if (!CATALOG) {
    taskHistoryMessage = "后端未连接，无法重新运行任务。";
    renderTaskHistoryDrawer();
    setJobMessage("后端未连接，无法重新运行任务。");
    return;
  }
  taskHistoryBusyId = taskId;
  taskHistoryMessage = "正在提交同配置任务...";
  renderTaskHistoryDrawer();
  try {
    const job = await postJson(row.rerunEndpoint, row.rerunPayload);
    CURRENT_JOB = row.kind === "inference" ? { kind: "inference", ...job } : job;
    await loadBackendCatalog();
    taskHistoryBusyId = "";
    taskHistoryMessage = `已重新启动${row.kind === "inference" ? "推理" : "训练"}任务 ${job.id}`;
    taskHistoryOpen = false;
    if (row.kind === "inference") {
      activeMode = "inference";
      render();
      setInferenceMessage(`已重新启动推理任务 ${job.id}`, "active");
      pollInferenceJob(job.id);
    } else {
      activeMode = "training";
      render();
      setJobMessage(`已重新启动训练任务 ${job.id}`);
      pollJob(job.id);
    }
  } catch (err) {
    taskHistoryBusyId = "";
    taskHistoryMessage = `重新运行失败：${err.message}`;
    renderTaskHistoryDrawer();
    const message = `重新运行失败：${err.message}`;
    if (row.kind === "inference") setInferenceMessage(message, "fail");
    else setJobMessage(message);
  }
}

function exportTaskAudit(taskId) {
  const row = lastTaskHistoryRows.find((item) => item.id === taskId);
  if (!row) return;
  const audit = {
    ...(row.auditJson || {}),
    exported_at: new Date().toISOString(),
  };
  const content = JSON.stringify(audit, null, 2);
  const blob = new Blob([content], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${row.kind || "task"}-${row.id || "audit"}-audit.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  taskHistoryMessage = `已导出审计 JSON：${link.download}`;
  renderTaskHistoryDrawer();
}

function currentModelCatalog() {
  const byName = new Map();
  (DATA?.modelCatalog?.models || []).forEach((item) => byName.set(item.name, item));
  (CATALOG?.models || []).forEach((item) => byName.set(item.name, item));
  return { models: [...byName.values()] };
}

function renderInferencePanel(dsOptions) {
  if (!$("inference-panel")) return;
  const models = modelOptions(currentModelCatalog());
  if (!inferenceSelection.models.size && models.length) {
    preferredInferenceModels(models, DATA?.inference || {}).forEach((name) => inferenceSelection.models.add(name));
  }
  if (!inferenceSelection.datasets.size) dsOptions.forEach((item) => inferenceSelection.datasets.add(item.name));

  renderResourceList(
    "inference-models",
    models,
    inferenceSelection.models,
    (item) => `${item.estimator || "模型"} / ${item.featureset || "特征"}`,
    () => renderInferencePanel(dsOptions),
  );
  renderResourceList(
    "inference-datasets",
    dsOptions,
    inferenceSelection.datasets,
    (item) => `${item.rows.toLocaleString()} 行`,
    () => renderInferencePanel(dsOptions),
  );

  const rows = inferenceRows(
    DATA.inference || {},
    [...inferenceSelection.models],
    [...inferenceSelection.datasets],
  );
  const summary = inferenceSummary({ records: rows });
  renderInferenceCompatibilityPanel(models, dsOptions);
  renderInferenceInsights(inferenceInsightSummary(rows), currentInferenceAuditSummary());
  $("inference-results").innerHTML = renderInferenceSummary(summary) + renderInferenceTable(rows);
  const actionState = inferenceActionState({
    backendConnected: Boolean(CATALOG),
    selectedModelCount: models.filter((item) => inferenceSelection.models.has(item.name)).length,
    selectedDatasetCount: dsOptions.filter((item) => inferenceSelection.datasets.has(item.name)).length,
    currentJob: CURRENT_JOB?.kind === "inference" ? CURRENT_JOB : null,
  });
  renderInferenceActionState(actionState);
  renderInferencePolicyControls(models, dsOptions);
  const start = $("start-inference");
  if (start) start.onclick = CATALOG ? startInferenceEvaluationWithPolicy : null;
  if (start) start.disabled = actionState.disabled;
}

function renderInferenceActionState(actionState) {
  const node = $("inference-job-state");
  if (!node) return;
  node.textContent = actionState.message;
  node.className = `job-state inference-job-state ${actionState.tone || "ready"}`;
}

function renderInferencePolicyControls(models, dsOptions) {
  ensureInferencePolicyUi();
  document.querySelectorAll("[data-inference-strategy]").forEach((button) => {
    const selected = button.dataset.inferenceStrategy === inferenceLaunchStrategy;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", selected ? "true" : "false");
    button.onclick = () => {
      inferenceLaunchStrategy = button.dataset.inferenceStrategy || "compatible_only";
      renderInferencePanel(currentInferenceDatasetOptions());
    };
  });
  const note = $("inference-policy-note");
  if (!note) return;
  const plan = inferenceLaunchPlan({
    modelNames: [...inferenceSelection.models],
    datasetNames: [...inferenceSelection.datasets],
    models,
    datasets: dsOptions,
    strategy: inferenceLaunchStrategy,
  });
  note.textContent = inferenceLaunchStrategy === "compatible_only"
    ? `${plan.payload.datasets.length} 个数据集将提交；${plan.skippedCount ? plan.message : "未发现需要跳过的数据集。"}`
    : "将按当前勾选提交，后端运行时校验不兼容组合。";
}

function ensureInferencePolicyUi() {
  const start = $("start-inference");
  if (!start) return;
  if (!document.querySelector("[data-inference-strategy]")) {
    const actions = document.createElement("div");
    actions.className = "inference-actions";
    const policy = document.createElement("div");
    policy.className = "strategy-control";
    policy.setAttribute("aria-label", "推理启动策略");
    policy.innerHTML =
      `<button type="button" data-inference-strategy="compatible_only" class="active" aria-pressed="true">仅运行兼容组合</button>` +
      `<button type="button" data-inference-strategy="include_risk" aria-pressed="false">包含风险组合</button>`;
    start.parentNode.insertBefore(actions, start);
    actions.appendChild(policy);
    actions.appendChild(start);
  }
  if (!$("inference-policy-note")) {
    const note = document.createElement("div");
    note.id = "inference-policy-note";
    note.className = "policy-note";
    note.setAttribute("aria-live", "polite");
    note.textContent = "默认仅提交 schema 兼容的数据集。";
    const jobState = $("inference-job-state");
    if (jobState?.parentNode) jobState.parentNode.insertBefore(note, jobState);
  }
}

function renderInferenceCompatibilityPanel(models, dsOptions) {
  const node = $("inference-compatibility");
  if (!node) return;
  const selectedModels = models.filter((item) => inferenceSelection.models.has(item.name));
  const selectedDatasets = dsOptions.filter((item) => inferenceSelection.datasets.has(item.name));
  const compatibility = inferenceCompatibilitySummary({ models: selectedModels, datasets: selectedDatasets });
  const riskRows = compatibility.rows.filter((row) => row.status !== "ok");
  const details = riskRows.slice(0, 8).map((row) =>
    `<div class="compat-row ${esc(row.status)}">` +
      `<b>${esc(row.model || "未知模型")}</b>` +
      `<span>${esc(row.dataset || "未知数据集")}</span>` +
      `<small>${row.status === "warn" ? `缺失：${esc(row.missingColumns.join(", "))}` : "缺少列清单，运行时确认"}</small>` +
    `</div>`,
  ).join("");
  const hiddenCount = Math.max(0, riskRows.length - 8);
  const noteKind = compatibility.status === "idle" ? "unknown" : compatibility.status;
  const noteTitle = compatibility.status === "warn"
    ? "发现 schema 风险"
    : (compatibility.status === "unknown" ? "需要运行时确认" : "可启动推理测评");

  node.innerHTML =
    `<div class="panel-title"><span>推理输入</span><b>兼容性预检</b></div>` +
    `<div class="panel-scroll">` +
    `<div class="selector-summary">` +
      artifactMetric("模型", `${selectedModels.length}/${models.length || 0}`, selectedModels.map((item) => item.name).slice(0, 3).join(" / ") || "暂无模型", "model") +
      artifactMetric("目标数据集", `${selectedDatasets.length}/${dsOptions.length || 0}`, selectedDatasets.map((item) => item.name).slice(0, 3).join(" / ") || "暂无数据集", "dataset") +
    `</div>` +
    `<div class="compat-note ${esc(noteKind)}">` +
      `<b>${esc(noteTitle)}</b>` +
      `<span>${esc(compatibility.message)}</span>` +
      `<small>${compatibility.okCount} 可推理 / ${compatibility.warnCount} 缺失列 / ${compatibility.unknownCount} 待确认</small>` +
    `</div>` +
    (details ? `<div class="compat-list">${details}${hiddenCount ? `<div class="compat-more">还有 ${hiddenCount} 个风险组合</div>` : ""}</div>` : "") +
    `</div>`;
}

function renderInferenceCompatibility(models, dsOptions) {
  const node = $("inference-compatibility");
  if (!node) return;
  const selectedModels = models.filter((item) => inferenceSelection.models.has(item.name));
  const selectedDatasets = dsOptions.filter((item) => inferenceSelection.datasets.has(item.name));
  const missingFeatureModels = selectedModels.filter((item) => !(item.featureColumns || []).length);
  node.innerHTML =
    `<div class="panel-title"><span>推理输入</span><b>兼容性预检</b></div>` +
    `<div class="panel-scroll">` +
    `<div class="selector-summary">` +
      artifactMetric("模型", `${selectedModels.length}/${models.length || 0}`, selectedModels.map((item) => item.name).slice(0, 3).join(" / ") || "暂无模型", "model") +
      artifactMetric("目标数据集", `${selectedDatasets.length}/${dsOptions.length || 0}`, selectedDatasets.map((item) => item.name).slice(0, 3).join(" / ") || "暂无数据集", "dataset") +
    `</div>` +
    `<div class="compat-note ${missingFeatureModels.length ? "warn" : "ok"}">` +
      `<b>${missingFeatureModels.length ? "需要复核特征 schema" : "可启动推理测评"}</b>` +
      `<span>${missingFeatureModels.length ? `${missingFeatureModels.length} 个模型缺少特征列清单，运行前请确认 manifest。` : "已选择模型和目标数据集，后端会在运行时再次校验特征列。"}</span>` +
    `</div>` +
    `</div>`;
}

function renderInferenceInsights(summary, audit = currentInferenceAuditSummary()) {
  const node = $("inference-insights");
  if (!node) return;
  node.innerHTML =
    `<div class="panel-title"><span>自动结论</span><b>迁移分析</b></div>` +
    `<div class="panel-scroll">` +
    `<div class="insight-stack">` +
      insightCard("推荐模型", summary.bestTransfer?.model || "暂无", summary.recommendation, "ok") +
      insightCard("速度参考", summary.fastest?.model || "暂无", summary.fastest ? `${fmt(summary.fastest.ms_per_1000_samples, 2)} ms / 1000 样本` : "暂无成功推理记录", "speed") +
      insightCard("风险提示", `${summary.failedCount} 个失败组合`, summary.risk, summary.failedCount ? "warn" : "ok") +
    `</div>` +
    renderInferenceAudit(audit) +
    `<button class="secondary-action" data-mode-shortcut="models">查看模型资产</button>` +
    `</div>`;
  const button = node.querySelector("[data-mode-shortcut='models']");
  if (button) {
    button.onclick = () => {
      activeMode = "models";
      render();
    };
  }
}

function renderInferenceAudit(audit) {
  if (!audit?.hasAudit) {
    return `<div class="audit-grid">` +
      `<div class="audit-card muted"><span>任务审计</span><b>暂无记录</b><small>启动推理任务后显示策略、过滤数据集和 run 目录。</small></div>` +
    `</div>`;
  }
  return `<div class="audit-grid" aria-label="推理任务审计">` +
    auditCard("策略", audit.strategyLabel, audit.sourceLabel) +
    auditCard("数据集提交", `${audit.submittedCount}/${audit.selectedCount}`, `跳过 ${audit.skippedCount} 个`) +
    auditCard("跳过数据集", audit.skippedLabel, audit.skippedCount ? "schema 风险过滤" : "未过滤") +
    auditCard("Run 目录", audit.runDir || audit.jobId || "暂无", audit.jobId ? `任务 ${audit.jobId}` : (audit.status || "等待任务")) +
  `</div>`;
}

function auditCard(label, value, hint) {
  return `<div class="audit-card"><span>${esc(label)}</span><b>${esc(value)}</b><small>${esc(hint)}</small></div>`;
}

function insightCard(label, value, hint, kind) {
  return `<div class="insight-card ${esc(kind)}">` +
    `<span>${esc(label)}</span>` +
    `<b>${esc(value)}</b>` +
    `<small>${esc(hint)}</small>` +
  `</div>`;
}

function renderInferenceSummary(summary) {
  const card = (label, row, value) =>
    `<div class="kpi inference-kpi"><div class="lab">${esc(label)}</div>` +
    `<div class="val">${row ? esc(row.model) : "暂无"}</div>` +
    `<div class="hint">${row ? esc(value(row)) : "暂无推理测速数据"}</div></div>`;
  return `<div class="inference-kpis">` +
    card("迁移精度", summary.bestTransfer, (row) => `MAE ${fmt(row.mae)}`) +
    card("最低延迟", summary.fastest, (row) => `${fmt(row.ms_per_1000_samples, 2)} ms / 1000 样本`) +
    card("最高吞吐", summary.highestThroughput, (row) => `${fmt(row.samples_per_second, 0)} 样本/秒`) +
    `</div>`;
}

function renderInferenceTable(rows) {
  if (!rows.length) {
    return '<div class="note">暂无推理测速数据。上传或选择模型后，可在新数据集上启动 inference-only 测评。</div>';
  }
  const body = rows.map((row) =>
    `<tr class="${row.status === "failed" ? "failed" : ""}">` +
    `<td><span class="algo">${esc(row.model)}</span><small>${esc(row.estimator || "")}</small></td>` +
    `<td>${esc(row.dataset)}</td>` +
    `<td>${esc(row.featureset || "full")}</td>` +
    `<td class="tnum">${fmt(row.mae)}</td>` +
    `<td class="tnum">${fmt(row.rmse)}</td>` +
    `<td class="tnum">${fmt(row.predict_ms, 2)}</td>` +
    `<td class="tnum">${fmt(row.ms_per_1000_samples, 2)}</td>` +
    `<td class="tnum">${fmt(row.samples_per_second, 0)}</td>` +
    `<td>${row.status === "succeeded" ? "成功" : esc(row.error || "失败")}</td>` +
    `</tr>`).join("");
  return `<div class="inference-table"><table><thead><tr>` +
    `<th>模型</th><th>数据集</th><th>特征集</th><th>MAE</th><th>RMSE</th>` +
    `<th>predict ms</th><th>ms/1000</th><th>样本/秒</th><th>状态</th>` +
    `</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderModelLibrary() {
  if (!$("model-library-table")) return;
  const models = modelOptions(currentModelCatalog());
  const assets = modelAssetRows(currentModelCatalog(), DATA?.inference || {});
  renderModelLibrarySummary(assets);
  if (!models.length) {
    $("model-library-table").innerHTML = '<div class="note">暂无模型。训练测评完成后会产出模型，也可以在推理测评页上传 Pickle 模型。</div>';
    return;
  }
  const rows = assets.map((item) =>
    `<tr>` +
    `<td><span class="algo">${esc(item.name)}</span><small>${esc(item.badge)}</small></td>` +
    `<td>${esc(item.estimator || "未知")}</td>` +
    `<td>${esc(item.featureset || "未知")}</td>` +
    `<td>${esc((item.trainedOn || []).join(" / ") || "外部上传")}</td>` +
    `<td class="tnum">${esc(String(item.featureColumnCount || 0))}</td>` +
    `<td class="tnum">${item.latestMae == null ? "暂无" : fmt(item.latestMae)}</td>` +
    `<td class="tnum">${item.latestSpeed == null ? "暂无" : `${fmt(item.latestSpeed, 2)} ms`}</td>` +
    `</tr>`).join("");
  $("model-library-table").innerHTML =
    `<div class="inference-table asset-table"><table><thead><tr>` +
    `<th>模型</th><th>算法</th><th>特征集</th><th>训练数据</th><th>特征数</th><th>最近 MAE</th><th>最近速度</th>` +
    `</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderModelLibrarySummary(assets = []) {
  const node = $("model-library-summary");
  if (!node) return;
  const generated = assets.filter((item) => item.source !== "upload").length;
  const uploaded = assets.filter((item) => item.source === "upload").length;
  const withInference = assets.filter((item) => item.latestMae != null).length;
  const best = assets.find((item) => item.latestMae != null);
  const audit = currentInferenceAuditSummary();
  node.innerHTML =
    artifactMetric("模型总数", `${assets.length}`, `${generated} 训练产出 / ${uploaded} 外部上传`, "model") +
    artifactMetric("已推理模型", `${withInference}`, "拥有最近 inference-only 指标", "job") +
    artifactMetric("最佳资产", best?.name || "暂无", best ? `最近 MAE ${fmt(best.latestMae)}` : "等待推理测评", "dataset") +
    renderModelAuditSummary(audit);
}

function renderModelAuditSummary(audit) {
  return artifactMetric(
    "最近策略",
    audit?.hasAudit ? audit.strategyLabel : "暂无",
    audit?.hasAudit ? `${audit.sourceLabel} / 跳过 ${audit.skippedCount} 个` : "等待推理任务",
    "job",
  );
}

async function uploadResource(kind) {
  if (kind === "model") {
    const modelFile = $("upload-model-file")?.files?.[0];
    const manifestFile = $("upload-model-manifest")?.files?.[0];
    if (!modelFile || !manifestFile) {
      setInferenceMessage("请选择模型 Pickle 文件和 model_manifest.json。", "warn");
      return;
    }
    const name = ($("upload-model-name").value || modelFile.name.replace(/\.[^.]+$/, "")).trim();
    try {
      setInferenceMessage("正在上传模型...", "active");
      const response = await postJson("/api/upload/model", {
        name,
        modelFilename: modelFile.name,
        modelContentBase64: await fileToBase64(modelFile),
        manifestFilename: manifestFile.name,
        manifestContentBase64: await fileToBase64(manifestFile),
      });
      inferenceSelection.models.add(response.resource.name);
      await loadBackendCatalog();
      render();
      setInferenceMessage(`已上传模型 ${response.resource.name}`, "ok");
    } catch (err) {
      setInferenceMessage(`模型上传失败：${err.message}`, "fail");
    }
    return;
  }

  const spec = {
    dataset: ["upload-dataset-name", "upload-dataset-file", "/api/upload/dataset", "datasets"],
    algorithm: ["upload-algorithm-name", "upload-algorithm-file", "/api/upload/algorithm", "estimators"],
    featureset: ["upload-featureset-name", "upload-featureset-file", "/api/upload/featureset", "featuresets"],
  }[kind];
  if (!spec) return;
  const [nameId, fileId, endpoint, selectionKey] = spec;
  const file = $(fileId)?.files?.[0];
  if (!file) {
    setJobMessage("请选择上传文件。");
    return;
  }
  const name = ($(nameId).value || file.name.replace(/\.[^.]+$/, "")).trim();
  try {
    setJobMessage("正在上传资源...");
    const resource = await postJson(endpoint, {
      name,
      filename: file.name,
      contentBase64: await fileToBase64(file),
    });
    resourceSelection[selectionKey].add(resource.resource.name);
    await loadBackendCatalog();
    render();
    setJobMessage(`已上传 ${resource.resource.name}`);
  } catch (err) {
    setJobMessage(`上传失败：${err.message}`);
  }
}

async function startEvaluation() {
  const payload = buildCurrentEvaluationPayload();
  if (!payload.datasets.length || !payload.estimators.length || !payload.featuresets.length) {
    setJobMessage("请至少选择一个数据集、算法和特征。");
    return;
  }
  try {
    CURRENT_JOB = await postJson("/api/jobs", payload);
    renderJobState();
    renderRunPlan();
    pollJob(CURRENT_JOB.id);
  } catch (err) {
    setJobMessage(`启动失败：${err.message}`);
  }
}

async function startInferenceEvaluationWithPolicy() {
  const models = modelOptions(currentModelCatalog());
  const dsOptions = currentInferenceDatasetOptions();
  const plan = inferenceLaunchPlan({
    modelNames: selectedValues("inference-models"),
    datasetNames: selectedValues("inference-datasets"),
    models,
    datasets: dsOptions,
    strategy: inferenceLaunchStrategy,
  });
  const payload = plan.payload;
  if (!payload.models.length || !payload.datasets.length) {
    setInferenceMessage(plan.skippedCount && !payload.datasets.length
      ? "兼容策略已跳过全部目标数据集，请改选数据集或切换为包含风险组合。"
      : "请至少选择一个模型和一个数据集。", "warn");
    return;
  }
  try {
    setInferenceMessage(plan.skippedCount
      ? `正在启动推理任务，${plan.message}`
      : "正在启动推理任务...", "active");
    CURRENT_JOB = { kind: "inference", ...(await postJson("/api/inference-jobs", payload)) };
    renderInferencePanel(currentInferenceDatasetOptions());
    pollInferenceJob(CURRENT_JOB.id);
  } catch (err) {
    setInferenceMessage(`推理任务启动失败：${err.message}`, "fail");
  }
}

async function startInferenceEvaluation() {
  const payload = buildInferencePayload({
    models: selectedValues("inference-models"),
    datasets: selectedValues("inference-datasets"),
  });
  if (!payload.models.length || !payload.datasets.length) {
    setInferenceMessage("请至少选择一个模型和一个数据集。", "warn");
    return;
  }
  try {
    setInferenceMessage("正在启动推理任务...", "active");
    CURRENT_JOB = { kind: "inference", ...(await postJson("/api/inference-jobs", payload)) };
    renderInferencePanel(currentInferenceDatasetOptions());
    pollInferenceJob(CURRENT_JOB.id);
  } catch (err) {
    setInferenceMessage(`推理任务启动失败：${err.message}`, "fail");
  }
}

function selectedValues(id) {
  return [...($(id)?.querySelectorAll("input[type='checkbox']:checked") || [])].map((input) => input.value);
}

async function pollJob(jobId) {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    try {
      const response = await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(jobId)}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      CURRENT_JOB = await response.json();
      renderWorkbench();
      if (CURRENT_JOB.status === "succeeded") {
        await load({ cacheBust: true });
        return;
      }
      if (CURRENT_JOB.status === "failed") return;
    } catch (err) {
      setJobMessage(`任务轮询失败：${err.message}`);
      return;
    }
  }
}

async function pollInferenceJob(jobId) {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    try {
      const response = await fetch(`${API_BASE}/api/inference-jobs/${encodeURIComponent(jobId)}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      CURRENT_JOB = { kind: "inference", ...(await response.json()) };
      renderInferencePanel(currentInferenceDatasetOptions());
      if (CURRENT_JOB.status === "succeeded") {
        await load({ cacheBust: true });
        activeMode = "inference";
        render();
        return;
      }
      if (CURRENT_JOB.status === "failed") return;
    } catch (err) {
      setInferenceMessage(`推理任务轮询失败：${err.message}`, "fail");
      return;
    }
  }
}

async function postJson(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(reader.error || new Error("文件读取失败"));
    reader.readAsDataURL(file);
  });
}

function setJobMessage(message) {
  const node = $("job-state");
  if (node) node.textContent = message;
}

function setInferenceMessage(message, tone = "ready") {
  const node = $("inference-job-state");
  if (!node) {
    setJobMessage(message);
    return;
  }
  node.textContent = message;
  node.className = `job-state inference-job-state ${tone}`;
}

function renderDatasetPanel(options) {
  const totalRows = options.reduce((sum, item) => sum + item.rows, 0);
  const selected = options.filter((item) => activeDatasets.has(item.name));
  const selectedRows = selected.reduce((sum, item) => sum + item.rows, 0);
  const metricMode = scoped.usedDatasetLevel ? "按所选数据集折聚合" : "全量聚合结果";
  $("dataset-summary").innerHTML = [
    metric("选中样本", selectedRows.toLocaleString(), totalRows ? `占 QC 样本 ${(100 * selectedRows / totalRows).toFixed(1)}%` : "行数不可用"),
    metric("选中数据集", `${selected.length}/${options.length}`, selected.map((item) => item.name).join(" / ") || "全部数据集"),
    metric("指标范围", metricMode, scoped.scopeLabel),
  ].join("");

  const allPressed = selected.length === options.length;
  $("dataset-controls").innerHTML =
    `<button class="dataset-chip all" aria-pressed="${allPressed}" data-dataset-action="all">全部</button>` +
    options.map((item) => {
      const pressed = activeDatasets.has(item.name);
      const share = totalRows ? `${(100 * item.rows / totalRows).toFixed(1)}%` : "无";
      const noMetric = item.hasMetrics ? "" : " no-metric";
      return `<div class="dataset-chip${noMetric}" aria-pressed="${pressed}">` +
        `<button class="dataset-main" data-dataset-only="${esc(item.name)}" aria-label="只看 ${esc(item.name)}">` +
          `<span class="dataset-name">${esc(item.name)}</span>` +
          `<span class="dataset-meta">${item.rows.toLocaleString()} 行 / ${share}</span>` +
          `<span class="dataset-action">只看</span>` +
        `</button>` +
        `<button class="dataset-toggle" aria-pressed="${pressed}" data-dataset-toggle="${esc(item.name)}" aria-label="${pressed ? "从多选中移除" : "加入多选"} ${esc(item.name)}">` +
          `${pressed ? "已选" : "多选"}` +
        `</button>` +
        `</div>`;
    }).join("");

  $("dataset-controls").querySelectorAll("[data-dataset-only]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeDatasets = new Set(nextDatasetSelection([...activeDatasets], btn.dataset.datasetOnly, options.map((item) => item.name), "only"));
      render();
    });
  });
  $("dataset-controls").querySelectorAll("[data-dataset-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeDatasets = new Set(nextDatasetSelection([...activeDatasets], btn.dataset.datasetToggle, options.map((item) => item.name), "toggle"));
      render();
    });
  });
  $("dataset-controls").querySelector("[data-dataset-action='all']").addEventListener("click", () => {
    activeDatasets = new Set(nextDatasetSelection([...activeDatasets], null, options.map((item) => item.name), "all"));
    render();
  });
}

function metric(label, value, hint) {
  return `<div class="scope-metric"><span>${esc(label)}</span><b>${esc(value)}</b><small>${esc(hint)}</small></div>`;
}

function renderStabilityPanel() {
  const rows = stabilityRows(DATA.config || {});
  if (!rows.length) {
    $("stability").innerHTML = '<div class="note">暂无多 seed 稳定性复跑记录。</div>';
    return;
  }

  const seeds = DATA.config.stability_seeds || [];
  const best = rows[0];
  const mostStable = [...rows].sort((a, b) => (a.std - b.std) || (a.mean - b.mean))[0];
  const meta = [
    metric("复跑种子", seeds.join(" / "), `${rows.length} 个 Top 组合`),
    metric("均值最优", best.label, `MAE ${fmt(best.mean)}`),
    metric("波动最小", mostStable.label, `标准差 ${fmt(mostStable.std)}`),
  ].join("");

  const body = rows.map((row) => {
    const cls = row.stabilityLabel === "较稳" ? "stable" : row.stabilityLabel === "中等" ? "medium" : "volatile";
    return `<tr>` +
      `<td><span class="algo">${esc(row.estimator)}</span><small>${esc(row.featureset)}</small></td>` +
      `<td class="tnum">${fmt(row.mean)}</td>` +
      `<td class="tnum">${fmt(row.std)}</td>` +
      `<td class="tnum">${esc(row.rangeText)}</td>` +
      `<td><span class="stability-badge ${cls}">${esc(row.stabilityLabel)}</span></td>` +
      `</tr>`;
  }).join("");

  $("stability").innerHTML =
    `<div class="stability-summary">${meta}</div>` +
    `<div class="note">多 seed 稳定性为全数据集聚合复跑结果，不随上方数据集筛选变化。</div>` +
    `<div class="stability-table"><table><thead><tr>` +
    `<th>组合</th><th>MAE 均值</th><th>seed 标准差</th><th>seed 范围</th><th>判定</th>` +
    `</tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderMatrix(lb, sets) {
  const rows = evaluationMatrixRows(lb, sets);
  const vals = rows.flatMap((row) => row.cells.map((cell) => cell.value)).filter((v) => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const best = Math.min(...vals);
  let h = "<table class='matrix'><thead><tr><th>算法</th>" +
    sets.map((s) => `<th>${esc(s)}</th>`).join("") + "</tr></thead><tbody>";
  rows.forEach((row) => {
    h += `<tr><td>${esc(row.label)}</td>`;
    row.cells.forEach(({ featureset, value: v }) => {
      if (v == null) { h += '<td class="na" title="不可用">无</td>'; return; }
      const t = hi > lo ? (v - lo) / (hi - lo) : 0;
      const isBest = Math.abs(v - best) < 1e-9;
      h += `<td class="cell${isBest ? " best" : ""}" style="background:${heat(t)}" title="${esc(row.label)} x ${esc(featureset)}: MAE ${v.toFixed(4)}">${v.toFixed(3)}</td>`;
    });
    h += "</tr>";
  });
  h += "</tbody></table>";
  $("matrix").innerHTML = h;
}

function renderTabs(sets) {
  $("fs-tabs").innerHTML = sets.map((s) =>
    `<button class="tab" role="tab" aria-selected="${s === activeFs}" data-fs="${esc(s)}">${esc(s)}</button>`).join("");
  $("fs-tabs").querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => { activeFs = t.dataset.fs; render(); }));
}

function renderInsights(lb) {
  $("insights-fs").textContent = activeFs;
  const rows = lb.filter((x) => (x.featureset || "full") === activeFs);
  const insights = algorithmInsights(rows);
  const item = (label, entry, value) => `<div class="insight"><span>${label}</span>` +
    (entry ? `<b>${esc(evaluationLabel(entry))}</b><small>${value(entry)}</small>` : "<b>无</b><small>无指标</small>") +
    "</div>";
  $("insights").innerHTML =
    item("精度", insights.bestAccuracy, (x) => `MAE ${fmt(metricMean(x, "mae"))}`) +
    item("校准", insights.bestCalibration, (x) => `PICP ${fmt(metricMean(x, "picp"))}`) +
    item("物理一致", insights.bestPhysical, (x) => `单调违例 ${fmt(metricMean(x, "monotonicity_violation_rate"))}`) +
    renderBestFeatureList(lb);
}

function renderBestFeatureList(lb) {
  const identities = uniq(lb.map((x) => evaluationIdentity(x)));
  const rows = identities.map((identity) => {
    const best = lb.filter((x) => evaluationIdentity(x) === identity && metricMean(x, "mae") != null)
      .sort((a, b) => metricMean(a, "mae") - metricMean(b, "mae"))[0];
    return `<tr><td>${best ? esc(evaluationLabel(best)) : "无"}</td><td>${best ? esc(best.featureset) : "无"}</td><td class="tnum">${best ? fmt(metricMean(best, "mae")) : "无"}</td></tr>`;
  }).join("");
  return `<div class="best-feature"><table><thead><tr><th>算法</th><th>最佳特征</th><th>MAE</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderLeaderboard(lb) {
  $("lb-fs").textContent = activeFs;
  const rows = lb.filter((x) => (x.featureset || "full") === activeFs)
    .filter((x) => metricMean(x, "mae") != null)
    .map((entry) => ({
      entry,
      coverage: leaderboardCoverage(entry, selectedDatasetNames(), DATA.datasetLeaderboard || []),
    }))
    .sort((a, b) =>
      (a.coverage.rank - b.coverage.rank) ||
      (b.coverage.count - a.coverage.count) ||
      (metricMean(a.entry, "mae") - metricMean(b.entry, "mae")));
  const maxMae = Math.max(...rows.map((x) => metricMean(x.entry, "mae")), 0.001);
  const vdot = (ok) => ok == null ? '<span class="dot na">-</span>' : ok ? '<span class="dot p">Y</span>' : '<span class="dot f">N</span>';
  const hasCoverageWarning = rows.some((row) => row.coverage.warning);
  let h = hasCoverageWarning
    ? '<div class="coverage-note">覆盖提示：局部覆盖结果不进入全量总榜，只在对应数据集范围内排序。</div>'
    : "";
  h += "<table class='lb'><thead><tr><th>算法</th><th>覆盖范围</th><th>MAE</th><th>RMSE</th><th>MBE</th><th>PICP</th><th>CI</th><th>判定</th></tr></thead><tbody>";
  rows.forEach(({ entry: x, coverage }, i) => {
    const mae = metricMean(x, "mae"), w = Math.round((mae / maxMae) * 90);
    const v = x.verdicts || {};
    const ci = x.ci_source || "conformal";
    h += `<tr class="${i === 0 ? "best" : ""}">` +
      `<td><span class="rank">${i + 1}</span><span class="algo">${esc(evaluationLabel(x))}</span></td>` +
      `<td><span class="coverage-badge ${esc(coverage.kind)}" title="${esc(coverage.detail)}">${esc(coverage.label)}</span></td>` +
      `<td><div class="bar-cell"><span class="bar ${i === rows.length - 1 && rows.length > 1 ? "bad" : ""}" style="width:${w}px"></span><span class="tnum">${fmt(mae)}</span></div></td>` +
      `<td class="tnum">${fmt(metricMean(x, "rmse"))}</td>` +
      `<td class="tnum">${fmt(metricMean(x, "mean_bias_error"), 3, true)}</td>` +
      `<td class="tnum">${fmt(metricMean(x, "picp"))}</td>` +
      `<td><span class="pill ${esc(ci)}">${esc(ci)}</span></td>` +
      `<td><span class="v">${vdot(v.generalization)}${vdot(v.calibration)}${vdot(v.physical_consistency)}</span></td>` +
      "</tr>";
  });
  h += "</tbody></table>";
  $("leaderboard").innerHTML = h;
}

function renderPicp(lb) {
  const rows = lb.filter((x) => (x.featureset || "full") === activeFs && metricMean(x, "picp") != null)
    .sort((a, b) => metricMean(b, "picp") - metricMean(a, "picp"));
  if (!rows.length) {
    $("picp").innerHTML = '<div class="note">该特征集暂无区间覆盖率记录。</div>';
    return;
  }
  $("picp").innerHTML = rows.map((x) => {
    const p = metricMean(x, "picp"), low = p < 0.9;
    return `<div class="picp-row"><span class="nm">${esc(evaluationLabel(x))}</span>` +
      `<div class="picp-track"><div class="picp-fill ${low ? "low" : ""}" style="width:${(p * 100).toFixed(1)}%"></div>` +
      '<div class="nominal" style="left:90%"></div></div>' +
      `<span class="picp-val">${p.toFixed(3)}</span></div>`;
  }).join("") + '<p class="note">橙色线 = 90% 目标。native = 模型原生区间；conformal = 分割保形；weighted_conformal = 加权保形。</p>';
}

function renderQc() {
  const qc = DATA.qc;
  if (!qc) { $("qc").innerHTML = '<div class="note">暂无 QC 报告。</div>'; return; }
  const line = (k, v, color) => `<div class="qc-line"><span>${esc(k)}</span><b class="tnum"${color ? ` style="color:${color}"` : ""}>${esc(v)}</b></div>`;
  const reasons = qc.drop_reasons || {};
  $("qc").innerHTML =
    line("输入行数", (qc.n_in ?? 0).toLocaleString()) +
    Object.entries(reasons).map(([k, v]) => line("剔除 - " + k, v, "var(--fail)")).join("") +
    line("剔除列", (qc.dropped_columns || []).join(", ") || "无") +
    line("通过率", (100 * (qc.pass_rate ?? 1)).toFixed(2) + "%", "var(--pass)");
}

function renderFoot() {
  const algos = uniq(DATA.leaderboard.map((x) => evaluationLabel(x))).join(" / ");
  const scope = scoped.usedDatasetLevel ? `已选数据集：${scoped.scopeLabel}` : "全量聚合结果";
  const profile = activeProfileOption?.label || "当前根目录";
  $("foot").textContent = `${DATA.fallback ? "示例数据" : "sohbench 实测运行"} / ${profile} / ${algos} / ${scope}`;
}

load();
