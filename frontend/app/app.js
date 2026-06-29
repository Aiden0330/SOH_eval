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
  datasetOptions,
  inferenceRows,
  inferenceSummary,
  leaderboardCoverage,
  metricMean,
  modelOptions,
  nextDatasetSelection,
  scopeSummary,
  stabilityRows,
  summarizeBySelection,
} from "./dashboard-core.mjs";

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

async function load(options = {}) {
  const suffix = `?v=${encodeURIComponent(DATA_VERSION)}${options.cacheBust ? `&t=${Date.now()}` : ""}`;
  try {
    const [lb, qc, cfg, byDataset, inference, modelCatalog] = await Promise.all([
      fetch("./data/leaderboard.json" + suffix).then((r) => { if (!r.ok) throw 0; return r.json(); }),
      fetch("./data/qc_report.json" + suffix).then((r) => (r.ok ? r.json() : null)).catch(() => null),
      fetch("./data/run_config.json" + suffix).then((r) => (r.ok ? r.json() : null)).catch(() => null),
      fetch("./data/dataset_leaderboard.json" + suffix).then((r) => (r.ok ? r.json() : [])).catch(() => []),
      fetch("./data/inference_benchmark.json" + suffix).then((r) => (r.ok ? r.json() : { records: [] })).catch(() => ({ records: [] })),
      fetch("./data/model_catalog.json" + suffix).then((r) => (r.ok ? r.json() : { models: [] })).catch(() => ({ models: [] })),
    ]);
    DATA = {
      leaderboard: lb,
      datasetLeaderboard: byDataset || [],
      qc,
      config: cfg || {},
      inference: inference || { records: [] },
      modelCatalog: modelCatalog || { models: [] },
    };
  } catch (_) {
    DATA = {
      leaderboard: FALLBACK.leaderboard,
      datasetLeaderboard: FALLBACK.datasetLeaderboard,
      qc: FALLBACK.qc,
      config: FALLBACK.config,
      inference: FALLBACK.inference,
      modelCatalog: FALLBACK.modelCatalog,
      fallback: true,
    };
  }
  await loadBackendCatalog();
  render();
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

  const dsOptions = datasetOptions(DATA.config, DATA.qc, DATA.datasetLeaderboard || []);
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
  renderWorkbench();
  renderInferencePanel(dsOptions);
  renderModelLibrary();
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

function renderRunMeta() {
  const c = DATA.config || {};
  const parts = [];
  if (c.datasets) parts.push(`数据集 <b>${c.datasets.length}</b>`);
  const protos = c.protocols || uniq(DATA.leaderboard.map((x) => x.protocol));
  if (protos?.length) parts.push(`协议 <b>${protos.join(",")}</b>`);
  if (c.seeds) parts.push(`种子 <b>${c.seeds.length}</b>`);
  parts.push(`来源 <b>${DATA.fallback ? "示例" : "实测"}</b>`);
  $("runmeta").innerHTML = parts.map((p) => `<span>${p}</span>`).join("");
}

function renderKpis(lb, dsOptions) {
  const withMae = lb.filter((x) => metricMean(x, "mae") != null);
  const best = bestComparableEntry(withMae, selectedDatasetNames(), DATA.datasetLeaderboard || []);
  const algos = uniq(lb.map((x) => x.estimator));
  const selected = dsOptions.filter((item) => activeDatasets.has(item.name));
  const selectedRows = selected.reduce((sum, item) => sum + item.rows, 0);
  const qc = DATA.qc;
  let pass = 0, total = 0;
  lb.forEach((x) => Object.values(x.verdicts || {}).forEach((v) => { total++; if (v) pass++; }));
  const kpi = (lab, val, hint, color) =>
    `<div class="kpi"><div class="lab">${lab}</div><div class="val"${color ? ` style="color:${color}"` : ""}>${val}</div><div class="hint">${hint}</div></div>`;
  $("kpis").innerHTML =
    kpi("可比最佳 MAE", best ? esc(best.estimator) : "无",
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
  $("scope-context").innerHTML =
    `<div class="scope-heading">` +
      `<span class="scope-eyebrow">全局评测上下文</span>` +
      `<strong>${esc(summary.metricMode)}</strong>` +
      `<span class="scope-notice">${esc(summary.notice)}</span>` +
    `</div>` +
    `<div class="scope-strip">` +
      scopePill("数据集", summary.datasetLabel, names || "全部数据集", summary.coverageKind) +
      scopePill("样本", summary.rowLabel, `占 QC 样本 ${summary.shareLabel}`, "rows") +
      scopePill("协议", summary.protocolLabel, `seed ${summary.seedLabel}`, "protocol") +
      scopePill("来源", summary.sourceLabel, summary.scopeLabel, DATA.fallback ? "warn" : "source") +
    `</div>`;
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
  document.querySelectorAll("[data-mode-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.modePanel !== activeMode;
  });
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
  if (!inferenceSelection.models.size && models.length) inferenceSelection.models.add(models[0].name);
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
  $("inference-results").innerHTML = renderInferenceSummary(summary) + renderInferenceTable(rows);
  const start = $("start-inference");
  if (start) start.onclick = CATALOG ? startInferenceEvaluation : null;
  if (start) start.disabled = !CATALOG;
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
  if (!models.length) {
    $("model-library-table").innerHTML = '<div class="note">暂无模型。训练测评完成后会产出模型，也可以在推理测评页上传 Pickle 模型。</div>';
    return;
  }
  const rows = models.map((item) =>
    `<tr>` +
    `<td><span class="algo">${esc(item.name)}</span><small>${esc(item.badge)}</small></td>` +
    `<td>${esc(item.estimator || "未知")}</td>` +
    `<td>${esc(item.featureset || "未知")}</td>` +
    `<td>${esc((item.trainedOn || []).join(" / ") || "外部上传")}</td>` +
    `<td class="tnum">${esc(String((item.featureColumns || []).length))}</td>` +
    `</tr>`).join("");
  $("model-library-table").innerHTML =
    `<div class="inference-table"><table><thead><tr>` +
    `<th>模型</th><th>算法</th><th>特征集</th><th>训练数据</th><th>特征数</th>` +
    `</tr></thead><tbody>${rows}</tbody></table></div>`;
}

async function uploadResource(kind) {
  if (kind === "model") {
    const modelFile = $("upload-model-file")?.files?.[0];
    const manifestFile = $("upload-model-manifest")?.files?.[0];
    if (!modelFile || !manifestFile) {
      setJobMessage("请选择模型 Pickle 文件和 model_manifest.json。");
      return;
    }
    const name = ($("upload-model-name").value || modelFile.name.replace(/\.[^.]+$/, "")).trim();
    try {
      setJobMessage("正在上传模型...");
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
      setJobMessage(`已上传模型 ${response.resource.name}`);
    } catch (err) {
      setJobMessage(`模型上传失败：${err.message}`);
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

async function startInferenceEvaluation() {
  const payload = buildInferencePayload({
    models: selectedValues("inference-models"),
    datasets: selectedValues("inference-datasets"),
  });
  if (!payload.models.length || !payload.datasets.length) {
    setJobMessage("请至少选择一个模型和一个数据集。");
    return;
  }
  try {
    CURRENT_JOB = await postJson("/api/inference-jobs", payload);
    renderInferencePanel(datasetOptions(DATA.config, DATA.qc, DATA.datasetLeaderboard || []));
    pollInferenceJob(CURRENT_JOB.id);
  } catch (err) {
    setJobMessage(`推理任务启动失败：${err.message}`);
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
      CURRENT_JOB = await response.json();
      renderInferencePanel(datasetOptions(DATA.config, DATA.qc, DATA.datasetLeaderboard || []));
      if (CURRENT_JOB.status === "succeeded") {
        await load({ cacheBust: true });
        activeMode = "inference";
        render();
        return;
      }
      if (CURRENT_JOB.status === "failed") return;
    } catch (err) {
      setJobMessage(`推理任务轮询失败：${err.message}`);
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
  const algos = uniq(lb.map((x) => x.estimator));
  const cell = {};
  lb.forEach((x) => { cell[x.estimator + "|" + (x.featureset || "full")] = metricMean(x, "mae"); });
  const vals = Object.values(cell).filter((v) => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const best = Math.min(...vals);
  let h = "<table class='matrix'><thead><tr><th>算法</th>" +
    sets.map((s) => `<th>${esc(s)}</th>`).join("") + "</tr></thead><tbody>";
  algos.forEach((a) => {
    h += `<tr><td>${esc(a)}</td>`;
    sets.forEach((s) => {
      const v = cell[a + "|" + s];
      if (v == null) { h += '<td class="na" title="不可用">无</td>'; return; }
      const t = hi > lo ? (v - lo) / (hi - lo) : 0;
      const isBest = Math.abs(v - best) < 1e-9;
      h += `<td class="cell${isBest ? " best" : ""}" style="background:${heat(t)}" title="${esc(a)} x ${esc(s)}: MAE ${v.toFixed(4)}">${v.toFixed(3)}</td>`;
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
    (entry ? `<b>${esc(entry.estimator)}</b><small>${value(entry)}</small>` : "<b>无</b><small>无指标</small>") +
    "</div>";
  $("insights").innerHTML =
    item("精度", insights.bestAccuracy, (x) => `MAE ${fmt(metricMean(x, "mae"))}`) +
    item("校准", insights.bestCalibration, (x) => `PICP ${fmt(metricMean(x, "picp"))}`) +
    item("物理一致", insights.bestPhysical, (x) => `单调违例 ${fmt(metricMean(x, "monotonicity_violation_rate"))}`) +
    renderBestFeatureList(lb);
}

function renderBestFeatureList(lb) {
  const algos = uniq(lb.map((x) => x.estimator));
  const rows = algos.map((algo) => {
    const best = lb.filter((x) => x.estimator === algo && metricMean(x, "mae") != null)
      .sort((a, b) => metricMean(a, "mae") - metricMean(b, "mae"))[0];
    return `<tr><td>${esc(algo)}</td><td>${best ? esc(best.featureset) : "无"}</td><td class="tnum">${best ? fmt(metricMean(best, "mae")) : "无"}</td></tr>`;
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
      `<td><span class="rank">${i + 1}</span><span class="algo">${esc(x.estimator)}</span></td>` +
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
    return `<div class="picp-row"><span class="nm">${esc(x.estimator)}</span>` +
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
  const algos = uniq(DATA.leaderboard.map((x) => x.estimator)).join(" / ");
  const scope = scoped.usedDatasetLevel ? `已选数据集：${scoped.scopeLabel}` : "全量聚合结果";
  $("foot").textContent = `${DATA.fallback ? "示例数据" : "sohbench 实测运行"} / ${algos} / ${scope}`;
}

load();
