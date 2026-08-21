/* sohbench dashboard: data-driven render of the Lab Light design.
   Loads benchmark artifacts from ./data and falls back to embedded sample data. */

"use strict";

import {
  algorithmInsights,
  auditNextAction,
  bestComparableEntry,
  batteryDiagnosticSummary,
  batteryCurvePreviewView,
  batteryScenarioRows,
  buildInferencePayload,
  buildEvaluationPayload,
  catalogOptions,
  evaluationPlanSummary,
  acceptanceDetailSummary,
  acceptanceKpiSummary,
  acceptancePresetOptions,
  batteryVisualAssetDetail,
  batteryVisualAssetSummary,
  caseContextProjection,
  casePipelineJobId,
  casePipelineProgress,
  requiresFreshTrainingCase,
  pipelineReconciliationTarget,
  pipelineCombinationConsistency,
  auditEvidenceDatasetScope,
  datasetOptions,
  inferenceInsightSummary,
  inferenceDatasetOptions,
  inferenceCompatibilitySummary,
  inferenceEvaluationEligibility,
  inferenceLaunchPlan,
  inferenceRows,
  inferenceAuditSummary,
  inferenceActionState,
  inferenceSummary,
  algorithmRecommendationView,
  leaderboardCoverage,
  modelAssetPage,
  modelAssetRows,
  metricMean,
  modelOptions,
  pipelineModelGroups,
  nextDatasetSelection,
  preferredInferenceModels,
  profileOptions,
  retestComparisonView,
  scopeSummary,
  stabilityRows,
  summarizeBySelection,
  taskHistoryRows,
  trainingArtifactSummary,
  trainingJobRuntimeState,
  trajectoryContextConsistency,
  auditCandidateViewProjection,
} from "./dashboard-core.mjs?v=phase46-label-safety-20260817a";

const $ = (id) => document.getElementById(id);
const URL_PARAMS = new URLSearchParams(location.search);
const API_BASE = URL_PARAMS.get("api") || "http://localhost:8001";
const DATA_VERSION = URL_PARAMS.get("v") || "live";
const INITIAL_CASE_ID = String(URL_PARAMS.get("case") || "").trim();
const DEFAULT_BATTERY_DASHBOARD_SOURCE = "D:\\battery_SOH\\dashboard";
const INITIAL_PROFILE = safeProfileName(URL_PARAMS.get("profile") || localStorage.getItem("sohbench.profile") || "root");
const CATALOG_MODEL_PREVIEW_LIMIT = 1200;
const CATALOG_JOB_PREVIEW_LIMIT = 20;
const MODEL_ASSET_PAGE_SIZE = 120;
const FS_ORDER = [
  "full", "electrochemical_diagnostics", "operating_stress", "degradation_kinetics",
  "voltage", "ic_dva", "partial_window", "capacity_time", "(byo)",
];
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
const INITIAL_MODE = URL_PARAMS.get("mode");
const APP_MODES = new Set(["training", "inference", "recommendations", "models", "datasets", "diagnostics", "audit"]);
let activeMode = APP_MODES.has(INITIAL_MODE) ? INITIAL_MODE : "training";
let scoped = null;
let CATALOG = null;
let FULL_MODELS_LOADED = false;
let FULL_MODELS_LOADING = false;
let WORKBENCH_ERROR = "";
let CURRENT_JOB = null;
let QUICKSTART_EXAMPLE = null;
let resourceSelection = {
  datasets: new Set(),
  estimators: new Set(),
  featuresets: new Set(),
};
let inferenceSelection = {
  models: new Set(),
  datasets: new Set(),
};
let inferenceModelSearch = "";
let inferenceLaunchStrategy = "compatible_only";
let taskHistoryOpen = false;
let acceptanceDetailOpen = false;
let batteryAssetOpen = false;
let activeBatteryCurveDataset = "";
let nativeDashboardContext = {};
let evidenceLiveContext = {};
let ACTIVE_EVALUATION_CASE = null;
let REVIEW_WORKSPACE = null;
let REVIEW_WORKFLOW_STATE = {
  status: "idle",
  message: "",
  busy: "",
};
let auditRetestPollTimer = null;
let auditScenarioState = { key: "", status: "idle", data: null, error: "" };
const auditScenarioCache = new Map();
let auditEvidenceTab = "trajectory";
let auditRawCurveState = { key: "", status: "idle", data: null, error: "" };
let casePersistenceTimer = null;
let casePersistenceInFlight = false;
let casePersistenceQueued = false;
let CASE_PERSISTENCE = {
  status: "idle",
  message: "",
};
let PIPELINE_STATUS = {
  tone: "ready",
  message: "",
};
let PIPELINE_JOBS = {
  training: null,
  inference: null,
};
const ACTIVE_PIPELINE_POLLS = new Set();
let API_SESSION = {
  token: "",
  legacyExecutionEnabled: false,
};
let MODEL_PACKAGE_STATUS = {
  tone: "ready",
  message: "",
};
let TRAINING_PREFLIGHT = {
  status: "idle",
  signature: "",
  result: null,
  error: "",
};
let trainingPreflightTimer = null;
let trainingPreflightRequest = 0;
let batteryCurveLoadState = {
  dataset: "",
  status: "idle",
  message: "",
  curve: null,
  totalCurves: 0,
  returnedCurves: 0,
};
let PACKAGE_STATUS = {
  busy: "",
  tone: "ready",
  message: "可同步 SOH 看板并生成离线包。",
  results: {},
};
let PACKAGE_REPORT = {
  busy: false,
  tone: "ready",
  message: "等待刷新验收",
  report: null,
};
let acceptancePreset = localStorage.getItem("sohbench.acceptancePreset") || "strict";
let trainingAutoSave = localStorage.getItem("sohbench.trainingAutoSave") !== "false";
let inferenceAutoSave = localStorage.getItem("sohbench.inferenceAutoSave") !== "false";
let SAVED_COMBINATION_STATUS = { busy: "", message: "", tone: "ready" };
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
let requestedCaseId = INITIAL_CASE_ID;

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
  await loadApiSession();
  await loadBackendCatalog();
  await resolveEvaluationCaseContext();
  await hydratePipelineJobs();
  await refreshPipelineModelCatalogForJobs();
  await reconcileHydratedPipelineCase();
  await hydratePipelineArtifacts();
  alignResourceSelectionToPipelineCase();
  await loadReviewWorkspace({ silent: true });
  await loadPackageReport({ silent: true });
  render();
  resumePipelinePolling();
}

async function loadProfileIndex(suffix = "") {
  try {
    const response = await fetch(`${API_BASE}/api/profiles${suffix}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (_) {
    try {
      const response = await fetch("./data/profiles/index.json" + suffix);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (_) {
      return { profiles: [] };
    }
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
  const manifest = profile === "root"
    ? {}
    : await fetchJson(`${base}/manifest.json${suffix}`, {});
  const benchmarkBase = isBatteryDiagnosticsProfile(profile, manifest) ? "./data" : base;
  const benchmark = await loadBenchmarkArtifacts(benchmarkBase, suffix).catch(async (err) => {
    if (profile === "root") throw err;
    return loadBenchmarkArtifacts("./data", suffix);
  });
  const batteryDiagnostics = await loadBatteryDiagnosticsArtifacts(base, suffix);
  const metricContributions = await fetchJson(
    `${base}/metric_contributions.json${suffix}`,
    { records: [] },
  );
  return {
    ...benchmark,
    profileManifest: manifest || {},
    batteryDiagnostics,
    metricContributions,
  };
}

function isBatteryDiagnosticsProfile(profile, manifest = {}) {
  if (profile === "root") return false;
  const indexed = (PROFILE_INDEX?.profiles || []).find((item) => item?.name === profile);
  return indexed?.kind === "battery_soh_diagnostics" || manifest?.kind === "battery_soh_diagnostics";
}

async function loadBenchmarkArtifacts(base, suffix) {
  const [
    lb,
    qc,
    cfg,
    byDataset,
    inference,
    inferenceConfig,
    modelCatalog,
    recommendation,
  ] = await Promise.all([
    fetchJson(`${base}/leaderboard.json${suffix}`),
    fetchJson(`${base}/qc_report.json${suffix}`, null),
    fetchJson(`${base}/run_config.json${suffix}`, {}),
    fetchJson(`${base}/dataset_leaderboard.json${suffix}`, []),
    fetchJson(`${base}/inference_benchmark.json${suffix}`, { records: [], summary: {} }),
    fetchJson(`${base}/inference_run_config.json${suffix}`, {}),
    fetchJson(`${base}/model_catalog.json${suffix}`, { models: [] }),
    fetchJson(`${base}/recommendation_summary.json${suffix}`, { scenarios: [], overall: {} }),
  ]);
  return {
    leaderboard: lb,
    datasetLeaderboard: byDataset || [],
    qc,
    config: cfg || {},
    inference: inference || { records: [], summary: {} },
    inferenceConfig: inferenceConfig || {},
    modelCatalog: modelCatalog || { models: [] },
    recommendation: recommendation || { scenarios: [], overall: {} },
  };
}

async function loadBatteryDiagnosticsArtifacts(base, suffix) {
  const [
    batteryManifest,
    batteryOverview,
    algorithmDesign,
    modelComparisons,
    featureWindows,
    curvePreviews,
  ] = await Promise.all([
    fetchJson(`${base}/battery_dashboard_manifest.json${suffix}`, { scenarios: [] }),
    fetchJson(`${base}/battery_dashboard_overview.json${suffix}`, { scenarios: [], transfers: [] }),
    fetchJson(`${base}/battery_algorithm_design.json${suffix}`, null),
    fetchJson(`${base}/battery_model_comparisons.json${suffix}`, null),
    fetchJson(`${base}/battery_feature_windows.json${suffix}`, null),
    fetchJson(`${base}/battery_curve_previews.json${suffix}`, null),
  ]);
  return {
    manifest: batteryManifest || { scenarios: [] },
    overview: batteryOverview || { scenarios: [], transfers: [] },
    algorithmDesign,
    modelComparisons,
    featureWindows,
    curvePreviews,
  };
}

async function fetchJson(url, fallback) {
  const response = await fetch(url);
  if (!response.ok) {
    if (arguments.length > 1) return fallback;
    let detail = "";
    try {
      detail = String((await response.json())?.error || "");
    } catch (_) {
      detail = "";
    }
    throw new Error(detail ? `${detail}（HTTP ${response.status}）` : `HTTP ${response.status}`);
  }
  return response.json();
}

function fallbackData() {
  // Deliberately empty: a failed load must surface an explicit error state,
  // never fabricated demo numbers (静态与实时不混合).
  return {
    leaderboard: [],
    datasetLeaderboard: [],
    qc: null,
    config: {},
    inference: { records: [], summary: {} },
    inferenceConfig: {},
    modelCatalog: { models: [] },
    profileManifest: {},
    batteryDiagnostics: {},
    metricContributions: { records: [] },
    fallback: true,
    loadError: true,
  };
}

async function loadBackendCatalog() {
  try {
    const params = new URLSearchParams({
      model_limit: String(CATALOG_MODEL_PREVIEW_LIMIT),
      job_limit: String(CATALOG_JOB_PREVIEW_LIMIT),
      compact_jobs: "1",
    });
    const response = await fetch(`${API_BASE}/api/catalog?${params.toString()}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    CATALOG = await response.json();
    try {
      const examplesResponse = await fetch(`${API_BASE}/api/examples`);
      if (examplesResponse.ok) {
        const examplesBody = await examplesResponse.json();
        QUICKSTART_EXAMPLE = (examplesBody.examples || [])[0] || null;
      }
    } catch (_) {
      QUICKSTART_EXAMPLE = null;
    }
    FULL_MODELS_LOADED = !CATALOG?.models_limited;
    WORKBENCH_ERROR = "";
    seedResourceSelection();
  } catch (err) {
    CATALOG = null;
    FULL_MODELS_LOADED = false;
    WORKBENCH_ERROR = "后端未连接";
  }
}

async function loadApiSession() {
  try {
    const response = await fetch(`${API_BASE}/api/session`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const body = await response.json();
    API_SESSION = {
      token: String(body.access_token || ""),
      legacyExecutionEnabled: body.legacy_execution_enabled === true,
    };
  } catch (_) {
    API_SESSION = { token: "", legacyExecutionEnabled: false };
  }
}

async function loadFullModelCatalog({ force = false } = {}) {
  if (!CATALOG || (!force && FULL_MODELS_LOADED) || FULL_MODELS_LOADING) return;
  FULL_MODELS_LOADING = true;
  try {
    const response = await fetch(`${API_BASE}/api/models`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const body = await response.json();
    CATALOG = {
      ...CATALOG,
      models: body.models || [],
      model_count: (body.models || []).length,
      models_limited: false,
    };
    FULL_MODELS_LOADED = true;
  } catch (err) {
    WORKBENCH_ERROR = `模型目录加载失败：${err.message}`;
  } finally {
    FULL_MODELS_LOADING = false;
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
    $("leaderboard").innerHTML = DATA.loadError
      ? '<div class="state">数据加载失败：读不到 data/ 工件，也未能连接后端 API。请启动服务后重试。 <button id="data-load-retry" type="button">重试</button></div>'
      : '<div class="state">暂无数据。</div>';
    const retry = $("data-load-retry");
    if (retry) retry.onclick = () => load({ cacheBust: true });
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
  prepareEvidenceContext(lb);
  renderCaseIdentity();
  renderModeTabs();
  wireProfileSelector(profileOptions(PROFILE_INDEX, activeProfile));
  renderWorkbench();
  if (activeMode === "inference") renderInferencePanel(inferenceDsOptions);
  if (activeMode === "models") renderModelLibrary();
  if (activeMode === "datasets") renderDatasetCatalogPage(dsOptions);
  if (activeMode === "recommendations") renderRecommendationPanel();
  renderAuditWorkspace();
  renderTaskHistoryDrawer();
  renderAcceptanceDetailDrawer(lb);
  renderBatteryAssetDrawer();
  renderBatteryDiagnosticsPanel();
  renderPackageOperationsPanel();
  renderPackageReportPanel();
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
  return datasetOptions(DATA.config, DATA.qc, DATA.datasetLeaderboard || [], CATALOG || {});
}

function renderDatasetCatalogPage(dsOptions = currentDatasetOptions()) {
  const summary = $("dataset-catalog-summary");
  const table = $("dataset-catalog-table");
  if (!summary || !table) return;
  const formatCount = (value) => Number(value || 0).toLocaleString("zh-CN");
  const knownChemistries = new Set(dsOptions.map((item) => item.chemistry).filter((value) => value && value !== "unknown"));
  const manifest = CATALOG?.data_manifest || {};
  const localReady = dsOptions.filter((item) => item.dataStatus === "ready").length;
  const totals = dsOptions.reduce((acc, item) => ({
    records: acc.records + Number(item.recordCount || item.rows || 0),
    cycles: acc.cycles + Number(item.cycleCount || 0),
    batteries: acc.batteries + Number(item.batteryCount || 0),
  }), { records: 0, cycles: 0, batteries: 0 });
  summary.innerHTML = [
    ["数据集", `${dsOptions.length} 个`, `${localReady} 个本地就绪`],
    ["数据总量", formatCount(totals.records), "处理后记录"],
    ["Cycle", formatCount(totals.cycles), "电池–Cycle 唯一组合"],
    ["电池", formatCount(totals.batteries), "跨数据集计数"],
    ["迁移校验", manifest.verified ? "已通过" : "待校验", manifest.migration_strategy || "未绑定清单"],
    ["已知化学体系", `${knownChemistries.size} 类`, [...knownChemistries].join(" / ") || "待补元数据"],
  ].map(([label, value, hint]) => `<div><span>${esc(label)}</span><b>${esc(value)}</b><small>${esc(hint)}</small></div>`).join("");

  table.innerHTML = dsOptions.map((item) => {
    const chemistry = item.chemistry && item.chemistry !== "unknown" ? item.chemistry : "待确认";
    const targets = dsOptions
      .filter((candidate) => candidate.name !== item.name && chemistry !== "待确认" && candidate.chemistry === item.chemistry)
      .map((candidate) => candidate.name);
    const capacitySource = ({ measured_capacity: "实测", rated_column: "额定列", rated_metadata: "额定元数据" })[item.capacitySource] || "待补";
    const capacity = item.maxCapacityAh == null
      ? "待补"
      : `${Number(item.maxCapacityAh).toFixed(3).replace(/0+$/, "").replace(/\.$/, "")} Ah`;
    return `<tr>` +
      `<td><b>${esc(item.name)}</b><small title="${esc(item.source || "")}">${item.uploaded ? "上传数据" : "项目数据"}</small></td>` +
      `<td><span class="scope-badge ${item.dataStatus === "ready" ? "ok" : "warn"}">${item.dataStatus === "ready" ? "处理表就绪" : "待检查"}</span><small>${item.rawLocal ? "原始数据本地" : "仅处理表"}${item.curveAvailable ? " · 曲线可用" : ""}</small></td>` +
      `<td>${formatCount(item.recordCount || item.rows)}</td>` +
      `<td>${formatCount(item.cycleCount)}</td>` +
      `<td>${formatCount(item.batteryCount)}</td>` +
      `<td><span class="chemistry-badge ${chemistry === "待确认" ? "unknown" : ""}">${esc(chemistry)}</span></td>` +
      `<td><b>${esc(capacity)}</b><small>${esc(capacitySource)}</small></td>` +
      `<td><span class="scope-badge ${item.heldOutReady ? "ok" : "warn"}">${item.heldOutReady ? "可按电池留出" : "需明确切分"}</span></td>` +
      `<td>${targets.length ? esc(targets.join(" / ")) : `<span class="muted">${chemistry === "待确认" ? "体系待确认" : "暂无同体系目标"}</span>`}</td>` +
    `</tr>`;
  }).join("") || `<tr><td colspan="9" class="state">后端目录尚未返回数据集信息。</td></tr>`;
}

function currentInferenceDatasetOptions(dsOptions = currentDatasetOptions()) {
  return inferenceDatasetOptions({
    baseOptions: dsOptions,
    catalog: CATALOG || {},
    inference: DATA?.inference || {},
  });
}

function currentInferenceAuditSummary() {
  const currentJob = currentInferenceJob();
  const catalog = ACTIVE_EVALUATION_CASE
    ? { ...(CATALOG || {}), inference_jobs: currentJob ? [currentJob] : [] }
    : (CATALOG || {});
  return inferenceAuditSummary({
    inferenceConfig: DATA?.inferenceConfig || {},
    catalog,
    currentJob,
  });
}

function pipelineJobMatchesActiveCase(job, kind) {
  if (!job) return false;
  if (!ACTIVE_EVALUATION_CASE?.id) return true;
  const expectedId = casePipelineJobId(ACTIVE_EVALUATION_CASE, activeProfileOption, kind);
  return Boolean(expectedId && String(job.id || "") === expectedId);
}

function currentTrainingJob() {
  if (CURRENT_JOB && CURRENT_JOB.kind !== "inference" && pipelineJobMatchesActiveCase(CURRENT_JOB, "training")) {
    return { kind: "training", ...CURRENT_JOB };
  }
  if (pipelineJobMatchesActiveCase(PIPELINE_JOBS.training, "training")) return PIPELINE_JOBS.training;
  const jobId = casePipelineJobId(ACTIVE_EVALUATION_CASE, activeProfileOption, "training");
  const job = (CATALOG?.jobs || []).find((item) => item.id === jobId);
  return job ? { kind: "training", ...job } : null;
}

function currentInferenceJob() {
  if (CURRENT_JOB?.kind === "inference" && pipelineJobMatchesActiveCase(CURRENT_JOB, "inference")) return CURRENT_JOB;
  if (pipelineJobMatchesActiveCase(PIPELINE_JOBS.inference, "inference")) return PIPELINE_JOBS.inference;
  const jobId = casePipelineJobId(ACTIVE_EVALUATION_CASE, activeProfileOption, "inference");
  const job = (CATALOG?.inference_jobs || []).find((item) => item.id === jobId);
  return job ? { kind: "inference", ...job } : null;
}

async function hydratePipelineJobs() {
  if (!CATALOG) return;
  const trainingId = casePipelineJobId(ACTIVE_EVALUATION_CASE, activeProfileOption, "training");
  const inferenceId = casePipelineJobId(ACTIVE_EVALUATION_CASE, activeProfileOption, "inference");
  const fetchJob = async (kind, id) => {
    if (!id) return null;
    try {
      const path = kind === "training" ? "/api/jobs/" : "/api/inference-jobs/";
      const response = await fetch(`${API_BASE}${path}${encodeURIComponent(id)}`);
      if (!response.ok) return null;
      const body = await response.json();
      return kind === "inference" ? { kind: "inference", ...body } : { kind: "training", ...body };
    } catch (_) {
      return null;
    }
  };
  const [training, inference] = await Promise.all([
    fetchJob("training", trainingId),
    fetchJob("inference", inferenceId),
  ]);
  PIPELINE_JOBS = { training, inference };
}

async function refreshPipelineModelCatalogForJobs({ force = false } = {}) {
  if (!CATALOG) return;
  const requiredNames = new Set([
    ...(currentTrainingJob()?.saved_models || []),
    ...(currentInferenceJob()?.request?.models || []),
  ].filter(Boolean));
  const knownNames = new Set((CATALOG.models || []).map((item) => item?.name).filter(Boolean));
  const missingLinkedModel = [...requiredNames].some((name) => !knownNames.has(name));
  if (force || missingLinkedModel) await loadFullModelCatalog({ force: true });
}

function pipelineCandidatesMatch(expected = []) {
  const actual = ACTIVE_EVALUATION_CASE?.candidates || [];
  if (actual.length !== expected.length) return false;
  const actualIds = new Set(actual.map((item) => item?.candidate_id).filter(Boolean));
  return expected.every((item) => actualIds.has(item.candidate_id));
}

async function reconcileHydratedPipelineCase() {
  if (!ACTIVE_EVALUATION_CASE?.id) return;
  const decision = pipelineReconciliationTarget({
    caseDocument: ACTIVE_EVALUATION_CASE,
    trainingJob: currentTrainingJob(),
    inferenceJob: currentInferenceJob(),
  });
  if (!decision.kind || !decision.jobId) return;
  const job = decision.kind === "training" ? currentTrainingJob() : currentInferenceJob();
  if (!job) return;
  const expectedCandidates = job.status === "succeeded"
    ? (decision.kind === "training" ? pipelineCandidatesFromTrainingJob(job) : pipelineCandidatesFromInferenceJob(job))
    : [];
  const candidateMismatch = job.status === "succeeded" && !pipelineCandidatesMatch(expectedCandidates);
  if (decision.needsIdentitySync || candidateMismatch) {
    await syncPipelineCaseWithJob(job, decision.kind);
  }
  if (decision.kind === "inference" && job.status === "succeeded") {
    const refreshedDecision = pipelineReconciliationTarget({
      caseDocument: ACTIVE_EVALUATION_CASE,
      trainingJob: currentTrainingJob(),
      inferenceJob: currentInferenceJob(),
    });
    if (decision.needsEvidenceBinding || refreshedDecision.needsEvidenceBinding) {
      await bindInferenceEvidenceToCase(job);
    }
  }
}

async function hydratePipelineArtifacts() {
  if (!ACTIVE_EVALUATION_CASE?.id) return;
  const fetchArtifacts = async (kind, job) => {
    if (!job?.id || job.status !== "succeeded") return null;
    try {
      const path = kind === "training" ? "/api/jobs/" : "/api/inference-jobs/";
      const response = await fetch(`${API_BASE}${path}${encodeURIComponent(job.id)}/artifacts`);
      if (!response.ok) return null;
      const body = await response.json();
      return body.job_id === job.id ? (body.artifacts || {}) : null;
    } catch (_) {
      return null;
    }
  };
  const [trainingArtifacts, inferenceArtifacts] = await Promise.all([
    fetchArtifacts("training", currentTrainingJob()),
    fetchArtifacts("inference", currentInferenceJob()),
  ]);
  if (trainingArtifacts) {
    DATA = {
      ...DATA,
      leaderboard: trainingArtifacts.leaderboard || DATA.leaderboard,
      datasetLeaderboard: trainingArtifacts.dataset_leaderboard || DATA.datasetLeaderboard,
      qc: trainingArtifacts.qc_report || DATA.qc,
      config: trainingArtifacts.run_config || DATA.config,
    };
  }
  if (inferenceArtifacts) {
    DATA = {
      ...DATA,
      inference: inferenceArtifacts.inference_benchmark || DATA.inference,
      inferenceConfig: inferenceArtifacts.inference_run_config || DATA.inferenceConfig,
    };
  }
}

function resumePipelinePolling() {
  const trainingJob = currentTrainingJob();
  const inferenceJob = currentInferenceJob();
  if (["queued", "running"].includes(trainingJob?.status)) void pollJob(trainingJob.id);
  if (["queued", "running"].includes(inferenceJob?.status)) void pollInferenceJob(inferenceJob.id);
}

function alignResourceSelectionToPipelineCase() {
  const request = currentTrainingJob()?.request || {};
  if (!ACTIVE_EVALUATION_CASE?.id || !request.datasets?.length) return;
  const options = catalogOptions(CATALOG || {});
  const known = {
    datasets: new Set(options.datasets.map((item) => item.name)),
    estimators: new Set(options.estimators.map((item) => item.name)),
    featuresets: new Set(options.featuresets.map((item) => item.name)),
  };
  resourceSelection = {
    datasets: new Set((request.datasets || []).filter((name) => known.datasets.has(name))),
    estimators: new Set((request.estimators || []).filter((name) => known.estimators.has(name))),
    featuresets: new Set((request.featuresets || []).filter((name) => known.featuresets.has(name))),
  };
}

function renderRunMeta() {
  const context = currentCaseProjection();
  if (context.hasCase) {
    const sourceLabel = context.sourceClass === "measured" ? "实测" : (context.sourceClass === "demo" ? "示例" : context.sourceClass || "未记录");
    const runLabel = context.execution.runId ? shortAuditRef(context.execution.runId) : statusLabel(context.execution.state);
    $("runmeta").innerHTML = [
      `案例 <b>v${esc(context.caseVersion || "—")}</b>`,
      `运行 <b title="${esc(context.execution.runId)}">${esc(runLabel)}</b>`,
      `来源 <b>${esc(sourceLabel)}</b>`,
    ].map((part) => `<span>${part}</span>`).join("");
    return;
  }
  const c = DATA.config || {};
  const parts = [];
  if (c.datasets) parts.push(`数据集 <b>${c.datasets.length}</b>`);
  const protos = c.protocols || uniq(DATA.leaderboard.map((x) => x.protocol));
  if (protos?.length) parts.push(`协议 <b>${protos.join(",")}</b>`);
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
  inferenceModelSearch = "";
  nativeDashboardContext = {};
  evidenceLiveContext = {};
  auditEvidenceTab = "trajectory";
  auditRawCurveState = { key: "", status: "idle", data: null, error: "" };
  ACTIVE_EVALUATION_CASE = null;
  CASE_PERSISTENCE = { status: "idle", message: "" };
  PIPELINE_STATUS = { tone: "ready", message: "" };
  PIPELINE_JOBS = { training: null, inference: null };
  CURRENT_JOB = null;
  requestedCaseId = "";
  try {
    const url = new URL(location.href);
    url.searchParams.delete("case");
    history.replaceState(null, "", url);
  } catch (_) {
    // Embedded and standalone builds may not expose a mutable browser history.
  }
  TRAINING_PREFLIGHT = { status: "idle", signature: "", result: null, error: "" };
  if (trainingPreflightTimer) clearTimeout(trainingPreflightTimer);
  trainingPreflightTimer = null;
  trainingPreflightRequest += 1;
  if (casePersistenceTimer) clearTimeout(casePersistenceTimer);
  casePersistenceTimer = null;
}

function renderKpis(lb, dsOptions) {
  const withMae = lb.filter((x) => metricMean(x, "mae") != null);
  const best = bestComparableEntry(withMae, selectedDatasetNames(), DATA.datasetLeaderboard || []);
  const algos = uniq(lb.map((x) => x.estimator));
  const selected = dsOptions.filter((item) => activeDatasets.has(item.name));
  const selectedRows = selected.reduce((sum, item) => sum + item.rows, 0);
  const qc = DATA.qc;
  const acceptance = acceptanceKpiSummary({
    mode: activeMode,
    rows: lb,
    datasetRows: DATA.datasetLeaderboard || [],
    inference: DATA.inference || {},
    preset: acceptancePreset,
  });
  const kpi = (lab, val, hint, color) =>
    `<div class="kpi"><div class="lab">${lab}</div><div class="val"${color ? ` style="color:${color}"` : ""}>${val}</div><div class="hint">${hint}</div></div>`;
  const context = currentCaseProjection(lb, dsOptions);
  if (context.hasCase) {
    const observedMae = context.verdict.observed?.mae;
    const candidateLabel = context.candidate.model || "待选择";
    const datasetLabel = context.dataset.name || "待固定";
    const verdictLabel = auditVerdictLabel(context.verdict.result);
    const verdictColor = context.verdict.result === "pass" ? "var(--pass)" :
      (context.verdict.result === "fail" ? "var(--fail)" : "var(--warn)");
    const closureLabels = {
      signed: "已签署",
      running: "复测中",
      failed: "复测失败",
      open: `${context.closure.openIssueCount} 项待闭环`,
      retested: "已完成复测",
      ready: "待进入闭环",
    };
    const closureColor = context.closure.state === "signed" || context.closure.state === "retested"
      ? "var(--pass)"
      : (context.closure.state === "failed" ? "var(--fail)" : "var(--warn)");
    $("kpis").innerHTML =
      kpi("当前候选", esc(candidateLabel), `MAE ${evidenceNumber(observedMae, 4)} · ${esc(context.candidate.featureset || "full")}`, "var(--primary)") +
      kpi("案例数据", esc(datasetLabel), `${esc(context.dataset.assetId || "未固定资产")} · ${esc(shortAuditRef(context.dataset.snapshotId))}`) +
      kpi("服务端判定", esc(verdictLabel), esc(context.verdict.criterionSetVersion || "未绑定判据"), verdictColor) +
      kpi("回归闭环", esc(closureLabels[context.closure.state] || context.closure.state), `${context.closure.retestMemberCount || 0} 条复测样本 · ${context.closure.retestCount} 次复测`, closureColor);
    return;
  }
  $("kpis").innerHTML =
    kpi("可比最佳 MAE", best ? esc(best.estimator) : "无",
      best ? `MAE ${fmt(metricMean(best, "mae"))} / ${esc(best.featureset)}` : "无指标",
      "var(--primary)") +
    kpi("数据集范围", `${selected.length}/${dsOptions.length}`,
      `${selectedRows.toLocaleString()} 条 QC 样本`) +
    kpi(acceptance.label, acceptance.value, acceptance.hint, acceptance.color) +
    kpi("数据 QC 通过率", qc ? (100 * qc.pass_rate).toFixed(2) + "%" : "无",
      qc ? `剔除 ${qc.n_out} 行` : "无 QC", "var(--pass)");
}

function renderScopeContext(dsOptions) {
  const context = currentCaseProjection(scoped?.entries || DATA?.leaderboard || [], dsOptions);
  if (context.hasCase) {
    const group = (label, value, title = value, kind = "") =>
      `<div class="scope-group"><span class="scope-label">${esc(label)}</span><div class="chips"><span class="chip ${esc(kind)}" title="${esc(title)}">${esc(value || "未固定")}</span></div></div>`;
    $("scope-context").innerHTML =
      group("案例", `${context.caseId} · v${context.caseVersion}`, context.caseId) +
      group("数据", context.dataset.name || "未固定", context.dataset.snapshotId || context.dataset.name) +
      group("候选", `${context.candidate.model || "未选择"} / ${context.candidate.featureset || "full"}`, context.candidate.id, "green") +
      group("协议", shortAuditRef(context.protocolRef), context.protocolRef) +
      group("模型产物", shortAuditRef(context.candidate.artifactId), context.candidate.artifactId, "amber");
    return;
  }
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
    `</div>`;
}

function currentCaseProjection(lb = scoped?.entries || DATA?.leaderboard || [], dsOptions = currentDatasetOptions()) {
  const trainingJob = currentTrainingJob();
  const inferenceJob = currentInferenceJob();
  const trainingRequest = trainingJob?.request || {};
  const inferenceRequest = inferenceJob?.request || {};
  const pipelineDatasets = inferenceRequest.submitted_datasets
    || inferenceRequest.selected_datasets
    || inferenceRequest.datasets
    || trainingRequest.datasets
    || [];
  const pipelineProtocols = trainingRequest.protocols || [];
  const pipelineRunId = trainingJob?.run_id || trainingJob?.id || "";
  const resultSummary = trainingJob?.result_summary || {};
  const runCandidates = (ACTIVE_EVALUATION_CASE?.candidates || [])
    .filter((item) => !pipelineRunId || item?.run_id === pipelineRunId);
  const pipelineCandidate = runCandidates.find((item) =>
    item?.model === resultSummary.best_estimator
    && (item?.featureset || "full") === (resultSummary.best_featureset || "full")
    && item?.role === "champion"
  ) || runCandidates.find((item) => item?.role === "champion") || runCandidates[0] || null;
  const best = bestComparableEntry(
    (lb || []).filter((item) => metricMean(item, "mae") != null),
    selectedDatasetNames(),
    DATA?.datasetLeaderboard || [],
  );
  return caseContextProjection(ACTIVE_EVALUATION_CASE || {}, REVIEW_WORKSPACE || {}, {
    caseId: currentCaseId(),
    profileName: activeProfileOption?.name || activeProfile || "root",
    sourceClass: DATA?.fallback ? "demo" : "measured",
    executionState: evaluationExecutionState(),
    runId: inferenceJob?.run_id || inferenceJob?.id || pipelineRunId || CURRENT_JOB?.run_id || CURRENT_JOB?.id || "",
    datasets: pipelineDatasets.length
      ? pipelineDatasets
      : (selectedDatasetNames().length ? selectedDatasetNames() : (dsOptions || []).map((item) => item.name)),
    protocols: pipelineProtocols.length ? pipelineProtocols : (DATA?.config?.protocols || []),
    criterionSetVersion: acceptancePreset,
    candidate: pipelineCandidate ? {
      id: pipelineCandidate.candidate_id || "",
      model: pipelineCandidate.model || pipelineCandidate.estimator || "",
      featureset: pipelineCandidate.featureset || "full",
      artifactId: pipelineCandidate.model_artifact_id || pipelineCandidate.artifact_id || "",
    } : (best ? {
      model: best.estimator || "",
      featureset: best.featureset || "full",
      artifactId: best.model_artifact_id || best.artifact_id || "",
    } : {}),
  });
}

function renderGlobalCaseContext() {
  if (!DATA) return;
  const dsOptions = currentDatasetOptions();
  const rows = scoped?.entries || DATA.leaderboard || [];
  renderRunMeta();
  renderKpis(rows, dsOptions);
  renderScopeContext(dsOptions);
  renderCaseIdentity();
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
    renderQuickstartExample(false);
    renderTrainingWorkflow(evaluationPlanSummary(buildCurrentEvaluationPayload()));
    renderTrainingArtifacts();
    jobState.textContent = "后端未连接，静态结果仍可浏览。";
    wireWorkbenchActions(false);
    return;
  }

  const options = catalogOptions(CATALOG);
  state.textContent = API_SESSION.legacyExecutionEnabled ? "已连接 · 可信执行" : "已连接 · 仅浏览";
  state.className = API_SESSION.legacyExecutionEnabled ? "api-state ok" : "api-state warn";
  renderResourceList("run-datasets", options.datasets, resourceSelection.datasets, (item) => `${item.rows.toLocaleString()} 行`);
  renderResourceList(
    "run-estimators",
    options.estimators,
    resourceSelection.estimators,
    (item) => item.methodCategory === "joint_method"
      ? `联合执行 · 固定 ${item.preferredFeatureset || "内部特征管线"}`
      : (item.methodCategory === "recommended_preset"
        ? `推荐预设 · ${item.preferredFeatureset || "可改选特征"}`
        : `自由组合 · ${item.description || item.featureMode || item.backend}`),
  );
  renderResourceList(
    "run-featuresets",
    options.featuresets,
    resourceSelection.featuresets,
    (item) => item.family === "mechanism_derived"
      ? `机理代理 · ${item.columns?.length || 0} 列`
      : (item.columns?.length ? `${item.columns.length} 列` : "选择器"),
  );
  renderRunPlan();
  renderQuickstartExample(true);
  renderTrainingWorkflow(evaluationPlanSummary(buildCurrentEvaluationPayload()));
  renderTrainingArtifacts();
  renderJobState();
  wireWorkbenchActions(true);
}

function renderQuickstartExample(enabled) {
  const stepsNode = $("quickstart-example-steps");
  const resultNode = $("quickstart-example-result");
  if (!resultNode) return;
  const example = QUICKSTART_EXAMPLE;
  const reported = new Map((example?.steps || []).map((step) => [step.id, step]));
  if (stepsNode) stepsNode.replaceChildren();

  if (!example) {
    resultNode.textContent = enabled ? "约数秒完成，不使用本地数据目录中的大数据集" : "后端未连接";
  } else if (example.status === "succeeded") {
    const result = example.result || {};
    const mae = Number.isFinite(Number(result.mae)) ? Number(result.mae).toFixed(4) : "无";
    const latency = Number.isFinite(Number(result.ms_per_1000_samples))
      ? `${Number(result.ms_per_1000_samples).toFixed(2)} ms/千样本`
      : "无测速";
    resultNode.innerHTML = `<strong>闭环通过</strong><span>MAE ${esc(mae)} · ${esc(latency)}</span>`;
  } else if (example.status === "failed") {
    resultNode.innerHTML = `<strong>运行失败</strong><span>${esc(example.error || "请查看后端日志")}</span>`;
  } else {
    const activeStep = reported.get(example.current_step)?.label || "准备资源";
    resultNode.innerHTML = `<strong>${example.status === "queued" ? "等待执行" : "正在执行"}</strong><span>${esc(activeStep)}</span>`;
  }
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
  scheduleTrainingPreflight(payload, plan);
  const preflight = currentTrainingPreflight(payload);
  const verified = preflight.status === "ready" && preflight.result?.ready;
  const blocked = preflight.status === "blocked";
  const summary = preflight.result?.summary || {};
  const runInstances = Number(summary.run_instances || 0);
  const cls = verified ? "ready" : (blocked || preflight.status === "failed" ? "blocked" : "incomplete");
  const primaryLabel = verified
    ? `${runInstances.toLocaleString()} 个已预检实例`
    : (preflight.status === "checking" ? "正在核对真实数据" : plan.comboLabel);
  const planNote = verified
    ? `${Number(summary.rows || 0).toLocaleString()} 条 QC 样本 · ${summary.runnable_combinations || 0} 个组合 · 数据快照与特征契约已固定。`
    : (blocked
      ? trainingPreflightBlockReason(preflight.result)
      : (preflight.status === "failed"
        ? `预检失败：${preflight.error}`
        : (plan.ready ? "正在由后端读取真实数据、协议分组和特征列。" : "请至少选择一个数据集、一个特征方案、一个算法、协议和随机种子。")));
  node.innerHTML =
    `<div class="run-plan-main ${cls}">` +
      `<span>运行计划</span>` +
      `<b>${esc(primaryLabel)}</b>` +
      `<small>${esc(plan.partsLabel)}</small>` +
    `</div>` +
    `<div class="run-plan-note ${blocked ? "fail" : ""}">${esc(planNote)}</div>`;
  const start = $("start-eval");
  if (start) {
    const executionEnabled = Boolean(CATALOG) && API_SESSION.legacyExecutionEnabled;
    const trainingJob = currentTrainingJob();
    const readyForInference = trainingJob?.status === "succeeded"
      && trainingJobMatchesPayload(trainingJob, payload)
      && Boolean(trainingJob.saved_models?.length);
    if (readyForInference) {
      start.textContent = "进入推理验证";
      start.disabled = false;
      start.title = `${trainingJob.saved_models.length} 个折次产物将按候选组聚合，并自动选出推荐产物`;
      start.onclick = () => continueTrainingToInference(trainingJob);
    } else {
      start.textContent = verified ? `运行 ${runInstances.toLocaleString()} 个实例` : "等待组合预检";
      start.disabled = !executionEnabled || !plan.ready || !verified;
      start.title = !executionEnabled
        ? "当前为仅浏览模式，请使用可信本机执行方式启动后端"
        : (!plan.ready
          ? "请先完成数据、特征、算法、协议和随机种子选择"
          : (verified ? "" : planNote));
      start.onclick = executionEnabled && plan.ready && verified ? startEvaluation : null;
    }
  }
  renderTrainingCombinationPreview();
  renderWorkbenchSteps(plan);
  renderTrainingWorkflow(plan);
  renderTrainingArtifacts();
  renderCompletionSaveControls();
}

function renderTrainingCombinationPreview() {
  const node = $("training-combination-preview");
  if (!node) return;
  const options = catalogOptions(CATALOG || {});
  const datasets = options.datasets.filter((item) => resourceSelection.datasets.has(item.name));
  const estimators = options.estimators.filter((item) => resourceSelection.estimators.has(item.name));
  const featuresets = options.featuresets.filter((item) => resourceSelection.featuresets.has(item.name));
  if (!estimators.length || !featuresets.length) {
    node.innerHTML = '<div class="combo-preview-empty">选择特征方案和算法后，这里会形成待运行组合矩阵。</div>';
    return;
  }
  const state = currentTrainingPreflight();
  if (state.status === "checking") {
    node.innerHTML = '<div class="combo-preview-empty checking"><b>正在执行组合预检</b><span>读取 QC 后真实列覆盖、LOCO/LODO/LCHO 分组和运行依赖。</span></div>';
    return;
  }
  if (state.status === "failed") {
    node.innerHTML = `<div class="combo-preview-empty failed"><b>组合预检失败</b><span>${esc(state.error)}</span></div>`;
    return;
  }
  const result = state.result;
  if (!result) {
    node.innerHTML = '<div class="combo-preview-empty">等待后端读取真实数据后确认组合。</div>';
    return;
  }
  const visibleEstimators = estimators.slice(0, 6);
  const visibleFeaturesets = featuresets.slice(0, 5);
  const coverageByName = new Map((result.featuresets || []).map((item) => [item.name, item]));
  const header = visibleFeaturesets.map((feature) => {
    const item = coverageByName.get(feature.name) || {};
    const coverage = item.dataset_coverage || [];
    const readyCount = coverage.filter((row) => row.status === "ready").length;
    const hasBlocked = coverage.some((row) => row.status === "blocked");
    const coverageLabel = `${item.selected_columns?.length || 0} 列 · ${readyCount}/${coverage.length || datasets.length} 完整`;
    return `<th><b>${esc(feature.name)}</b><small class="${hasBlocked ? "fail" : (readyCount === coverage.length ? "pass" : "warn")}">${esc(coverageLabel)}</small></th>`;
  }).join("");
  const rows = visibleEstimators.map((estimator) =>
    `<tr><th><b>${esc(estimator.name)}</b><small>${esc(estimator.backend || "external")}</small></th>` +
    visibleFeaturesets.map((feature) => {
      const combinationFeature = estimator.featureMode === "byo" ? "(byo)" : feature.name;
      const matches = (result.combinations || []).filter((item) =>
        item.estimator === estimator.name && item.featureset === combinationFeature);
      const notApplicable = matches.length > 0 && matches.every((item) => item.status === "not_applicable");
      const status = notApplicable
        ? "not-applicable"
        : (matches.some((item) => item.status === "blocked")
        ? "blocked"
        : (matches.some((item) => item.status === "warning") ? "review" : "ready"));
      const detail = matches.flatMap((item) => [...(item.reasons || []), ...(item.warnings || [])]);
      const label = estimator.featureMode === "byo"
        ? "算法自带特征"
        : (status === "not-applicable"
          ? "方法包不采用"
          : (status === "blocked" ? "已阻断" : (status === "review" ? "可运行·有提示" : "可运行")));
      return `<td><span class="combo-cell ${status}" title="${esc(detail.join("；") || `${estimator.name} × ${feature.name} 已通过`)}">${esc(label)}</span></td>`;
    }).join("") + `</tr>`
  ).join("");
  const hiddenCount = Math.max(0, estimators.length - visibleEstimators.length) * featuresets.length +
    Math.max(0, featuresets.length - visibleFeaturesets.length) * visibleEstimators.length;
  const blocked = (result.combinations || []).filter((item) => item.status === "blocked");
  const warnings = (result.combinations || []).filter((item) => item.status === "warning");
  const notApplicable = (result.combinations || []).filter((item) => item.status === "not_applicable");
  const preflightToken = String(result.preflight_id || "").replace("preflight@sha256:", "").slice(0, 12);
  node.innerHTML =
    `<div class="combo-preview-head"><div><span>权威组合预检</span><b>${estimators.length} 个算法 × ${featuresets.length} 个特征方案</b></div>` +
      `<small>${Number(result.summary?.runnable_combinations || 0)} 个实际组合 · ${Number(result.summary?.rows || 0).toLocaleString()} 条真实样本 · ${esc(preflightToken)}</small></div>` +
    `<div class="combo-preview-table"><table><thead><tr><th>算法 / 特征</th>${header}</tr></thead><tbody>${rows}</tbody></table></div>` +
    (hiddenCount ? `<div class="combo-preview-note">矩阵仅预览前 ${visibleEstimators.length} 个算法和 ${visibleFeaturesets.length} 个特征方案；运行计划包含全部选择。</div>` : "") +
    (blocked.length
      ? `<div class="combo-coverage-warning fail">${esc(trainingPreflightBlockReason(result))}</div>`
      : (warnings.length
        ? `<div class="combo-coverage-warning">${warnings.length} 个组合可运行但存在覆盖提示；悬停单元格查看原因。</div>`
        : `<div class="combo-coverage-pass">后端预检通过：${notApplicable.length ? `${notApplicable.length} 个不适用的笛卡尔组合已自动跳过；` : ""}数据快照、特征列和协议分组均可执行。</div>`));
}

function renderWorkbenchSteps(plan = null) {
  const node = $("workbench-steps");
  if (!node) return;
  const trainingJob = currentTrainingJob();
  const runtime = trainingJobRuntimeState(trainingJob);
  const jobStatus = trainingJob?.status || "";
  const preflight = currentTrainingPreflight();
  const preflightState = preflight.status === "ready" ? "complete" : (preflight.status === "checking" ? "active" : "pending");
  const items = [
    ["预检", preflightState, preflight.status === "ready" ? "真实数据与组合已固定" : (preflight.status === "blocked" ? "组合被阻断" : "等待权威预检")],
    ["运行", jobStatus === "running" || jobStatus === "queued" ? "active" : (jobStatus === "succeeded" ? "complete" : "pending"), trainingJob ? runtime.hint : "尚未启动"],
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
  const trainingJob = currentTrainingJob();
  const runtime = trainingJobRuntimeState(trainingJob);
  const jobStatus = trainingJob?.status || "";
  const preflight = currentTrainingPreflight();
  const model = modelOptions(currentModelCatalog())[0] || {};
  const dataset = (DATA?.config?.datasets || [])[0] || selectedDatasetNames()[0] || "";
  const steps = [
    ["资源接入", CATALOG ? "complete" : "pending", CATALOG ? "资源目录已加载" : "等待后端连接"],
    ["组合预检", preflight.status === "ready" ? "complete" : (preflight.status === "checking" ? "active" : "pending"), preflight.status === "ready" ? "真实列覆盖与协议分组通过" : (preflight.status === "blocked" ? "存在不可运行组合" : plan?.partsLabel || "请选择数据、算法、特征")],
    ["训练执行", jobStatus === "running" || jobStatus === "queued" ? "active" : (jobStatus === "succeeded" ? "complete" : "pending"), trainingJob ? runtime.hint : "尚未启动"],
    ["模型发布", jobStatus === "succeeded" ? "complete" : "pending", jobStatus === "succeeded" ? "内容寻址算法包已登记" : "训练产物进入算法包管理"],
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
    `<div class="note">源码算法、外部 Pickle 和 manifest 统一在“算法包管理”接入；本页只负责选择可运行组合。</div>` +
    sohDrilldownButton({
      dataset,
      source: "training",
      model: model.name || model.estimator || "",
      featureset: model.featureset || activeFs || "full",
      label: "进入比较与证据",
      className: "secondary-action soh-panel-action",
    }) +
    `</div>`;
}

function renderTrainingArtifacts() {
  const node = $("training-artifacts");
  if (!node) return;
  const summary = trainingArtifactSummary(DATA?.config || {}, currentModelCatalog());
  const trainingJob = currentTrainingJob();
  const runtime = trainingJobRuntimeState(trainingJob);
  const jobStatus = trainingJob?.status || "未启动";
  const selectedFeatures = resourceSelection.featuresets.size;
  const selectedEstimators = resourceSelection.estimators.size;
  const selectedDatasets = resourceSelection.datasets.size;
  const preflight = currentTrainingPreflight();
  const preflightSummary = preflight.result?.summary || {};
  const resultSummary = trainingJob?.result_summary || {};
  node.innerHTML =
    `<div class="panel-title"><span>当前实验</span><b>组合摘要</b></div>` +
    `<div class="panel-scroll">` +
    `<div class="artifact-stack">` +
      artifactMetric("特征方案", `${selectedFeatures} 个`, "决定模型输入表达与可解释性", "dataset") +
      artifactMetric("算法候选", `${selectedEstimators} 个`, "基线与深度算法可并行对比", "model") +
      artifactMetric("数据范围", `${selectedDatasets} 个数据集`, "用于检查特征列覆盖", "dataset") +
      artifactMetric("组合预检", preflight.status === "ready" ? "已通过" : (preflight.status === "blocked" ? "已阻断" : "待确认"), preflight.status === "ready" ? `${preflightSummary.run_instances || 0} 个运行实例 / 快照已固定` : "由后端读取真实数据确认", "job") +
      artifactMetric("任务状态", jobStatus, trainingJob ? `${trainingJob.id} / ${runtime.hint}` : "等待新任务", "job") +
      (trainingJob?.status === "succeeded" ? artifactMetric("真实结果", resultSummary.best_mae == null ? "已产出" : `MAE ${fmt(Number(resultSummary.best_mae), 4)}`, `${resultSummary.model_artifact_count || 0} 个不可变模型产物`, "model") : "") +
      artifactMetric("历史模型包", summary.label, `${summary.generatedCount} 训练产出 / ${summary.uploadedCount} 外部导入`, "model") +
    `</div>` +
    `<div class="note">外部 Pickle 与 manifest 请在“算法包管理”中导入；训练成功后无需手工上传。</div>` +
    `</div>`;
}

function artifactMetric(label, value, hint, kind) {
  return `<div class="artifact-metric ${esc(kind)}">` +
    `<span>${esc(label)}</span>` +
    `<b>${esc(value)}</b>` +
    `<small>${esc(hint)}</small>` +
  `</div>`;
}

function renderBatteryDiagnosticsPanel() {
  const node = $("battery-diagnostics-summary");
  if (!node) return;
  const diagnostics = DATA?.batteryDiagnostics || {};
  const summary = batteryDiagnosticSummary(diagnostics);
  const visualSummary = batteryVisualAssetSummary({
    ...diagnostics,
    profileManifest: DATA?.profileManifest || {},
  });
  const assetDetail = batteryVisualAssetDetail({
    ...diagnostics,
    profileManifest: DATA?.profileManifest || {},
  });
  const curvePreview = batteryCurvePreviewView(diagnostics, activeBatteryCurveDataset);
  if (!activeBatteryCurveDataset && curvePreview.selectedDataset) {
    activeBatteryCurveDataset = curvePreview.selectedDataset;
  }
  const scenarios = Array.isArray(diagnostics?.overview?.scenarios) && diagnostics.overview.scenarios.length
    ? diagnostics.overview.scenarios
    : (Array.isArray(diagnostics?.manifest?.scenarios) ? diagnostics.manifest.scenarios : []);
  if (!summary.available) {
    node.innerHTML = '<div class="note battery-diagnostics-empty">尚未导入 SOH diagnostics。导入 SOH dashboard artifacts 后将在这里显示场景、迁移和曲线资产。</div>';
    return;
  }

  const scenarioRows = batteryScenarioRows(scenarios).slice(0, 6).map((item) =>
    `<tr>` +
      `<td><span class="algo">${esc(item.name)}</span><small>${esc(item.kind)}</small></td>` +
      `<td>${esc(item.source)}</td>` +
      `<td>${esc(item.verdict)}</td>` +
      `<td class="tnum">${esc(item.metric)}</td>` +
    `</tr>`
  ).join("");
  const table = scenarioRows
    ? `<div class="battery-scenario-table"><table><thead><tr><th>场景</th><th>来源</th><th>判定</th><th>指标</th></tr></thead><tbody>${scenarioRows}</tbody></table></div>`
    : '<div class="note">overview.scenarios 暂无 SOH 场景明细。</div>';

  node.innerHTML =
    (curvePreview.available ? renderBatteryCurvePreview(curvePreview) : "") +
    `<div class="battery-diagnostics-grid">` +
      artifactMetric("来源", summary.sourceLabel, `${summary.scenarioCount} 个场景`, "model") +
      artifactMetric("实验 / 现场", `${summary.labScenarioCount} / ${summary.fieldScenarioCount}`, `${summary.passedCount} 个场景通过`, "dataset") +
      artifactMetric("迁移记录", `${summary.transferCount}`, "overview.transfers", "job") +
      artifactMetric("算法/曲线资产", visualSummary.available ? "可下钻" : "待迁移",
        visualSummary.curveArtifactCount
          ? `${visualSummary.statusLabel}；曲线 ${visualSummary.curveArtifactCount} 索引 / ${visualSummary.curveCopiedCount} 内嵌 / ${visualSummary.curveTotalLabel}`
          : visualSummary.statusLabel,
        visualSummary.available ? "model" : "job") +
    `</div>` +
    (visualSummary.available
      ? `<details class="battery-asset-inline"><summary>算法 / 特征 / 曲线索引</summary>${renderBatteryAssetSections(assetDetail, { available: false })}</details>`
      : "") +
    table;
  wireBatteryCurvePreview();
}

function renderPackageOperationsPanel() {
  const status = $("package-operation-status");
  const results = $("package-operation-results");
  const source = $("package-source-dashboard");
  if (!status || !results) return;
  if (source && !source.value) source.value = DEFAULT_BATTERY_DASHBOARD_SOURCE;

  const connected = Boolean(CATALOG);
  const executionEnabled = connected && API_SESSION.legacyExecutionEnabled;
  const busy = Boolean(PACKAGE_STATUS.busy);
  const message = executionEnabled
    ? (PACKAGE_STATUS.message || "可同步 SOH 看板并生成离线包。")
    : (connected
      ? "安全模式已启用：本机同步与打包入口未开放。"
      : "后端未连接，暂不能执行本机文件操作。");
  status.textContent = message;
  status.className = `package-status ${executionEnabled ? PACKAGE_STATUS.tone : "fail"}`;

  const order = ["sync", "portable", "standalone"];
  const empty = '<div class="note">暂无同步或打包记录。</div>';
  results.innerHTML = order
    .filter((key) => PACKAGE_STATUS.results[key])
    .map((key) => renderPackageResult(key, PACKAGE_STATUS.results[key]))
    .join("") || empty;

  const bind = (id, action) => {
    const button = $(id);
    if (!button) return;
    button.disabled = !executionEnabled || busy;
    button.title = executionEnabled ? "" : "安全模式下未启用旧执行入口";
    button.onclick = executionEnabled && !busy ? () => runPackageOperation(action) : null;
  };
  bind("sync-native-dashboard", "sync");
  bind("build-portable-package", "portable");
  bind("build-standalone-package", "standalone");
}

function renderPackageResult(kind, record) {
  const manifest = record?.manifest || {};
  const title = {
    sync: "SOH 看板同步",
    portable: "目录包",
    standalone: "单 HTML",
  }[kind] || kind;
  const output = manifest.output || manifest.target || "未记录输出路径";
  const count = manifest.file_count != null
    ? `${manifest.file_count} 个文件`
    : manifest.artifact_count != null
      ? `${manifest.artifact_count} 个 JSON`
      : manifest.copied_files
        ? `${manifest.copied_files.length} 个文件`
        : "文件数未记录";
  const size = manifest.total_bytes != null ? formatBytes(manifest.total_bytes) : "";
  const curves = manifest.battery_dashboard?.soh_curve_file_count ?? (manifest.include_soh_curves ? "已包含" : "未包含");
  const detection = manifest.excluded_detection_assets || manifest.battery_dashboard?.excluded_detection_assets
    ? "ESS detection 已排除"
    : "ESS detection 边界待核验";
  return `<div class="package-result">` +
    `<b>${esc(title)}</b>` +
    `<small>${esc(record.time || "")}</small>` +
    `<span title="${esc(output)}">${esc(output)}</span>` +
    `<em>${esc([count, size, `曲线 ${curves}`, detection].filter(Boolean).join(" / "))}</em>` +
  `</div>`;
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(2)} MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${value} B`;
}

function renderPackageReportPanel() {
  const summary = $("package-report-summary");
  const checks = $("package-report-checks");
  const refresh = $("refresh-package-report");
  if (!summary || !checks) return;

  const connected = Boolean(CATALOG);
  const busy = Boolean(PACKAGE_REPORT.busy);
  summary.textContent = connected
    ? (PACKAGE_REPORT.message || "等待刷新验收")
    : "后端未连接，暂不能读取交付包验收报告。";
  summary.className = `package-status ${connected ? PACKAGE_REPORT.tone : "fail"}`;

  const reportChecks = Array.isArray(PACKAGE_REPORT.report?.checks) ? PACKAGE_REPORT.report.checks : [];
  checks.innerHTML = reportChecks.length
    ? reportChecks.map(renderPackageCheck).join("")
    : '<div class="note">暂无验收报告。完成同步和打包后点击刷新。</div>';

  if (refresh) {
    refresh.disabled = !connected || busy;
    refresh.onclick = connected && !busy ? () => loadPackageReport() : null;
  }
}

function renderPackageCheck(check) {
  const state = check?.passed ? "pass" : "fail";
  const label = check?.label || check?.id || "验收项";
  const detail = check?.detail || "";
  return `<div class="package-check ${state}">` +
    `<span>${check?.passed ? "通过" : "待处理"}</span>` +
    `<b>${esc(label)}</b>` +
    `<small>${esc(detail)}</small>` +
  `</div>`;
}

async function loadPackageReport(options = {}) {
  const silent = Boolean(options.silent);
  if (!CATALOG) {
    PACKAGE_REPORT = {
      busy: false,
      tone: "fail",
      message: "后端未连接，暂不能读取交付包验收报告。",
      report: null,
    };
    if (!silent) renderPackageReportPanel();
    return;
  }

  PACKAGE_REPORT = {
    ...PACKAGE_REPORT,
    busy: true,
    tone: "active",
    message: "正在刷新交付包验收...",
  };
  if (!silent) renderPackageReportPanel();

  try {
    const response = await fetch(`${API_BASE}/api/package/status`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.ok) throw new Error(body.error || `HTTP ${response.status}`);
    const report = body.report || {};
    const summary = report.summary || {};
    const ready = Boolean(summary.ready);
    PACKAGE_REPORT = {
      busy: false,
      tone: ready ? "ok" : "fail",
      message: ready
        ? `验收通过：${summary.passed || 0}/${summary.total || 0} 项已通过`
        : `验收未通过：${summary.failed || 0} 项需处理`,
      report,
    };
  } catch (err) {
    PACKAGE_REPORT = {
      busy: false,
      tone: "fail",
      message: `验收报告读取失败：${err.message}`,
      report: null,
    };
  }
  if (!silent) renderPackageReportPanel();
}

async function runPackageOperation(action) {
  const source = ($("package-source-dashboard")?.value || DEFAULT_BATTERY_DASHBOARD_SOURCE).trim() || DEFAULT_BATTERY_DASHBOARD_SOURCE;
  const specs = {
    sync: {
      endpoint: "/api/sync/native-dashboard",
      label: "同步 SOH 看板",
      payload: { source, include_curves: true },
    },
    portable: {
      endpoint: "/api/package/portable",
      label: "生成目录包",
      payload: {},
    },
    standalone: {
      endpoint: "/api/package/standalone",
      label: "生成单 HTML",
      payload: {},
    },
  };
  const spec = specs[action];
  if (!spec) return;

  PACKAGE_STATUS = {
    ...PACKAGE_STATUS,
    busy: action,
    tone: "active",
    message: `正在${spec.label}...`,
  };
  renderPackageOperationsPanel();
  try {
    const response = await postJson(spec.endpoint, spec.payload);
    PACKAGE_STATUS = {
      busy: "",
      tone: "ok",
      message: `${spec.label}完成。`,
      results: {
        ...PACKAGE_STATUS.results,
        [action]: {
          time: new Date().toLocaleString("zh-CN"),
          manifest: response.manifest || {},
        },
      },
    };
    if (action === "sync") {
      await load({ cacheBust: true });
    } else {
      await loadPackageReport({ silent: true });
      renderPackageOperationsPanel();
      renderPackageReportPanel();
    }
  } catch (err) {
    PACKAGE_STATUS = {
      ...PACKAGE_STATUS,
      busy: "",
      tone: "fail",
      message: `${spec.label}失败：${err.message}`,
    };
    renderPackageOperationsPanel();
  }
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

function trainingPayloadSignature(payload) {
  return JSON.stringify({
    datasets: payload.datasets || [],
    estimators: payload.estimators || [],
    featuresets: payload.featuresets || [],
    protocols: payload.protocols || [],
    seeds: payload.seeds || [],
  });
}

function currentTrainingPreflight(payload = buildCurrentEvaluationPayload()) {
  const signature = trainingPayloadSignature(payload);
  return TRAINING_PREFLIGHT.signature === signature ? TRAINING_PREFLIGHT : {
    status: "idle",
    signature,
    result: null,
    error: "",
  };
}

function trainingPreflightBlockReason(result) {
  const blocked = (result?.combinations || []).find((item) => item.status === "blocked");
  if (blocked?.reasons?.[0]) return blocked.reasons[0];
  const skipped = (result?.combinations || []).find((item) => item.status === "not_applicable");
  return skipped?.warnings?.[0] || "当前组合未通过后端预检";
}

function scheduleTrainingPreflight(payload, plan) {
  const signature = trainingPayloadSignature(payload);
  if (!plan.ready || !CATALOG) {
    if (trainingPreflightTimer) clearTimeout(trainingPreflightTimer);
    trainingPreflightTimer = null;
    TRAINING_PREFLIGHT = { status: "idle", signature, result: null, error: "" };
    return;
  }
  if (TRAINING_PREFLIGHT.signature === signature && ["checking", "ready", "blocked", "failed"].includes(TRAINING_PREFLIGHT.status)) {
    return;
  }
  if (trainingPreflightTimer) clearTimeout(trainingPreflightTimer);
  const requestNumber = ++trainingPreflightRequest;
  TRAINING_PREFLIGHT = { status: "checking", signature, result: null, error: "" };
  trainingPreflightTimer = setTimeout(async () => {
    try {
      const result = await postJson("/api/training/preflight", payload);
      if (requestNumber !== trainingPreflightRequest || signature !== trainingPayloadSignature(buildCurrentEvaluationPayload())) return;
      TRAINING_PREFLIGHT = {
        status: result.ready ? "ready" : "blocked",
        signature,
        result,
        error: "",
      };
    } catch (err) {
      if (requestNumber !== trainingPreflightRequest) return;
      TRAINING_PREFLIGHT = { status: "failed", signature, result: null, error: err.message };
    }
    renderRunPlan();
  }, 180);
}

function renderJobState() {
  const node = $("job-state");
  if (!node) return;
  const trainingJob = currentTrainingJob();
  if (!trainingJob) {
    node.textContent = "组合通过真实数据预检后，可启动一轮新的本机评测。";
    node.className = "job-state";
    return;
  }
  const runtime = trainingJobRuntimeState(trainingJob);
  const bits = [`任务 ${trainingJob.id}`, `状态 ${trainingJob.status}`, runtime.hint];
  if (trainingJob.run_dir) bits.push(`结果 ${trainingJob.run_dir}`);
  if (trainingJob.preflight_id) bits.push(`预检 ${String(trainingJob.preflight_id).replace("preflight@sha256:", "").slice(0, 12)}`);
  if (trainingJob.status === "succeeded") {
    const summary = trainingJob.result_summary || {};
    bits.push(`真实结果 ${summary.evaluations || 0} 个 fold 实例`);
    if (summary.best_mae != null) bits.push(`最佳 MAE ${Number(summary.best_mae).toFixed(4)}`);
    bits.push(`${summary.model_artifact_count || 0} 个不可变模型产物`);
    bits.push("已发布到仪表盘");
  }
  if (trainingJob.error) bits.push(`错误 ${trainingJob.error}`);
  if (PIPELINE_STATUS.message && ["succeeded", "failed"].includes(trainingJob.status)) bits.push(PIPELINE_STATUS.message);
  node.className = `job-state ${runtime.tone}`;
  node.innerHTML = bits.map((bit, i) => i === 0 ? `<strong>${esc(bit)}</strong>` : esc(bit)).join(" / ");
}

function wireWorkbenchActions(enabled) {
  const executionEnabled = enabled && API_SESSION.legacyExecutionEnabled;
  document.querySelectorAll("[data-upload-kind]").forEach((button) => {
    button.disabled = !executionEnabled;
    button.title = executionEnabled ? "" : "当前为仅浏览模式，请使用可信本机执行方式启动后端";
    button.onclick = executionEnabled ? () => uploadResource(button.dataset.uploadKind) : null;
  });
  const batterySohImport = $("import-battery-soh-plugins");
  if (batterySohImport) {
    batterySohImport.disabled = !executionEnabled;
    batterySohImport.title = executionEnabled ? "" : "当前为仅浏览模式，请使用可信本机执行方式启动后端";
    batterySohImport.onclick = executionEnabled ? importBatterySohPlugins : null;
  }
  const start = $("start-eval");
  if (start) {
    const payload = buildCurrentEvaluationPayload();
    const plan = evaluationPlanSummary(payload);
    const preflight = currentTrainingPreflight(payload);
    const verified = preflight.status === "ready" && preflight.result?.ready;
    const trainingJob = currentTrainingJob();
    const readyForInference = trainingJob?.status === "succeeded"
      && trainingJobMatchesPayload(trainingJob, payload)
      && Boolean(trainingJob.saved_models?.length);
    if (readyForInference) {
      start.disabled = false;
      start.title = "模型、数据范围和训练任务编号将自动带入同一案例";
      start.onclick = () => continueTrainingToInference(trainingJob);
    } else {
      start.disabled = !executionEnabled || !plan.ready || !verified;
      start.title = !executionEnabled
        ? "当前为仅浏览模式，请使用可信本机执行方式启动后端"
        : (!plan.ready ? "请先完成数据、特征、算法、协议和随机种子选择" : (verified ? "" : "等待后端组合预检通过"));
      start.onclick = executionEnabled && plan.ready && verified ? startEvaluation : null;
    }
  }
  const quickstart = $("start-quickstart-example");
  if (quickstart) {
    const busy = ["queued", "running"].includes(QUICKSTART_EXAMPLE?.status);
    const succeeded = QUICKSTART_EXAMPLE?.status === "succeeded";
    quickstart.disabled = !executionEnabled || busy;
    quickstart.title = executionEnabled ? "" : "当前为仅浏览模式，请使用可信本机执行方式启动后端";
    quickstart.textContent = busy ? "运行中" : (succeeded ? "查看结果" : (QUICKSTART_EXAMPLE?.status === "failed" ? "重新运行" : "运行内置示例"));
    quickstart.onclick = executionEnabled && !busy ? (succeeded ? async () => {
      activeProfile = QUICKSTART_EXAMPLE?.profile_name || "quickstart-e2e";
      localStorage.setItem("sohbench.profile", activeProfile);
      await load({ cacheBust: true });
      activeMode = "inference";
      render();
    } : startQuickstartExample) : null;
  }
  document.querySelectorAll("input[name='run-protocol']").forEach((input) => {
    input.onchange = renderRunPlan;
  });
  const seeds = $("job-seeds");
  if (seeds) seeds.oninput = renderRunPlan;
}

async function startQuickstartExample() {
  try {
    QUICKSTART_EXAMPLE = await postJson("/api/examples/quickstart", {});
    renderQuickstartExample(true);
    wireWorkbenchActions(true);
    await pollQuickstartExample(QUICKSTART_EXAMPLE.id);
  } catch (err) {
    QUICKSTART_EXAMPLE = {
      status: "failed",
      error: err.message,
      steps: [],
    };
    renderQuickstartExample(true);
    wireWorkbenchActions(Boolean(CATALOG));
  }
}

async function pollQuickstartExample(exampleId) {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    try {
      const response = await fetch(`${API_BASE}/api/examples/${encodeURIComponent(exampleId)}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      QUICKSTART_EXAMPLE = await response.json();
      renderQuickstartExample(true);
      wireWorkbenchActions(true);
      if (QUICKSTART_EXAMPLE.status === "succeeded") {
        await load({ cacheBust: true });
        return;
      }
      if (QUICKSTART_EXAMPLE.status === "failed") return;
    } catch (err) {
      QUICKSTART_EXAMPLE = { ...QUICKSTART_EXAMPLE, status: "failed", error: `状态轮询失败：${err.message}` };
      renderQuickstartExample(true);
      wireWorkbenchActions(Boolean(CATALOG));
      return;
    }
  }
}

function currentCaseId() {
  if (ACTIVE_EVALUATION_CASE?.id) return ACTIVE_EVALUATION_CASE.id;
  const jobId = CURRENT_JOB?.job_id || CURRENT_JOB?.id || activeProfileOption?.trainingJobId ||
    activeProfileOption?.inferenceJobId || "";
  const raw = jobId || activeProfileOption?.name || activeProfile || "LIVE";
  const text = String(raw || "LIVE");
  return jobId && text.length > 18 ? text.slice(-18) : text;
}

function evaluationIdToken(value) {
  return String(value || "unknown")
    .replace(/[^A-Za-z0-9_.-]+/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "")
    .slice(0, 48) || "unknown";
}

function evaluationCandidateId(candidate) {
  if (candidate?.candidate_id) return String(candidate.candidate_id);
  const artifactId = candidate?.model_artifact_id || candidate?.artifact_id || candidateModelArtifactId(candidate);
  const pinned = (ACTIVE_EVALUATION_CASE?.candidates || [])
    .find((item) => artifactId && item.model_artifact_id === artifactId);
  if (pinned?.candidate_id) return pinned.candidate_id;
  return `legacy:${evaluationIdToken(candidate?.estimator || candidate?.model)}:${evaluationIdToken(candidate?.featureset || "full")}`;
}

function evaluationExecutionState() {
  const state = String(CURRENT_JOB?.status || "").toLowerCase();
  if (state === "succeeded" || state === "completed") return "completed";
  if (["queued", "running", "failed"].includes(state)) return state;
  if (activeProfileOption?.trainingJobId || activeProfileOption?.inferenceJobId) return "completed";
  return "idle";
}

function evaluationCaseIdentityPayload() {
  const candidates = evidenceCandidateRows(DATA?.leaderboard || []).map((row, index) => ({
    candidate_id: evaluationCandidateId(row),
    model: row.estimator || "unknown",
    featureset: row.featureset || "full",
    role: index === 0 ? "champion" : "candidate",
    model_artifact_id: row.model_artifact_id || row.artifact_id || candidateModelArtifactId(row),
    run_id: row.run_id || CURRENT_JOB?.run_id || "",
  }));
  return {
    profile_name: activeProfileOption?.name || activeProfile || "root",
    idempotency_key: evaluationCaseIdempotencyKey(),
    title: `${activeProfileOption?.label || activeProfile || "SOH"} · 评测案例`,
    source_class: DATA?.fallback ? "demo" : "measured",
    execution: {
      state: evaluationExecutionState(),
      training_job_id: activeProfileOption?.trainingJobId || (CURRENT_JOB?.kind === "training" ? CURRENT_JOB?.id || "" : ""),
      inference_job_id: activeProfileOption?.inferenceJobId || (CURRENT_JOB?.kind === "inference" ? CURRENT_JOB?.id || "" : ""),
      run_id: CURRENT_JOB?.run_id || "",
      failure_reason: CURRENT_JOB?.status === "failed" ? String(CURRENT_JOB?.error || "") : "",
    },
    candidates,
  };
}

function candidateModelArtifactId(row) {
  const models = currentModelCatalog().models || [];
  const runModels = new Set(DATA?.config?.saved_models || []);
  const matching = models.filter((item) =>
    item.estimator === row.estimator &&
    (item.featureset || "full") === (row.featureset || "full") &&
    item.artifact_id
  );
  if (matching.length === 1) return matching[0].artifact_id;
  const sameRun = matching.filter((item) => runModels.has(item.name));
  return sameRun.length === 1 ? sameRun[0].artifact_id : "";
}

function evaluationCaseIdempotencyKey() {
  const profile = activeProfileOption?.name || activeProfile || "root";
  if (ACTIVE_EVALUATION_CASE?.profile_name === profile && ACTIVE_EVALUATION_CASE?.idempotency_key) {
    return ACTIVE_EVALUATION_CASE.idempotency_key;
  }
  if (DATA?.profileManifest?.kind === "phase30_release_gate" && DATA?.config?.comparison_id) {
    return `phase30:${DATA.config.comparison_id}`;
  }
  if (DATA?.profileManifest?.kind === "phase24_case_pipeline" && DATA?.config?.comparison_id) {
    return `phase24:${DATA.config.comparison_id}`;
  }
  if (DATA?.profileManifest?.kind === "phase22_golden" && DATA?.config?.comparison_id) {
    return `phase22:${DATA.config.comparison_id}`;
  }
  const comparison = [
    ...(DATA?.config?.protocols || []),
    ...(DATA?.config?.datasets || []),
  ].sort().join("-") || "unscoped";
  return `pipeline:${evaluationIdToken(profile)}|scope:${evaluationIdToken(comparison)}`;
}

function evaluationContextBindingId(caseDocument = ACTIVE_EVALUATION_CASE) {
  return String(caseDocument?.context_binding?.binding_id || "");
}

function evaluationCaseMutationEnvelope(caseDocument = ACTIVE_EVALUATION_CASE) {
  return {
    expected_version: caseDocument?.version,
    expected_context_binding_id: evaluationContextBindingId(caseDocument),
  };
}

function nextPipelineCaseKey() {
  const profile = evaluationIdToken(activeProfileOption?.name || activeProfile || "root");
  const random = globalThis.crypto?.randomUUID?.().replace(/-/g, "").slice(0, 12)
    || `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  return `pipeline:${profile}|branch:${random}`;
}

function hasSignedCurrentCase() {
  const version = Number(ACTIVE_EVALUATION_CASE?.version || 0);
  return Boolean((REVIEW_WORKSPACE?.evidence_packs || []).some((item) =>
    item.state === "signed" && Number(item.case_version) === version
  ));
}

async function ensureWritablePipelineCase({ forNewTraining = false } = {}) {
  if (!ACTIVE_EVALUATION_CASE) await resolveEvaluationCaseContext();
  const immutable = hasSignedCurrentCase()
    || ["completed", "failed", "archived"].includes(ACTIVE_EVALUATION_CASE?.lifecycle_state);
  const freshTraining = forNewTraining && requiresFreshTrainingCase(
    ACTIVE_EVALUATION_CASE,
    REVIEW_WORKSPACE || {},
  );
  if (!immutable && !freshTraining) return ACTIVE_EVALUATION_CASE;

  const key = nextPipelineCaseKey();
  const payload = evaluationCaseIdentityPayload();
  payload.idempotency_key = key;
  payload.title = `${activeProfileOption?.label || activeProfile || "SOH"} · 新评测分支`;
  payload.execution = {
    state: "idle",
    training_job_id: "",
    inference_job_id: "",
    run_id: "",
    failure_reason: "",
  };
  payload.candidates = [];
  const body = await postJson("/api/v2/cases/resolve", payload);
  ACTIVE_EVALUATION_CASE = body.case || null;
  requestedCaseId = ACTIVE_EVALUATION_CASE?.id || "";
  updateModeUrl(activeMode);
  CURRENT_JOB = null;
  PIPELINE_JOBS = { training: null, inference: null };
  REVIEW_WORKSPACE = null;
  PIPELINE_STATUS = {
    tone: "ok",
    message: freshTraining
      ? "已为新训练创建案例分支；推理验证与评测证据均已重置为待开始。"
      : "已从终态案例创建新的评测分支。",
  };
  await loadReviewWorkspace({ silent: true, force: true });
  return ACTIVE_EVALUATION_CASE;
}

async function fetchCurrentEvaluationCase() {
  if (!ACTIVE_EVALUATION_CASE?.id) return null;
  const response = await fetch(`${API_BASE}/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  ACTIVE_EVALUATION_CASE = await response.json();
  return ACTIVE_EVALUATION_CASE;
}

async function updateEvaluationCaseIdentity(patch, reason) {
  if (!ACTIVE_EVALUATION_CASE?.id) return null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const body = await postJson(
        `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/identity`,
        {
          ...patch,
          ...evaluationCaseMutationEnvelope(),
          actor: "local-ui",
          reason,
        },
      );
      ACTIVE_EVALUATION_CASE = body.case || body;
      return ACTIVE_EVALUATION_CASE;
    } catch (err) {
      if (attempt === 0 && /conflict|版本|409/i.test(String(err.message || ""))) {
        await fetchCurrentEvaluationCase();
        continue;
      }
      throw err;
    }
  }
  return ACTIVE_EVALUATION_CASE;
}

function savedCombinations() {
  return Array.isArray(ACTIVE_EVALUATION_CASE?.saved_combinations)
    ? ACTIVE_EVALUATION_CASE.saved_combinations
    : [];
}

function activeSavedCombination() {
  const id = String(ACTIVE_EVALUATION_CASE?.active_saved_combination_id || "");
  return savedCombinations().find((item) => item.saved_combination_id === id) || null;
}

function savedCombinationsForJob(stage, job) {
  if (!job?.id) return [];
  const jobKey = stage === "training" ? "training_job_id" : "inference_job_id";
  return savedCombinations().filter((item) =>
    item.source_stage === stage && String(item[jobKey] || "") === String(job.id)
  );
}

async function updateCompletedCombinationSave(stage, job, shouldSave, options = {}) {
  if (!ACTIVE_EVALUATION_CASE?.id || !job?.id || job.status !== "succeeded") return null;
  SAVED_COMBINATION_STATUS = {
    busy: `${stage}:${job.id}`,
    message: shouldSave ? "正在加入已完成组合…" : "正在移出已完成组合…",
    tone: "active",
  };
  renderCompletionSaveControls();
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const body = await postJson(
        `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/saved-combinations`,
        {
          action: shouldSave ? "save" : "remove",
          source_stage: stage,
          job_id: job.id,
          ...evaluationCaseMutationEnvelope(),
          criterion_set_id: acceptancePreset || "engineering",
          ...(stage === "inference" && shouldSave ? {
            model_names: job.request?.models || [],
            datasets: job.request?.submitted_datasets || job.request?.datasets || [],
          } : {}),
          actor: "local-ui",
          reason: shouldSave
            ? `用户将已完成${stage === "training" ? "训练" : "推理"}组合加入第 3 页`
            : `用户将已完成${stage === "training" ? "训练" : "推理"}组合移出第 3 页`,
        },
      );
      ACTIVE_EVALUATION_CASE = body.case || body;
      const count = savedCombinationsForJob(stage, job).length;
      SAVED_COMBINATION_STATUS = {
        busy: "",
        message: shouldSave ? `已加入 ${count} 个完成组合` : "已移出完成组合，任务记录仍保留",
        tone: "ok",
      };
      renderCompletionSaveControls();
      if (!options.silent) render();
      return ACTIVE_EVALUATION_CASE;
    } catch (err) {
      if (attempt === 0 && /conflict|版本|409/i.test(String(err.message || ""))) {
        await fetchCurrentEvaluationCase();
        continue;
      }
      SAVED_COMBINATION_STATUS = { busy: "", message: `保存失败：${err.message}`, tone: "fail" };
      renderCompletionSaveControls();
      if (!options.silent) render();
      throw err;
    }
  }
  return ACTIVE_EVALUATION_CASE;
}

function renderCompletionSaveControls() {
  const bind = (stage, node, job, preference) => {
    if (!node) return;
    const completed = job?.status === "succeeded";
    const stored = completed ? savedCombinationsForJob(stage, job).length : 0;
    node.checked = completed ? stored > 0 : preference;
    node.disabled = Boolean(SAVED_COMBINATION_STATUS.busy) || Boolean(job && !completed && ["queued", "running"].includes(job.status));
    const label = node.closest("label")?.querySelector("small");
    if (label) {
      label.textContent = completed
        ? (stored ? `已保存 ${stored} 项` : "完成但未保存")
        : "第 3 页可选";
    }
    node.onchange = async () => {
      if (stage === "training") {
        trainingAutoSave = node.checked;
        localStorage.setItem("sohbench.trainingAutoSave", String(trainingAutoSave));
      } else {
        inferenceAutoSave = node.checked;
        localStorage.setItem("sohbench.inferenceAutoSave", String(inferenceAutoSave));
      }
      if (completed) {
        try {
          await updateCompletedCombinationSave(stage, job, node.checked);
        } catch (_) {
          // Inline status already explains the authoritative backend failure.
        }
      }
    };
  };
  bind("training", $("training-save-toggle"), currentTrainingJob(), trainingAutoSave);
  bind("inference", $("inference-save-toggle"), currentInferenceJob(), inferenceAutoSave);
}

async function activateSavedCombination(savedId) {
  if (!ACTIVE_EVALUATION_CASE?.id || !savedId) return;
  SAVED_COMBINATION_STATUS = { busy: `activate:${savedId}`, message: "正在切换完成组合…", tone: "active" };
  renderAuditWorkspace();
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const body = await postJson(
        `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/saved-combinations/activate`,
        {
          saved_combination_id: savedId,
          ...evaluationCaseMutationEnvelope(),
          actor: "local-ui",
          reason: "用户在第 3 页切换已完成组合",
        },
      );
      ACTIVE_EVALUATION_CASE = body.case || body;
      evidenceLiveContext = {};
      nativeDashboardContext = {};
      invalidateAuditRawCurves();
      await hydratePipelineJobs();
      await hydratePipelineArtifacts();
      await loadReviewWorkspace({ silent: true, force: true });
      const item = activeSavedCombination();
      SAVED_COMBINATION_STATUS = {
        busy: "",
        message: item?.source_stage === "training"
          ? "已切换到训练内评估；完成推理后可查看泛化证据"
          : "已切换到真实推理证据",
        tone: "ok",
      };
      render();
      return;
    } catch (err) {
      if (attempt === 0 && /conflict|版本|409/i.test(String(err.message || ""))) {
        await fetchCurrentEvaluationCase();
        continue;
      }
      SAVED_COMBINATION_STATUS = { busy: "", message: `切换失败：${err.message}`, tone: "fail" };
      render();
      return;
    }
  }
}

function pipelineCandidateId(item) {
  const identity = item?.artifact_id || item?.model_artifact_id || item?.name || item?.estimator || "model";
  return `artifact:${evaluationIdToken(identity)}`;
}

function pipelineCandidatesFromTrainingJob(job) {
  const artifacts = job?.model_artifacts || [];
  const bestEstimator = job?.result_summary?.best_estimator || "";
  const bestFeatureset = job?.result_summary?.best_featureset || "";
  const scope = pipelineModelGroups({
    trainingJob: job || {},
    models: modelOptions({ models: artifacts }),
  });
  return scope.groups.map((group) => {
    const item = group.models.find((model) => model.name === group.recommendedModelName) || group.models[0] || {};
    return {
    candidate_id: pipelineCandidateId(item),
    model: group.estimator || item.estimator || item.name || "unknown",
    featureset: group.featureset || item.featureset || "full",
    role: group.estimator === bestEstimator && (group.featureset || "full") === (bestFeatureset || "full")
      ? "champion"
      : "candidate",
    model_artifact_id: group.recommendedArtifactId || item.artifact_id || item.model_artifact_id || "",
    run_id: job.run_id || job.id || "",
    };
  });
}

function pipelineCandidatesFromInferenceJob(job) {
  return (job?.request?.model_bindings || [])
    .map((item, index) => ({
      candidate_id: pipelineCandidateId(item),
      model: item.estimator || item.name || "unknown",
      featureset: item.featureset || "full",
      role: index === 0 ? "champion" : "candidate",
      model_artifact_id: item.artifact_id || "",
      run_id: item.created_by_run || ACTIVE_EVALUATION_CASE?.execution?.training_job_id || "",
    }));
}

function pipelineExecutionState(job) {
  const status = String(job?.status || "idle");
  if (status === "succeeded") return "completed";
  if (["queued", "running", "failed"].includes(status)) return status;
  return "idle";
}

async function syncPipelineCaseWithJob(job, kind) {
  if (!job || !ACTIVE_EVALUATION_CASE) return;
  if (kind === "training") PIPELINE_JOBS.training = { kind: "training", ...job };
  if (kind === "inference") PIPELINE_JOBS.inference = { kind: "inference", ...job };
  const current = ACTIVE_EVALUATION_CASE.execution || {};
  const execution = {
    ...current,
    state: pipelineExecutionState(job),
    run_id: job.run_id || job.id || current.run_id || "",
    failure_reason: job.status === "failed" ? String(job.error || "任务失败") : "",
  };
  const trainingStartsNewGeneration = Boolean(
    kind === "training"
    && job.id
    && String(job.id) !== String(current.training_job_id || ""),
  );
  if (kind === "training") {
    execution.training_job_id = job.id || "";
    if (trainingStartsNewGeneration) {
      execution.inference_job_id = "";
      PIPELINE_JOBS.inference = null;
    }
  }
  if (kind === "inference") execution.inference_job_id = job.id || "";
  const candidates = job.status === "succeeded"
    ? (kind === "training" ? pipelineCandidatesFromTrainingJob(job) : pipelineCandidatesFromInferenceJob(job))
    : [];
  await updateEvaluationCaseIdentity(
    {
      execution,
      ...(candidates.length || trainingStartsNewGeneration ? { candidates } : {}),
      ...(candidates.length || trainingStartsNewGeneration ? { replace_candidates: true } : {}),
    },
    `${kind} job ${job.id || ""} ${job.status || "updated"}`,
  );
  PIPELINE_STATUS = {
    tone: job.status === "failed" ? "fail" : (job.status === "succeeded" ? "ok" : "active"),
    message: kind === "training"
      ? (job.status === "succeeded" ? "训练产物已绑定到当前案例，可继续推理验证。" : "训练任务已绑定到当前案例。")
      : (job.status === "succeeded" ? "推理结果已绑定到当前案例，可进入评测与证据。" : "推理任务已绑定到当前案例。"),
  };
  renderPipelineNavigation(activeMode);
}

async function bindInferenceEvidenceToCase(job) {
  if (!ACTIVE_EVALUATION_CASE || job?.status !== "succeeded") return;
  const selectedName = job.request?.models?.[0] || "";
  const model = (currentModelCatalog().models || []).find((item) => item.name === selectedName) || {};
  const caseCandidates = ACTIVE_EVALUATION_CASE.candidates || [];
  const candidate = (
    model.artifact_id
      ? caseCandidates.find((item) => item.model_artifact_id === model.artifact_id)
      : null
  );
  const dataset = job.request?.submitted_datasets?.[0] || job.request?.datasets?.[0] || "";
  if (!candidate || !dataset) return;
  nativeDashboardContext = {
    ...nativeDashboardContext,
    dataset,
    source: "inference",
    model: candidate.model,
    featureset: candidate.featureset || "full",
    cell: "",
    cycle: "",
  };
  evidenceLiveContext = {};
  const body = await postJson(
    `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/evidence-selection`,
    {
      ...evaluationCaseMutationEnvelope(),
      actor: "local-ui",
      reason: `inference job ${job.id} linked to evidence scope`,
      criterion_set_id: acceptancePreset,
      selection: {
        candidate_id: candidate.candidate_id,
        dataset,
        asset_id: "",
        sample_id: "",
        cycle_id: null,
        evidence_tab: "overview",
      },
    },
  );
  ACTIVE_EVALUATION_CASE = body.case || body;
  CASE_PERSISTENCE = { status: "saved", message: "推理范围已绑定" };
}

function restoreEvaluationCaseSelection(caseDocument) {
  const selection = caseDocument?.evidence_selection || {};
  const candidate = (caseDocument?.candidates || []).find((item) => item.candidate_id === selection.candidate_id);
  nativeDashboardContext = {
    ...nativeDashboardContext,
    dataset: selection.dataset || nativeDashboardContext.dataset || "",
    source: "evidence",
    model: candidate?.model || nativeDashboardContext.model || "",
    featureset: candidate?.featureset || nativeDashboardContext.featureset || "full",
    cell: selection.asset_id || nativeDashboardContext.cell || "",
    cycle: selection.cycle_id ?? nativeDashboardContext.cycle ?? "",
  };
  evidenceLiveContext = {};
  auditEvidenceTab = selection.evidence_tab === "raw" ? "raw" : "trajectory";
  invalidateAuditRawCurves();
  if (caseDocument?.verdict?.criterion_set_version) {
    acceptancePreset = criterionBaseId(caseDocument.verdict.criterion_set_version);
  }
}

function evidenceViewMatchesPinnedCase() {
  if (!ACTIVE_EVALUATION_CASE) return false;
  const selection = ACTIVE_EVALUATION_CASE.evidence_selection || {};
  const context = currentEvidenceContext();
  const candidate = (ACTIVE_EVALUATION_CASE.candidates || [])
    .find((item) => item.candidate_id === selection.candidate_id);
  const viewCandidate = selectedEvidenceCandidate(auditCandidateRows(), context);
  const candidateIdentityMatches = viewCandidate?.candidate_id
    ? String(viewCandidate.candidate_id) === String(candidate?.candidate_id || "")
    : (viewCandidate?.model_artifact_id
      ? String(viewCandidate.model_artifact_id) === String(candidate?.model_artifact_id || "")
      : (String(candidate?.model || "") === String(context.model || "")
        && String(candidate?.featureset || "full") === String(context.featureset || "full")));
  return String(selection.dataset || "") === String(context.dataset || "") &&
    String(selection.asset_id || "") === String(context.cell || "") &&
    String(selection.cycle_id ?? "") === String(context.cycle ?? "") &&
    candidateIdentityMatches &&
    criterionBaseId(ACTIVE_EVALUATION_CASE.verdict?.criterion_set_version) ===
      criterionBaseId(acceptancePreset);
}

function alignEvidenceViewToPinnedCase(caseDocument = ACTIVE_EVALUATION_CASE) {
  if (!caseDocument) return;
  const selection = caseDocument.evidence_selection || {};
  const candidate = (caseDocument.candidates || [])
    .find((item) => item.candidate_id === selection.candidate_id);
  nativeDashboardContext = {
    ...nativeDashboardContext,
    dataset: selection.dataset || nativeDashboardContext.dataset || "",
    source: "evidence",
    model: candidate?.model || nativeDashboardContext.model || "",
    featureset: candidate?.featureset || nativeDashboardContext.featureset || "full",
    cell: selection.asset_id || "",
    cycle: selection.cycle_id ?? "",
  };
  evidenceLiveContext = {
    ...evidenceLiveContext,
    dataset: selection.dataset || evidenceLiveContext.dataset || "",
    model: candidate?.model || evidenceLiveContext.model || "",
    featureset: candidate?.featureset || evidenceLiveContext.featureset || "full",
    cell: selection.asset_id || "",
    cycle: selection.cycle_id ?? "",
    recordId: selection.sample_id || "",
    rawAvailable: selection.curve_ref?.availability === "available",
  };
}

function markEvidenceViewState() {
  if (!ACTIVE_EVALUATION_CASE || ["local", "error", "resolving"].includes(CASE_PERSISTENCE.status)) return;
  CASE_PERSISTENCE = evidenceViewMatchesPinnedCase()
    ? { status: "saved", message: "当前视图已固定" }
    : { status: "dirty", message: "正在固定当前视图" };
  if (CASE_PERSISTENCE.status === "dirty") queueEvaluationCasePersistence();
}

async function resolveEvaluationCaseContext() {
  if (!DATA) return;
  CASE_PERSISTENCE = { status: "resolving", message: "正在恢复案例" };
  try {
    let body;
    if (requestedCaseId) {
      const response = await fetch(`${API_BASE}/api/v2/cases/${encodeURIComponent(requestedCaseId)}`);
      if (!response.ok) throw new Error(`案例 ${requestedCaseId} 不可用（HTTP ${response.status}）`);
      const requestedCase = await response.json();
      if (requestedCase.profile_name !== (activeProfileOption?.name || activeProfile)) {
        throw new Error(`案例 ${requestedCaseId} 不属于当前数据源`);
      }
      body = { case: requestedCase, created: false };
    } else {
      body = await postJson("/api/v2/cases/resolve", evaluationCaseIdentityPayload());
    }
    ACTIVE_EVALUATION_CASE = body.case || null;
    if (ACTIVE_EVALUATION_CASE) {
      requestedCaseId = ACTIVE_EVALUATION_CASE.id || "";
      restoreEvaluationCaseSelection(ACTIVE_EVALUATION_CASE);
      updateModeUrl(activeMode);
    }
    const viewMatches = evidenceViewMatchesPinnedCase();
    CASE_PERSISTENCE = !viewMatches
      ? { status: "dirty", message: "当前视图等待服务端固定" }
      : { status: "saved", message: body.created ? "案例已建立" : "案例已恢复" };
    if (!viewMatches) queueEvaluationCasePersistence();
  } catch (err) {
    ACTIVE_EVALUATION_CASE = null;
    CASE_PERSISTENCE = { status: "local", message: `本地模式：${err.message}` };
  }
}

function currentEvidencePersistencePayload() {
  if (!ACTIVE_EVALUATION_CASE || !DATA) return null;
  const context = currentEvidenceContext();
  const viewCandidate = selectedEvidenceCandidate(auditCandidateRows(), context);
  const caseCandidates = ACTIVE_EVALUATION_CASE.candidates || [];
  const candidate = caseCandidates.find((item) =>
    viewCandidate?.candidate_id && item.candidate_id === viewCandidate.candidate_id
  ) || caseCandidates.find((item) =>
    viewCandidate?.model_artifact_id && item.model_artifact_id === viewCandidate.model_artifact_id
  ) || (!viewCandidate?.candidate_id && !viewCandidate?.model_artifact_id
    ? caseCandidates.find((item) =>
      item.model === context.model
      && (item.featureset || "full") === (context.featureset || "full")
    )
    : null);
  if (!candidate) return null;
  const payload = {
    ...evaluationCaseMutationEnvelope(),
    actor: "local-ui",
    reason: "user pinned evidence context",
    criterion_set_id: acceptancePreset,
    selection: {
      candidate_id: candidate.candidate_id,
      dataset: context.dataset || "",
      asset_id: context.cell || "",
      sample_id: context.recordId || "",
      cycle_id: context.cycle === "" ? null : context.cycle,
      evidence_tab: auditEvidenceTab === "raw" ? "raw" : "overview",
    },
  };
  const verdict = ACTIVE_EVALUATION_CASE.verdict || {};
  if (
    String(verdict.criterion_set_version || "").includes("@custom-")
    && criterionBaseId(verdict.criterion_set_version) === criterionBaseId(acceptancePreset)
  ) {
    payload.criterion_thresholds = { ...(verdict.thresholds || {}) };
  }
  return payload;
}

function evidenceSelectionFingerprint(selection = {}) {
  return JSON.stringify([
    selection.candidate_id || "",
    selection.dataset || "",
    selection.asset_id || "",
    selection.sample_id || "",
    selection.cycle_id ?? "",
    selection.evidence_tab || "",
  ]);
}

function queueEvaluationCasePersistence() {
  if (!ACTIVE_EVALUATION_CASE) return;
  const signedPack = (REVIEW_WORKSPACE?.evidence_packs || [])
    .find((item) => item.state === "signed");
  if (signedPack) {
    CASE_PERSISTENCE = { status: "locked", message: "已签署案例不可原位改写；请创建新案例版本" };
    renderAuditWorkspace();
    return;
  }
  if (casePersistenceInFlight) {
    casePersistenceQueued = true;
    return;
  }
  if (casePersistenceTimer) clearTimeout(casePersistenceTimer);
  casePersistenceTimer = setTimeout(() => {
    casePersistenceTimer = null;
    persistEvaluationCaseEvidence();
  }, 300);
}

async function persistEvaluationCaseEvidence() {
  const payload = currentEvidencePersistencePayload();
  if (!payload || casePersistenceInFlight) return;
  const requestSelectionFingerprint = evidenceSelectionFingerprint(payload.selection);
  casePersistenceInFlight = true;
  CASE_PERSISTENCE = { status: "saving", message: "正在保存案例" };
  renderAuditWorkspace();
  try {
    const response = await fetch(
      `${API_BASE}/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/evidence-selection`,
      {
        method: "POST",
        headers: apiJsonHeaders(),
        body: JSON.stringify(payload),
      },
    );
    const body = await response.json().catch(() => ({}));
    if (response.status === 409) {
      const current = await fetch(
        `${API_BASE}/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}`,
      ).then((item) => item.ok ? item.json() : null).catch(() => null);
      if (current) {
        ACTIVE_EVALUATION_CASE = current;
        alignEvidenceViewToPinnedCase(current);
      }
      throw new Error("案例已在其他窗口更新，请确认当前内容后再次固定证据");
    }
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    const pendingPayload = currentEvidencePersistencePayload();
    const pendingViewChanged = Boolean(
      pendingPayload
      && evidenceSelectionFingerprint(pendingPayload.selection) !== requestSelectionFingerprint
    );
    ACTIVE_EVALUATION_CASE = body.case || body;
    if (pendingViewChanged) {
      casePersistenceQueued = true;
      CASE_PERSISTENCE = { status: "dirty", message: "正在固定最后一次选择" };
    } else {
      alignEvidenceViewToPinnedCase();
      CASE_PERSISTENCE = { status: "saved", message: "案例已保存" };
    }
  } catch (err) {
    CASE_PERSISTENCE = { status: "error", message: `保存失败：${err.message}` };
  } finally {
    casePersistenceInFlight = false;
    renderAuditWorkspace();
    renderGlobalCaseContext();
    if (casePersistenceQueued) {
      casePersistenceQueued = false;
      queueEvaluationCasePersistence();
    }
  }
}

function renderCaseIdentity() {
  const node = $("case-context-id");
  if (node) node.textContent = currentCaseId();
  const auditNode = $("audit-breadcrumb-case");
  if (auditNode) auditNode.textContent = currentCaseId();
}

function evidenceCandidateRows(rows = scoped?.entries || DATA?.leaderboard || []) {
  const seen = new Set();
  return (rows || [])
    .filter((row) => metricMean(row, "mae") != null)
    .sort((a, b) => metricMean(a, "mae") - metricMean(b, "mae"))
    .filter((row) => {
      const key = `${row.estimator || ""}|${row.featureset || "full"}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(0, 6);
}

function auditCandidateRows() {
  const inferenceJob = currentInferenceJob();
  const inferenceJobId = String(inferenceJob?.id || ACTIVE_EVALUATION_CASE?.execution?.inference_job_id || "");
  const activeDataset = String(
    currentEvidenceContext().dataset
    || ACTIVE_EVALUATION_CASE?.evidence_selection?.dataset
    || "",
  );
  const savedRows = savedCombinations().filter((item) =>
    item.source_stage === "inference"
    && (!inferenceJobId || String(item.inference_job_id || "") === inferenceJobId)
    && (!activeDataset || String(item.target_dataset || "") === activeDataset)
    && item.candidate?.model_artifact_id
  );
  if (savedRows.length) {
    const metric = (value) => Number.isFinite(Number(value)) ? { mean: Number(value), std: 0 } : null;
    const seen = new Set();
    return savedRows
      .filter((item) => {
        const key = String(item.candidate?.candidate_id || item.candidate?.model_artifact_id || "");
        if (!key || seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .map((item) => ({
        candidate_id: String(item.candidate?.candidate_id || ""),
        estimator: item.candidate?.model || "unknown",
        featureset: item.candidate?.featureset || "full",
        protocol: item.protocol || "",
        model_name: item.candidate?.model || "",
        model_artifact_id: item.candidate?.model_artifact_id || "",
        source: "saved_inference",
        target_dataset: item.target_dataset || "",
        metrics: {
          mae: metric(item.metrics?.mae),
          rmse: metric(item.metrics?.rmse),
          mean_bias_error: metric(item.metrics?.mean_bias_error),
          picp: metric(item.metrics?.picp),
          monotonicity_violation_rate: metric(item.metrics?.monotonicity ?? item.metrics?.monotonicity_violation_rate),
          value_range_violation: metric(item.metrics?.value_range ?? item.metrics?.value_range_violation),
        },
        verdicts: {},
      }))
      .sort((left, right) => metricMean(left, "mae") - metricMean(right, "mae"));
  }
  const requestedModels = inferenceJob?.request?.models || [];
  const requestedDatasets = inferenceJob?.request?.submitted_datasets
    || inferenceJob?.request?.datasets
    || [];
  const records = inferenceRows(DATA?.inference || {}, requestedModels, requestedDatasets)
    .filter((row) => row.status === "succeeded" && Number.isFinite(Number(row.mae)));
  if (!records.length) return evidenceCandidateRows(scoped?.entries || DATA?.leaderboard || []);
  const models = currentModelCatalog().models || [];
  return records.map((row) => {
    const model = models.find((item) => item.name === row.model) || {};
    const metric = (value) => Number.isFinite(Number(value)) ? { mean: Number(value), std: 0 } : null;
    return {
      estimator: row.estimator || model.estimator || row.model || "unknown",
      featureset: row.featureset || model.featureset || "full",
      protocol: model.protocol || currentTrainingJob()?.request?.protocols?.[0] || "",
      model_name: row.model || model.name || "",
      model_artifact_id: model.artifact_id || "",
      source: "inference",
       metrics: {
         mae: metric(row.mae),
         rmse: metric(row.rmse),
         mean_bias_error: metric(row.mean_bias_error),
         picp: metric(row.picp),
         monotonicity_violation_rate: metric(row.monotonicity_violation_rate),
         value_range_violation: metric(row.value_range_violation),
       },
      verdicts: {},
    };
  }).sort((left, right) => metricMean(left, "mae") - metricMean(right, "mae"));
}

function prepareEvidenceContext(rows = scoped?.entries || DATA?.leaderboard || []) {
  const candidates = evidenceCandidateRows(rows);
  const requested = nativeDashboardContext || {};
  const binding = ACTIVE_EVALUATION_CASE?.context_binding || {};
  const requestedModel = requested.model || binding.model || "";
  const requestedFeatureset = requested.featureset || binding.featureset || "full";
  const selected = candidates.find((row) =>
    row.estimator === requestedModel
    && (row.featureset || "full") === requestedFeatureset
  );
  const datasets = selectedDatasetNames();
  nativeDashboardContext = {
    dataset: requested.dataset || datasets[0] || DATA?.config?.datasets?.[0] || "",
    source: requested.source || "evidence",
    model: requestedModel || selected?.estimator || "",
    featureset: requestedFeatureset || selected?.featureset || "full",
    cell: requested.cell || "",
    cycle: requested.cycle ?? "",
  };
}

function currentEvidenceContext() {
  const requested = nativeDashboardContext || {};
  const live = evidenceLiveContext || {};
  return {
    dataset: live.dataset || requested.dataset || "",
    requestedDataset: requested.dataset || "",
    scenario: live.scenario || "",
    source: live.source || requested.source || "evidence",
    model: requested.model || live.model || "",
    featureset: requested.featureset || live.featureset || "full",
    cell: live.cell || requested.cell || "",
    cycle: live.cycle ?? requested.cycle ?? "",
    recordId: live.recordId || "",
    trueSoh: live.trueSoh ?? null,
    predSoh: live.predSoh ?? null,
    lower: live.lower ?? null,
    upper: live.upper ?? null,
    rawAvailable: live.rawAvailable === true,
    featureWindow: Array.isArray(live.featureWindow) ? live.featureWindow : null,
    reason: live.reason || "",
  };
}

function selectedEvidenceCandidate(candidates, context) {
  return candidates.find((row) =>
    row.estimator === context.model
    && (row.featureset || "full") === (context.featureset || "full")
  ) || null;
}

function evidenceAcceptance(candidate) {
  const detail = acceptanceDetailSummary({
    rows: scoped?.entries || DATA?.leaderboard || [],
    datasetRows: DATA?.datasetLeaderboard || [],
    preset: acceptancePreset,
  });
  const row = candidate ? detail.rows.find((item) =>
    item.estimator === candidate.estimator && item.featureset === (candidate.featureset || "full")) : null;
  const known = Object.values(row?.verdicts || {}).filter((value) => value != null);
  if (!row || !known.length) return { detail, row, label: "待判定", tone: "pending" };
  if (known.every(Boolean)) return { detail, row, label: "通过", tone: "pass" };
  return { detail, row, label: "不通过", tone: "fail" };
}

function evidenceNumber(value, digits = 4) {
  if (value == null || value === "") return "—";
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function openSohDashboardDrilldown(context = {}) {
  nativeDashboardContext = {
    dataset: context.dataset || "",
    source: context.source || "",
    model: context.model || context.estimator || "",
    featureset: context.featureset || "",
    cell: context.cell || "",
    cycle: context.cycle ?? "",
  };
  evidenceLiveContext = {};
  auditEvidenceTab = context.cycle !== "" && context.cycle != null ? "raw" : "trajectory";
  invalidateAuditRawCurves();
  activateMode("audit");
}

function handleSohDashboardDrilldownClick(event) {
  const button = event.target.closest("[data-soh-dashboard-open]");
  if (!button) return;
  openSohDashboardDrilldown(button.dataset);
}

async function loadReviewWorkspace({ force = false, silent = false } = {}) {
  const caseId = ACTIVE_EVALUATION_CASE?.id;
  if (!caseId) {
    REVIEW_WORKSPACE = null;
    REVIEW_WORKFLOW_STATE = { status: "local", message: "服务端案例不可用", busy: "" };
    return;
  }
  if (!force && REVIEW_WORKSPACE?.case_id === caseId &&
      REVIEW_WORKSPACE?.case_version === ACTIVE_EVALUATION_CASE.version) return;
  if (!silent) REVIEW_WORKFLOW_STATE = { status: "loading", message: "正在读取闭环状态", busy: "load" };
  try {
    const bindingId = evaluationContextBindingId();
    const response = await fetch(
      `${API_BASE}/api/v2/cases/${encodeURIComponent(caseId)}/review-workspace?context_binding_id=${encodeURIComponent(bindingId)}`,
    );
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    REVIEW_WORKSPACE = body;
    REVIEW_WORKFLOW_STATE = { status: "ready", message: "闭环状态已同步", busy: "" };
  } catch (err) {
    REVIEW_WORKSPACE = null;
    REVIEW_WORKFLOW_STATE = { status: "error", message: `闭环状态不可用：${err.message}`, busy: "" };
  }
}

async function refreshActiveEvaluationCase() {
  if (!ACTIVE_EVALUATION_CASE?.id) return;
  const response = await fetch(
    `${API_BASE}/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}`,
  );
  if (!response.ok) return;
  ACTIVE_EVALUATION_CASE = await response.json();
}

async function runReviewMutation(action, path, payload, { updatesCase = false } = {}) {
  REVIEW_WORKFLOW_STATE = { status: "loading", message: "正在提交变更", busy: action };
  renderAuditWorkspace();
  try {
    const body = await postJson(path, payload);
    if (updatesCase) ACTIVE_EVALUATION_CASE = body.case || body;
    else await refreshActiveEvaluationCase();
    await loadReviewWorkspace({ force: true, silent: true });
    const retestId = action === "request-retest" ? body?.id : "";
    REVIEW_WORKFLOW_STATE = {
      status: "ready",
      message: retestId ? "受控复测已排队，结果将自动回填" : "变更已写入不可变审计轨迹",
      busy: "",
    };
    if (retestId) scheduleAuditRetestPolling(retestId);
  } catch (err) {
    await refreshActiveEvaluationCase();
    await loadReviewWorkspace({ force: true, silent: true });
    REVIEW_WORKFLOW_STATE = { status: "error", message: `操作未完成：${err.message}`, busy: "" };
  }
  renderAuditWorkspace();
  renderGlobalCaseContext();
}

function scheduleAuditRetestPolling(retestId, attempt = 0) {
  if (auditRetestPollTimer) clearTimeout(auditRetestPollTimer);
  const terminal = (REVIEW_WORKSPACE?.retests || []).find((item) => item.id === retestId);
  if (terminal && ["completed", "failed"].includes(terminal.status)) {
    REVIEW_WORKFLOW_STATE = {
      status: terminal.status === "completed" ? "ready" : "error",
      message: terminal.status === "completed" ? "受控复测已完成并自动回填" : `受控复测失败：${terminal.result?.error || "请查看审计轨迹"}`,
      busy: "",
    };
    renderAuditWorkspace();
    renderGlobalCaseContext();
    return;
  }
  if (attempt >= 120 || activeMode !== "audit") return;
  auditRetestPollTimer = setTimeout(async () => {
    await loadReviewWorkspace({ force: true, silent: true });
    renderAuditWorkspace();
    renderGlobalCaseContext();
    scheduleAuditRetestPolling(retestId, attempt + 1);
  }, 500);
}

function auditPrimaryAction(caseDocument, reviewWorkspace, readiness) {
  return auditNextAction({ caseDocument, reviewWorkspace, readiness });
}

function savedCombinationValidationLabel(item) {
  return ({
    training_internal: "训练内评估",
    replay: "同源回放",
    cross_dataset: "跨数据集验证",
  })[item?.validation_kind] || "已完成";
}

function savedCombinationGroups(items = savedCombinations()) {
  const groups = new Map();
  items.forEach((item) => {
    const candidate = item.candidate || {};
    const key = candidate.model_artifact_id || `${candidate.model}|${candidate.featureset}`;
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        model: candidate.model || "未知算法",
        featureset: candidate.featureset || "full",
        protocol: item.protocol || "",
        items: [],
      });
    }
    groups.get(key).items.push(item);
  });
  return [...groups.values()]
    .map((group) => ({
      ...group,
      items: group.items.sort((left, right) => {
        const order = { training_internal: 0, replay: 1, cross_dataset: 2 };
        return (order[left.validation_kind] ?? 9) - (order[right.validation_kind] ?? 9)
          || String(left.target_dataset || "").localeCompare(String(right.target_dataset || ""));
      }),
    }))
    .sort((left, right) => left.model.localeCompare(right.model)
      || left.featureset.localeCompare(right.featureset));
}

function latestSavedTrainingJobId(items = savedCombinations()) {
  return [...items]
    .filter((item) => item.training_job_id)
    .sort((left, right) => {
      const jobOrder = String(right.training_job_id).localeCompare(String(left.training_job_id));
      if (jobOrder) return jobOrder;
      const leftTime = Date.parse(left.saved_at || "") || 0;
      const rightTime = Date.parse(right.saved_at || "") || 0;
      return rightTime - leftTime;
    })[0]?.training_job_id || "";
}

function renderSavedModelGroups(groups, activeId) {
  return groups.map((group) =>
    `<section class="saved-model-group"><header><span>${esc(group.model)}</span><b title="${esc(group.key)}">${esc(group.featureset)}</b><small>${esc(group.protocol || "未记录协议")}</small></header>` +
    `<div class="saved-result-options">${group.items.map((item) => {
      const active = item.saved_combination_id === activeId;
      const mae = Number(item.metrics?.mae);
      const dataset = item.validation_kind === "training_internal"
        ? (item.training_datasets || []).join(" + ") || "训练范围"
        : item.target_dataset || "目标数据";
      return `<button type="button" class="saved-result-option ${active ? "active" : ""} ${esc(item.validation_kind)}" ` +
        `data-saved-combination="${esc(item.saved_combination_id)}" aria-pressed="${active}" ` +
        `${SAVED_COMBINATION_STATUS.busy ? "disabled" : ""}>` +
        `<span>${esc(savedCombinationValidationLabel(item))}</span><b>${esc(dataset)}</b>` +
        `<small>${Number.isFinite(mae) ? `MAE ${esc(fmt(mae, 4))}` : "已完成"} · ${esc(auditVerdictLabel(item.verdict_result))}</small></button>`;
    }).join("")}</div></section>`
  ).join("");
}

function renderSavedCombinationRegistry(node = $("audit-saved-combinations")) {
  if (!node) return;
  const allItems = savedCombinations();
  const latestTrainingJobId = latestSavedTrainingJobId(allItems);
  const currentItems = latestTrainingJobId
    ? allItems.filter((item) => String(item.training_job_id || "") === String(latestTrainingJobId))
    : allItems;
  const historyItems = allItems.filter((item) => !currentItems.includes(item));
  const groups = savedCombinationGroups(currentItems);
  const historyGroups = savedCombinationGroups(historyItems);
  const activeId = String(ACTIVE_EVALUATION_CASE?.active_saved_combination_id || "");
  if (!ACTIVE_EVALUATION_CASE) {
    node.innerHTML = "";
    node.hidden = true;
    return;
  }
  node.hidden = false;
  if (!groups.length) {
    node.innerHTML = `<div class="saved-registry-head"><div><span>已完成组合</span><b>尚未收藏结果</b></div>` +
      `<small>训练或推理完成后勾选“加入已完成组合”，结果才会进入第 3 页。</small></div>`;
    return;
  }
  const itemCount = groups.reduce((sum, group) => sum + group.items.length, 0);
  const historyItemCount = historyGroups.reduce((sum, group) => sum + group.items.length, 0);
  node.innerHTML =
    `<div class="saved-registry-head"><div><span>本轮结果</span><b>${groups.length} 个模型身份 · ${itemCount} 项结果</b></div>` +
      `<small>${esc(SAVED_COMBINATION_STATUS.message || "默认只展示最近一次训练代次；历史结果仍可展开选择。")}</small></div>` +
    `<div class="saved-registry-body"><div class="saved-registry-scroll">${renderSavedModelGroups(groups, activeId)}</div>` +
      (historyGroups.length
        ? `<details class="saved-registry-history"><summary>历史组合 ${historyGroups.length} 个模型身份 · ${historyItemCount} 项结果</summary>` +
          `<div class="saved-registry-history-scroll">${renderSavedModelGroups(historyGroups, activeId)}</div></details>`
        : "") +
    `</div>`;
  node.querySelectorAll("[data-saved-combination]").forEach((button) => {
    button.onclick = () => activateSavedCombination(button.dataset.savedCombination);
  });
}

function enterInferenceForSavedCombination(item = activeSavedCombination()) {
  const artifactId = String(item?.candidate?.model_artifact_id || "");
  const model = (currentModelCatalog().models || []).find((entry) =>
    String(entry.artifact_id || "") === artifactId
  );
  if (model?.name) inferenceSelection.models = new Set([model.name]);
  activeMode = "inference";
  render();
}

function renderTrainingSavedAuditState(item, nodes) {
  const candidate = item.candidate || {};
  const metrics = item.metrics || {};
  const datasets = item.training_datasets || [];
  const primaryReview = $("audit-review-primary");
  if (primaryReview) {
    primaryReview.textContent = "进入推理验证";
    primaryReview.disabled = false;
    primaryReview.title = "沿用当前模型产物选择一个或多个目标数据集";
    primaryReview.dataset.auditAction = "go-inference";
    const primaryHint = $("audit-primary-hint");
    if (primaryHint) {
      primaryHint.textContent = "训练已完成，下一步验证泛化";
      primaryHint.hidden = false;
    }
  }
  nodes.stageRail.innerHTML = [
    ["✓", "训练完成", "模型与训练内指标已固定", "done"],
    ["2", "推理验证", "选择更多目标数据集验证泛化", "active"],
    ["3", "证据评审", "推理完成后进入", ""],
  ].map(([number, label, detail, state]) =>
    `<div class="fusion-stage ${state}"><span>${number}</span><div><b>${label}</b><small>${detail}</small></div></div>`
  ).join("");
  nodes.contextRibbon.innerHTML = [
    ["结果范围", "训练内评估", "尚未形成目标数据集泛化结论", "combination incomplete"],
    ["训练数据", datasets.join(" + ") || "—", item.dataset_snapshot_id || datasets.join(", "), "dataset"],
    ["算法", candidate.model || "—", candidate.model_artifact_id || "", "algorithm"],
    ["特征", candidate.featureset || "full", candidate.model_artifact_id || "", "feature"],
    ["协议 / 判据", `${item.protocol || "—"} · ${item.criterion_set_version || "—"}`, item.run_id || "", "protocol"],
  ].map(([label, value, title, className]) =>
    `<div class="fusion-context-cell ${esc(className)}"><span>${esc(label)}</span><b title="${esc(title)}">${esc(value)}</b></div>`
  ).join("");
  nodes.scopeNode.innerHTML =
    `<section class="fusion-scope-block"><div class="fusion-block-label"><span>已选训练组合</span><small>不可变模型身份</small></div>` +
      `<div class="training-saved-scope"><b>${esc(candidate.model)} × ${esc(candidate.featureset || "full")}</b>` +
      `<small title="${esc(candidate.model_artifact_id)}">${esc(shortAuditRef(candidate.model_artifact_id))}</small></div></section>` +
    `<section class="fusion-scope-block"><div class="fusion-block-label"><span>下一步</span><small>泛化验证</small></div>` +
      `<p class="training-saved-guidance">沿用该模型产物，在一个或多个新数据集上运行推理；第 3 页会追加同源回放或跨数据集结果。</p></section>`;
  nodes.kpiNode.innerHTML = [
    ["当前候选", candidate.model || "—", candidate.featureset || "full", "primary"],
    ["训练内 MAE", evidenceNumber(metrics.mae, 4), "交叉验证聚合", ""],
    ["训练内 PICP@90", evidenceNumber(metrics.picp, 3), "仅代表训练协议范围", ""],
    ["泛化状态", "待验证", "尚未对目标数据集执行推理", "warn"],
  ].map(([label, value, note, tone]) =>
    `<div class="fusion-kpi ${tone}"><span>${label}</span><b>${esc(value)}</b><small>${esc(note)}</small></div>`
  ).join("");
  nodes.evidenceHead.innerHTML = `<div><b>训练内评估摘要</b><span>不绘制伪造的逐 Cycle 轨迹；真实曲线需来自完成的推理记录。</span></div>`;
  nodes.trajectoryNode.hidden = false;
  nodes.rawEvidenceNode.hidden = true;
  nodes.trajectoryNode.innerHTML =
    `<div class="training-evidence-placeholder"><span>训练内评估</span><b>${esc(candidate.model)} × ${esc(candidate.featureset || "full")}</b>` +
      `<p>模型产物和聚合指标已完成，但还没有目标数据集上的 PredictionRecord、SOH 轨迹和原始 V / I / T / IC / DV 证据。</p>` +
      `<button type="button" class="primary-action" data-mode-shortcut="inference">进入推理验证</button></div>`;
  nodes.readinessNode.innerHTML =
    `<div class="fusion-verdict-hero warn"><div><span>当前范围</span><b>训练完成 · 泛化待验证</b></div>` +
      `<p>训练结果可以保留和比较，但不能替代跨数据集推理证据，也不能直接签署 EvidencePack。</p></div>` +
    `<section class="fusion-inspector-section"><div class="fusion-block-label"><span>已固定内容</span><small>训练任务 ${esc(item.training_job_id)}</small></div>` +
      `<div class="training-fixed-list"><span>✓ 模型产物</span><span>✓ 数据快照</span><span>✓ 特征方案</span><span>✓ 训练协议</span><span>○ 目标数据推理</span><span>○ 逐点证据</span></div></section>`;
  nodes.closureNode.innerHTML = `<div class="training-next-step"><b>建议一次选择多个目标数据集</b><span>同一模型只加载一次，批量得到同源回放与跨数据集泛化结果。</span></div>`;
  nodes.trailNode.innerHTML = renderAuditTrail();
  wireAuditActions();
  document.querySelectorAll("[data-mode-shortcut]").forEach((button) => {
    button.onclick = () => {
      if (button.dataset.modeShortcut === "inference") enterInferenceForSavedCombination(item);
      else {
        activeMode = button.dataset.modeShortcut;
        render();
      }
    };
  });
}

function renderAuditWorkspace() {
  const summary = $("audit-case-summary");
  const savedRegistryNode = $("audit-saved-combinations");
  const stageRail = $("audit-stage-rail");
  const contextRibbon = $("audit-context-ribbon");
  const scopeNode = $("audit-scope-body");
  const kpiNode = $("audit-kpi-strip");
  const evidenceHead = $("audit-evidence-head");
  const trajectoryNode = $("audit-soh-trajectory");
  const rawEvidenceNode = $("audit-raw-evidence");
  const readinessNode = $("audit-readiness-body");
  const closureNode = $("audit-closure-body");
  const trailNode = $("audit-trail-body");
  if (!summary || !savedRegistryNode || !stageRail || !contextRibbon || !scopeNode || !kpiNode || !evidenceHead || !trajectoryNode || !rawEvidenceNode || !readinessNode || !closureNode || !trailNode) return;

  renderSavedCombinationRegistry(savedRegistryNode);

  const review = ACTIVE_EVALUATION_CASE?.review || { state: "not_started" };
  const readiness = REVIEW_WORKSPACE?.signing_readiness || { gates: {}, blockers: [], can_sign: false };
  const stateTone = REVIEW_WORKFLOW_STATE.status === "error" ? "fail" :
    (REVIEW_WORKFLOW_STATE.status === "loading" ? "pending" : (readiness.can_sign ? "pass" : "warn"));
  const latestSignedPack = (REVIEW_WORKSPACE?.evidence_packs || [])
    .find((item) => item.state === "signed" &&
      Number(item.case_version) === Number(ACTIVE_EVALUATION_CASE?.version));
  const archiveIncompleteCase = $("archive-incomplete-case");
  if (archiveIncompleteCase) {
    const canArchive = Boolean(
      ACTIVE_EVALUATION_CASE
      && ["draft", "failed"].includes(ACTIVE_EVALUATION_CASE.lifecycle_state)
      && !latestSignedPack
    );
    archiveIncompleteCase.hidden = !canArchive;
    archiveIncompleteCase.disabled = !canArchive || REVIEW_WORKFLOW_STATE.busy === "archive-case";
  }
  const finalizeFailedCase = $("finalize-failed-case");
  if (finalizeFailedCase) {
    const canFinalizeFailure = Boolean(
      ACTIVE_EVALUATION_CASE
      && ACTIVE_EVALUATION_CASE.lifecycle_state === "active"
      && ACTIVE_EVALUATION_CASE.verdict?.result === "fail"
      && ["in_review", "rejected"].includes(ACTIVE_EVALUATION_CASE.review?.state)
    );
    finalizeFailedCase.hidden = !canFinalizeFailure;
    finalizeFailedCase.disabled = !canFinalizeFailure || REVIEW_WORKFLOW_STATE.busy === "finalize-failed";
  }
  const isGoldenProfile = DATA?.profileManifest?.kind === "phase22_golden";
  summary.innerHTML = ACTIVE_EVALUATION_CASE
    ? `<span class="audit-chip"><b>${esc(ACTIVE_EVALUATION_CASE.id)}</b></span>` +
      `<span class="audit-chip">v${esc(ACTIVE_EVALUATION_CASE.version)}</span>` +
      (isGoldenProfile ? `<span class="audit-chip golden">真实黄金案例</span>` : "") +
      `<span class="audit-chip ${stateTone}">${esc(latestSignedPack ? "已签署" : (readiness.can_sign ? "可签署" : "受控阻断"))}</span>`
    : `<span class="audit-chip fail">本地预览</span>`;

  const primaryReview = $("audit-review-primary");
  if (primaryReview) {
    const primary = auditPrimaryAction(ACTIVE_EVALUATION_CASE, REVIEW_WORKSPACE, readiness);
    primaryReview.textContent = primary.label;
    primaryReview.disabled = primary.disabled;
    primaryReview.title = primary.hint || primary.label;
    if (primary.action) primaryReview.dataset.auditAction = primary.action;
    else delete primaryReview.dataset.auditAction;
    if (primary.target) primaryReview.dataset.reviewTarget = primary.target;
    else delete primaryReview.dataset.reviewTarget;
    if (primary.packId) primaryReview.dataset.packId = primary.packId;
    else delete primaryReview.dataset.packId;
    if (primary.issueId) primaryReview.dataset.issueId = primary.issueId;
    else delete primaryReview.dataset.issueId;
    if (primary.setId) primaryReview.dataset.setId = primary.setId;
    else delete primaryReview.dataset.setId;
    if (Number.isInteger(primary.neighborRadius)) {
      primaryReview.dataset.neighborRadius = String(primary.neighborRadius);
    } else delete primaryReview.dataset.neighborRadius;
    if (primary.auto) primaryReview.dataset.auditAuto = "true";
    else delete primaryReview.dataset.auditAuto;
    const primaryHint = $("audit-primary-hint");
    if (primaryHint) {
      primaryHint.textContent = primary.hint || "";
      primaryHint.hidden = !primary.hint;
    }
  }

  if (!ACTIVE_EVALUATION_CASE) {
    const unavailable = `<div class="audit-empty"><b>尚未连接服务端案例</b><span>${esc(REVIEW_WORKFLOW_STATE.message || "请启动 API 后刷新页面。")}</span></div>`;
    readinessNode.innerHTML = unavailable;
    closureNode.innerHTML = unavailable;
    trailNode.innerHTML = unavailable;
    scopeNode.innerHTML = unavailable;
    kpiNode.innerHTML = unavailable;
    evidenceHead.innerHTML = unavailable;
    trajectoryNode.innerHTML = unavailable;
    rawEvidenceNode.innerHTML = unavailable;
    return;
  }

  const caseDocument = ACTIVE_EVALUATION_CASE;
  const activeSaved = activeSavedCombination();
  if (activeSaved?.source_stage === "training") {
    renderTrainingSavedAuditState(activeSaved, {
      stageRail,
      contextRibbon,
      scopeNode,
      kpiNode,
      evidenceHead,
      trajectoryNode,
      rawEvidenceNode,
      readinessNode,
      closureNode,
      trailNode,
    });
    return;
  }
  const projectedCase = currentCaseProjection();
  const selection = caseDocument.evidence_selection || {};
  const coverage = caseDocument.evidence_coverage || { available_count: 0, required_count: 0, items: [] };
  const verdict = caseDocument.verdict || { result: "pending", observed: {}, thresholds: {} };
  const context = currentEvidenceContext();
  const candidates = auditCandidateRows();
  const candidate = selectedEvidenceCandidate(candidates, context);
  const acceptance = evidenceAcceptance(candidate);
  const viewMatchesPinnedCase = evidenceViewMatchesPinnedCase();
  const viewVerdict = auditCandidateViewProjection({
    candidate,
    caseVerdict: verdict,
    pinned: viewMatchesPinnedCase,
  });
  const issues = REVIEW_WORKSPACE?.issues || [];
  const retests = REVIEW_WORKSPACE?.retests || [];
  const activeRetest = retests.find((item) => ["requested", "running"].includes(item.status));
  const completedRetest = retests.find((item) => item.status === "completed");
  const reviewDone = review.state === "approved" && issues.every((item) => ["resolved", "closed"].includes(item.status));
  const preparationDone = Boolean(
    selection.split_assignment_id &&
    selection.dataset_snapshot_id &&
    caseDocument.execution?.state === "completed"
  );
  const evidenceDone = Boolean(selection.candidate_id && coverage.state === "complete");
  const stages = [
    ["1", "准备", "协议 · 数据 · 运行", preparationDone],
    ["2", "证据", "比较 · 下钻 · 固定", evidenceDone],
    ["3", "评审", readiness.can_sign ? "可签署发布" : "判定 · 问题 · 闭环", reviewDone && readiness.can_sign],
  ];
  const currentStage = !preparationDone ? 1 : (!evidenceDone ? 2 : 3);
  stageRail.innerHTML = stages.map(([number, label, detail, done], index) =>
    `<div class="fusion-stage ${done ? "done" : ""} ${index + 1 === currentStage ? "active" : ""}"><span>${done ? "✓" : number}</span><div><b>${label}</b><small>${esc(detail)}</small></div></div>`
  ).join("");

  const protocolLabel = projectedCase.protocolRef || "未固定";
  const combinationConsistency = pipelineCombinationConsistency({
    caseDocument,
    trainingJob: currentTrainingJob(),
    inferenceJob: currentInferenceJob(),
    models: modelOptions(currentModelCatalog()),
  });
  const consistencyLabels = {
    consistent: "三页组合一致",
    mismatch: "跨页组合不一致",
    incomplete: "组合待核验",
  };
  const pageCheckLabel = (item) => !item.available ? "待核验" : (item.matches ? "一致" : "不一致");
  const consistencyDetail = [
    `服务端绑定：g${combinationConsistency.pageChecks.audit.generation || 0} · ${shortAuditRef(combinationConsistency.pageChecks.audit.bindingId)}`,
    `设计与训练：${pageCheckLabel(combinationConsistency.pageChecks.training)}`,
    `推理验证：${pageCheckLabel(combinationConsistency.pageChecks.inference)}`,
    `评测与证据：${pageCheckLabel(combinationConsistency.pageChecks.audit)}`,
    ...combinationConsistency.reasons,
  ].join("；");
  const combination = combinationConsistency.combination;
  contextRibbon.innerHTML = [
    ["跨页组合", viewMatchesPinnedCase ? consistencyLabels[combinationConsistency.state] : "当前候选待固定", viewMatchesPinnedCase ? consistencyDetail : `当前正在预览 ${candidate?.estimator || "候选"} × ${candidate?.featureset || "full"}；固定后才会更新三页权威组合。`, `combination ${viewMatchesPinnedCase ? combinationConsistency.state : "incomplete"}`],
    ["数据集", context.dataset || combination.dataset || projectedCase.dataset.name || "—", context.dataset || projectedCase.dataset.snapshotId || combination.dataset, "dataset"],
    ["算法", candidate?.estimator || combination.estimator || projectedCase.candidate.model || "—", candidate?.model_artifact_id || projectedCase.candidate.artifactId || combination.estimator, "algorithm"],
    ["特征", candidate?.featureset || combination.featureset || projectedCase.candidate.featureset || "full", candidate?.model_artifact_id || projectedCase.candidate.artifactId || combination.featureset, "feature"],
    ["协议 / 判据", `${shortAuditRef(protocolLabel)} · ${projectedCase.verdict.criterionSetVersion || "未绑定"}`, `${protocolLabel} · ${projectedCase.verdict.criterionSetVersion || "未绑定"}`, "protocol"],
  ].map(([label, value, title, className]) => `<div class="fusion-context-cell ${esc(className)}"><span>${esc(label)}</span><b title="${esc(title)}">${esc(value)}</b></div>`).join("");

  const datasetChoices = auditEvidenceDatasetScope({
    caseDocument,
    inferenceJob: currentInferenceJob(),
  });
  if (!datasetChoices.length && context.dataset) datasetChoices.push(context.dataset);
  scopeNode.innerHTML =
    `<section class="fusion-scope-block"><div class="fusion-block-label"><span>判据集</span><small>${esc(verdict.criterion_set_version || acceptancePreset)}</small></div>` +
      `<select id="audit-criterion-select" aria-label="判据集">${acceptancePresetOptions().map((item) => `<option value="${esc(item.id)}" ${item.id === acceptancePreset ? "selected" : ""}>${esc(item.label)}</option>`).join("")}</select></section>` +
    `<section class="fusion-scope-block"><div class="fusion-block-label"><span>候选算法</span><small>同协议重测</small></div><div class="fusion-candidate-list">` +
      (candidates.length ? candidates.slice(0, 3).map((row, index) => {
        const selected = row === candidate;
        const mae = metricMean(row, "mae");
        return `<button type="button" class="fusion-candidate ${selected ? "selected" : ""}" data-audit-candidate="${index}" aria-pressed="${selected}"><span class="fusion-rank">${index + 1}</span><span><b>${esc(row.estimator)}</b><small>${esc(row.featureset || "full")} · ${esc(row.protocol || protocolLabel)}</small></span><strong>${evidenceNumber(mae, 3)}</strong></button>`;
      }).join("") : `<div class="audit-empty compact"><span>暂无候选</span></div>`) +
      `</div></section>` +
    `<section class="fusion-scope-block"><div class="fusion-block-label"><span>证据数据集</span><small>选择会同步画布</small></div><select id="audit-dataset-select" aria-label="证据数据集">${datasetChoices.map((name) => `<option value="${esc(name)}" ${name === context.dataset ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></section>` +
    `<section class="fusion-scope-block"><div class="fusion-block-label"><span>代表电芯</span><small>按证据贡献定位</small></div><div class="fusion-asset-card selected"><span><b>${esc(context.cell || selection.asset_id || "等待画布选择")}</b><small>${esc(context.dataset || selection.dataset || "—")} · cycle ${esc(context.cycle ?? selection.cycle_id ?? "—")}</small></span><strong>${viewMatchesPinnedCase ? "已固定" : "待固定"}</strong></div></section>`;

  const observed = viewVerdict.observed || {};
  const rank = candidates.findIndex((item) => item === candidate) + 1;
  const visibleCandidateArtifact = String(candidate?.model_artifact_id || activeAuditTrajectoryIdentity().modelArtifactId || "");
  const worstReason = (viewVerdict.reasons || [])[0] || (viewVerdict.mode === "preview" ? "当前候选尚未固定到案例" : "无已知超限项");
  const previewVerdictLabel = viewVerdict.result === "fail"
    ? "预览不通过"
    : (viewVerdict.result === "pass" ? "预览通过" : "预览待补充");
  kpiNode.innerHTML = [
    ["当前候选", candidate?.estimator || (rank > 0 ? `排名 #${rank}` : "待比较"), `${candidate?.featureset || "full"} · ${shortAuditRef(visibleCandidateArtifact)}`, "primary"],
    ["MAE", evidenceNumber(observed.mae, 4), `门槛 ≤ ${evidenceNumber(viewVerdict.thresholds?.mae_max, 3)}`, ""],
    ["PICP@90", evidenceNumber(observed.picp, 3), `门槛 ≥ ${evidenceNumber(viewVerdict.thresholds?.picp_min, 2)}`, ""],
    [viewVerdict.mode === "preview" ? "候选状态" : "闭环状态", viewVerdict.mode === "preview" ? previewVerdictLabel : (activeRetest ? "批量复测中" : (completedRetest ? `${projectedCase.closure.retestMemberCount || 0} 条已回填` : `${projectedCase.closure.openIssueCount} 个问题`)), viewVerdict.mode === "preview" ? "固定候选后写入权威判定" : (activeRetest?.run_id || completedRetest?.run_id || worstReason), viewVerdict.mode === "preview" ? "warn" : (activeRetest ? "warn" : (completedRetest ? "pass" : "fail"))],
  ].map(([label, value, note, tone]) => `<div class="fusion-kpi ${tone}"><span>${label}</span><b>${esc(value)}</b><small title="${esc(note)}">${esc(note)}</small></div>`).join("");

  const visibleAsset = context.cell || selection.asset_id;
  const visibleCycle = context.cycle ?? selection.cycle_id;
  const pointAvailable = Boolean(visibleAsset && visibleCycle !== null && visibleCycle !== "");
  const evidenceContext = {
    dataset: context.dataset || selection.dataset,
    asset: context.cell || selection.asset_id,
    cycle: context.cycle ?? selection.cycle_id,
    recordId: context.recordId || selection.sample_id || "",
  };
  const evidenceScenarioKey = auditScenarioKey(evidenceContext.dataset, evidenceContext.asset);
  if (evidenceContext.dataset && (
    auditScenarioState.key !== evidenceScenarioKey
    || auditScenarioState.status === "idle"
  )) {
    loadAuditScenario(evidenceContext.dataset, evidenceContext.asset);
  }
  evidenceHead.innerHTML = `<div><b>比较证据</b><span>聚合、切片、电芯、Cycle 与原始曲线使用同一记录主键</span></div>` +
    `<div class="fusion-evidence-tabs" role="tablist" aria-label="证据视图">` +
      `<button type="button" role="tab" data-audit-evidence-tab="trajectory" class="${auditEvidenceTab === "trajectory" ? "active" : ""}" aria-selected="${auditEvidenceTab === "trajectory"}">SOH 轨迹</button>` +
      `<button type="button" role="tab" data-audit-evidence-tab="raw" class="${auditEvidenceTab === "raw" ? "active" : ""}" aria-selected="${auditEvidenceTab === "raw"}">原始 V / I / T / IC / DV</button>` +
    `</div>${auditCycleSelector(evidenceContext.dataset, evidenceContext.asset, evidenceContext.cycle)}` +
    `<span class="fusion-live-pill ${viewMatchesPinnedCase ? "live" : (pointAvailable ? "dirty" : "")}">${pointAvailable ? `${esc(visibleAsset)} / cycle ${esc(visibleCycle)} · ${viewMatchesPinnedCase ? "已固定" : "待固定"}` : "等待选择 Cycle"}</span>`;
  trajectoryNode.hidden = auditEvidenceTab !== "trajectory";
  rawEvidenceNode.hidden = auditEvidenceTab !== "raw";
  if (auditEvidenceTab === "trajectory") renderAuditTrajectory(trajectoryNode, evidenceContext);
  else renderAuditRawEvidence(rawEvidenceNode, evidenceContext);

  const gateLabels = {
    execution: "执行已完成",
    evidence: "必需证据完整",
    verdict: "判定允许发布",
    review: "人工评审批准",
    issues: "问题全部闭环",
    reconciliation: "新旧 KPI 一致",
  };
  const criteria = auditVerdictCriteria(viewVerdict);
  const coverageItems = coverage.items || [];
  const verdictHeroLabel = viewVerdict.mode === "preview" ? previewVerdictLabel : auditVerdictLabel(viewVerdict.result);
  const verdictHeroMessage = viewVerdict.mode === "preview"
    ? ((viewVerdict.reasons || []).map(auditVerdictReasonLabel).join("；") || "当前为候选预览；固定后由服务端生成权威判定。")
    : ((viewVerdict.reasons || []).map(auditVerdictReasonLabel).join("；") || "当前服务端判定无额外失败原因。");
  const visibleIdentity = activeAuditTrajectoryIdentity();
  readinessNode.innerHTML =
    `<div class="fusion-verdict-hero ${auditVerdictTone(viewVerdict.result)}"><div><span>${viewVerdict.mode === "preview" ? "当前预览" : "当前候选"}</span><b>${esc(verdictHeroLabel)}</b></div><p>${esc(verdictHeroMessage)}</p></div>` +
    `<section class="fusion-inspector-section"><div class="fusion-block-label"><span>${viewVerdict.mode === "preview" ? "预览判据检查" : "判据检查"}</span><small>${esc(viewVerdict.criterion_set_version || verdict.criterion_set_version || "未记录")}</small></div><div class="fusion-criteria-list">${criteria.map((item) => `<div class="fusion-criterion ${item.passed === true ? "pass" : (item.passed === false ? "fail" : "pending")}"><span>${item.passed === true ? "✓" : (item.passed === false ? "!" : "·")}</span><div><b>${esc(item.label)}</b><small>${esc(item.detail)}</small></div><strong>${esc(item.value)}</strong></div>`).join("")}</div>${auditThresholdEditor(viewVerdict)}</section>` +
    `<section class="fusion-inspector-section"><div class="fusion-block-label"><span>${viewMatchesPinnedCase ? "证据完整度" : "已固定证据完整度"}</span><small>${coverage.available_count || 0} / ${coverage.required_count || 0}</small></div><div class="fusion-coverage-bar">${coverageItems.map((item) => `<span class="${item.available ? "complete" : "missing"}" title="${esc(item.kind)}"></span>`).join("")}</div><div class="fusion-coverage-grid">${coverageItems.map((item) => `<span class="${item.available ? "complete" : "missing"}">${item.available ? "●" : "○"} ${esc(auditCoverageLabel(item.kind))}</span>`).join("")}</div></section>` +
    `<section class="fusion-inspector-section"><div class="fusion-block-label"><span>当前证据血缘</span><small>从 Cycle 回溯</small></div><div class="fusion-lineage">${[
      ["PredictionRecord", evidenceContext.recordId || selection.sample_id || "未固定"],
      ["ModelArtifact", visibleIdentity.modelArtifactId || "缺失"],
      ["DatasetSnapshot", viewMatchesPinnedCase ? shortAuditRef(selection.dataset_snapshot_id) : "待固定"],
      ["Protocol", shortAuditRef(selection.split_assignment_id)],
    ].map(([label, value]) => `<div><span></span><p><b>${label}</b><code title="${esc(value)}">${esc(value)}</code></p></div>`).join("")}</div></section>` +
    `<div class="audit-readiness-hero ${readiness.can_sign ? "pass" : "blocked"}">` +
      `<span>EvidencePack</span><b>${readiness.can_sign ? "已满足签署条件" : "暂不可签署"}</b>` +
      `<small>${esc(REVIEW_WORKFLOW_STATE.message || "以服务端门禁为准")}</small>` +
    `</div>` +
    `<div class="audit-gate-list">${Object.entries(gateLabels).map(([key, label]) => {
      const passed = readiness.gates?.[key] === true;
      return `<div class="audit-gate ${passed ? "pass" : "blocked"}"><span>${passed ? "✓" : "!"}</span><div><b>${label}</b><small>${passed ? "已满足" : "阻断发布"}</small></div></div>`;
    }).join("")}</div>` +
    (readiness.blockers?.length
      ? `<div class="audit-blockers"><b>阻断原因</b>${readiness.blockers.map((item) => `<span>${esc(signingBlockerLabel(item))}</span>`).join("")}</div>`
      : `<div class="audit-blockers pass"><b>门禁通过</b><span>可以进入签署动作。</span></div>`);

  closureNode.innerHTML = renderAuditClosure(review, readiness);
  trailNode.innerHTML = renderAuditTrail();
  wireAuditActions();
  wireAuditFusionControls(candidates);
  if (activeRetest) scheduleAuditRetestPolling(activeRetest.id);
}

function shortAuditRef(value) {
  const text = String(value || "未固定");
  if (text.length <= 28) return text;
  return `${text.slice(0, 13)}…${text.slice(-10)}`;
}

function criterionBaseId(value) {
  return String(value || "strict").split("@custom-", 1)[0];
}

function auditThresholdDefinitions() {
  return [
    { key: "mae_max", label: "MAE 上限", observed: "mae", direction: "max", digits: 4 },
    { key: "picp_min", label: "PICP@90 下限", observed: "picp", direction: "min", digits: 3 },
    { key: "monotonicity_max", label: "单调违例上限", observed: "monotonicity", direction: "max", digits: 4 },
    { key: "value_range_max", label: "值域越界上限", observed: "value_range", direction: "max", digits: 4 },
  ];
}

function auditThresholdEditor(verdict) {
  const limits = verdict.thresholds || {};
  const observed = verdict.observed || {};
  const custom = String(verdict.criterion_set_version || "").includes("@custom-");
  const signed = (REVIEW_WORKSPACE?.evidence_packs || []).some((item) => item.state === "signed");
  const locked = signed
    || CASE_PERSISTENCE.status === "saving"
    || ["completed", "archived", "failed"].includes(ACTIVE_EVALUATION_CASE?.lifecycle_state);
  const fields = auditThresholdDefinitions().map((field) => {
    const numeric = Number(limits[field.key]);
    const value = Number.isFinite(numeric) ? numeric : 0;
    const current = evidenceNumber(observed[field.observed], field.digits);
    return `<label class="fusion-threshold-field"><span>${esc(field.label)}<small>观测 ${esc(current)}</small></span>` +
      `<input id="audit-threshold-${esc(field.key)}" data-audit-threshold="${esc(field.key)}" type="number" min="0" max="1" step="0.001" value="${esc(value)}" ${locked ? "disabled" : ""} /></label>`;
  }).join("");
  return `<details class="fusion-threshold-editor" ${custom ? "open" : ""}>` +
    `<summary><span>调整通过阈值</span><small>${custom ? "当前为案例自定义版本" : "按当前算法组合生效"}</small></summary>` +
    `<div class="fusion-threshold-editor-body">` +
      `<div class="fusion-threshold-grid">${fields}</div>` +
      `<div class="fusion-threshold-preview"><span>修改预览</span><b id="audit-threshold-preview">${esc(auditVerdictLabel(verdict.result))}</b><small>仅改变判定口径，不修改原始测评结果</small></div>` +
      `<label class="fusion-threshold-reason"><span>修改理由</span><input id="audit-threshold-reason" type="text" maxlength="200" required value="根据当前算法与目标数据域调整通过阈值" ${locked ? "disabled" : ""} /></label>` +
      `<div class="fusion-threshold-actions"><button id="save-audit-thresholds" type="button" ${locked ? "disabled" : ""}>保存新版本并重新判定</button>` +
      (custom ? `<button id="reset-audit-thresholds" class="secondary" type="button" ${locked ? "disabled" : ""}>恢复${esc(criterionBaseId(verdict.criterion_set_version))}预设</button>` : "") +
      `</div><p>保存后写入审计轨迹；如曾批准评审，将自动退回重新批准。</p>` +
    `</div></details>`;
}

function auditThresholdValues({ report = false } = {}) {
  const values = {};
  for (const field of auditThresholdDefinitions()) {
    const input = $(`audit-threshold-${field.key}`);
    if (!input || !input.checkValidity()) {
      if (report) input?.reportValidity();
      return null;
    }
    const value = Number(input.value);
    if (!Number.isFinite(value) || value < 0 || value > 1) return null;
    values[field.key] = value;
  }
  return values;
}

function updateAuditThresholdPreview() {
  const target = $("audit-threshold-preview");
  if (!target) return;
  const thresholds = auditThresholdValues();
  if (!thresholds) {
    target.textContent = "输入无效";
    target.className = "fail";
    return;
  }
  const observed = ACTIVE_EVALUATION_CASE?.verdict?.observed || {};
  let known = 0;
  const failed = [];
  auditThresholdDefinitions().forEach((field) => {
    const value = Number(observed[field.observed]);
    if (!Number.isFinite(value)) return;
    known += 1;
    const passed = field.direction === "max"
      ? value <= thresholds[field.key]
      : value >= thresholds[field.key];
    if (!passed) failed.push(field.label);
  });
  target.textContent = !known ? "待判定" : (failed.length ? `不通过 · ${failed.join("、")}` : "通过");
  target.className = !known ? "pending" : (failed.length ? "fail" : "pass");
}

async function saveAuditCriterionThresholds(thresholds, reason) {
  if (!ACTIVE_EVALUATION_CASE?.id) return;
  const payload = currentEvidencePersistencePayload();
  if (!payload) return;
  payload.expected_version = ACTIVE_EVALUATION_CASE.version;
  payload.actor = "local-reviewer";
  payload.reason = reason || "调整当前算法组合的通过阈值";
  payload.criterion_set_id = criterionBaseId(acceptancePreset);
  payload.criterion_thresholds = thresholds;
  CASE_PERSISTENCE = { status: "saving", message: "正在保存判据新版本并重新判定" };
  renderAuditWorkspace();
  try {
    const body = await postJson(
      `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/evidence-selection`,
      payload,
    );
    ACTIVE_EVALUATION_CASE = body.case || body;
    acceptancePreset = criterionBaseId(ACTIVE_EVALUATION_CASE.verdict?.criterion_set_version);
    localStorage.setItem("sohbench.acceptancePreset", acceptancePreset);
    alignEvidenceViewToPinnedCase();
    await loadReviewWorkspace({ force: true, silent: true });
    CASE_PERSISTENCE = {
      status: "saved",
      message: thresholds ? "阈值新版本已保存，判定已更新" : "已恢复预设阈值，判定已更新",
    };
  } catch (err) {
    await refreshActiveEvaluationCase();
    await loadReviewWorkspace({ force: true, silent: true });
    CASE_PERSISTENCE = { status: "error", message: `阈值保存失败：${err.message}` };
  }
  renderAuditWorkspace();
  renderGlobalCaseContext();
  renderPipelineNavigation();
}

function auditVerdictLabel(result) {
  return ({ pass: "通过", conditional: "有条件通过", fail: "不通过", pending: "待判定", not_applicable: "不适用" })[result] || result;
}

function auditVerdictTone(result) {
  return result === "pass" ? "pass" : (result === "fail" ? "fail" : "warn");
}

function auditVerdictReasonLabel(reason) {
  const labels = { generalization: "整体误差超过门槛", calibration: "区间覆盖率未通过", monotonicity: "单调性约束未通过", value_range: "值域约束未通过" };
  return labels[reason] || reason;
}

function auditVerdictCriteria(verdict) {
  const observed = verdict.observed || {};
  const limits = verdict.thresholds || {};
  const criteria = [
    ["整体 MAE", "mae", "mae_max", "max", 4],
    ["PICP@90", "picp", "picp_min", "min", 3],
    ["物理一致性", "monotonicity", "monotonicity_max", "max", 4],
    ["值域约束", "value_range", "value_range_max", "max", 4],
  ];
  return criteria.map(([label, observedKey, limitKey, direction, digits]) => {
    const value = observed[observedKey];
    const limit = limits[limitKey];
    const known = Number.isFinite(Number(value)) && Number.isFinite(Number(limit));
    const passed = !known ? null : (direction === "max" ? Number(value) <= Number(limit) : Number(value) >= Number(limit));
    return {
      label,
      passed,
      value: known ? evidenceNumber(value, digits) : "—",
      detail: known ? `${direction === "max" ? "≤" : "≥"} ${evidenceNumber(limit, digits)}` : "缺少可审计数值",
    };
  });
}

function auditCoverageLabel(kind) {
  return ({
    run_spec: "RunSpec", dataset_snapshot: "数据快照", split_assignment: "划分指纹",
    comparison_scope: "比较范围", model_artifact: "模型产物", prediction_record: "逐点预测",
    raw_curve: "原始 V/I/T", criterion_set: "判据版本",
  })[kind] || kind;
}

function wireAuditFusionControls(candidates) {
  const criterion = $("audit-criterion-select");
  if (criterion) criterion.onchange = async () => {
    acceptancePreset = criterion.value;
    localStorage.setItem("sohbench.acceptancePreset", acceptancePreset);
    markEvidenceViewState();
    await saveAuditCriterionThresholds(
      null,
      `切换到${criterion.options[criterion.selectedIndex]?.text || acceptancePreset}预设判据`,
    );
  };
  document.querySelectorAll("[data-audit-threshold]").forEach((input) => {
    input.oninput = updateAuditThresholdPreview;
  });
  const saveThresholds = $("save-audit-thresholds");
  if (saveThresholds) saveThresholds.onclick = async () => {
    const thresholds = auditThresholdValues({ report: true });
    if (!thresholds) return;
    const reason = $("audit-threshold-reason")?.value.trim();
    if (!reason) {
      $("audit-threshold-reason")?.reportValidity();
      return;
    }
    await saveAuditCriterionThresholds(thresholds, reason);
  };
  const resetThresholds = $("reset-audit-thresholds");
  if (resetThresholds) resetThresholds.onclick = async () => {
    await saveAuditCriterionThresholds(
      null,
      `恢复${criterionBaseId(ACTIVE_EVALUATION_CASE?.verdict?.criterion_set_version)}预设阈值`,
    );
  };
  const dataset = $("audit-dataset-select");
  if (dataset) dataset.onchange = () => {
    nativeDashboardContext = { ...nativeDashboardContext, dataset: dataset.value, source: "evidence", cell: "", cycle: "" };
    evidenceLiveContext = {};
    markEvidenceViewState();
    invalidateAuditRawCurves();
    renderAuditWorkspace();
  };
  const cycle = $("audit-cycle-input");
  if (cycle) {
    const applyCycle = () => {
      if (!cycle.value) return;
      const selected = selectAuditCycle(
        cycle.value,
        cycle.dataset.auditDataset || "",
        cycle.dataset.auditAsset || "",
      );
      cycle.setCustomValidity(selected ? "" : "该 Cycle 没有预测记录");
      if (!selected) cycle.reportValidity();
    };
    cycle.onchange = applyCycle;
    cycle.onkeydown = (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        applyCycle();
      }
    };
  }
  document.querySelectorAll("[data-audit-candidate]").forEach((button) => {
    button.onclick = () => {
      const candidate = candidates[Number(button.dataset.auditCandidate)];
      if (!candidate) return;
      nativeDashboardContext = { ...nativeDashboardContext, source: "evidence", model: candidate.estimator || "", featureset: candidate.featureset || "full" };
      evidenceLiveContext = {};
      markEvidenceViewState();
      invalidateAuditRawCurves();
      renderAuditWorkspace();
    };
  });
  document.querySelectorAll("[data-audit-evidence-tab]").forEach((button) => {
    button.onclick = () => {
      auditEvidenceTab = button.dataset.auditEvidenceTab === "raw" ? "raw" : "trajectory";
      renderAuditWorkspace();
    };
  });
  document.querySelectorAll("[data-mode-shortcut]").forEach((button) => {
    button.onclick = () => { activeMode = button.dataset.modeShortcut; render(); };
  });
}

function auditScenarioPointRows(dataset = "", asset = "") {
  const key = auditScenarioKey(dataset, asset);
  if (auditScenarioState.key !== key || auditScenarioState.status !== "ready") {
    return { asset: String(asset || ""), points: [] };
  }
  const allPoints = auditScenarioState.data?.points || [];
  const assets = uniq(allPoints.map((item) => String(item.cell_id || item.asset_id || "")).filter(Boolean));
  const selectedAsset = assets.includes(String(asset || "")) ? String(asset) : (assets[0] || "");
  const points = allPoints
    .filter((item) => String(item.cell_id || item.asset_id || "") === selectedAsset)
    .map((item) => {
      const recordId = String(item.prediction_record_id || item.sample_id || "");
      const contribution = (DATA?.metricContributions?.records || []).find((row) =>
        row.prediction_record_id === recordId || row.sample_id === item.sample_id
      );
      return {
        cycle: Number(item.cycle_index ?? item.cycle_id),
        truth: Number(item.true_soh),
        pred: Number(item.pred_soh),
        lower: item.lower == null ? null : Number(item.lower),
        upper: item.upper == null ? null : Number(item.upper),
        contribution: Number(
          item.metric_contributions?.mae_contribution
          ?? item.mae_contribution
          ?? contribution?.mae_contribution
        ),
        absoluteError: Number(item.absolute_error),
        rawAvailable: item.raw_curve_available !== false,
        recordId,
      };
    })
    .filter((item) => Number.isFinite(item.cycle) && Number.isFinite(item.truth) && Number.isFinite(item.pred))
    .sort((left, right) => left.cycle - right.cycle);
  return { asset: selectedAsset, points };
}

function selectAuditCycle(cycle, dataset = "", asset = "") {
  const context = currentEvidenceContext();
  const selectedDataset = String(dataset || context.dataset || ACTIVE_EVALUATION_CASE?.evidence_selection?.dataset || "");
  const selectedAsset = String(asset || context.cell || ACTIVE_EVALUATION_CASE?.evidence_selection?.asset_id || "");
  const scenario = auditScenarioPointRows(selectedDataset, selectedAsset);
  const point = scenario.points.find((item) => String(item.cycle) === String(cycle));
  if (!point) return false;
  nativeDashboardContext = {
    ...nativeDashboardContext,
    dataset: selectedDataset,
    source: "evidence",
    cell: scenario.asset,
    cycle: point.cycle,
  };
  evidenceLiveContext = {
    dataset: selectedDataset,
    source: "evidence",
    cell: scenario.asset,
    cycle: point.cycle,
    recordId: point.recordId || `${selectedDataset}:${scenario.asset}:${point.cycle}`,
    trueSoh: point.truth,
    predSoh: point.pred,
    lower: point.lower,
    upper: point.upper,
    rawAvailable: point.rawAvailable,
  };
  invalidateAuditRawCurves();
  markEvidenceViewState();
  renderAuditWorkspace();
  return true;
}

function auditCycleSelector(dataset = "", asset = "", cycle = "") {
  const key = auditScenarioKey(dataset, asset);
  const scenario = auditScenarioPointRows(dataset, asset);
  const ready = auditScenarioState.key === key && auditScenarioState.status === "ready" && scenario.points.length;
  const selectedCycle = String(cycle ?? "");
  const placeholder = auditScenarioState.key === key && auditScenarioState.status === "error"
    ? "Cycle 不可用"
    : "载入 Cycle…";
  const firstCycle = ready ? scenario.points[0].cycle : "";
  const lastCycle = ready ? scenario.points[scenario.points.length - 1].cycle : "";
  const rangeLabel = ready ? `${firstCycle}–${lastCycle}` : placeholder;
  return `<label class="fusion-cycle-picker" title="${ready ? `${scenario.points.length} 个可选 Cycle` : esc(placeholder)}"><span>Cycle</span><input id="audit-cycle-input" type="number" inputmode="numeric" aria-label="选择 Cycle" data-audit-dataset="${esc(dataset)}" data-audit-asset="${esc(scenario.asset || asset)}" value="${ready ? esc(selectedCycle) : ""}" min="${firstCycle}" max="${lastCycle}" step="1" placeholder="${esc(rangeLabel)}" ${ready ? "" : "disabled"}></label>`;
}

function activeAuditTrajectoryIdentity() {
  const context = currentEvidenceContext();
  const selection = ACTIVE_EVALUATION_CASE?.evidence_selection || {};
  const savedCandidate = activeSavedCombination()?.candidate || {};
  const selectedRow = selectedEvidenceCandidate(auditCandidateRows(), context) || {};
  const caseCandidates = ACTIVE_EVALUATION_CASE?.candidates || [];
  const selectedCaseCandidate = caseCandidates.find((item) =>
    String(item.candidate_id || "") === String(selection.candidate_id || "")
  ) || {};
  let modelArtifactId = String(selectedRow.model_artifact_id || "");
  if (!modelArtifactId &&
      String(savedCandidate.model || "") === String(context.model || "") &&
      String(savedCandidate.featureset || "full") === String(context.featureset || "full")) {
    modelArtifactId = String(savedCandidate.model_artifact_id || "");
  }
  if (!modelArtifactId &&
      String(selectedCaseCandidate.model || "") === String(context.model || "") &&
      String(selectedCaseCandidate.featureset || "full") === String(context.featureset || "full")) {
    modelArtifactId = String(selectedCaseCandidate.model_artifact_id || "");
  }
  const artifactCandidate = caseCandidates.find((item) =>
    modelArtifactId && String(item.model_artifact_id || "") === modelArtifactId
  ) || null;
  const contextCandidates = caseCandidates.filter((item) =>
    String(item.model || "") === String(context.model || "") &&
    String(item.featureset || "full") === String(context.featureset || "full")
  );
  const candidate = artifactCandidate
    || (contextCandidates.length === 1 ? contextCandidates[0] : null)
    || selectedCaseCandidate;
  return {
    candidateId: String(selectedRow.candidate_id || candidate?.candidate_id || savedCandidate.candidate_id || ""),
    modelArtifactId: String(modelArtifactId || candidate?.model_artifact_id || ""),
  };
}

function auditScenarioKey(dataset, asset = "") {
  const selection = ACTIVE_EVALUATION_CASE?.evidence_selection || {};
  const inferenceJobId = activeEvidenceInferenceJobId();
  const identity = activeAuditTrajectoryIdentity();
  return [
    ACTIVE_EVALUATION_CASE?.id || activeProfileOption?.name || activeProfile || "root",
    inferenceJobId,
    evaluationContextBindingId(),
    identity.candidateId,
    identity.modelArtifactId,
    dataset || "",
    asset || selection.asset_id || "",
  ].join("|");
}

function activeEvidenceInferenceJobId() {
  const saved = activeSavedCombination();
  if (saved?.source_stage === "inference" && saved.inference_job_id) {
    return String(saved.inference_job_id);
  }
  return String(ACTIVE_EVALUATION_CASE?.execution?.inference_job_id || "");
}

function auditScenarioUrl(dataset, asset = "") {
  const raw = String(dataset || "").trim();
  const caseId = String(ACTIVE_EVALUATION_CASE?.id || "");
  const inferenceJobId = activeEvidenceInferenceJobId();
  if (caseId && inferenceJobId && raw) {
    const identity = activeAuditTrajectoryIdentity();
    const query = new URLSearchParams({
      inference_job_id: inferenceJobId,
      dataset: raw,
      compact: "1",
      context_binding_id: evaluationContextBindingId(),
    });
    if (identity.candidateId) query.set("candidate_id", identity.candidateId);
    if (identity.modelArtifactId) query.set("model_artifact_id", identity.modelArtifactId);
    const selectedAsset = String(
      asset
      || (raw === String(ACTIVE_EVALUATION_CASE?.evidence_selection?.dataset || "")
        ? ACTIVE_EVALUATION_CASE?.evidence_selection?.asset_id
        : "")
      || "",
    );
    if (selectedAsset) query.set("asset_id", selectedAsset);
    return `${API_BASE}/api/v2/cases/${encodeURIComponent(caseId)}/trajectory?${query}`;
  }
  const token = raw === "calce_cs2" ? "calce" : raw;
  if (!/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(token)) return "";
  return `${profileDataPath(activeProfileOption?.name || activeProfile || "root")}/battery_dashboard_${encodeURIComponent(token)}.json?v=${encodeURIComponent(DATA_VERSION)}`;
}

async function loadAuditScenario(dataset, asset = "") {
  const key = auditScenarioKey(dataset, asset);
  if (auditScenarioState.key === key && ["loading", "ready"].includes(auditScenarioState.status)) return;
  const cached = auditScenarioCache.get(key);
  if (cached) {
    auditScenarioState = { key, status: "ready", data: cached, error: "" };
    return;
  }
  const url = auditScenarioUrl(dataset, asset);
  if (!url) {
    auditScenarioState = { key, status: "error", data: null, error: "当前数据集名称不可用于证据场景" };
    return;
  }
  auditScenarioState = { key, status: "loading", data: null, error: "" };
  try {
    const data = await fetchJson(url);
    const identity = activeAuditTrajectoryIdentity();
    const consistency = trajectoryContextConsistency(data, {
      case_id: String(ACTIVE_EVALUATION_CASE?.id || ""),
      inference_job_id: activeEvidenceInferenceJobId(),
      dataset: String(dataset || ""),
      candidate_id: identity.candidateId,
      model_artifact_id: identity.modelArtifactId,
      context_binding_id: evaluationContextBindingId(),
    });
    if (!consistency.matches) {
      const details = consistency.mismatches.map((item) => `${item.field}: ${item.actual || "缺失"}`).join("；");
      throw new Error(`轨迹上下文不一致（${details}），已阻止显示`);
    }
    if (!Array.isArray(data?.points) || !data.points.length) throw new Error("场景没有逐 Cycle 预测记录");
    auditScenarioCache.set(key, data);
    if (auditScenarioCache.size > 12) {
      auditScenarioCache.delete(auditScenarioCache.keys().next().value);
    }
    if (key === auditScenarioKey(dataset, asset)) {
      auditScenarioState = { key, status: "ready", data, error: "" };
    }
  } catch (err) {
    if (key === auditScenarioKey(dataset, asset)) {
      auditScenarioState = { key, status: "error", data: null, error: err.message };
    }
  }
  if (activeMode === "audit") renderAuditWorkspace();
}

function invalidateAuditRawCurves() {
  auditRawCurveState = { key: "", status: "idle", data: null, error: "" };
}

function auditRawCurveContext({ dataset = "", asset = "", cycle = "", recordId = "" } = {}) {
  const resolvedRecordId = String(recordId || (dataset && asset && cycle !== "" && cycle != null
    ? `${dataset}:${asset}:${cycle}`
    : ""));
  const caseId = String(ACTIVE_EVALUATION_CASE?.id || "");
  const caseVersion = String(ACTIVE_EVALUATION_CASE?.version ?? "");
  return {
    caseId,
    recordId: resolvedRecordId,
    key: `${caseId}|${caseVersion}|${resolvedRecordId}`,
  };
}

async function loadAuditRawCurves(context) {
  const request = auditRawCurveContext(context);
  if (!request.caseId || !request.recordId) return;
  if (auditRawCurveState.key === request.key && ["loading", "ready"].includes(auditRawCurveState.status)) return;
  auditRawCurveState = { key: request.key, status: "loading", data: null, error: "" };
  try {
    const query = new URLSearchParams({
      case_id: request.caseId,
      context_binding_id: evaluationContextBindingId(),
      signals: "voltage,current,temperature,ic,dv",
      resolution: "viewport",
      max_points: "480",
    });
    const response = await fetch(
      `${API_BASE}/api/v2/prediction-records/${encodeURIComponent(request.recordId)}/curves?${query}`,
    );
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
    auditRawCurveState = { key: request.key, status: "ready", data: body, error: "" };
  } catch (err) {
    auditRawCurveState = { key: request.key, status: "error", data: null, error: err.message };
  }
  if (activeMode === "audit" && auditEvidenceTab === "raw") renderAuditWorkspace();
}

function compactCurveNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const absolute = Math.abs(number);
  if ((absolute > 0 && absolute < 0.001) || absolute >= 10000) return number.toExponential(2);
  if (absolute >= 100) return number.toFixed(1);
  if (absolute >= 10) return number.toFixed(2);
  return number.toFixed(3);
}

function rawCurveDerivationLabel(value) {
  if (value === "measured") return "实测";
  if (value === "source") return "源文件";
  if (value === "recorded feature value") return "推理记录标量";
  if (value === "integrated from measured current/time") return "由实测电流 / 时间积分";
  if (value === "derived from measured time") return "由实测时间派生";
  if (value === "measured voltage underlying statistic") return "电压统计量的原始曲线";
  if (value === "measured current underlying statistic") return "电流统计量的原始曲线";
  if (value === "measured temperature underlying statistic") return "温度统计量的原始曲线";
  if (value === "measured voltage inside feature window") return "特征窗内实测电压";
  if (value === "derived voltage/current proxy") return "由 V / I 派生代理量";
  if (value === "measured voltage proxy") return "实测电压代理曲线";
  if (String(value || "").startsWith("derived")) return "由实测 V / I / time 派生";
  return value || "未记录";
}

function curveHighlightRange(signal, featureWindow = {}) {
  const xName = String(signal?.x_name || "");
  if (xName === "time_s") return featureWindow.cycle_time_s || [];
  if (xName === "sample_index") return featureWindow.sample_index || [];
  if (xName === "voltage_v") return featureWindow.voltage_v || [];
  if (xName === "charge_ah") return featureWindow.charge_ah || [];
  return [];
}

function renderCurveHighlight(range, x, xMin, xMax, margin, plotWidth, plotHeight, color = "#F4B72C") {
  if (!Array.isArray(range) || range.length !== 2) return "";
  const low = Math.max(xMin, Number(range[0]));
  const high = Math.min(xMax, Number(range[1]));
  if (!Number.isFinite(low) || !Number.isFinite(high) || high < low) return "";
  const left = x(low);
  const right = x(high);
  const bandWidth = Math.max(right - left, 2);
  return `<rect class="curve-highlight-band" x="${left.toFixed(1)}" y="${margin.top}" width="${bandWidth.toFixed(1)}" height="${plotHeight}" fill="${color}" opacity=".22"/>` +
    `<line x1="${left.toFixed(1)}" y1="${margin.top}" x2="${left.toFixed(1)}" y2="${margin.top + plotHeight}" stroke="${color}" stroke-width="1.2" stroke-dasharray="4 3"/>` +
    `<line x1="${right.toFixed(1)}" y1="${margin.top}" x2="${right.toFixed(1)}" y2="${margin.top + plotHeight}" stroke="${color}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
}

function renderRawSignalChart(name, signal, featureWindow) {
  const labels = {
    voltage: ["电压 V", "#0A8FCE"],
    current: ["电流 I", "#7257D9"],
    temperature: ["温度 T", "#E44D44"],
    ic: ["增量容量 IC", "#07966A"],
    dv: ["微分电压 DV", "#C37A09"],
  };
  const [label, color] = labels[name] || [name, "#4E6B82"];
  const sourceX = (signal?.x || []).map((value) => Number(value));
  const timeOrigin = signal?.x_name === "time_s" && Number.isFinite(sourceX[0]) ? sourceX[0] : 0;
  const pairs = sourceX.map((value, index) => [value - timeOrigin, Number(signal?.y?.[index])])
    .filter(([xValue, yValue]) => Number.isFinite(xValue) && Number.isFinite(yValue));
  if (!pairs.length) {
    return `<article class="raw-signal-card unavailable"><div class="raw-signal-head"><span>${esc(label)}</span><em>不可用</em></div><p>当前 CurveSegmentRef 未返回可绘制数据。</p></article>`;
  }
  const width = 360;
  const height = 154;
  const margin = { left: 45, right: 12, top: 12, bottom: 30 };
  const xValues = pairs.map(([value]) => value);
  const yValues = pairs.map(([, value]) => value);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const rawYMin = Math.min(...yValues);
  const rawYMax = Math.max(...yValues);
  const yPadding = Math.max((rawYMax - rawYMin) * 0.08, Math.abs(rawYMax || 1) * 0.01, 1e-9);
  const yMin = rawYMin - yPadding;
  const yMax = rawYMax + yPadding;
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (value) => margin.left + ((value - xMin) / Math.max(xMax - xMin, 1e-9)) * plotWidth;
  const y = (value) => margin.top + (1 - (value - yMin) / Math.max(yMax - yMin, 1e-9)) * plotHeight;
  const points = pairs.map(([xValue, yValue]) => `${x(xValue).toFixed(1)},${y(yValue).toFixed(1)}`).join(" ");
  const grid = Array.from({ length: 4 }, (_, index) => yMin + ((yMax - yMin) * index) / 3);
  const highlightRange = curveHighlightRange(signal, featureWindow);
  const featureBand = renderCurveHighlight(
    highlightRange,
    x,
    xMin,
    xMax,
    margin,
    plotWidth,
    plotHeight,
  );
  const yUnit = String(signal.y_unit || "");
  const xUnit = String(signal.x_unit || "");
  const xLabel = signal.x_name === "time_s" ? "Cycle 时间" : (signal.x_name || "x");
  return `<article class="raw-signal-card" data-raw-signal="${esc(name)}">` +
    `<div class="raw-signal-head"><span>${esc(label)}</span><em>${esc(rawCurveDerivationLabel(signal.derivation))}</em></div>` +
    `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(label)} 曲线">` +
      `<rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" rx="4" fill="#FBFDFF"/>${featureBand}` +
      grid.map((value) => `<line x1="${margin.left}" y1="${y(value).toFixed(1)}" x2="${width - margin.right}" y2="${y(value).toFixed(1)}" stroke="#E6EDF4"/><text x="${margin.left - 6}" y="${(y(value) + 3).toFixed(1)}" text-anchor="end">${esc(compactCurveNumber(value))}</text>`).join("") +
      `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>` +
      `<text x="${margin.left}" y="${height - 9}">${esc(compactCurveNumber(xMin))}</text><text x="${width - margin.right}" y="${height - 9}" text-anchor="end">${esc(compactCurveNumber(xMax))}</text>` +
      `<text x="${margin.left + plotWidth / 2}" y="${height - 9}" text-anchor="middle">${esc(xLabel)} · ${esc(xUnit)}</text>` +
    `</svg>` +
    `<div class="raw-signal-foot"><span>${pairs.length} / ${signal.source_points || pairs.length} 点</span><span>${esc(yUnit)} · ${esc(compactCurveNumber(rawYMin))}–${esc(compactCurveNumber(rawYMax))}</span></div>` +
  `</article>`;
}

function featureCurveLabel(name) {
  const labels = {
    charge_capacity_max: "充电容量累计",
    time_s_max: "时间轴上限",
    cycle_index_feature: "Cycle 序号",
    cycle_fraction: "寿命进度",
    q_win_ah: "窗口充入电量",
    t_win_s: "窗口持续时间",
    ic_peak_ah_per_v: "IC 峰值",
    ic_peak_voltage_v: "IC 峰值电压",
    voltage_mean: "电压均值",
    voltage_std: "电压波动",
    voltage_min: "最低电压",
    voltage_max: "最高电压",
    current_mean: "电流均值",
    current_std: "电流波动",
    abs_current_mean: "电流绝对均值",
    temperature_mean: "温度均值",
    battery_soh_cycle_norm: "Cycle 归一位置",
    battery_soh_voltage_sag: "电压下垂",
    battery_soh_resistance_growth: "阻抗增长代理",
  };
  return labels[name] || name.replaceAll("_", " ");
}

function featureScopeMeta(scope) {
  if (scope === "feature_window") return ["局部特征窗", "window", "#E5A514"];
  if (scope === "full_cycle") return ["全 Cycle", "full", "#3A8FD8"];
  return ["跨 Cycle 上下文", "context", "#788B9D"];
}

function formatFeatureInterval(feature, featureWindow = {}) {
  const scope = String(feature?.scope || "");
  if (scope === "cycle_context") return "不对应 Cycle 内局部时间窗";
  const range = feature?.highlight?.cycle_time_s || [];
  if (!Array.isArray(range) || range.length !== 2) return "时间窗不可用";
  const unit = feature?.x_unit === "index" ? "点" : "s";
  return `${compactCurveNumber(range[0])}–${compactCurveNumber(range[1])} ${unit}` +
    (scope === "feature_window" && Number.isFinite(Number(featureWindow.duration_s))
      ? ` · Δ ${compactCurveNumber(featureWindow.duration_s)} s`
      : "");
}

function renderFeatureCurveChart(feature, featureWindow = {}) {
  const name = String(feature?.name || "feature");
  const [scopeLabel, scopeClass, bandColor] = featureScopeMeta(feature?.scope);
  const pairs = (feature?.x || []).map((value, index) => [Number(value), Number(feature?.y?.[index])])
    .filter(([xValue, yValue]) => Number.isFinite(xValue) && Number.isFinite(yValue));
  const value = compactCurveNumber(feature?.value);
  const unit = String(feature?.y_unit || "");
  const interval = formatFeatureInterval(feature, featureWindow);
  if (!pairs.length) {
    return `<article class="feature-curve-card unavailable" data-feature-curve="${esc(name)}">` +
      `<div class="feature-curve-head"><div><span>${esc(featureCurveLabel(name))}</span><code>${esc(name)}</code></div><b>${esc(value)} ${esc(unit)}</b></div>` +
      `<div class="feature-curve-meta"><em class="${scopeClass}">${esc(scopeLabel)}</em><span>${esc(interval)}</span></div>` +
      `<p>推理记录包含该特征值，但当前 CurveSegmentRef 缺少对应原始信号。</p>` +
    `</article>`;
  }
  const width = 420;
  const height = 168;
  const margin = { left: 48, right: 12, top: 13, bottom: 30 };
  const xValues = pairs.map(([item]) => item);
  const yValues = pairs.map(([, item]) => item);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const rawYMin = Math.min(...yValues);
  const rawYMax = Math.max(...yValues);
  const yPadding = Math.max((rawYMax - rawYMin) * 0.08, Math.abs(rawYMax || 1) * 0.01, 1e-9);
  const yMin = rawYMin - yPadding;
  const yMax = rawYMax + yPadding;
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (item) => margin.left + ((item - xMin) / Math.max(xMax - xMin, 1e-9)) * plotWidth;
  const y = (item) => margin.top + (1 - (item - yMin) / Math.max(yMax - yMin, 1e-9)) * plotHeight;
  const points = pairs.map(([xValue, yValue]) => `${x(xValue).toFixed(1)},${y(yValue).toFixed(1)}`).join(" ");
  const grid = Array.from({ length: 4 }, (_, index) => yMin + ((yMax - yMin) * index) / 3);
  const highlightRange = feature?.highlight?.cycle_time_s || [];
  const highlight = feature?.scope === "cycle_context" ? "" : renderCurveHighlight(
    highlightRange,
    x,
    xMin,
    xMax,
    margin,
    plotWidth,
    plotHeight,
    bandColor,
  );
  return `<article class="feature-curve-card ${scopeClass}" data-feature-curve="${esc(name)}">` +
    `<div class="feature-curve-head"><div><span>${esc(featureCurveLabel(name))}</span><code>${esc(name)}</code></div><b>${esc(value)} ${esc(unit)}</b></div>` +
    `<div class="feature-curve-meta"><em class="${scopeClass}">${esc(scopeLabel)}</em><span>${esc(interval)}</span></div>` +
    `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(featureCurveLabel(name))}特征曲线，使用区间 ${esc(interval)}">` +
      `<rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" rx="4" fill="#FBFDFF"/>${highlight}` +
      grid.map((item) => `<line x1="${margin.left}" y1="${y(item).toFixed(1)}" x2="${width - margin.right}" y2="${y(item).toFixed(1)}" stroke="#E7EDF3"/><text x="${margin.left - 6}" y="${(y(item) + 3).toFixed(1)}" text-anchor="end">${esc(compactCurveNumber(item))}</text>`).join("") +
      `<polyline points="${points}" fill="none" stroke="#126FA7" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>` +
      `<text x="${margin.left}" y="${height - 9}">${esc(compactCurveNumber(xMin))}</text><text x="${width - margin.right}" y="${height - 9}" text-anchor="end">${esc(compactCurveNumber(xMax))}</text>` +
      `<text x="${margin.left + plotWidth / 2}" y="${height - 9}" text-anchor="middle">Cycle 时间 · ${esc(feature.x_unit || "s")}</text>` +
    `</svg>` +
    `<div class="feature-curve-foot"><span>${esc(rawCurveDerivationLabel(feature.derivation))}</span><span>${esc(feature.source_signal || "recorded_scalar")}</span></div>` +
  `</article>`;
}

function renderAuditRawEvidence(node, context = {}) {
  const request = auditRawCurveContext(context);
  if (!request.caseId || !request.recordId) {
    node.innerHTML = `<div class="fusion-trajectory-state"><span>1</span><b>先在 SOH 轨迹选择一个 Cycle</b><small>选择后将直接读取同一 PredictionRecord 的 V / I / T / IC / DV</small></div>`;
    return;
  }
  if (context.rawAvailable === false) {
    node.innerHTML = `<div class="fusion-trajectory-state"><span>i</span><b>该 Cycle 没有迁移到本地的原始曲线</b><small>SOH 预测轨迹仍可审计；请选择轨迹上带原始证据的 Cycle 查看 V / I / T / IC / DV。</small></div>`;
    return;
  }
  if (auditRawCurveState.key !== request.key || auditRawCurveState.status === "idle") {
    node.innerHTML = `<div class="fusion-trajectory-state"><span class="audit-run-spinner"></span><b>正在读取规范曲线证据</b><small>${esc(shortAuditRef(request.recordId))}</small></div>`;
    loadAuditRawCurves(context);
    return;
  }
  if (auditRawCurveState.status === "loading") {
    node.innerHTML = `<div class="fusion-trajectory-state"><span class="audit-run-spinner"></span><b>正在读取规范曲线证据</b><small>${esc(shortAuditRef(request.recordId))}</small></div>`;
    return;
  }
  if (auditRawCurveState.status === "error") {
    node.innerHTML = `<div class="fusion-trajectory-state error"><span>!</span><b>曲线证据暂不可用</b><small>${esc(auditRawCurveState.error)}</small><button type="button" data-audit-raw-retry>重新读取</button></div>`;
    const retry = node.querySelector("[data-audit-raw-retry]");
    if (retry) retry.onclick = () => {
      invalidateAuditRawCurves();
      renderAuditRawEvidence(node, context);
    };
    return;
  }
  const data = auditRawCurveState.data || {};
  const record = data.prediction_record || {};
  const sampling = data.sampling || {};
  const signals = data.signals || {};
  const order = ["voltage", "current", "temperature", "ic", "dv"];
  const featureWindow = data.feature_window || {};
  const featureCurves = data.feature_curves || [];
  const missing = data.missing_signals || [];
  const cycleWindow = featureWindow.cycle_time_s;
  const featureWindowText = Array.isArray(cycleWindow) && cycleWindow.length === 2
    ? `${compactCurveNumber(cycleWindow[0])}–${compactCurveNumber(cycleWindow[1])} s · Δ ${compactCurveNumber(featureWindow.duration_s)} s`
    : "未映射到 Cycle 时间";
  node.innerHTML = `<div class="raw-evidence-summary" data-raw-record-id="${esc(record.prediction_record_id || record.sample_id || request.recordId)}">` +
      `<div><span>规范 PredictionRecord</span><b title="${esc(record.prediction_record_id || record.sample_id || request.recordId)}">${esc(shortAuditRef(record.prediction_record_id || record.sample_id || request.recordId))}</b></div>` +
      `<div><span>CurveSegmentRef</span><b title="${esc(data.curve_segment_ref?.source_uri || "")}">${esc(shortAuditRef(data.curve_segment_ref?.source_uri || "已校验"))}</b></div>` +
      `<div><span>采样</span><b>${esc(sampling.returned_points ?? "—")} / ${esc(sampling.source_points ?? "—")} 点 · ${esc(sampling.resolution || "viewport")}</b></div>` +
      `<div><span>Cycle 特征时间窗</span><b title="${esc(featureWindow.source || "")}">${esc(featureWindowText)}</b></div>` +
    `</div>` +
    (missing.length ? `<div class="raw-evidence-warning">未返回：${missing.map((name) => esc(name.toUpperCase())).join(" / ")}</div>` : "") +
    `<section class="raw-curve-section">` +
      `<div class="raw-section-head"><div><span>原始与派生信号</span><small>同一 Cycle 的 V / I / T / IC / DV</small></div><div class="curve-window-legend"><i></i>特征窗 ${Array.isArray(featureWindow.voltage_v) ? `${esc(featureWindow.voltage_v[0])}–${esc(featureWindow.voltage_v[1])} V` : "未记录"}</div></div>` +
      `<div class="raw-signal-grid">${order.filter((name) => signals[name]).map((name) => renderRawSignalChart(name, signals[name], featureWindow)).join("")}</div>` +
    `</section>` +
    `<section class="feature-curve-section">` +
      `<div class="raw-section-head"><div><span>当前模型特征曲线 <b>${featureCurves.length}</b></span><small>${esc(record.featureset || "未命名特征方案")} · 与本次 PredictionRecord 完全一致</small></div><div class="feature-scope-legend"><span class="window">局部特征窗</span><span class="full">全 Cycle</span><span class="context">跨 Cycle 上下文</span></div></div>` +
      (featureCurves.length
        ? `<div class="feature-curve-grid">${featureCurves.map((feature) => renderFeatureCurveChart(feature, featureWindow)).join("")}</div>`
        : `<div class="raw-evidence-warning">当前 PredictionRecord 未记录 feature_values，无法证明模型实际使用的特征曲线。</div>`) +
    `</section>`;
}

function renderAuditTrajectory(node, { dataset = "", asset = "", cycle = "" } = {}) {
  const key = auditScenarioKey(dataset, asset);
  if (auditScenarioState.key !== key || auditScenarioState.status === "idle") {
    node.innerHTML = `<div class="fusion-trajectory-state"><span class="audit-run-spinner"></span><b>正在装载同一事实源的 SOH 轨迹</b><small>${esc(dataset || "未选择数据集")}</small></div>`;
    loadAuditScenario(dataset, asset);
    return;
  }
  if (auditScenarioState.status === "loading") {
    node.innerHTML = `<div class="fusion-trajectory-state"><span class="audit-run-spinner"></span><b>正在装载同一事实源的 SOH 轨迹</b><small>${esc(dataset || "未选择数据集")}</small></div>`;
    return;
  }
  if (auditScenarioState.status === "error") {
    node.innerHTML = `<div class="fusion-trajectory-state error"><span>!</span><b>轨迹证据不可用</b><small>${esc(auditScenarioState.error)}</small></div>`;
    return;
  }

  const scenario = auditScenarioPointRows(dataset, asset);
  const selectedAsset = scenario.asset;
  const points = scenario.points;
  if (!points.length) {
    node.innerHTML = `<div class="fusion-trajectory-state error"><span>!</span><b>当前电芯没有可绘制记录</b><small>${esc(selectedAsset || "—")}</small></div>`;
    return;
  }

  const width = 920;
  const height = 286;
  const margin = { left: 54, right: 24, top: 20, bottom: 38 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xMin = points[0].cycle;
  const xMax = points[points.length - 1].cycle || xMin + 1;
  const yValues = points.flatMap((item) => [item.truth, item.pred, item.lower, item.upper]).filter(Number.isFinite);
  const rawMin = Math.min(...yValues);
  const rawMax = Math.max(...yValues);
  const yMin = Math.max(0, Math.min(rawMin - 0.025, 0.78));
  const yMax = Math.min(1.08, Math.max(rawMax + 0.02, 1.01));
  const x = (value) => margin.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * plotWidth;
  const y = (value) => margin.top + (1 - (value - yMin) / Math.max(yMax - yMin, 0.01)) * plotHeight;
  const selectedPoint = points.find((item) => String(item.cycle) === String(cycle));
  const sampleStep = Math.max(1, Math.ceil(points.length / 90));
  const sampledBase = points.filter((_, index) => index % sampleStep === 0 || index === points.length - 1);
  const sampled = selectedPoint && !sampledBase.some((item) => item.cycle === selectedPoint.cycle)
    ? [...sampledBase, selectedPoint].sort((left, right) => left.cycle - right.cycle)
    : sampledBase;
  const truthLine = sampled.map((item) => `${x(item.cycle).toFixed(1)},${y(item.truth).toFixed(1)}`).join(" ");
  const predLine = sampled.map((item) => `${x(item.cycle).toFixed(1)},${y(item.pred).toFixed(1)}`).join(" ");
  const intervalPoints = sampled.filter((item) => Number.isFinite(item.lower) && Number.isFinite(item.upper));
  const band = intervalPoints.length > 1
    ? [...intervalPoints.map((item) => `${x(item.cycle).toFixed(1)},${y(item.upper).toFixed(1)}`), ...intervalPoints.slice().reverse().map((item) => `${x(item.cycle).toFixed(1)},${y(item.lower).toFixed(1)}`)].join(" ")
    : "";
  const yGrid = Array.from({ length: 5 }, (_, index) => yMin + ((yMax - yMin) * index) / 4);
  const xGrid = Array.from({ length: 6 }, (_, index) => xMin + ((xMax - xMin) * index) / 5);
  const predictionTargets = sampled.map((item) => {
    const selected = selectedPoint && item.cycle === selectedPoint.cycle;
    const delta = item.pred - item.truth;
    return `<circle class="fusion-cycle-point ${selected ? "selected" : ""} ${item.rawAvailable ? "raw-backed" : "trajectory-only"}" cx="${x(item.cycle).toFixed(1)}" cy="${y(item.pred).toFixed(1)}" r="${selected ? 6 : 4}" data-audit-cycle="${item.cycle}" data-audit-asset="${esc(selectedAsset)}" data-audit-series="prediction" tabindex="0" role="button" aria-label="Cycle ${item.cycle}，预测 SOH ${(item.pred * 100).toFixed(2)}%，真值 SOH ${(item.truth * 100).toFixed(2)}%，差异 ${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} 个百分点"></circle>`;
  }).join("");
  const truthTargets = sampled.map((item) => {
    const selected = selectedPoint && item.cycle === selectedPoint.cycle;
    return `<circle class="fusion-truth-point ${selected ? "selected" : ""}" cx="${x(item.cycle).toFixed(1)}" cy="${y(item.truth).toFixed(1)}" r="${selected ? 5.5 : 7}" data-audit-cycle="${item.cycle}" data-audit-asset="${esc(selectedAsset)}" data-audit-series="truth" tabindex="-1" aria-hidden="true"></circle>`;
  }).join("");
  const selectedReadout = selectedPoint ? (() => {
    const selectedX = x(selectedPoint.cycle);
    const truthY = y(selectedPoint.truth);
    const predictionY = y(selectedPoint.pred);
    const delta = selectedPoint.pred - selectedPoint.truth;
    const readoutWidth = 146;
    const readoutHeight = 68;
    const readoutX = selectedX > width - margin.right - readoutWidth - 18
      ? selectedX - readoutWidth - 12
      : selectedX + 12;
    const readoutY = Math.max(
      margin.top + 4,
      Math.min(Math.min(truthY, predictionY) - 34, height - margin.bottom - readoutHeight - 4),
    );
    return `<g class="fusion-selected-comparison" aria-hidden="true">` +
      `<line x1="${selectedX.toFixed(1)}" y1="${truthY.toFixed(1)}" x2="${selectedX.toFixed(1)}" y2="${predictionY.toFixed(1)}"/>` +
      `<g class="fusion-point-readout" transform="translate(${readoutX.toFixed(1)} ${readoutY.toFixed(1)})">` +
        `<rect width="${readoutWidth}" height="${readoutHeight}" rx="7"/>` +
        `<text x="9" y="15" class="title">Cycle ${selectedPoint.cycle}</text>` +
        `<text x="9" y="29">真值 SOH <tspan>${(selectedPoint.truth * 100).toFixed(2)}%</tspan></text>` +
        `<text x="9" y="43">预测 SOH <tspan>${(selectedPoint.pred * 100).toFixed(2)}%</tspan></text>` +
        `<text x="9" y="57" class="delta">差异（预测−真值） <tspan>${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} pp</tspan></text>` +
      `</g>` +
    `</g>`;
  })() : "";
  const eolY = yMin <= 0.8 && yMax >= 0.8 ? y(0.8) : null;
  const selectedDetail = selectedPoint
    ? `<div class="cycle-key"><span>电芯 / Cycle</span><b>${esc(selectedAsset)} / ${selectedPoint.cycle}</b></div><div class="soh-reading truth"><span>真值 SOH</span><b>${selectedPoint.truth.toFixed(4)}</b></div><div class="soh-reading prediction"><span>预测 SOH</span><b>${selectedPoint.pred.toFixed(4)}</b></div><div class="soh-reading difference"><span>差异（预测−真值）</span><b>${(selectedPoint.pred - selectedPoint.truth >= 0 ? "+" : "")}${(selectedPoint.pred - selectedPoint.truth).toFixed(4)}</b></div><div><span>MAE 贡献</span><b>${Number.isFinite(selectedPoint.contribution) ? selectedPoint.contribution.toFixed(6) : "—"}</b></div><div><span>原始证据</span><b>${selectedPoint.rawAvailable ? "可用" : "未迁移"}</b></div><div class="record-key"><span>记录主键</span><b title="${esc(selectedPoint.recordId)}">${esc(selectedPoint.recordId || `${dataset}:${selectedAsset}:${selectedPoint.cycle}`)}</b></div><button type="button" data-audit-pin-cycle="${selectedPoint.cycle}">${(REVIEW_WORKSPACE?.evidence_packs || []).some((item) => item.state === "signed") ? "创建新案例版本以固定" : "固定当前 Cycle"}</button>`
    : `<div class="empty"><span>交互提示</span><b>点击蓝色预测点或同一横坐标的黑色真值点，查看两项 SOH 与差异</b></div>`;
  node.innerHTML =
    `<div class="fusion-trajectory-chart"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(selectedAsset)} SOH 真值与模型预测轨迹">` +
      `<rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" fill="#FFFFFF"/>` +
      yGrid.map((value) => `<line x1="${margin.left}" y1="${y(value).toFixed(1)}" x2="${width - margin.right}" y2="${y(value).toFixed(1)}" stroke="#E5EBF2"/><text x="${margin.left - 9}" y="${(y(value) + 4).toFixed(1)}" text-anchor="end">${(value * 100).toFixed(0)}</text>`).join("") +
      xGrid.map((value) => `<line x1="${x(value).toFixed(1)}" y1="${margin.top}" x2="${x(value).toFixed(1)}" y2="${height - margin.bottom}" stroke="#EEF2F6"/><text x="${x(value).toFixed(1)}" y="${height - 14}" text-anchor="middle">${Math.round(value)}</text>`).join("") +
      (eolY == null ? "" : `<line x1="${margin.left}" y1="${eolY.toFixed(1)}" x2="${width - margin.right}" y2="${eolY.toFixed(1)}" stroke="#E45A5A" stroke-dasharray="6 5"/><text x="${width - margin.right - 5}" y="${(eolY - 6).toFixed(1)}" text-anchor="end" class="eol-label">EOL 80%</text>`) +
      (band ? `<polygon points="${band}" fill="#DDF3FC" opacity=".9"/>` : "") +
      `<polyline points="${truthLine}" fill="none" stroke="#25394E" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>` +
      `<polyline points="${predLine}" fill="none" stroke="#0A94D1" stroke-width="2.8" stroke-dasharray="8 5" stroke-linecap="round" stroke-linejoin="round"/>` +
      truthTargets + predictionTargets + selectedReadout +
      `<text x="${margin.left + plotWidth / 2}" y="${height - 1}" text-anchor="middle" class="axis-label">循环数</text><text transform="translate(14 ${margin.top + plotHeight / 2}) rotate(-90)" text-anchor="middle" class="axis-label">SOH (%)</text>` +
    `</svg><div class="fusion-chart-legend"><span class="truth">标签真值</span><span class="pred">模型预测</span>${band ? `<span class="band">95% 区间</span>` : ""}<small>${points.length} 个预测记录 · ${esc(selectedAsset)}</small></div></div>` +
    `<div class="fusion-cycle-detail" aria-live="polite">${selectedDetail}</div>`;
  wireAuditTrajectoryActions(selectedAsset, dataset);
}

function wireAuditTrajectoryActions(asset, dataset) {
  const selectCycle = (cycle) => selectAuditCycle(cycle, dataset, asset);
  document.querySelectorAll("[data-audit-cycle]").forEach((point) => {
    point.onclick = () => selectCycle(point.dataset.auditCycle);
    point.onkeydown = (event) => {
      if (["Enter", " "].includes(event.key)) {
        event.preventDefault();
        selectCycle(point.dataset.auditCycle);
      }
    };
  });
  document.querySelectorAll("[data-audit-pin-cycle]").forEach((button) => {
    button.onclick = () => {
      selectCycle(button.dataset.auditPinCycle);
      queueEvaluationCasePersistence();
    };
  });
  document.querySelectorAll("[data-mode-shortcut]").forEach((button) => {
    button.onclick = () => { activeMode = button.dataset.modeShortcut; render(); };
  });
}

function renderAuditClosure(review, readiness) {
  const issues = REVIEW_WORKSPACE?.issues || [];
  const regressionSets = REVIEW_WORKSPACE?.regression_sets || [];
  const retests = REVIEW_WORKSPACE?.retests || [];
  const reconciliations = REVIEW_WORKSPACE?.reconciliations || [];
  const packs = REVIEW_WORKSPACE?.evidence_packs || [];
  const busy = Boolean(REVIEW_WORKFLOW_STATE.busy);
  const exactEvidence = Boolean(
    ACTIVE_EVALUATION_CASE?.evidence_selection?.sample_id &&
    ACTIVE_EVALUATION_CASE?.evidence_selection?.cycle_id !== null &&
    ACTIVE_EVALUATION_CASE?.evidence_selection?.cycle_id !== ""
  );

  const issueCards = issues.length ? issues.map((issue) => {
    const regressionSet = regressionSets.find((item) => item.issue_id === issue.id);
    const issueRetests = retests.filter((item) => item.issue_id === issue.id);
    const latestRetest = issueRetests[0];
    const completedRetest = issueRetests.find((item) => item.status === "completed");
    const context = issue.context || {};
    let nextAction = "";
    if (!["resolved", "closed"].includes(issue.status)) {
      if (!regressionSet) {
        nextAction = `<button type="button" data-audit-action="create-regression" data-issue-id="${esc(issue.id)}">建立回归集</button>`;
      } else if (!(regressionSet.members || []).length) {
        nextAction = `<div class="audit-batch-action"><label><span>回归范围</span><select id="audit-batch-${esc(regressionSet.id)}" aria-label="批量回归范围"><option value="2" selected>当前 ±2 Cycle · 最多 5 条</option><option value="0">仅当前 Cycle</option><option value="5">当前 ±5 Cycle · 最多 11 条</option></select></label><button type="button" data-audit-action="add-member" data-set-id="${esc(regressionSet.id)}" ${exactEvidence ? "" : "disabled"}>生成批量回归集</button></div>` +
          (!exactEvidence ? `<small class="audit-action-hint">需先在比较与证据页固定到精确 Cycle</small>` : "");
      } else if (!latestRetest) {
        const canReinfer = Boolean(context.model_artifact_id);
        nextAction = `<div class="audit-batch-ready"><span>${regressionSet.members.length} 条固定样本</span><button type="button" data-audit-action="request-retest" data-issue-id="${esc(issue.id)}" data-set-id="${esc(regressionSet.id)}">${canReinfer ? "运行批量同协议复测" : "运行冻结证据重放"}</button></div>` +
          (!canReinfer ? `<small class="audit-action-hint">缺少不可变模型产物：只重放冻结预测，不能完成复测门禁</small>` : "");
      } else if (["requested", "running"].includes(latestRetest.status)) {
        const mode = latestRetest.result?.dispatch?.execution_mode;
        nextAction = `<div class="audit-controlled-run running"><span class="audit-run-spinner" aria-hidden="true"></span><div><b>${mode === "evidence_replay" ? "冻结证据重放中" : (latestRetest.status === "running" ? "批量模型重新推理中" : "批量复测已排队")}</b><small>${regressionSet.members.length} 条样本 · ${esc(latestRetest.run_id || "控制面正在分配 run_id")} · 完成后自动回填</small></div></div>`;
      } else if (completedRetest && issue.status === "in_retest") {
        const result = completedRetest.result || {};
        nextAction = renderRetestComparison(result, completedRetest.run_id) +
          `<div class="audit-inline-form audit-resolve-form"><input id="audit-resolve-${esc(issue.id)}" type="text" value="批量同协议复测完成，差异与失败定位已自动回填" aria-label="问题闭环依据" /><button type="button" data-audit-action="resolve-issue" data-issue-id="${esc(issue.id)}">确认闭环</button></div>`;
      } else if (latestRetest?.status === "failed") {
        const replay = latestRetest.result?.execution_mode === "evidence_replay" && latestRetest.result?.replay_completed;
        nextAction = `<div class="audit-controlled-run failed"><span>!</span><div><b>${replay ? "证据重放已完成 · 复测门禁仍阻断" : "受控复测失败"}</b><small>${esc(replay ? "需绑定不可变可执行模型产物后重新推理" : (latestRetest.result?.error || "请查看审计轨迹"))}</small></div></div><button type="button" data-audit-action="request-retest" data-issue-id="${esc(issue.id)}" data-set-id="${esc(regressionSet.id)}">修复后重新发起</button>`;
      }
    }
    if (["resolved", "closed"].includes(issue.status) && completedRetest) {
      nextAction = renderRetestComparison(completedRetest.result || {}, completedRetest.run_id);
    }
    return `<article class="audit-issue-card ${esc(issue.status)}">` +
      `<div class="audit-card-head"><div><span>${esc(severityLabel(issue.severity))}</span><b>${esc(issue.title)}</b></div><em>${esc(issueStateLabel(issue.status))}</em></div>` +
      `<p>${esc(issue.description || "未填写补充说明")}</p>` +
      `<div class="audit-pin-row"><span>${esc(context.dataset || "未固定数据集")}</span><span>${esc(context.asset_id || "未固定资产")}</span><span>Cycle ${esc(context.cycle_id ?? "—")}</span></div>` +
      `<div class="audit-progress-row"><span class="${regressionSet ? "done" : ""}">回归集</span><span class="${regressionSet?.members?.length ? "done" : ""}">固定样本${regressionSet?.members?.length ? ` · ${regressionSet.members.length}` : ""}</span><span class="${completedRetest ? "done" : ""}">复测结果</span><span class="${["resolved", "closed"].includes(issue.status) ? "done" : ""}">问题闭环</span></div>` +
      `<div class="audit-card-actions">${nextAction}</div>` +
    `</article>`;
  }).join("") : `<div class="audit-empty compact"><b>当前案例没有问题单</b><span>如果结论或证据存在风险，可从这里建立闭环。</span></div>`;

  const latestReconciliation = reconciliations[0];
  const latestPack = packs[0];
  const latestPackCurrent = Boolean(
    latestPack && Number(latestPack.case_version) === Number(ACTIVE_EVALUATION_CASE?.version)
  );
  const reviewCanApprove = ACTIVE_EVALUATION_CASE?.evidence_coverage?.state === "complete" &&
    ["pass", "conditional"].includes(ACTIVE_EVALUATION_CASE?.verdict?.result);
  return `<section class="audit-section">` +
      `<div class="audit-section-head"><div><span>01</span><b>问题与复测</b><small>${issues.length} 项</small></div><details class="audit-secondary-details"><summary>记录问题</summary><div class="audit-new-issue"><input id="audit-issue-title" type="text" value="评测结论需要进入回归闭环" aria-label="问题标题" /><select id="audit-issue-severity" aria-label="问题严重度"><option value="major">主要</option><option value="critical">严重</option><option value="minor">轻微</option></select><input id="audit-issue-description" type="text" placeholder="风险或弱证据说明" aria-label="问题说明" /><button type="button" data-audit-action="create-issue">创建问题单</button></div></details></div>` +
      `<div class="audit-issue-list">${issueCards}</div>` +
    `</section>` +
    `<section class="audit-section audit-dual-grid">` +
      `<div class="audit-decision-card"><div class="audit-section-head"><div><span>02</span><b>KPI 对账</b></div><em class="${esc(latestReconciliation?.status || "pending")}">${esc(reconciliationLabel(latestReconciliation?.status))}</em></div>` +
        `<p>${latestReconciliation ? esc((latestReconciliation.reasons || []).map(reconciliationReasonLabel).join("；") || `容差 ${latestReconciliation.tolerance}`) : "评审批准后，页面主操作会自动推进服务端对账。"}</p></div>` +
      `<div class="audit-decision-card"><div class="audit-section-head"><div><span>03</span><b>人工评审</b></div><em class="${esc(review.state)}">${esc(reviewStateLabel(review.state))}</em></div>` +
        `<p>使用页面右上角主操作推进评审；仅在需要修改评审人、说明或驳回时展开。</p>` +
        `<details class="audit-secondary-details audit-review-details"><summary>评审信息</summary><div class="audit-review-form"><input id="audit-reviewer" type="text" value="local-reviewer" aria-label="评审人" /><input id="audit-review-note" type="text" placeholder="评审说明（决定时必填）" aria-label="评审说明" />${renderReviewActions(review.state, reviewCanApprove)}</div></details></div>` +
    `</section>` +
    `<section class="audit-section audit-pack-section">` +
      `<div class="audit-section-head"><div><span>04</span><b>EvidencePack</b><small>${packs.length} 个版本</small></div><em class="${esc(latestPackCurrent ? (latestPack?.state || "pending") : "stale")}">${esc(latestPack ? (latestPackCurrent ? packStateLabel(latestPack.state) : "案例已变更 · 需重建") : "尚未生成")}</em></div>` +
      (latestPack ? `<div class="audit-pack-digest"><span>${esc(latestPack.id)}</span><code>${esc(latestPack.digest)}</code><small>固定案例 v${esc(latestPack.case_version)}</small></div>` : "") +
      `<p class="audit-stage-guidance">${latestPack ? "主操作会在门禁满足后签署当前草稿。" : "完成评审与 KPI 对账后，主操作会生成不可变草稿。"}</p>` +
      (latestPack?.state === "draft" ? `<details class="audit-secondary-details"><summary>签署信息</summary><div class="audit-pack-actions"><input id="audit-signer" type="text" value="local-reviewer" aria-label="签署人" /><input id="audit-sign-reason" type="text" value="证据、评审与 KPI 对账门禁均已通过" aria-label="签署依据" /></div></details>` : "") +
    `</section>` +
    (busy ? `<div class="audit-busy" role="status">正在提交，请稍候…</div>` : "");
}

function renderRetestComparison(result, runId) {
  const view = retestComparisonView(result);
  if (!view.available) {
    return `<div class="audit-result-summary"><div><span>受控运行</span><b>${esc(runId || "—")}</b></div><div><span>MAE / PICP</span><b>${evidenceNumber(result.metrics?.mae, 4)} / ${evidenceNumber(result.metrics?.picp, 3)}</b></div><div><span>复测判定</span><b class="${result.verdict === "pass" ? "pass" : "fail"}">${esc(auditVerdictLabel(result.verdict))}</b></div></div>`;
  }
  const changeTone = view.regressedCount > 0 ? "fail" : (view.improvedCount > 0 ? "pass" : "stable");
  const criteria = view.localization.failedCriteria.length
    ? view.localization.failedCriteria.map(auditCriterionLabel).join(" / ")
    : "无失败判据";
  const tags = [
    view.localization.datasets.join(" / ") || "未记录数据集",
    shortAuditRef(view.localization.splitAssignmentId || view.localization.protocolRef),
    shortAuditRef(view.localization.modelArtifactId),
    criteria,
  ];
  return `<div class="audit-retest-comparison ${changeTone}">` +
    `<div class="audit-retest-head"><div><span>批量同集对比 · ${view.memberCount} 条</span><b title="${esc(runId || "")}">${esc(shortAuditRef(runId || "—"))}</b></div><em class="${result.verdict === "pass" ? "pass" : "fail"}">${esc(auditVerdictLabel(result.verdict))}</em></div>` +
    `<div class="audit-comparison-kpis">` +
      `<div><span>MAE 基线 → 复测</span><b>${evidenceNumber(view.baselineMae, 4)} → ${evidenceNumber(view.retestMae, 4)}</b><small class="${auditDeltaTone(view.maeDelta, false)}">Δ ${auditDeltaNumber(view.maeDelta, 4)}</small></div>` +
      `<div><span>PICP 基线 → 复测</span><b>${evidenceNumber(view.baselinePicp, 3)} → ${evidenceNumber(view.retestPicp, 3)}</b><small class="${auditDeltaTone(view.picpDelta, true)}">Δ ${auditDeltaNumber(view.picpDelta, 3)}</small></div>` +
      `<div><span>逐点变化</span><b>${view.improvedCount} 改善 · ${view.unchangedCount} 持平 · ${view.regressedCount} 退化</b><small>${view.localization.thresholdExceedingCount} 条触及失败条件</small></div>` +
    `</div>` +
    `<div class="audit-localization-tags">${tags.map((item) => `<span title="${esc(item)}">${esc(item)}</span>`).join("")}</div>` +
    (view.topRecords.length ? `<div class="audit-delta-table"><div><span>Cycle</span><span>基线误差</span><span>复测误差</span><span>Δ误差</span></div>${view.topRecords.map((item) => `<div class="${esc(item.change)}"><b>${esc(item.cycleId ?? item.sampleId)}</b><span>${evidenceNumber(item.baselineError, 4)}</span><span>${evidenceNumber(item.retestError, 4)}</span><strong>${auditDeltaNumber(item.errorDelta, 4)}</strong></div>`).join("")}</div>` : "") +
  `</div>`;
}

function auditDeltaNumber(value, digits) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  return `${numeric > 0 ? "+" : ""}${numeric.toFixed(digits)}`;
}

function auditDeltaTone(value, higherIsBetter) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || Math.abs(numeric) <= 1e-12) return "stable";
  const improved = higherIsBetter ? numeric > 0 : numeric < 0;
  return improved ? "pass" : "fail";
}

function auditCriterionLabel(value) {
  return ({ mae_max: "MAE 超限", picp_min: "PICP 不足" })[value] || value;
}

function renderReviewActions(state, canApprove) {
  if (state === "in_review") return `<button type="button" data-audit-action="review" data-review-target="rejected">驳回评审</button>`;
  return `<span class="audit-inline-guidance">${canApprove ? "主操作可继续推进" : "等待证据门禁满足"}</span>`;
}

function renderAuditTrail() {
  const events = REVIEW_WORKSPACE?.events || [];
  if (!events.length) return `<div class="audit-empty"><b>暂无闭环事件</b><span>创建问题单后，事件摘要会写入内容寻址存储。</span></div>`;
  return `<div class="audit-timeline">${events.map((event) =>
    `<article><span></span><div><b>${esc(auditEventLabel(event.event_type))}</b><small>${esc(event.actor || "system")} · ${esc(formatAuditTime(event.created_at))}</small><code title="${esc(event.payload_digest || "")}">${esc((event.payload_digest || "").slice(0, 24))}</code></div></article>`
  ).join("")}</div>`;
}

function focusAuditSection(id) {
  const node = $(id);
  if (!node) return;
  node.scrollIntoView({ behavior: "smooth", block: "start" });
  node.classList.remove("audit-focus-pulse");
  requestAnimationFrame(() => node.classList.add("audit-focus-pulse"));
  setTimeout(() => node.classList.remove("audit-focus-pulse"), 1400);
}

function automaticAuditIssueText() {
  const verdict = ACTIVE_EVALUATION_CASE?.verdict || {};
  const reasons = Array.isArray(verdict.reasons) ? verdict.reasons : [];
  const reasonText = reasons.join("；") || "当前判定或证据门禁未满足";
  return {
    title: verdict.result === "fail" ? "当前候选未通过评测门禁" : "当前证据需要受控闭环",
    description: `自动记录自服务端判定：${reasonText}`,
  };
}

async function finalizeFailedEvaluationCase() {
  if (!ACTIVE_EVALUATION_CASE?.id) return;
  const confirmed = window.confirm("将当前案例固定为“不通过”并结束。该操作会写入审计轨迹，是否继续？");
  if (!confirmed) return;
  REVIEW_WORKFLOW_STATE = { status: "loading", message: "正在固定不通过结论", busy: "finalize-failed" };
  renderAuditWorkspace();
  try {
    let current = ACTIVE_EVALUATION_CASE;
    if (current.review?.state === "in_review") {
      const rejected = await postJson(`/api/v2/cases/${encodeURIComponent(current.id)}/review`, {
        ...evaluationCaseMutationEnvelope(current),
        state: "rejected",
        reviewer: $("audit-reviewer")?.value || "local-reviewer",
        note: $("audit-review-note")?.value || "当前候选未通过服务端判据，确认结束为不通过",
      });
      current = rejected.case || rejected;
    }
    const failed = await postJson(`/api/v2/cases/${encodeURIComponent(current.id)}/lifecycle`, {
      ...evaluationCaseMutationEnvelope(current),
      lifecycle_state: "failed",
      actor: "local-reviewer",
      reason: "不通过结论已复核，结束当前评测案例",
    });
    ACTIVE_EVALUATION_CASE = failed.case || failed;
    await loadReviewWorkspace({ force: true, silent: true });
    REVIEW_WORKFLOW_STATE = { status: "ready", message: "案例已按不通过结论结束", busy: "" };
  } catch (err) {
    await refreshActiveEvaluationCase();
    await loadReviewWorkspace({ force: true, silent: true });
    REVIEW_WORKFLOW_STATE = { status: "error", message: `结束案例失败：${err.message}`, busy: "" };
  }
  renderAuditWorkspace();
  renderGlobalCaseContext();
}

function wireAuditActions() {
  document.querySelectorAll("[data-audit-action]").forEach((button) => {
    button.onclick = async () => {
      if (button.disabled || !ACTIVE_EVALUATION_CASE) return;
      const version = ACTIVE_EVALUATION_CASE.version;
      const action = button.dataset.auditAction;
      if (action === "go-inference") {
        enterInferenceForSavedCombination();
      } else if (action === "focus-closure") {
        focusAuditSection("audit-closure-body");
      } else if (action === "focus-readiness") {
        focusAuditSection("audit-readiness-body");
      } else if (action === "finalize-failed") {
        await finalizeFailedEvaluationCase();
      } else if (action === "archive-case") {
        await runReviewMutation(action, `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/lifecycle`, {
          ...evaluationCaseMutationEnvelope(),
          lifecycle_state: "archived",
          actor: "local-ui",
          reason: "归档未进入正式评测流程的草稿或失败案例",
        }, { updatesCase: true });
      } else if (action === "create-issue") {
        const automatic = button.dataset.auditAuto === "true";
        const issueText = automaticAuditIssueText();
        await runReviewMutation(action, "/api/v2/review-actions/issues", {
          case_id: ACTIVE_EVALUATION_CASE.id,
          expected_case_version: version,
          title: automatic ? issueText.title : $("audit-issue-title")?.value,
          severity: automatic ? "major" : $("audit-issue-severity")?.value,
          description: automatic ? issueText.description : $("audit-issue-description")?.value,
          actor: "local-reviewer",
        });
      } else if (action === "create-regression") {
        await runReviewMutation(action, "/api/v2/regression-sets", {
          issue_id: button.dataset.issueId,
          expected_case_version: version,
          actor: "local-reviewer",
        });
      } else if (action === "add-member") {
        const batchRadius = Number.parseInt(
          button.dataset.neighborRadius ?? $(`audit-batch-${button.dataset.setId}`)?.value ?? "0",
          10,
        );
        await runReviewMutation(action, `/api/v2/regression-sets/${encodeURIComponent(button.dataset.setId)}/members`, {
          expected_case_version: version,
          neighbor_radius: Number.isInteger(batchRadius) ? batchRadius : 0,
          actor: "local-reviewer",
        });
      } else if (action === "request-retest") {
        await runReviewMutation(action, `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/retests`, {
          expected_case_version: version,
          issue_id: button.dataset.issueId,
          regression_set_id: button.dataset.setId,
          actor: "local-reviewer",
        });
      } else if (action === "resolve-issue") {
        const automatic = button.dataset.auditAuto === "true";
        await runReviewMutation(action, `/api/v2/review-actions/issues/${encodeURIComponent(button.dataset.issueId)}/status`, {
          expected_case_version: version,
          status: "resolved",
          reason: automatic
            ? "受控复测结果已回填并完成复核"
            : $(`audit-resolve-${button.dataset.issueId}`)?.value,
          actor: "local-reviewer",
        });
      } else if (action === "reconcile") {
        await runReviewMutation(action, `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/reconcile`, {
          expected_case_version: version,
          tolerance: 0.000001,
          actor: "local-reviewer",
        });
      } else if (action === "review") {
        const reviewer = $("audit-reviewer")?.value || "local-reviewer";
        const note = $("audit-review-note")?.value || (
          button.dataset.reviewTarget === "approved"
            ? "证据与受控复测闭环已复核"
            : (button.dataset.reviewTarget === "rejected" ? "评审驳回，需补齐证据" : "提交评审")
        );
        await runReviewMutation(action, `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/review`, {
          ...evaluationCaseMutationEnvelope(),
          state: button.dataset.reviewTarget,
          reviewer,
          note,
        }, { updatesCase: true });
      } else if (action === "create-pack") {
        await runReviewMutation(action, `/api/v2/cases/${encodeURIComponent(ACTIVE_EVALUATION_CASE.id)}/evidence-packs`, {
          expected_case_version: version,
          actor: "local-reviewer",
        });
      } else if (action === "sign-pack") {
        await runReviewMutation(action, `/api/v2/evidence-packs/${encodeURIComponent(button.dataset.packId)}/sign`, {
          expected_case_version: version,
          signer: $("audit-signer")?.value || "local-reviewer",
          reason: $("audit-sign-reason")?.value || "证据、评审与 KPI 对账门禁均已通过",
        });
      }
    };
  });
}

function reviewStateLabel(state) {
  return ({ not_started: "未发起", in_review: "评审中", approved: "已批准", rejected: "已驳回" })[state] || state;
}

function issueStateLabel(state) {
  return ({ open: "待处理", in_retest: "复测中", resolved: "已闭环", closed: "已关闭" })[state] || state;
}

function severityLabel(severity) {
  return ({ minor: "轻微", major: "主要", critical: "严重" })[severity] || severity;
}

function reconciliationLabel(status) {
  return ({ match: "一致", mismatch: "超出容差", unavailable: "不可对账", pending: "待运行" })[status || "pending"];
}

function reconciliationReasonLabel(reason) {
  const labels = {
    "legacy dashboard scenario is unavailable": "旧看板场景数据不可用",
    "selected candidate is unavailable": "当前候选模型不可用",
    "legacy dashboard model identity does not match selected candidate": "旧看板模型标识与当前候选不一致",
    "canonical verdict has no observed KPI values": "案例事实表尚无可对账 KPI",
    "canonical and legacy sources have no comparable KPI fields": "新旧数据源没有同名可比 KPI",
    "one or more KPI deltas exceed tolerance": "至少一项 KPI 差值超出容差",
  };
  return labels[reason] || reason;
}

function packStateLabel(state) {
  return ({ draft: "不可变草稿", signed: "已签署" })[state] || state;
}

function signingBlockerLabel(blocker) {
  const labels = {
    "only measured cases can be signed": "仅实测案例允许签署",
    "execution is not completed": "评测执行尚未完成",
    "required evidence is incomplete": "必需证据不完整",
    "verdict is not pass or conditional": "判定未达到通过或有条件通过",
    "review is not approved": "人工评审尚未批准",
    "open review issues remain": "仍有未闭环问题单",
    "retest is not completed": "同协议复测尚未完成",
    "canonical and legacy KPI reconciliation has not matched the current case version": "当前案例版本与旧看板 KPI 尚未对齐",
  };
  return labels[blocker] || blocker;
}

function auditEventLabel(eventType) {
  const labels = {
    "case.created": "评测案例已创建",
    "case.identity_updated": "案例执行血缘已更新",
    "case.legacy_migrated": "旧案例已迁移",
    "evidence.pinned": "证据上下文已固定",
    "criterion.updated": "判据阈值已更新并重新判定",
    "review.in_review": "人工评审已发起",
    "review.approved": "人工评审已批准",
    "review.rejected": "人工评审已驳回",
    "issue.created": "问题单已创建",
    "issue.in_retest": "问题进入复测",
    "issue.resolved": "问题已闭环",
    "issue.closed": "问题已关闭",
    "regression_set.created": "回归集已建立",
    "regression_set.member_added": "固定证据已加入回归集",
    "regression_set.members_added": "批量固定证据已加入回归集",
    "retest.requested": "已申请同协议复测",
    "retest.running": "受控同协议复测运行中",
    "retest.completed": "复测结果已自动回填",
    "retest.failed": "复测失败",
    "reconciliation.match": "KPI 对账一致",
    "reconciliation.mismatch": "KPI 对账超出容差",
    "reconciliation.unavailable": "KPI 无法对账",
    "evidence_pack.created": "EvidencePack 草稿已生成",
    "evidence_pack.signed": "EvidencePack 已签署",
  };
  return labels[eventType] || eventType;
}

function formatAuditTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
}

function renderModeTabs() {
  document.body.classList.toggle("audit-focus-mode", activeMode === "audit");
  const navigationMode = activeMode;
  document.querySelectorAll("[data-mode]").forEach((button) => {
    const selected = button.dataset.mode === navigationMode;
    button.setAttribute("aria-selected", String(selected));
    button.onclick = () => activateMode(button.dataset.mode);
  });
  document.querySelectorAll("[data-side-mode]").forEach((button) => {
    const selected = button.dataset.sideMode === navigationMode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-current", selected ? "page" : "false");
    button.onclick = () => activateMode(button.dataset.sideMode);
  });
  document.querySelectorAll("[data-mode-shortcut]").forEach((button) => {
    button.onclick = () => activateMode(button.dataset.modeShortcut);
  });
  renderPipelineNavigation(navigationMode);
  document.querySelectorAll("[data-mode-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.modePanel !== activeMode;
  });
}

function renderPipelineNavigation(navigationMode = activeMode) {
  const progress = casePipelineProgress({
    caseDocument: ACTIVE_EVALUATION_CASE || {},
    reviewWorkspace: REVIEW_WORKSPACE || {},
    trainingJob: currentTrainingJob(),
    inferenceJob: currentInferenceJob(),
  });
  progress.stages.forEach((stage) => {
    const button = document.querySelector(`[data-pipeline-stage="${stage.id}"]`);
    const status = document.querySelector(`[data-pipeline-status="${stage.id}"]`);
    if (button) {
      button.dataset.stageState = stage.state;
      button.title = stage.detail;
    }
    if (status) {
      const prefix = navigationMode === stage.id ? "当前 · " : "";
      const label = stage.state === "complete"
        ? "已完成"
        : (stage.state === "active"
          ? (stage.id === "audit" ? "发布待闭环" : "进行中")
          : (stage.state === "blocked" ? "需处理" : "待开始"));
      status.textContent = `${prefix}${label}`;
    }
  });
  const tools = document.querySelector(".sidenav-tools");
  if (tools && ["datasets", "diagnostics", "models", "recommendations"].includes(navigationMode)) tools.open = true;
}

function renderTaskHistoryDrawer() {
  const drawer = $("task-history-drawer");
  const backdrop = $("task-history-backdrop");
  if (!drawer || !backdrop) return;
  drawer.hidden = !taskHistoryOpen;
  backdrop.hidden = !taskHistoryOpen;
  drawer.setAttribute("aria-hidden", taskHistoryOpen ? "false" : "true");
  wireTaskHistoryShell();
  if (!taskHistoryOpen) return;

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

function wireTaskHistoryShell() {
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
}

function wireTaskHistoryDrawer() {
  wireTaskHistoryShell();
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

function wireAcceptanceControls() {
  const options = acceptancePresetOptions();
  if (!options.some((item) => item.id === acceptancePreset)) acceptancePreset = "strict";
  const select = $("acceptance-preset");
  if (select) {
    select.value = acceptancePreset;
    select.onchange = () => {
      acceptancePreset = select.value;
      localStorage.setItem("sohbench.acceptancePreset", acceptancePreset);
      markEvidenceViewState();
      render();
    };
  }
  const open = $("open-acceptance-detail");
  if (open) {
    open.onclick = () => {
      acceptanceDetailOpen = true;
      renderAcceptanceDetailDrawer();
    };
  }
}

function renderAcceptanceDetailDrawer(rows = null) {
  const drawer = $("acceptance-detail-drawer");
  const backdrop = $("acceptance-detail-backdrop");
  if (!drawer || !backdrop) return;
  const visible = acceptanceDetailOpen && activeMode !== "inference";
  drawer.hidden = !visible;
  backdrop.hidden = !visible;
  drawer.setAttribute("aria-hidden", visible ? "false" : "true");
  wireAcceptanceDetailShell();
  if (!visible) return;

  const detail = acceptanceDetailSummary({
    rows: rows || scoped?.entries || DATA?.leaderboard || [],
    datasetRows: DATA?.datasetLeaderboard || [],
    preset: acceptancePreset,
  });
  const body = $("acceptance-detail-body");
  if (!body) return;
  const thresholds = detail.thresholds;
  const failingRows = detail.failingRows.slice(0, 20);
  body.innerHTML =
    `<div class="acceptance-detail-summary">` +
      metric("当前口径", detail.preset.label, detail.preset.description) +
      metric("聚合通过", detail.aggregate.label, `${detail.aggregate.fail} 个子项未通过`) +
      metric("数据集级通过", detail.dataset.label, "按逐数据集 leaderboard 复核") +
    `</div>` +
    `<div class="acceptance-thresholds" aria-label="验收阈值">` +
      `<span>MAE ≤ ${esc(numberText(thresholds.maeMax))}</span>` +
      `<span>PICP ≥ ${esc(numberText(thresholds.picpMin))}</span>` +
      `<span>单调违例 ≤ ${esc(numberText(thresholds.monotonicityMax))}</span>` +
      `<span>越界 ≤ ${esc(numberText(thresholds.rangeMax))}</span>` +
    `</div>` +
    `<div class="acceptance-note">严格口径适合论文级验收；工程可用和探索口径用于筛选候选模型，不改变原始测评数据。</div>` +
    `<div class="acceptance-table-wrap">` +
      `<table class="acceptance-table">` +
        `<thead><tr><th>组合</th><th>MAE</th><th>PICP</th><th>单调</th><th>判定</th><th>原因</th></tr></thead>` +
        `<tbody>` +
          (failingRows.length ? failingRows.map(renderAcceptanceFailureRow).join("") :
            `<tr><td colspan="6" class="acceptance-empty">当前口径下聚合组合全部通过。</td></tr>`) +
        `</tbody>` +
      `</table>` +
    `</div>`;
}

function wireAcceptanceDetailShell() {
  const close = $("close-acceptance-detail");
  if (close) close.onclick = closeAcceptanceDetailDrawer;
  const backdrop = $("acceptance-detail-backdrop");
  if (backdrop) backdrop.onclick = closeAcceptanceDetailDrawer;
}

function closeAcceptanceDetailDrawer() {
  acceptanceDetailOpen = false;
  renderAcceptanceDetailDrawer();
}

function renderBatteryAssetDrawer() {
  const drawer = $("battery-asset-drawer");
  const backdrop = $("battery-asset-backdrop");
  if (!drawer || !backdrop) return;
  drawer.hidden = !batteryAssetOpen;
  backdrop.hidden = !batteryAssetOpen;
  drawer.setAttribute("aria-hidden", batteryAssetOpen ? "false" : "true");
  wireBatteryAssetShell();
  if (!batteryAssetOpen) return;

  const detail = batteryVisualAssetDetail({
    ...(DATA?.batteryDiagnostics || {}),
    profileManifest: DATA?.profileManifest || {},
  });
  const curvePreview = batteryCurvePreviewView(DATA?.batteryDiagnostics || {}, activeBatteryCurveDataset);
  const body = $("battery-asset-body");
  if (!body) return;
  body.innerHTML = renderBatteryAssetSections(detail, curvePreview);
  wireBatteryCurvePreview();
}

function renderBatteryAssetSections(detail, curvePreview = { available: false }) {
  const comparisons = detail.comparisons.length
    ? detail.comparisons.map((item) =>
      `<tr><td><b>${esc(item.title)}</b><small>${esc(item.scope)}</small></td><td>${esc(item.maeLabel || "暂无")}</td><td>${esc(item.rmseLabel || "暂无")}</td><td>${esc(item.r2Label || "暂无")}</td></tr>`
    ).join("")
    : `<tr><td colspan="4" class="acceptance-empty">暂无模型对比资产。</td></tr>`;
  const featureRows = detail.featureWindows.length
    ? detail.featureWindows.map((item) =>
      `<tr><td>${esc(item.dataset)}</td><td>${esc(item.windowLabel)}</td><td>${item.rows.toLocaleString("en-US")}</td><td>${item.curves.toLocaleString("en-US")}</td></tr>`
    ).join("")
    : `<tr><td colspan="4" class="acceptance-empty">暂无特征窗口资产。</td></tr>`;
  const curveRows = detail.curves.length
    ? detail.curves.slice(0, 12).map((item) =>
      `<tr><td>${esc(item.dataset)}</td><td><b>${esc(item.filename)}</b><small>${esc(item.source)}</small></td><td>${esc(item.sizeLabel)}</td><td>${esc(item.copiedLabel)}</td></tr>`
    ).join("")
    : `<tr><td colspan="4" class="acceptance-empty">暂无曲线索引。</td></tr>`;
  return `<section class="battery-asset-section">` +
      `<h3>算法说明</h3>` +
      `<div class="battery-asset-cards">` +
        artifactMetric("模型族", detail.algorithm.modelFamily || "暂无", detail.algorithm.title, "model") +
        artifactMetric("任务", detail.algorithm.task || "暂无", detail.algorithm.sourceFile || "未记录来源", "dataset") +
        artifactMetric("特征", detail.algorithm.features.length.toString(), detail.algorithm.featuresLabel, "job") +
        artifactMetric("禁用项", detail.algorithm.forbidden.length.toString(), detail.algorithm.forbiddenLabel, "job") +
      `</div>` +
    `</section>` +
    `<section class="battery-asset-section">` +
      `<h3>模型对比</h3>` +
      `<div class="battery-asset-table-wrap"><table class="battery-asset-table"><thead><tr><th>对比</th><th>MAE</th><th>RMSE</th><th>R2</th></tr></thead><tbody>${comparisons}</tbody></table></div>` +
    `</section>` +
    `<section class="battery-asset-section">` +
      `<h3>特征窗口</h3>` +
      `<div class="battery-asset-table-wrap"><table class="battery-asset-table"><thead><tr><th>数据集</th><th>电压窗</th><th>行数</th><th>曲线</th></tr></thead><tbody>${featureRows}</tbody></table></div>` +
    `</section>` +
    `<section class="battery-asset-section">` +
      `<h3>曲线索引</h3>` +
      `<div class="acceptance-note">当前记录曲线文件索引和大小；曲线本体未内嵌时，后续应按数据集懒加载。</div>` +
      `<div class="battery-asset-table-wrap"><table class="battery-asset-table"><thead><tr><th>数据集</th><th>文件</th><th>大小</th><th>状态</th></tr></thead><tbody>${curveRows}</tbody></table></div>` +
    `</section>` +
    renderBatteryCurvePreview(curvePreview);
}

function renderBatteryCurvePreview(preview) {
  if (!preview?.available) return "";
  const loadedState = batteryCurveLoadState.dataset === preview.selectedDataset ? batteryCurveLoadState : null;
  const loaded = loadedState?.status === "loaded" ? loadedState : null;
  const loadBusy = loadedState?.status === "loading";
  const curve = loaded?.curve || preview.curves[0];
  const options = preview.options.map((item) =>
    `<option value="${esc(item.value)}"${item.value === preview.selectedDataset ? " selected" : ""}>${esc(item.label)} · ${esc(item.kindLabel)}</option>`
  ).join("");
  const series = curve?.series || [];
  const legend = series.slice(0, 3).map((item, index) =>
    `<span><i style="background:${curveColor(index)}"></i>${esc(item.label)} <small>${esc(numberText(item.min))} - ${esc(numberText(item.max))}</small></span>`
  ).join("");
  const loadedLabel = loaded
    ? `完整曲线 ${loaded.returnedCurves.toLocaleString("en-US")} / ${loaded.totalCurves.toLocaleString("en-US")}`
    : "导入抽样预览";
  const loadMessage = loadedState?.message || "在线前后端模式可按当前数据集懒加载完整曲线。";
  return `<section class="battery-asset-section battery-curve-preview">` +
    `<div class="battery-curve-head">` +
      `<h3>曲线预览</h3>` +
      `<label>数据集 <select data-battery-curve-dataset>${options}</select></label>` +
    `</div>` +
    `<div class="battery-curve-meta">` +
      `<span>${esc(preview.kindLabel)}</span>` +
      `<span>${esc(preview.totalCurvesLabel)}</span>` +
      `<span>${esc(curve?.id || "暂无曲线")}</span>` +
      `<span>${esc(loadedLabel)}</span>` +
    `</div>` +
    `<div class="battery-curve-actions">` +
      `<button type="button" data-battery-curve-load${loadBusy ? " disabled" : ""}>${loadBusy ? "加载中..." : "加载完整曲线"}</button>` +
      `<span>${esc(loadMessage)}</span>` +
    `</div>` +
    `<div class="battery-curve-chart">${renderBatteryCurveSvg(curve)}</div>` +
    `<div class="battery-curve-legend">${legend}</div>` +
    `<small class="battery-curve-source">${esc(preview.sourceFile || "预览数据来自导入时抽样")}</small>` +
  `</section>`;
}

function renderBatteryCurveSvg(curve) {
  const series = (curve?.series || []).slice(0, 3);
  const allPoints = series.flatMap((item) => item.points || []);
  if (!series.length || !allPoints.length) {
    return `<div class="acceptance-empty">暂无可绘制曲线预览。</div>`;
  }
  const width = 520;
  const height = 180;
  const pad = { left: 34, right: 12, top: 14, bottom: 26 };
  const xs = allPoints.map((point) => point[0]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const xRange = maxX - minX || 1;
  const x = (value) => pad.left + ((value - minX) / xRange) * (width - pad.left - pad.right);
  const lines = series.map((item, index) => {
    const ys = item.points.map((point) => point[1]);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const yRange = maxY - minY || 1;
    const y = (value) => pad.top + (1 - ((value - minY) / yRange)) * (height - pad.top - pad.bottom);
    const points = item.points.map((point) => `${x(point[0]).toFixed(1)},${y(point[1]).toFixed(1)}`).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${curveColor(index)}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(curve?.id || "曲线预览")}">` +
    `<rect x="0" y="0" width="${width}" height="${height}" rx="8" fill="#fff"/>` +
    `<path d="M${pad.left} ${pad.top}V${height - pad.bottom}H${width - pad.right}" fill="none" stroke="#CBD5E1" stroke-width="1"/>` +
    `<path d="M${pad.left} ${height - pad.bottom - 45}H${width - pad.right}M${pad.left} ${height - pad.bottom - 90}H${width - pad.right}" stroke="#E5EAF2" stroke-width="1"/>` +
    lines +
    `<text x="${pad.left}" y="${height - 7}" fill="#64748B" font-size="10">${esc(curve?.xKey || "x")}</text>` +
    `<text x="${width - pad.right}" y="${height - 7}" text-anchor="end" fill="#64748B" font-size="10">${esc(numberText(maxX))}</text>` +
  `</svg>`;
}

function curveColor(index) {
  return ["#3B5BDB", "#2F9E44", "#F08C00"][index % 3];
}

function wireBatteryCurvePreview() {
  document.querySelectorAll("[data-battery-curve-dataset]").forEach((select) => {
    select.onchange = () => {
      activeBatteryCurveDataset = select.value;
      if (batteryCurveLoadState.dataset !== activeBatteryCurveDataset) {
        batteryCurveLoadState = {
          dataset: activeBatteryCurveDataset,
          status: "idle",
          message: "",
          curve: null,
          totalCurves: 0,
          returnedCurves: 0,
        };
      }
      renderBatteryDiagnosticsPanel();
      renderBatteryAssetDrawer();
    };
  });
  document.querySelectorAll("[data-battery-curve-load]").forEach((button) => {
    button.setAttribute("data-battery-curve-wired", "true");
    button.onclick = handleBatteryCurveButtonClick;
  });
}

function handleBatteryCurveButtonClick(event) {
  if (event?.currentTarget?.disabled) return;
  event?.preventDefault?.();
  event?.stopPropagation?.();
  loadFullBatteryCurveDataset();
}

function handleBatteryCurveDocumentClick(event) {
  if (event.defaultPrevented) return;
  const target = event.target?.closest ? event.target : event.target?.parentElement;
  const button = target?.closest?.("[data-battery-curve-load]");
  if (!button || button.disabled) return;
  event.preventDefault();
  loadFullBatteryCurveDataset();
}

async function loadFullBatteryCurveDataset() {
  const dataset = activeBatteryCurveDataset || document.querySelector("[data-battery-curve-dataset]")?.value || "";
  if (!dataset) return;
  batteryCurveLoadState = {
    dataset,
    status: "loading",
    message: "正在从后端读取完整曲线...",
    curve: null,
    totalCurves: 0,
    returnedCurves: 0,
  };
  renderBatteryDiagnosticsPanel();
  try {
    const params = new URLSearchParams({
      profile: activeProfileOption?.name || activeProfile || "battery-soh-diagnostics",
      dataset,
      limit: "5",
    });
    const response = await fetch(`${API_BASE}/api/battery-curves?${params.toString()}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const curve = curveFromFullBatteryPayload(payload);
    if (!curve) throw new Error("未找到可绘制曲线");
    batteryCurveLoadState = {
      dataset,
      status: "loaded",
      message: payload.truncated ? "已加载前 5 条完整曲线用于快速查看。" : "完整曲线已加载。",
      curve,
      totalCurves: Number(payload.total_curves) || 0,
      returnedCurves: Number(payload.returned_curves) || 0,
    };
  } catch (error) {
    batteryCurveLoadState = {
      dataset,
      status: "failed",
      message: `完整曲线加载失败：${error.message}`,
      curve: null,
      totalCurves: 0,
      returnedCurves: 0,
    };
  }
  renderBatteryDiagnosticsPanel();
  renderBatteryAssetDrawer();
}

function curveFromFullBatteryPayload(payload) {
  const curves = payload?.curves && typeof payload.curves === "object" ? Object.entries(payload.curves) : [];
  const first = curves.find(([, value]) => value && typeof value === "object");
  if (!first) return null;
  const [id, raw] = first;
  const essSeries = seriesFromEssCurve(raw);
  if (essSeries.length) {
    return { id, xKey: "sample", series: essSeries };
  }
  const tValues = numericList(raw.t_s || raw.time_s || raw.time || raw.cycle);
  const series = [
    seriesFromRawValues("voltage_v", "电压", tValues, raw.voltage_v || raw.voltage),
    seriesFromRawValues("current_a", "电流", tValues, raw.current_a || raw.current),
    seriesFromRawValues("temperature_c", "温度", tValues, raw.temperature_c || raw.temperature),
  ].filter(Boolean);
  return series.length ? { id, xKey: "t_s", series } : null;
}

function seriesFromEssCurve(raw) {
  const dev = raw?.dev_mv && typeof raw.dev_mv === "object" ? raw.dev_mv : null;
  if (!dev) return [];
  const xValues = numericList(raw.sample);
  return Object.entries(dev).slice(0, 3).map(([key, values]) =>
    seriesFromRawValues(key, key, xValues, values)
  ).filter(Boolean);
}

function seriesFromRawValues(key, label, xValues, values) {
  const yValues = numericList(values);
  const xs = xValues.length ? xValues : yValues.map((_, index) => index);
  const limit = Math.min(xs.length, yValues.length);
  if (!limit) return null;
  const step = Math.max(1, Math.ceil(limit / 500));
  const points = [];
  for (let index = 0; index < limit; index += step) {
    points.push([xs[index], yValues[index]]);
  }
  const ys = points.map((point) => point[1]);
  return {
    key,
    label,
    min: Math.min(...ys),
    max: Math.max(...ys),
    pointCount: limit,
    points,
  };
}

function numericList(values) {
  return Array.isArray(values) ? values.map(Number).filter(Number.isFinite) : [];
}

function wireBatteryAssetShell() {
  const close = $("close-battery-asset");
  if (close) close.onclick = closeBatteryAssetDrawer;
  const backdrop = $("battery-asset-backdrop");
  if (backdrop) backdrop.onclick = closeBatteryAssetDrawer;
}

function closeBatteryAssetDrawer() {
  batteryAssetOpen = false;
  renderBatteryAssetDrawer();
}

function renderAcceptanceFailureRow(row) {
  return `<tr>` +
    `<td><b>${esc(row.name)}</b><small>${esc(row.scope || row.protocol || "aggregate")}</small></td>` +
    `<td>${esc(numberText(row.metrics.mae))}</td>` +
    `<td>${esc(numberText(row.metrics.picp))}</td>` +
    `<td>${esc(numberText(row.metrics.monotonicity))}</td>` +
    `<td><div class="acceptance-verdicts">${renderAcceptanceVerdict(row, "generalization", "泛化")}${renderAcceptanceVerdict(row, "calibration", "校准")}${renderAcceptanceVerdict(row, "physical_consistency", "物理")}</div></td>` +
    `<td>${esc(row.reasons.join("；") || "已通过")}</td>` +
  `</tr>`;
}

function renderAcceptanceVerdict(row, key, label) {
  const value = row.verdicts?.[key];
  const cls = value === true ? "pass" : (value === false ? "fail" : "na");
  return `<span class="acceptance-verdict ${cls}">${esc(label)}</span>`;
}

function numberText(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "暂无";
  return numeric.toFixed(Math.abs(numeric) >= 10 ? 2 : 4).replace(/0+$/, "").replace(/\.$/, "");
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

function currentPipelineModelScope(models) {
  return pipelineModelGroups({
    trainingJob: currentTrainingJob() || {},
    inferenceJob: currentInferenceJob() || {},
    models,
  });
}

function renderInferenceModelScope(scope) {
  const node = $("inference-model-scope");
  if (!node) return;
  if (!scope.linked) {
    node.innerHTML =
      `<div class="pipeline-model-empty"><b>尚无本次训练产物</b>` +
      `<span>可返回“设计与训练”运行组合，或在下方高级选择中改选已有算法包。</span></div>`;
    return;
  }
  node.innerHTML =
    `<div class="pipeline-model-head"><span>自动继承</span><b>${esc(shortAuditRef(scope.runId || "当前案例"))}</b></div>` +
    `<div class="pipeline-model-groups">` +
    scope.groups.map((group) => {
      const selected = group.models.filter((item) => inferenceSelection.models.has(item.name));
      const checked = selected.length > 0;
      const recommended = selected[0]?.name || group.recommendedModelName;
      const heldOut = group.recommendedHeldOutKey ? ` · 推荐留出 ${group.recommendedHeldOutKey}` : "";
      return `<label class="pipeline-model-group ${checked ? "selected" : ""}">` +
        `<input type="checkbox" data-pipeline-model-group="${esc(group.key)}" value="${esc(recommended)}" ${checked ? "checked" : ""} />` +
        `<span><b>${esc(group.estimator)} × ${esc(group.featureset)}</b>` +
        `<small>${esc((group.protocol || "协议未记录").toUpperCase())} · ${group.artifactCount} 个折次产物${esc(heldOut)}</small>` +
        `<code title="${esc(recommended)}">${esc(recommended)}</code></span>` +
      `</label>`;
    }).join("") +
    `</div>` +
    `<p>已按训练任务聚合折次模型；每个候选组只提交一个推荐产物。</p>`;
  node.querySelectorAll("input[data-pipeline-model-group]").forEach((input) => {
    input.onchange = () => {
      const group = scope.groups.find((item) => item.key === input.dataset.pipelineModelGroup);
      if (!group) return;
      group.models.forEach((item) => inferenceSelection.models.delete(item.name));
      if (input.checked && input.value) inferenceSelection.models.add(input.value);
      renderInferencePanel(currentInferenceDatasetOptions());
    };
  });
}

function renderAdvancedInferenceModels(models, scope) {
  const runNames = new Set(scope.runModelNames || []);
  const query = inferenceModelSearch.trim().toLowerCase();
  const candidates = models.filter((item) => {
    if (runNames.has(item.name)) return false;
    if (!query) return true;
    return [item.name, item.estimator, item.featureset, item.protocol]
      .some((value) => String(value || "").toLowerCase().includes(query));
  });
  const selectedOutsideRun = models.filter((item) =>
    !runNames.has(item.name) && inferenceSelection.models.has(item.name)
  );
  const visibleMap = new Map();
  [...selectedOutsideRun, ...candidates.slice(0, 80)].forEach((item) => visibleMap.set(item.name, item));
  const visible = [...visibleMap.values()];
  renderResourceList(
    "inference-models",
    visible,
    inferenceSelection.models,
    (item) => `${item.estimator || "模型"} / ${item.featureset || "特征"}`,
    () => renderInferencePanel(currentInferenceDatasetOptions()),
  );
  const input = $("inference-model-search");
  if (input) {
    input.value = inferenceModelSearch;
    input.oninput = () => {
      inferenceModelSearch = input.value;
      renderAdvancedInferenceModels(models, scope);
    };
  }
  const count = $("inference-advanced-count");
  if (count) count.textContent = scope.otherModelCount ? `${scope.otherModelCount} 个` : "";
  const note = $("inference-model-search-note");
  if (note) {
    note.textContent = candidates.length > visible.length
      ? `显示 ${visible.length} / ${candidates.length} 个匹配项，请继续输入关键词缩小范围。`
      : `显示 ${visible.length} 个匹配项。`;
  }
}

function renderInferencePanel(dsOptions) {
  if (!$("inference-panel")) return;
  const chemistryByDataset = new Map(dsOptions.map((item) => [item.name, item.chemistry]));
  const models = modelOptions(currentModelCatalog()).map((item) => ({
    ...item,
    sourceChemistries: uniq((item.trainedOn || [])
      .map((name) => chemistryByDataset.get(name))
      .filter((value) => value && value !== "unknown")),
  }));
  const modelScope = currentPipelineModelScope(models);
  if (!inferenceSelection.models.size && models.length) {
    const knownNames = new Set(models.map((item) => item.name));
    modelScope.recommendedModelNames
      .filter((name) => knownNames.has(name))
      .forEach((name) => inferenceSelection.models.add(name));
    if (!inferenceSelection.models.size) {
      preferredInferenceModels(models, DATA?.inference || {}).forEach((name) => inferenceSelection.models.add(name));
    }
  }
  if (!inferenceSelection.datasets.size) {
    const linkedJob = currentInferenceJob() || currentTrainingJob();
    const linkedDatasets = linkedJob?.request?.selected_datasets || linkedJob?.request?.datasets || [];
    const knownDatasets = new Set(dsOptions.map((item) => item.name));
    linkedDatasets.filter((name) => knownDatasets.has(name)).forEach((name) => inferenceSelection.datasets.add(name));
    if (!inferenceSelection.datasets.size) {
      const selectedModels = models.filter((item) => inferenceSelection.models.has(item.name));
      dsOptions
        .filter((dataset) => selectedModels.length && selectedModels.some((model) => inferenceEvaluationEligibility(model, dataset).eligible))
        .forEach((item) => inferenceSelection.datasets.add(item.name));
    }
  }

  renderInferenceModelScope(modelScope);
  renderAdvancedInferenceModels(models, modelScope);
  renderResourceList(
    "inference-datasets",
    dsOptions,
    inferenceSelection.datasets,
    (item) => {
      const chemistry = item.chemistry && item.chemistry !== "unknown" ? item.chemistry : "体系待确认";
      const selectedModels = models.filter((model) => inferenceSelection.models.has(model.name));
      const scopes = selectedModels.map((model) => inferenceEvaluationEligibility(model, item));
      const role = scopes.some((scope) => scope.scope === "held_out_test")
        ? "训练留出测试"
        : (scopes.some((scope) => scope.scope === "same_chemistry_transfer" || scope.scope === "runtime_chemistry_check") ? "同体系迁移" : "风险范围");
      return `${item.rows.toLocaleString()} 行 · ${chemistry} · ${role}`;
    },
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
  const inferenceJob = currentInferenceJob();
  const currentJobMatchesSelection = inferenceJobMatchesSelection(inferenceJob);
  const actionState = inferenceActionState({
    backendConnected: Boolean(CATALOG),
    selectedModelCount: models.filter((item) => inferenceSelection.models.has(item.name)).length,
    selectedDatasetCount: dsOptions.filter((item) => inferenceSelection.datasets.has(item.name)).length,
    currentJob: inferenceJob,
    currentJobMatchesSelection,
  });
  renderInferenceActionState(actionState);
  renderInferencePolicyControls(models, dsOptions);
  const start = $("start-inference");
  const executionEnabled = Boolean(CATALOG) && API_SESSION.legacyExecutionEnabled;
  const readyForAudit = inferenceJob?.status === "succeeded" && currentJobMatchesSelection;
  if (start && readyForAudit) {
    start.textContent = "进入评测与证据";
    start.onclick = () => continueInferenceToAudit(inferenceJob);
    start.disabled = false;
    start.removeAttribute("aria-busy");
    start.title = "预测结果、模型产物与数据范围将沿用当前案例";
    renderInferenceActionState({
      disabled: false,
      message: PIPELINE_STATUS.message || "推理完成，结果已绑定到当前案例。",
      tone: "ok",
    });
  } else if (start) {
    const selectionChanged = inferenceJob?.status === "succeeded" && !currentJobMatchesSelection;
    const running = ["queued", "running"].includes(inferenceJob?.status);
    start.textContent = running ? "推理运行中…" : (selectionChanged ? "运行新推理" : (inferenceJob?.status === "failed" ? "重新运行推理" : "启动推理验证"));
    start.onclick = executionEnabled ? startInferenceEvaluationWithPolicy : null;
    start.disabled = actionState.disabled || !executionEnabled;
    if (running) start.setAttribute("aria-busy", "true");
    else start.removeAttribute("aria-busy");
    start.title = executionEnabled ? "" : "当前为仅浏览模式，请使用可信本机执行方式启动后端";
  }
  if (!executionEnabled && !readyForAudit) {
    renderInferenceActionState({
      disabled: true,
      message: "当前为仅浏览模式；算法包仍可查看，但推理执行未启用。",
      tone: "warn",
    });
  }
  renderCompletionSaveControls();
}

function renderInferenceActionState(actionState) {
  const node = $("inference-job-state");
  if (!node) return;
  node.textContent = actionState.message;
  node.className = `job-state inference-job-state ${actionState.tone || "ready"}`;
}

function renderInferencePolicyControls(models, dsOptions) {
  ensureInferencePolicyUi();
  const strategy = $("inference-strategy");
  if (strategy) {
    strategy.value = inferenceLaunchStrategy;
    strategy.onchange = () => {
      inferenceLaunchStrategy = strategy.value || "compatible_only";
      renderInferencePanel(currentInferenceDatasetOptions());
    };
  }
  const note = $("inference-policy-note");
  if (!note) return;
  const plan = inferenceLaunchPlan({
    modelNames: [...inferenceSelection.models],
    datasetNames: [...inferenceSelection.datasets],
    models,
    datasets: dsOptions,
    featureSets: CATALOG?.featuresets || [],
    strategy: inferenceLaunchStrategy,
  });
  note.textContent = inferenceLaunchStrategy === "compatible_only"
    ? `${plan.payload.models.length} 个模型 × ${plan.payload.datasets.length} 个数据集将提交；同源仅使用训练留出，同体系数据用于迁移。${plan.skippedCount ? plan.message : ""}`
    : "研究模式会包含训练回放、跨化学体系或元数据未知的范围，结果必须单独标注风险。";
}

function ensureInferencePolicyUi() {
  const start = $("start-inference");
  if (!start) return;
  const jobState = $("inference-job-state");
  const titleBlock = start.closest(".action-title")?.querySelector(".panel-title");
  if (jobState && titleBlock && jobState.parentNode !== titleBlock) {
    titleBlock.appendChild(jobState);
  }
  if (!$("inference-strategy")) {
    const actions = document.createElement("div");
    actions.className = "inference-actions";
    const policy = document.createElement("label");
    policy.className = "strategy-control";
    policy.innerHTML =
      `<span>运行策略</span><select id="inference-strategy" aria-label="推理启动策略">` +
      `<option value="compatible_only">留出测试 / 同体系迁移</option>` +
      `<option value="include_risk">研究模式：包含风险范围</option></select>`;
    start.parentNode.insertBefore(actions, start);
    actions.appendChild(policy);
    const saveToggle = document.createElement("label");
    saveToggle.className = "completion-save-toggle inference-save-toggle";
    saveToggle.htmlFor = "inference-save-toggle";
    saveToggle.title = "只收藏完成结果，不影响任务审计记录";
    saveToggle.innerHTML = `<input id="inference-save-toggle" type="checkbox" checked />` +
      `<span><b>加入已完成组合</b><small>第 3 页可选</small></span>`;
    actions.appendChild(saveToggle);
    actions.appendChild(start);
  }
  if (!$("inference-policy-note")) {
    const note = document.createElement("div");
    note.id = "inference-policy-note";
    note.className = "policy-note";
    note.setAttribute("aria-live", "polite");
    note.textContent = "默认同源仅提交训练未使用的留出切分；跨数据集迁移限定相同化学体系。";
    const panel = $("inference-panel");
    const workbench = panel?.querySelector(".inference-workbench");
    if (panel && workbench) panel.insertBefore(note, workbench);
  }
}

function renderInferenceCompatibilityPanel(models, dsOptions) {
  const node = $("inference-compatibility");
  if (!node) return;
  const selectedModels = models.filter((item) => inferenceSelection.models.has(item.name));
  const selectedDatasets = dsOptions.filter((item) => inferenceSelection.datasets.has(item.name));
  const compatibility = inferenceCompatibilitySummary({
    models: selectedModels,
    datasets: selectedDatasets,
    featureSets: CATALOG?.featuresets || [],
  });
  const evaluationRows = selectedModels.flatMap((model) => selectedDatasets.map((dataset) => ({
    model: model.name,
    dataset: dataset.name,
    ...inferenceEvaluationEligibility(model, dataset),
  })));
  const riskRows = compatibility.rows.filter((row) => {
    const evaluation = evaluationRows.find((item) => item.model === row.model && item.dataset === row.dataset);
    return row.status !== "ok" || !evaluation?.eligible;
  });
  const details = riskRows.slice(0, 8).map((row) =>
    {
      const evaluation = evaluationRows.find((item) => item.model === row.model && item.dataset === row.dataset);
      const detail = row.status === "warn"
        ? `缺失：${row.missingColumns.join(", ")}`
        : (!evaluation?.eligible ? evaluation?.reason : "缺少列清单，运行时确认");
      return (
    `<div class="compat-row ${esc(row.status)}">` +
      `<b>${esc(row.model || "未知模型")}</b>` +
      `<span>${esc(row.dataset || "未知数据集")}</span>` +
      `<small>${esc(detail)}</small>` +
    `</div>`
      );
    },
  ).join("");
  const hiddenCount = Math.max(0, riskRows.length - 8);
  const noteKind = compatibility.status === "idle" ? "unknown" : compatibility.status;
  const policyRiskCount = evaluationRows.filter((row) => !row.eligible).length;
  const noteTitle = compatibility.status === "warn"
    ? "发现 schema 风险"
    : (policyRiskCount ? "发现训练重叠或迁移边界风险" : (compatibility.status === "unknown" ? "需要运行时确认" : "可启动推理测评"));

  node.innerHTML =
    `<div class="panel-title"><span>推理输入</span><b>兼容性预检</b></div>` +
    `<div class="panel-scroll">` +
    `<div class="selector-summary">` +
      artifactMetric("模型", `${selectedModels.length}/${models.length || 0}`, selectedModels.map((item) => item.name).slice(0, 3).join(" / ") || "暂无模型", "model") +
      artifactMetric("目标数据集", `${selectedDatasets.length}/${dsOptions.length || 0}`, selectedDatasets.map((item) => item.name).slice(0, 3).join(" / ") || "暂无数据集", "dataset") +
    `</div>` +
    `<div class="compat-note ${esc(noteKind)}">` +
      `<b>${esc(noteTitle)}</b>` +
      `<span>${esc(policyRiskCount ? `${policyRiskCount} 个组合不满足留出测试或同化学体系迁移规则。` : compatibility.message)}</span>` +
      `<small>${compatibility.okCount} schema 可用 / ${policyRiskCount} 范围风险 / ${compatibility.unknownCount} 待确认</small>` +
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
    `</div>`;
}

function renderInferenceAudit(audit) {
  if (!audit?.hasAudit) {
    return `<div class="audit-grid">` +
      `<div class="audit-card muted"><span>任务审计</span><b>暂无记录</b><small>启动推理任务后显示策略、过滤数据集和 run 目录。</small></div>` +
    `</div>`;
  }
  const skippedDetail = [
    audit.skippedModelCount ? `模型：${audit.skippedModelLabel}` : "",
    audit.skippedDatasetCount ? `数据集：${audit.skippedLabel}` : "",
  ].filter(Boolean).join("；") || "未过滤";
  return `<div class="audit-grid" aria-label="推理任务审计">` +
    auditCard("策略", audit.strategyLabel, audit.sourceLabel) +
    auditCard("组合提交", `${audit.submittedModelCount}/${audit.selectedModelCount} 模型 × ${audit.submittedCount}/${audit.selectedCount} 数据集`, `过滤 ${audit.skippedCount} 项`) +
    auditCard("兼容过滤", skippedDetail, audit.skippedCount ? "schema 风险过滤" : "未过滤") +
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
  return `<div class="inference-summary-block">` +
    `<div class="inference-kpis">` +
      card("迁移精度", summary.bestTransfer, (row) => `MAE ${fmt(row.mae)}`) +
      card("最低延迟", summary.fastest, (row) => `${fmt(row.ms_per_1000_samples, 2)} ms / 1000 样本`) +
      card("最高吞吐", summary.highestThroughput, (row) => `${fmt(row.samples_per_second, 0)} 样本/秒`) +
    `</div>` +
    `</div>`;
}

function sohDrilldownButton({
  dataset = "",
  source = "",
  model = "",
  featureset = "",
  cell = "",
  label = "查看 SOH",
  className = "soh-drilldown",
} = {}) {
  return `<button class="${esc(className)}" type="button" data-soh-dashboard-open` +
    ` data-dataset="${esc(dataset)}" data-source="${esc(source)}"` +
    ` data-model="${esc(model)}" data-featureset="${esc(featureset)}" data-cell="${esc(cell)}"` +
    ` title="在比较与证据阶段查看 ${esc(dataset || model || "当前结果")}">${esc(label)}</button>`;
}

function firstCoverageDataset(coverage = {}) {
  return String(coverage.detail || "")
    .split("/")
    .map((item) => item.trim())
    .find((item) => item && !item.startsWith("未")) || "";
}

function renderInferenceTable(rows) {
  if (!rows.length) {
    return '<div class="note">暂无推理测速数据。请先从算法包管理选择已校验模型，再在目标数据集上启动 inference-only 测评。</div>';
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
  renderModelPackageState();
  const models = modelOptions(currentModelCatalog());
  const assets = modelAssetRows(currentModelCatalog(), DATA?.inference || {});
  renderModelLibrarySummary(assets);
  renderModelLibraryGovernance(assets);
  if (!models.length) {
    $("model-library-table").innerHTML = '<div class="note">暂无算法包。可在左侧导入 Pickle + manifest，训练成功后也会自动登记模型包。</div>';
    return;
  }
  const page = modelAssetPage(assets, { limit: MODEL_ASSET_PAGE_SIZE });
  const rows = page.rows.map((item) => {
    const artifact = String(item.artifactId || "").replace("model@sha256:", "");
    return (
    `<tr>` +
    `<td><span class="algo">${esc(item.name)}</span><small>${esc(item.badge)}</small></td>` +
    `<td>${esc(item.estimator || "未知")}</td>` +
    `<td>${esc(item.featureset || "未知")}</td>` +
    `<td>${esc((item.trainedOn || []).join(" / ") || "外部上传")}</td>` +
    `<td class="tnum">${esc(String(item.featureColumnCount || 0))}</td>` +
    `<td><span class="artifact-id">${artifact ? esc(artifact.slice(0, 12)) : "未固定"}</span><small>${esc(item.contractVersion || "legacy")}</small></td>` +
    `<td class="tnum">${item.latestMae == null ? "暂无" : fmt(item.latestMae)}</td>` +
    `<td class="tnum">${item.latestSpeed == null ? "暂无" : `${fmt(item.latestSpeed, 2)} ms`}</td>` +
    `</tr>`);
  }).join("");
  $("model-library-table").innerHTML =
    `<div class="asset-table-note">${esc(page.note)}</div>` +
    `<div class="inference-table asset-table"><table><thead><tr>` +
    `<th>模型</th><th>算法</th><th>特征集</th><th>训练数据</th><th>特征数</th><th>不可变身份</th><th>最近 MAE</th><th>最近速度</th>` +
    `</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderModelPackageState() {
  const node = $("model-package-state");
  if (!node) return;
  const connected = Boolean(CATALOG);
  const executionEnabled = connected && API_SESSION.legacyExecutionEnabled;
  const defaultMessage = !connected
    ? "后端未连接，暂不能导入算法包。"
    : (executionEnabled
      ? "可信本机执行已开启，可导入并校验算法包。"
      : "当前为仅浏览模式；已注册包可查看，外部包导入未启用。");
  node.textContent = MODEL_PACKAGE_STATUS.message || defaultMessage;
  node.className = `package-inline-state ${MODEL_PACKAGE_STATUS.message ? MODEL_PACKAGE_STATUS.tone : (executionEnabled ? "ok" : "warn")}`;
}

function renderModelLibrarySummary(assets = []) {
  const node = $("model-library-summary");
  if (!node) return;
  const generated = assets.filter((item) => item.source !== "upload").length;
  const uploaded = assets.filter((item) => item.source === "upload").length;
  const immutable = assets.filter((item) => item.artifactId).length;
  const withInference = assets.filter((item) => item.latestMae != null).length;
  const best = assets.find((item) => item.latestMae != null);
  const audit = currentInferenceAuditSummary();
  node.innerHTML =
    artifactMetric("算法包总数", `${assets.length}`, `${generated} 训练产出 / ${uploaded} 外部导入`, "model") +
    artifactMetric("身份固定", `${immutable}/${assets.length}`, "内容摘要可用于历史复验", "job") +
    artifactMetric("最佳资产", best?.name || "暂无", best ? `最近 MAE ${fmt(best.latestMae)}` : "等待推理测评", "dataset") +
    renderModelAuditSummary(audit);
}

function renderModelLibraryGovernance(assets = []) {
  const node = $("model-library-governance");
  if (!node) return;
  const immutable = assets.filter((item) => item.artifactId).length;
  const contracted = assets.filter((item) => item.contractVersion && item.contractVersion !== "legacy").length;
  const latest = assets.find((item) => item.artifactId) || {};
  const digest = String(latest.artifactId || "").replace("model@sha256:", "").slice(0, 16);
  node.innerHTML =
    `<div class="asset-governance-note"><span>身份固定</span><strong>${immutable}/${assets.length} 个包具有内容摘要</strong></div>` +
    `<div class="asset-governance-note green"><span>契约校验</span><strong>${contracted} 个包使用版本化 manifest</strong></div>` +
    `<div class="asset-governance-note amber"><span>最近摘要</span><strong>${esc(digest || "等待新算法包")}</strong></div>` +
    `<p class="package-security-note">同名算法包再次生成前，旧字节会进入不可变历史归档；已有案例仍按 model@sha256 精确加载。</p>`;
}

function renderModelAuditSummary(audit) {
  return artifactMetric(
    "最近策略",
    audit?.hasAudit ? audit.strategyLabel : "暂无",
    audit?.hasAudit ? `${audit.sourceLabel} / 跳过 ${audit.skippedCount} 个` : "等待推理任务",
    "job",
  );
}

function renderRecommendationPanel() {
  const overview = $("recommendation-overview");
  const scenariosNode = $("recommendation-scenarios");
  if (!overview || !scenariosNode) return;
  const view = algorithmRecommendationView(DATA?.recommendation || {});
  if (!view.available) {
    overview.innerHTML = '<div class="state">暂无算法推荐数据。完成 LODO/LOCO 对比后运行推荐摘要脚本。</div>';
    scenariosNode.innerHTML = "";
    return;
  }
  const datasetLabel = (view.scope.datasets || []).join(" / ") || "暂无";
  overview.innerHTML = `
    <div class="recommendation-hero">
      <div>
        <span>默认推荐</span>
        <strong class="mono">${esc(view.defaultAlgorithm)}</strong>
        <small>${esc(datasetLabel)}</small>
      </div>
      <div class="recommendation-points">
        ${view.why.map((item) => `<p>${esc(item)}</p>`).join("") || "<p>暂无推荐理由。</p>"}
      </div>
      <div class="recommendation-caveats">
        ${view.caveats.slice(0, 3).map((item) => `<p>${esc(item)}</p>`).join("") || "<p>暂无明显约束。</p>"}
      </div>
    </div>
  `;
  scenariosNode.innerHTML = view.scenarioCards.map((scenario) => `
    <article class="recommendation-card">
      <header>
        <div>
          <span>${esc(scenario.protocolLabel)}</span>
          <h3>${esc(scenario.title)}</h3>
        </div>
        <b class="mono">${esc(scenario.recommendedAlgorithm)}</b>
      </header>
      <p>${esc(scenario.recommendation || scenario.description || "")}</p>
      <div class="recommendation-metrics">
        ${recommendationMetric("精度最佳", scenario.accuracyWinner, "平均 MAE 最低")}
        ${recommendationMetric("速度最快", scenario.speedWinner, "推理 ms/1000 最低")}
        ${recommendationMetric("成功率", scenario.successLabel, `${scenario.recordLabel} 条记录`)}
      </div>
      <div class="recommendation-table">
        <table>
          <thead><tr><th>#</th><th>算法</th><th>MAE</th><th>RMSE</th><th>速度</th></tr></thead>
          <tbody>
            ${scenario.ranking.map((row) => `
              <tr>
                <td>${row.rank}</td>
                <td class="mono">${esc(row.algorithm)}</td>
                <td>${esc(row.maeLabel)}</td>
                <td>${esc(row.rmseLabel)}</td>
                <td>${esc(row.speedLabel)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </article>
  `).join("");
}

function recommendationMetric(label, value, hint) {
  return `<div><span>${esc(label)}</span><b>${esc(value)}</b><small>${esc(hint)}</small></div>`;
}

async function uploadResource(kind) {
  if (kind === "model") {
    const modelFile = $("upload-model-file")?.files?.[0];
    const manifestFile = $("upload-model-manifest")?.files?.[0];
    if (!modelFile || !manifestFile) {
      setModelPackageMessage("请选择 Pickle 模型和对应的 model_manifest.json。", "warn");
      return;
    }
    const name = ($("upload-model-name").value || modelFile.name.replace(/\.[^.]+$/, "")).trim();
    try {
      setModelPackageMessage("正在读取并校验算法包...", "active");
      const response = await postJson("/api/upload/model", {
        name,
        modelFilename: modelFile.name,
        modelContentBase64: await fileToBase64(modelFile),
        manifestFilename: manifestFile.name,
        manifestContentBase64: await fileToBase64(manifestFile),
      });
      inferenceSelection.models.add(response.resource.name);
      await loadBackendCatalog();
      await loadFullModelCatalog({ force: true });
      MODEL_PACKAGE_STATUS = { tone: "ok", message: `算法包 ${response.resource.name} 已导入并登记。` };
      render();
    } catch (err) {
      setModelPackageMessage(`算法包导入失败：${err.message}`, "fail");
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
    setModelPackageMessage("请选择需要导入的文件。", "warn");
    return;
  }
  const name = ($(nameId).value || file.name.replace(/\.[^.]+$/, "")).trim();
  try {
    setModelPackageMessage("正在校验并导入资源...", "active");
    const resource = await postJson(endpoint, {
      name,
      filename: file.name,
      contentBase64: await fileToBase64(file),
    });
    resourceSelection[selectionKey].add(resource.resource.name);
    await loadBackendCatalog();
    MODEL_PACKAGE_STATUS = { tone: "ok", message: `${resource.resource.name} 已导入，可在训练组合中选择。` };
    render();
  } catch (err) {
    setModelPackageMessage(`资源导入失败：${err.message}`, "fail");
  }
}

async function importBatterySohPlugins() {
  const batterySohRoot = (($("battery-soh-root")?.value || "D:\\battery_SOH").trim() || "D:\\battery_SOH");
  try {
    setModelPackageMessage(`正在从 ${batterySohRoot} 接入算法与特征...`, "active");
    const response = await postJson("/api/import/battery-soh-plugins", { replace: true, battery_soh_root: batterySohRoot });
    const estimatorName = response.plugins?.estimator?.name || "battery_soh_tabular_ensemble";
    const featuresetName = response.plugins?.featureset?.name || "battery_soh_operational";
    resourceSelection.estimators.add(estimatorName);
    resourceSelection.featuresets.add(featuresetName);
    await loadBackendCatalog();
    MODEL_PACKAGE_STATUS = { tone: "ok", message: `${estimatorName} / ${featuresetName} 已接入训练组合目录。` };
    render();
  } catch (err) {
    setModelPackageMessage(`battery_SOH 接入失败（${batterySohRoot}）：${err.message}`, "fail");
  }
}

function normalizedValues(values = []) {
  return [...new Set((values || []).map((item) => String(item)))].sort();
}

function sameValues(left = [], right = []) {
  const a = normalizedValues(left);
  const b = normalizedValues(right);
  return a.length === b.length && a.every((item, index) => item === b[index]);
}

function trainingJobMatchesPayload(job, payload) {
  const request = job?.request || {};
  return ["datasets", "estimators", "featuresets", "protocols", "seeds"]
    .every((key) => sameValues(request[key] || [], payload[key] || []));
}

function inferenceJobMatchesSelection(job) {
  const request = job?.request || {};
  const currentModels = [...inferenceSelection.models];
  const currentDatasets = [...inferenceSelection.datasets];
  const modelMatches = [request.selected_models, request.submitted_models, request.models]
    .some((values) => Array.isArray(values) && values.length && sameValues(values, currentModels));
  const datasetMatches = [request.selected_datasets, request.submitted_datasets, request.datasets]
    .some((values) => Array.isArray(values) && values.length && sameValues(values, currentDatasets));
  return modelMatches && datasetMatches;
}

function updateModeUrl(mode) {
  try {
    const url = new URL(location.href);
    url.searchParams.set("mode", mode);
    if (ACTIVE_EVALUATION_CASE?.id) url.searchParams.set("case", ACTIVE_EVALUATION_CASE.id);
    else url.searchParams.delete("case");
    history.replaceState(null, "", url);
  } catch (_) {
    // Embedded and standalone builds may not expose a mutable browser history.
  }
}

async function activateMode(mode) {
  if (!APP_MODES.has(mode)) return;
  activeMode = mode;
  updateModeUrl(mode);
  render();
  if (mode === "models" || mode === "inference") await loadFullModelCatalog();
  if (mode === "audit") await loadReviewWorkspace({ force: true });
  render();
}

async function continueTrainingToInference(job = currentTrainingJob()) {
  if (!job?.saved_models?.length) return;
  await refreshPipelineModelCatalogForJobs({ force: true });
  const models = modelOptions(currentModelCatalog());
  const knownModels = new Set(models.map((item) => item.name));
  const knownDatasets = new Set(currentInferenceDatasetOptions().map((item) => item.name));
  const scope = pipelineModelGroups({ trainingJob: job, models });
  inferenceSelection = {
    models: new Set(scope.recommendedModelNames.filter((name) => knownModels.has(name))),
    datasets: new Set((job.request?.datasets || []).filter((name) => knownDatasets.has(name))),
  };
  if (!inferenceSelection.models.size) {
    const fallback = job.saved_models.find((name) => knownModels.has(name)) || job.saved_models[0];
    if (fallback) inferenceSelection.models.add(fallback);
  }
  if (!inferenceSelection.datasets.size) {
    currentInferenceDatasetOptions().forEach((item) => inferenceSelection.datasets.add(item.name));
  }
  inferenceLaunchStrategy = "compatible_only";
  PIPELINE_STATUS = {
    tone: "ok",
    message: `已从 ${job.saved_models.length} 个折次产物中自动选出 ${inferenceSelection.models.size} 个候选，无需重新选择。`,
  };
  await activateMode("inference");
}

async function continueInferenceToAudit(job = currentInferenceJob()) {
  if (job?.status === "succeeded") await bindInferenceEvidenceToCase(job);
  PIPELINE_STATUS = { tone: "ok", message: "已进入同一案例的评测与证据阶段。" };
  await activateMode("audit");
}

async function startEvaluation() {
  const payload = buildCurrentEvaluationPayload();
  if (!payload.datasets.length || !payload.estimators.length || !payload.featuresets.length) {
    setJobMessage("请至少选择一个数据集、算法和特征。");
    return;
  }
  const preflight = currentTrainingPreflight(payload);
  if (preflight.status !== "ready" || !preflight.result?.ready) {
    setJobMessage(preflight.status === "blocked"
      ? `组合已阻断：${trainingPreflightBlockReason(preflight.result)}`
      : "请等待真实数据组合预检通过后再运行。");
    renderRunPlan();
    return;
  }
  try {
    await ensureWritablePipelineCase({ forNewTraining: true });
    setJobMessage("预检已固定，正在提交真实训练任务...");
    CURRENT_JOB = await postJson("/api/jobs", {
      ...payload,
      preflight_id: preflight.result.preflight_id,
      case_id: ACTIVE_EVALUATION_CASE?.id || "",
      expected_context_binding_id: evaluationContextBindingId(),
    });
    await syncPipelineCaseWithJob(CURRENT_JOB, "training");
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
    modelNames: [...inferenceSelection.models],
    datasetNames: [...inferenceSelection.datasets],
    models,
    datasets: dsOptions,
    featureSets: CATALOG?.featuresets || [],
    strategy: inferenceLaunchStrategy,
  });
  const payload = plan.payload;
  if (!payload.models.length || !payload.datasets.length) {
    setInferenceMessage(plan.skippedCount
      ? `没有可安全提交的模型 × 数据集组合。${plan.message}；请减少选择或切换为“研究模式：包含风险范围”。`
      : "请至少选择一个模型和一个数据集。", "warn");
    return;
  }
  try {
    const start = $("start-inference");
    if (start) {
      start.textContent = "正在提交…";
      start.disabled = true;
      start.setAttribute("aria-busy", "true");
    }
    setInferenceMessage(plan.skippedCount
      ? `正在提交 ${payload.models.length} 个模型 × ${payload.datasets.length} 个数据集；${plan.message}`
      : `正在提交 ${payload.models.length} 个模型 × ${payload.datasets.length} 个数据集…`, "active");
    await ensureWritablePipelineCase();
    CURRENT_JOB = { kind: "inference", ...(await postJson("/api/inference-jobs", {
      ...payload,
      case_id: ACTIVE_EVALUATION_CASE?.id || "",
      expected_context_binding_id: evaluationContextBindingId(),
      profile_name: ACTIVE_EVALUATION_CASE?.profile_name || activeProfileOption?.name || activeProfile || "root",
    })) };
    await syncPipelineCaseWithJob(CURRENT_JOB, "inference");
    renderInferencePanel(currentInferenceDatasetOptions());
    pollInferenceJob(CURRENT_JOB.id);
  } catch (err) {
    renderInferencePanel(currentInferenceDatasetOptions());
    setInferenceMessage(`推理任务启动失败：${err.message}`, "fail");
  }
}

async function startInferenceEvaluation() {
  const payload = buildInferencePayload({
    models: [...inferenceSelection.models],
    datasets: [...inferenceSelection.datasets],
  });
  if (!payload.models.length || !payload.datasets.length) {
    setInferenceMessage("请至少选择一个模型和一个数据集。", "warn");
    return;
  }
  try {
    const start = $("start-inference");
    if (start) {
      start.textContent = "正在提交…";
      start.disabled = true;
      start.setAttribute("aria-busy", "true");
    }
    setInferenceMessage("正在启动推理任务...", "active");
    await ensureWritablePipelineCase();
    CURRENT_JOB = { kind: "inference", ...(await postJson("/api/inference-jobs", {
      ...payload,
      case_id: ACTIVE_EVALUATION_CASE?.id || "",
      expected_context_binding_id: evaluationContextBindingId(),
      profile_name: ACTIVE_EVALUATION_CASE?.profile_name || activeProfileOption?.name || activeProfile || "root",
    })) };
    await syncPipelineCaseWithJob(CURRENT_JOB, "inference");
    renderInferencePanel(currentInferenceDatasetOptions());
    pollInferenceJob(CURRENT_JOB.id);
  } catch (err) {
    renderInferencePanel(currentInferenceDatasetOptions());
    setInferenceMessage(`推理任务启动失败：${err.message}`, "fail");
  }
}

function selectedValues(id) {
  return [...($(id)?.querySelectorAll("input[type='checkbox']:checked") || [])].map((input) => input.value);
}

async function pollJob(jobId) {
  const pollKey = `training:${jobId}`;
  if (ACTIVE_PIPELINE_POLLS.has(pollKey)) return;
  const ownerCaseId = ACTIVE_EVALUATION_CASE?.id || "";
  ACTIVE_PIPELINE_POLLS.add(pollKey);
  try {
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      if (ownerCaseId && ACTIVE_EVALUATION_CASE?.id !== ownerCaseId) return;
      const response = await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(jobId)}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      CURRENT_JOB = await response.json();
      renderWorkbench();
      if (CURRENT_JOB.status === "succeeded") {
        await refreshPipelineModelCatalogForJobs({ force: true });
        await syncPipelineCaseWithJob(CURRENT_JOB, "training");
        if (trainingAutoSave) {
          try {
            await updateCompletedCombinationSave("training", CURRENT_JOB, true, { silent: true });
          } catch (_) {
            // Training remains complete even if optional curation fails.
          }
        }
        await load({ cacheBust: true });
        activeMode = "training";
        render();
        return;
      }
      if (CURRENT_JOB.status === "failed") {
        await syncPipelineCaseWithJob(CURRENT_JOB, "training");
        render();
        return;
      }
    }
  } catch (err) {
    setJobMessage(`任务轮询失败：${err.message}`);
  } finally {
    ACTIVE_PIPELINE_POLLS.delete(pollKey);
  }
}

async function pollInferenceJob(jobId) {
  const pollKey = `inference:${jobId}`;
  if (ACTIVE_PIPELINE_POLLS.has(pollKey)) return;
  const ownerCaseId = ACTIVE_EVALUATION_CASE?.id || "";
  ACTIVE_PIPELINE_POLLS.add(pollKey);
  try {
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      if (ownerCaseId && ACTIVE_EVALUATION_CASE?.id !== ownerCaseId) return;
      const response = await fetch(`${API_BASE}/api/inference-jobs/${encodeURIComponent(jobId)}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      CURRENT_JOB = { kind: "inference", ...(await response.json()) };
      renderInferencePanel(currentInferenceDatasetOptions());
      if (CURRENT_JOB.status === "succeeded") {
        const completedJob = CURRENT_JOB;
        const shouldAdvanceToAudit = activeMode === "inference";
        await syncPipelineCaseWithJob(completedJob, "inference");
        if (inferenceAutoSave) {
          try {
            await updateCompletedCombinationSave("inference", completedJob, true, { silent: true });
          } catch (_) {
            // Inference remains complete even if optional curation fails.
          }
        }
        await load({ cacheBust: true });
        if (shouldAdvanceToAudit) {
          await continueInferenceToAudit(completedJob);
        } else {
          await bindInferenceEvidenceToCase(completedJob);
          render();
        }
        return;
      }
      if (CURRENT_JOB.status === "failed") {
        await syncPipelineCaseWithJob(CURRENT_JOB, "inference");
        render();
        return;
      }
    }
  } catch (err) {
    setInferenceMessage(`推理任务轮询失败：${err.message}`, "fail");
  } finally {
    ACTIVE_PIPELINE_POLLS.delete(pollKey);
  }
}

async function postJson(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: apiJsonHeaders(),
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function apiJsonHeaders() {
  const headers = { "Content-Type": "application/json" };
  if (API_SESSION.token) headers["X-SOHBench-Token"] = API_SESSION.token;
  return headers;
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

function setModelPackageMessage(message, tone = "ready") {
  MODEL_PACKAGE_STATUS = { message, tone };
  renderModelPackageState();
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
  h += "<table class='lb'><thead><tr><th>算法</th><th>覆盖范围</th><th>MAE</th><th>RMSE</th><th>MBE</th><th>PICP</th><th>CI</th><th>判定</th><th>SOH</th></tr></thead><tbody>";
  rows.forEach(({ entry: x, coverage }, i) => {
    const mae = metricMean(x, "mae"), w = Math.round((mae / maxMae) * 90);
    const v = x.verdicts || {};
    const ci = x.ci_source || "conformal";
    const dataset = firstCoverageDataset(coverage);
    h += `<tr class="${i === 0 ? "best" : ""}">` +
      `<td><span class="rank">${i + 1}</span><span class="algo">${esc(x.estimator)}</span></td>` +
      `<td><span class="coverage-badge ${esc(coverage.kind)}" title="${esc(coverage.detail)}">${esc(coverage.label)}</span></td>` +
      `<td><div class="bar-cell"><span class="bar ${i === rows.length - 1 && rows.length > 1 ? "bad" : ""}" style="width:${w}px"></span><span class="tnum">${fmt(mae)}</span></div></td>` +
      `<td class="tnum">${fmt(metricMean(x, "rmse"))}</td>` +
      `<td class="tnum">${fmt(metricMean(x, "mean_bias_error"), 3, true)}</td>` +
      `<td class="tnum">${fmt(metricMean(x, "picp"))}</td>` +
      `<td><span class="pill ${esc(ci)}">${esc(ci)}</span></td>` +
      `<td><span class="v">${vdot(v.generalization)}${vdot(v.calibration)}${vdot(v.physical_consistency)}</span></td>` +
      `<td>${sohDrilldownButton({ dataset, source: "training", model: x.estimator, featureset: x.featureset || "full" })}</td>` +
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
  const profile = activeProfileOption?.label || "当前根目录";
  $("foot").textContent = `${DATA.fallback ? "示例数据" : "sohbench 实测运行"} / ${profile} / ${algos} / ${scope}`;
}

document.addEventListener("click", handleBatteryCurveDocumentClick);
document.addEventListener("click", handleSohDashboardDrilldownClick);
load().catch((error) => {
  const detail = String(error?.message || error || "未知错误");
  WORKBENCH_ERROR = `初始化失败：${detail.slice(0, 320)}${detail.length > 320 ? "…" : ""}`;
  const state = $("api-state");
  if (state) {
    state.textContent = WORKBENCH_ERROR;
    state.className = "api-state fail";
  }
  console.error("SOH workbench initialization failed", error);
});
