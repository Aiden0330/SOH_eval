import assert from "node:assert/strict";
import {
  summarizeBySelection,
  datasetOptions,
  evaluationCellKey,
  evaluationIdentity,
  evaluationLabel,
  evaluationMatrixRows,
  algorithmInsights,
  bestComparableEntry,
  buildEvaluationPayload,
  buildInferencePayload,
  catalogOptions,
  evaluationPlanSummary,
  inferenceRows,
  inferenceAuditSummary,
  inferenceActionState,
  inferenceCompatibilitySummary,
  inferenceDatasetOptions,
  inferenceLaunchPlan,
  inferenceInsightSummary,
  inferenceSummary,
  isTerminalJobStatus,
  preferredInferenceModels,
  profileOptions,
  leaderboardCoverage,
  modelAssetRows,
  modelOptions,
  scopeSummary,
  stabilityRows,
  taskHistoryRows,
  trainingArtifactSummary,
  nextDatasetSelection,
} from "./dashboard-core.mjs";

{
  assert.equal(isTerminalJobStatus("queued"), false);
  assert.equal(isTerminalJobStatus("running"), false);
  assert.equal(isTerminalJobStatus("succeeded"), true);
  assert.equal(isTerminalJobStatus("partial"), true);
  assert.equal(isTerminalJobStatus("failed"), true);
}

const aggregate = [
  entry("ridge", "full", 0.30, { picp: 0.72, mono: 0.10 }),
  entry("gbdt", "full", 0.20, { picp: 0.60, mono: 0.30 }),
  entry("ridge", "ic_dva", 0.18, { picp: 0.82, mono: 0.12 }),
  entry("gbdt", "ic_dva", 0.12, { picp: 0.65, mono: 0.20 }),
];

const byDataset = [
  entry("ridge", "full", 0.40, { held: "nasa", picp: 0.50, mono: 0.20 }),
  entry("gbdt", "full", 0.10, { held: "nasa", picp: 0.70, mono: 0.35 }),
  entry("ridge", "full", 0.20, { held: "hust", picp: 0.90, mono: 0.10 }),
  entry("gbdt", "full", 0.30, { held: "hust", picp: 0.40, mono: 0.25 }),
];

{
  const options = profileOptions(
    {
      profiles: [
        { name: "smoke", label: "Smoke 闭环", kind: "smoke_acceptance" },
        { name: "full-7-datasets", label: "7 数据集全量", kind: "full_benchmark" },
        { name: "smoke", label: "Duplicate", kind: "ignored" },
      ],
    },
    "smoke",
  );
  assert.deepEqual(options.map((item) => item.name), ["root", "smoke", "full-7-datasets"]);
  assert.equal(options[0].label, "当前根目录");
  assert.equal(options[1].active, true);
  assert.equal(options[1].kind, "smoke_acceptance");
}

function entry(estimator, featureset, mae, opts = {}) {
  return {
    estimator,
    featureset,
    protocol: "lodo",
    track: opts.track ?? "unified",
    held_out_key: opts.held,
    ...(opts.candidateId ? { candidate_id: opts.candidateId } : {}),
    ci_source: "weighted_conformal",
    metrics: {
      mae: { mean: mae, std: 0 },
      picp: { mean: opts.picp ?? 0.5, std: 0 },
      monotonicity_violation_rate: { mean: opts.mono ?? 0.2, std: 0 },
      value_range_violation: { mean: 0, std: 0 },
    },
    verdicts: { generalization: false, calibration: false, physical_consistency: false },
  };
}

{
  const candidateRows = [
    entry("ridge", "full", 0.10, { held: "nasa", candidateId: "candidate-a" }),
    entry("ridge", "full", 0.20, { held: "nasa", candidateId: "candidate-b" }),
  ];
  const selected = summarizeBySelection([], candidateRows, ["nasa"], ["nasa", "hust"]);
  assert.equal(selected.entries.length, 2);
  assert.deepEqual(
    selected.entries.map((row) => [row.candidate_id, row.metrics.mae.mean]).sort(),
    [["candidate-a", 0.10], ["candidate-b", 0.20]],
  );
  assert.notEqual(evaluationIdentity(candidateRows[0]), evaluationIdentity(candidateRows[1]));
  assert.notEqual(evaluationCellKey(candidateRows[0]), evaluationCellKey(candidateRows[1]));
  assert.equal(evaluationLabel(candidateRows[0]), "candidate-a (ridge)");
  assert.equal(evaluationLabel(entry("ridge", "full", 0.30)), "ridge");
  assert.deepEqual(
    evaluationMatrixRows(candidateRows, ["full"]).map((row) => [row.label, row.cells[0].value]),
    [["candidate-a (ridge)", 0.10], ["candidate-b (ridge)", 0.20]],
  );
  const trackRows = [
    entry("ridge", "full", 0.10, { candidateId: "candidate-a", track: "unified" }),
    entry("ridge", "full", 0.20, { candidateId: "candidate-a", track: "byo" }),
  ];
  assert.deepEqual(
    evaluationMatrixRows(trackRows, ["full"]).map((row) => [row.label, row.cells[0].value]),
    [["candidate-a (ridge)", 0.10], ["candidate-a (ridge) [byo]", 0.20]],
  );
}

{
  const options = datasetOptions(
    { datasets: ["nasa", "hust", "oxford"] },
    { per_dataset: { nasa: { rows: 10 }, hust: { rows: 30 } } },
    byDataset,
  );
  assert.deepEqual(options.map((d) => [d.name, d.rows, d.hasMetrics]), [
    ["hust", 30, true],
    ["nasa", 10, true],
    ["oxford", 0, false],
  ]);
}

{
  const selected = summarizeBySelection(aggregate, byDataset, ["nasa"], ["nasa", "hust"]);
  assert.equal(selected.usedDatasetLevel, true);
  assert.equal(selected.entries.length, 2);
  assert.equal(selected.entries.find((e) => e.estimator === "gbdt").metrics.mae.mean, 0.10);
}

{
  const all = ["nasa", "hust", "oxford"];
  assert.deepEqual(nextDatasetSelection(["nasa", "hust", "oxford"], "hust", all, "only"), ["hust"]);
  assert.deepEqual(nextDatasetSelection(["hust"], "nasa", all, "toggle").sort(), ["hust", "nasa"]);
  assert.deepEqual(nextDatasetSelection(["hust", "nasa"], "hust", all, "toggle"), ["nasa"]);
  assert.deepEqual(nextDatasetSelection(["hust"], "hust", all, "toggle"), ["nasa", "hust", "oxford"]);
  assert.deepEqual(nextDatasetSelection(["hust"], null, all, "all"), ["nasa", "hust", "oxford"]);
}

{
  const selected = summarizeBySelection(aggregate, byDataset, ["nasa", "hust"], ["nasa", "hust"]);
  assert.equal(selected.usedDatasetLevel, false);
  assert.equal(selected.entries, aggregate);
}

{
  const selected = summarizeBySelection(aggregate, byDataset, ["nasa", "hust"], ["nasa", "hust", "oxford"]);
  const ridge = selected.entries.find((e) => e.estimator === "ridge" && e.featureset === "full");
  assert.equal(selected.usedDatasetLevel, true);
  assert.ok(Math.abs(ridge.metrics.mae.mean - 0.30) < 1e-12);
  assert.ok(Math.abs(ridge.metrics.picp.mean - 0.70) < 1e-12);
}

{
  const insights = algorithmInsights(aggregate.filter((e) => e.featureset === "full"));
  assert.equal(insights.bestAccuracy.estimator, "gbdt");
  assert.equal(insights.bestCalibration.estimator, "ridge");
  assert.equal(insights.bestPhysical.estimator, "ridge");
}

{
  const rows = stabilityRows({
    stability_seeds: [0, 1, 2],
    stability_summary: [
      {
        estimator: "mlp",
        featureset: "partial_window",
        mae_mean: 0.097021,
        mae_std: 0.02279,
        mae_min: 0.070739,
        mae_max: 0.111313,
      },
      {
        estimator: "gbdt",
        featureset: "ic_dva",
        mae_mean: 0.100977,
        mae_std: 0.005939,
        mae_min: 0.096657,
        mae_max: 0.10775,
      },
    ],
  });
  assert.equal(rows.length, 2);
  assert.equal(rows[0].label, "mlp x partial_window");
  assert.equal(rows[0].seedCount, 3);
  assert.equal(rows[0].rangeText, "0.071-0.111");
  assert.equal(rows[0].stabilityLabel, "波动较大");
  assert.equal(rows[1].label, "gbdt x ic_dva");
  assert.equal(rows[1].stabilityLabel, "较稳");
}

{
  const options = catalogOptions({
    datasets: [{ name: "nasa", rows: 10 }, { name: "new_public", rows: 5 }],
    estimators: [{ name: "ridge" }, { name: "trusted_algo", uploaded: true }],
    featuresets: [{ name: "full" }, { name: "trusted_shape", has_transform: true }],
  });
  assert.deepEqual(options.datasets.map((item) => item.name), ["nasa", "new_public"]);
  assert.equal(options.estimators.find((item) => item.name === "trusted_algo").badge, "上传");
  assert.equal(options.featuresets.find((item) => item.name === "trusted_shape").badge, "变换");
}

{
  const payload = buildEvaluationPayload({
    datasets: ["new_public"],
    estimators: ["trusted_algo"],
    featuresets: ["trusted_shape"],
    protocols: ["loco"],
    seedsText: "0, 2 4",
  });
  assert.deepEqual(payload, {
    datasets: ["new_public"],
    estimators: ["trusted_algo"],
    featuresets: ["trusted_shape"],
    protocols: ["loco"],
    seeds: [0, 2, 4],
  });
}

{
  const locoRows = [
    entry("trusted_huber_decay", "rwth_cycle_shape", 0.02, { held: "rwth_isea_public/rwth_cell_001" }),
    entry("trusted_huber_decay", "rwth_cycle_shape", 0.04, { held: "rwth_isea_public/rwth_cell_002" }),
  ];
  const options = datasetOptions(
    { datasets: ["rwth_isea_public"] },
    { per_dataset: { rwth_isea_public: { rows: 100 } } },
    locoRows,
  );
  const selected = summarizeBySelection([], locoRows, ["rwth_isea_public"], ["rwth_isea_public", "nasa"]);
  assert.deepEqual(options.map((item) => item.name), ["rwth_isea_public"]);
  assert.equal(selected.entries[0].metrics.mae.mean, 0.03);
}

{
  const options = datasetOptions(
    { datasets: ["nasa", "hust", "rwth_isea_public"], protocols: ["loco"], seeds: [0, 1] },
    { per_dataset: { nasa: { rows: 10 }, hust: { rows: 30 }, rwth_isea_public: { rows: 5 } } },
    byDataset,
  );
  const selected = summarizeBySelection(aggregate, byDataset, ["nasa", "hust"], options.map((item) => item.name));
  const summary = scopeSummary({
    datasetOptions: options,
    selectedDatasets: ["nasa", "hust"],
    scoped: selected,
    config: { protocols: ["loco"], seeds: [0, 1] },
    fallback: false,
  });
  assert.equal(summary.datasetLabel, "2/3");
  assert.equal(summary.rowLabel, "40");
  assert.equal(summary.shareLabel, "88.9%");
  assert.equal(summary.protocolLabel, "loco");
  assert.equal(summary.seedLabel, "2");
  assert.equal(summary.sourceLabel, "实测");
  assert.equal(summary.coverageKind, "partial");
  assert.equal(summary.notice, "当前结果按所选 held-out 数据集重新聚合。");
}

{
  const datasetRows = [
    entry("gbdt", "full", 0.10, { held: "nasa" }),
    entry("gbdt", "full", 0.20, { held: "hust" }),
    entry("trusted_huber_decay", "rwth_cycle_shape", 0.03, { held: "rwth_isea_public/rwth_cell_001" }),
  ];
  const full = leaderboardCoverage(entry("gbdt", "full", 0.10), ["nasa", "hust"], datasetRows);
  const local = leaderboardCoverage(entry("trusted_huber_decay", "rwth_cycle_shape", 0.03), ["nasa", "hust", "rwth_isea_public"], datasetRows);
  const assumed = leaderboardCoverage(entry("ridge", "full", 0.12), ["nasa", "hust"], []);
  assert.deepEqual(
    { label: full.label, kind: full.kind, rank: full.rank, detail: full.detail },
    { label: "2/2", kind: "full", rank: 0, detail: "nasa / hust" },
  );
  assert.deepEqual(
    { label: local.label, kind: local.kind, rank: local.rank, detail: local.detail },
    { label: "1/3", kind: "local", rank: 2, detail: "rwth_isea_public" },
  );
  assert.deepEqual(
    { label: assumed.label, kind: assumed.kind, rank: assumed.rank, detail: assumed.detail },
    { label: "全量", kind: "assumed", rank: 0, detail: "未记录数据集级明细" },
  );
}

{
  const plan = evaluationPlanSummary({
    datasets: ["nasa", "hust"],
    estimators: ["ridge", "gbdt"],
    featuresets: ["full", "ic_dva", "rwth_cycle_shape"],
    protocols: ["loco"],
    seeds: [0, 1],
  });
  assert.equal(plan.comboCount, 24);
  assert.equal(plan.comboLabel, "24 个组合");
  assert.equal(plan.ready, true);
  assert.equal(plan.partsLabel, "2 数据集 x 2 算法 x 3 特征 x 1 协议 x 2 seeds");
}

{
  const rows = [
    entry("gbdt", "ic_dva", 0.10),
    entry("trusted_huber_decay", "rwth_cycle_shape", 0.03),
  ];
  const datasetRows = [
    entry("gbdt", "ic_dva", 0.10, { held: "nasa" }),
    entry("gbdt", "ic_dva", 0.11, { held: "hust" }),
    entry("trusted_huber_decay", "rwth_cycle_shape", 0.03, { held: "rwth_isea_public/rwth_cell_001" }),
  ];
  const best = bestComparableEntry(rows, ["nasa", "hust", "rwth_isea_public"], datasetRows);
  assert.equal(best.estimator, "gbdt");
}

{
  const rows = modelOptions({
    models: [
      { name: "ridge_full", source: "training", estimator: "ridge", featureset: "full" },
      { name: "external_mlp", source: "upload", estimator: "mlp", featureset: "ic_dva" },
    ],
  });
  assert.deepEqual(rows.map((row) => row.name), ["external_mlp", "ridge_full"]);
  assert.equal(rows[0].badge, "上传模型");
  assert.equal(rows[1].badge, "训练产物");
}

{
  const options = inferenceDatasetOptions({
    baseOptions: datasetOptions(
      { datasets: ["nasa"] },
      { per_dataset: { nasa: { rows: 2705 } } },
      [],
    ),
    catalog: {
      datasets: [
        { name: "calce_cs2", rows: 15411 },
        { name: "nasa", rows: 2705 },
      ],
    },
    inference: {
      records: [
        { model: "ridge_voltage", dataset: "calce_cs2", status: "succeeded" },
        { model: "ridge_voltage", dataset: "nasa", status: "succeeded" },
      ],
    },
  });
  assert.deepEqual(options.map((item) => [item.name, item.rows, item.hasInference]), [
    ["calce_cs2", 15411, true],
    ["nasa", 2705, true],
  ]);
}

{
  const preferred = preferredInferenceModels(
    [
      { name: "byo_old" },
      { name: "ridge_voltage_loco_nasa_B0005_seed0" },
    ],
    {
      records: [
        { model: "ridge_voltage_loco_nasa_B0005_seed0", dataset: "nasa", status: "succeeded" },
      ],
    },
  );
  assert.deepEqual(preferred, ["ridge_voltage_loco_nasa_B0005_seed0"]);
}

{
  const payload = {
    records: [
      { model: "slow", dataset: "nasa", status: "succeeded", ms_per_1000_samples: 20, mae: 0.2 },
      { model: "fast", dataset: "nasa", status: "succeeded", ms_per_1000_samples: 5, mae: 0.3 },
      { model: "fast", dataset: "hust", status: "failed", ms_per_1000_samples: null, mae: null },
    ],
  };
  const rows = inferenceRows(payload, ["fast", "slow"], ["nasa"]);
  assert.deepEqual(rows.map((row) => row.model), ["fast", "slow"]);
}

{
  const summary = inferenceSummary({
    records: [
      { model: "a", status: "succeeded", ms_per_1000_samples: 10, samples_per_second: 100000, mae: 0.2 },
      { model: "b", status: "succeeded", ms_per_1000_samples: 5, samples_per_second: 200000, mae: 0.3 },
    ],
  });
  assert.equal(summary.fastest.model, "b");
  assert.equal(summary.bestTransfer.model, "a");
  assert.equal(summary.highestThroughput.model, "b");
}

{
  const payload = buildInferencePayload({
    models: ["m1", "m2"],
    datasets: ["nasa"],
  });
  assert.deepEqual(payload.models, ["m1", "m2"]);
  assert.deepEqual(payload.datasets, ["nasa"]);
}

{
  const compatibility = inferenceCompatibilitySummary({
    models: [
      {
        name: "ridge_full",
        featureColumns: ["voltage_mean", "ambient_temp_c"],
      },
      {
        name: "ridge_voltage",
        featureColumns: ["voltage_mean"],
      },
    ],
    datasets: [
      {
        name: "nasa",
        columns: ["dataset", "cell_id", "soh", "voltage_mean"],
      },
      {
        name: "uconn",
      },
    ],
  });
  assert.equal(compatibility.status, "warn");
  assert.equal(compatibility.okCount, 1);
  assert.equal(compatibility.warnCount, 1);
  assert.equal(compatibility.unknownCount, 2);
  assert.deepEqual(compatibility.rows.map((row) => [row.model, row.dataset, row.status, row.missingColumns]), [
    ["ridge_full", "nasa", "warn", ["ambient_temp_c"]],
    ["ridge_full", "uconn", "unknown", []],
    ["ridge_voltage", "nasa", "ok", []],
    ["ridge_voltage", "uconn", "unknown", []],
  ]);
  assert.match(compatibility.message, /ambient_temp_c/);
}

{
  const models = [
    { name: "ridge_voltage", featureColumns: ["voltage_mean"] },
    { name: "ridge_full", featureColumns: ["voltage_mean", "ambient_temp_c"] },
  ];
  const datasets = [
    { name: "nasa", columns: ["dataset", "soh", "voltage_mean", "ambient_temp_c"] },
    { name: "calce_cs2", columns: ["dataset", "soh", "voltage_mean"] },
    { name: "uconn" },
  ];
  const compatibleOnly = inferenceLaunchPlan({
    modelNames: ["ridge_voltage", "ridge_full"],
    datasetNames: ["nasa", "calce_cs2", "uconn"],
    models,
    datasets,
    strategy: "compatible_only",
  });
  assert.deepEqual(compatibleOnly.payload, {
    models: ["ridge_voltage", "ridge_full"],
    datasets: ["nasa"],
    launch_strategy: "compatible_only",
    selected_datasets: ["nasa", "calce_cs2", "uconn"],
    submitted_datasets: ["nasa"],
    skipped_datasets: ["calce_cs2", "uconn"],
  });
  assert.deepEqual(compatibleOnly.skippedDatasets, ["calce_cs2", "uconn"]);
  assert.equal(compatibleOnly.skippedCount, 2);
  assert.match(compatibleOnly.message, /2/);

  const includeRisk = inferenceLaunchPlan({
    modelNames: ["ridge_voltage", "ridge_full"],
    datasetNames: ["nasa", "calce_cs2", "uconn"],
    models,
    datasets,
    strategy: "include_risk",
  });
  assert.deepEqual(includeRisk.payload.datasets, ["nasa", "calce_cs2", "uconn"]);
  assert.equal(includeRisk.payload.launch_strategy, "include_risk");
  assert.deepEqual(includeRisk.payload.selected_datasets, ["nasa", "calce_cs2", "uconn"]);
  assert.deepEqual(includeRisk.payload.submitted_datasets, ["nasa", "calce_cs2", "uconn"]);
  assert.deepEqual(includeRisk.payload.skipped_datasets, []);
  assert.deepEqual(includeRisk.skippedDatasets, []);
}

{
  const staticSummary = inferenceAuditSummary({
    inferenceConfig: {
      job_id: "job-static",
      launch_strategy: "compatible_only",
      selected_datasets: ["nasa", "rwth_isea_public"],
      submitted_datasets: ["nasa"],
      skipped_datasets: ["rwth_isea_public"],
      run_dir: "runs/20260630T120000",
    },
  });
  assert.equal(staticSummary.hasAudit, true);
  assert.equal(staticSummary.strategyLabel, "仅运行兼容组合");
  assert.equal(staticSummary.selectedCount, 2);
  assert.equal(staticSummary.submittedCount, 1);
  assert.equal(staticSummary.skippedLabel, "rwth_isea_public");
  assert.equal(staticSummary.runDir, "runs/20260630T120000");

  const liveSummary = inferenceAuditSummary({
    inferenceConfig: {
      job_id: "job-static",
      launch_strategy: "compatible_only",
      skipped_datasets: ["rwth_isea_public"],
    },
    catalog: {
      inference_jobs: [{
        id: "job-legacy",
        status: "succeeded",
        request: {
          datasets: ["legacy"],
          models: ["old-model"],
        },
      }, {
        id: "job-live",
        status: "succeeded",
        run_dir: "runs/live",
        request: {
          launch_strategy: "include_risk",
          datasets: ["nasa", "rwth_isea_public"],
          selected_datasets: ["nasa", "rwth_isea_public"],
          submitted_datasets: ["nasa", "rwth_isea_public"],
          skipped_datasets: [],
        },
      }],
    },
  });
  assert.equal(liveSummary.jobId, "job-live");
  assert.equal(liveSummary.strategyLabel, "包含风险组合");
  assert.equal(liveSummary.skippedLabel, "无跳过数据集");
  assert.equal(liveSummary.sourceLabel, "后端任务记录");
}

{
  const history = taskHistoryRows(
    {
      jobs: [{
        id: "train-1",
        status: "succeeded",
        created_at: "2026-06-30 08:00:00",
        run_dir: "runs/train-1",
        request: {
          datasets: ["nasa", "hust"],
          estimators: ["ridge"],
          featuresets: ["voltage"],
        },
      }],
      inference_jobs: [{
        id: "infer-1",
        status: "succeeded",
        created_at: "2026-06-30 09:00:00",
        run_dir: "runs/infer-1",
        request: {
          models: ["ridge_voltage"],
          datasets: ["nasa", "hust"],
          launch_strategy: "compatible_only",
          selected_datasets: ["nasa", "hust", "rwth_isea_public"],
          submitted_datasets: ["nasa", "hust"],
          skipped_datasets: ["rwth_isea_public"],
        },
      }, {
        id: "infer-2",
        status: "failed",
        created_at: "2026-06-30 07:00:00",
        request: {
          models: ["ridge_full"],
          datasets: ["nasa"],
          launch_strategy: "include_risk",
        },
      }],
    },
    { kind: "inference", status: "all", strategy: "compatible_only", dataset: "rwth" },
  );

  assert.equal(history.totalCount, 3);
  assert.equal(history.filteredCount, 1);
  assert.deepEqual(history.rows.map((row) => row.id), ["infer-1"]);
  assert.equal(history.rows[0].kindLabel, "推理");
  assert.equal(history.rows[0].strategyLabel, "仅运行兼容组合");
  assert.equal(history.rows[0].datasetLabel, "nasa / hust / rwth_isea_public");
  assert.match(history.rows[0].detail, /ridge_voltage/);
  assert.equal(history.rows[0].rerunEndpoint, "/api/inference-jobs");
  assert.deepEqual(history.rows[0].rerunPayload, {
    models: ["ridge_voltage"],
    datasets: ["nasa", "hust"],
    launch_strategy: "compatible_only",
    selected_datasets: ["nasa", "hust", "rwth_isea_public"],
    submitted_datasets: ["nasa", "hust"],
    skipped_datasets: ["rwth_isea_public"],
  });
  assert.equal(history.rows[0].auditJson.kind, "inference");
  assert.equal(history.rows[0].auditJson.audit_version, 1);
  assert.equal(history.rows[0].auditJson.request.launch_strategy, "compatible_only");
  assert.equal(history.rows[0].auditJson.run_dir, "runs/infer-1");
  assert.equal(history.rows[0].auditJson.rerun_endpoint, "/api/inference-jobs");
  assert.deepEqual(history.rows[0].auditJson.rerun_payload, history.rows[0].rerunPayload);
  assert.match(history.rows[0].detailJson, /rwth_isea_public/);
  assert.deepEqual(history.statusOptions, ["failed", "succeeded"]);
}

{
  const history = taskHistoryRows(
    {
      jobs: [{
        id: "train-1",
        status: "failed",
        created_at: "2026-06-30 08:00:00",
        started_at: "2026-06-30 08:01:00",
        finished_at: "2026-06-30 08:02:00",
        raw_run_dir: "runs/raw-train-1",
        request: {
          datasets: ["nasa"],
          estimators: ["ridge"],
          featuresets: ["full"],
          protocols: ["loco", "lodo"],
          seeds: ["0", "1"],
        },
        error: "bad fold",
        traceback: "Traceback...",
        saved_models: ["ridge_full_loco"],
        model_warnings: ["warning"],
      }],
    },
    { kind: "training" },
  );
  const row = history.rows[0];
  assert.equal(row.rerunEndpoint, "/api/jobs");
  assert.deepEqual(row.rerunPayload, {
    datasets: ["nasa"],
    estimators: ["ridge"],
    featuresets: ["full"],
    protocols: ["loco", "lodo"],
    seeds: [0, 1],
  });
  assert.equal(row.auditJson.kind, "training");
  assert.equal(row.auditJson.raw_run_dir, "runs/raw-train-1");
  assert.equal(row.auditJson.traceback, "Traceback...");
  assert.deepEqual(row.auditJson.saved_models, ["ridge_full_loco"]);
  assert.deepEqual(row.auditJson.model_warnings, ["warning"]);
  assert.deepEqual(row.auditJson.rerun_payload, row.rerunPayload);
}

{
  const ready = inferenceActionState({
    backendConnected: true,
    selectedModelCount: 2,
    selectedDatasetCount: 3,
  });
  assert.equal(ready.disabled, false);
  assert.match(ready.message, /2 个模型 \/ 3 个数据集/);

  const running = inferenceActionState({
    backendConnected: true,
    selectedModelCount: 1,
    selectedDatasetCount: 1,
    currentJob: { kind: "inference", id: "job-1", status: "running" },
  });
  assert.equal(running.disabled, true);
  assert.match(running.message, /job-1/);
  assert.match(running.message, /正在执行/);
}

{
  const summary = trainingArtifactSummary(
    { datasets: ["nasa", "hust", "rwth_isea_public"] },
    {
      models: [
        { name: "ridge_full", source: "training" },
        { name: "gbdt_ic_dva", source: "training" },
        { name: "external_xgb", source: "upload" },
      ],
    },
  );
  assert.equal(summary.pickleCount, 3);
  assert.equal(summary.generatedCount, 2);
  assert.equal(summary.uploadedCount, 1);
  assert.equal(summary.datasetCount, 3);
  assert.equal(summary.label, "3 个 Pickle 模型");
}

{
  const insight = inferenceInsightSummary([
    { model: "gbdt", dataset: "nasa", status: "succeeded", mae: 0.1, ms_per_1000_samples: 6 },
    { model: "ridge", dataset: "nasa", status: "succeeded", mae: 0.2, ms_per_1000_samples: 3 },
    { model: "mlp", dataset: "rwth", status: "failed", error: "missing feature" },
  ]);
  assert.equal(insight.succeededCount, 2);
  assert.equal(insight.failedCount, 1);
  assert.equal(insight.bestTransfer.model, "gbdt");
  assert.equal(insight.fastest.model, "ridge");
  assert.match(insight.recommendation, /gbdt/);
  assert.match(insight.risk, /1/);
}

{
  const insight = inferenceInsightSummary([]);
  assert.equal(insight.succeededCount, 0);
  assert.equal(insight.failedCount, 0);
  assert.equal(insight.bestTransfer, null);
  assert.match(insight.risk, /暂无成功推理记录/);
}

{
  const assets = modelAssetRows(
    {
      models: [
        { name: "ridge_full", source: "training", estimator: "ridge", featureset: "full", trained_on: ["nasa"], feature_columns: ["a", "b"] },
        { name: "external_xgb", source: "upload", estimator: "xgb", featureset: "ic_dva", feature_columns: ["a"] },
      ],
    },
    {
      records: [
        { model: "ridge_full", status: "succeeded", mae: 0.2, ms_per_1000_samples: 5 },
        { model: "external_xgb", status: "succeeded", mae: 0.1, ms_per_1000_samples: 9 },
      ],
    },
  );
  assert.deepEqual(assets.map((row) => row.name), ["external_xgb", "ridge_full"]);
  assert.equal(assets[0].latestMae, 0.1);
  assert.equal(assets[0].featureColumnCount, 1);
  assert.equal(assets[1].trainedOn.join(","), "nasa");
}

{
  const partial = inferenceActionState({
    backendConnected: true,
    selectedModelCount: 1,
    selectedDatasetCount: 1,
    currentJob: { kind: "inference", id: "job-partial", status: "partial" },
  });
  assert.equal(partial.disabled, false);
  assert.equal(partial.tone, "warn");
  assert.match(partial.message, /job-partial/);
}

{
  const history = taskHistoryRows({
    jobs: [{
      id: "train-partial",
      status: "partial",
      request: { datasets: ["nasa"], estimators: ["ridge"], featuresets: ["full"] },
    }],
  });
  assert.deepEqual(history.statusOptions, ["partial"]);
  assert.equal(history.rows[0].statusLabel, "部分完成");
}
