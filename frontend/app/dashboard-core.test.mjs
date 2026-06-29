import assert from "node:assert/strict";
import {
  summarizeBySelection,
  datasetOptions,
  algorithmInsights,
  bestComparableEntry,
  buildEvaluationPayload,
  buildInferencePayload,
  catalogOptions,
  evaluationPlanSummary,
  inferenceRows,
  inferenceSummary,
  leaderboardCoverage,
  modelOptions,
  scopeSummary,
  stabilityRows,
  nextDatasetSelection,
} from "./dashboard-core.mjs";

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

function entry(estimator, featureset, mae, opts = {}) {
  return {
    estimator,
    featureset,
    protocol: "lodo",
    track: "unified",
    held_out_key: opts.held,
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
  assert.equal(rows[0].badge, "涓婁紶妯″瀷");
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
