"use strict";

export function metricMean(entry, name) {
  const value = entry?.metrics?.[name]?.mean;
  return value == null || Number.isNaN(value) ? null : value;
}

export function datasetOptions(config = {}, qc = {}, datasetRows = []) {
  const names = new Set(config.datasets || []);
  Object.keys(qc?.per_dataset || {}).forEach((name) => names.add(name));
  datasetRows.forEach((row) => {
    const name = heldOutDatasetName(row?.held_out_key);
    if (name) names.add(name);
  });

  const metricNames = new Set(datasetRows.map((row) => heldOutDatasetName(row.held_out_key)).filter(Boolean));
  const totalRows = Object.values(qc?.per_dataset || {})
    .reduce((sum, item) => sum + Number(item?.rows || 0), 0);

  return [...names].map((name) => {
    const rows = Number(qc?.per_dataset?.[name]?.rows || 0);
    return {
      name,
      rows,
      share: totalRows > 0 ? rows / totalRows : 0,
      hasMetrics: metricNames.has(name),
    };
  }).sort((a, b) => (b.rows - a.rows) || a.name.localeCompare(b.name));
}

export function summarizeBySelection(aggregateRows = [], datasetRows = [], selectedDatasets = [], allDatasets = []) {
  const all = new Set(allDatasets);
  const selected = normalizeSelection(selectedDatasets, allDatasets);
  const coversAll = all.size > 0 && selected.size === all.size && [...all].every((name) => selected.has(name));

  if (coversAll || !datasetRows.length) {
    return {
      entries: aggregateRows,
      selectedDatasets: [...selected],
      usedDatasetLevel: false,
      scopeLabel: "全部数据集",
    };
  }

  const filtered = datasetRows.filter((row) => selected.has(heldOutDatasetName(row.held_out_key)));
  if (!filtered.length) {
    return {
      entries: aggregateRows,
      selectedDatasets: [...selected],
      usedDatasetLevel: false,
      scopeLabel: "全部数据集",
    };
  }

  return {
    entries: rollupEntries(filtered),
    selectedDatasets: [...selected],
    usedDatasetLevel: true,
    scopeLabel: [...selected].join(", "),
  };
}

export function nextDatasetSelection(currentSelection = [], datasetName, allDatasets = [], action = "only") {
  const all = (allDatasets || []).filter(Boolean);
  if (action === "all") return [...all];

  if (!datasetName) return normalizeSelection(currentSelection, allDatasets);
  if (action === "only") return [datasetName];

  const selected = normalizeSelection(currentSelection, allDatasets);
  if (selected.has(datasetName)) {
    selected.delete(datasetName);
  } else {
    selected.add(datasetName);
  }

  return selected.size ? [...selected] : [...all];
}

export function catalogOptions(catalog = {}) {
  return {
    datasets: normalizeCatalogItems(catalog.datasets || [], (item) => ({
      rows: Number(item.rows || 0),
      columns: item.columns || [],
      badge: item.uploaded ? "上传" : "数据",
    })),
    estimators: normalizeCatalogItems(catalog.estimators || [], (item) => ({
      featureMode: item.feature_mode || "unified",
      backend: item.backend || "external",
      badge: item.uploaded ? "上传" : (item.backend || "内置"),
    })),
    featuresets: normalizeCatalogItems(catalog.featuresets || [], (item) => ({
      columns: item.columns || [],
      hasTransform: Boolean(item.has_transform),
      badge: item.has_transform ? "变换" : (item.uploaded ? "上传" : "内置"),
    })),
  };
}

export function modelOptions(source = {}) {
  const models = Array.isArray(source.models) ? source.models : [];
  return models
    .map((item) => ({
      name: String(item.name || ""),
      source: item.source || "training",
      estimator: item.estimator || "",
      featureset: item.featureset || "",
      featureColumns: item.feature_columns || item.featureColumns || [],
      trainedOn: item.trained_on || item.trainedOn || [],
      badge: item.source === "upload" ? "涓婁紶妯″瀷" : "璁粌浜х墿",
    }))
    .filter((item) => item.name)
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function inferenceRows(payload = {}, selectedModels = [], selectedDatasets = []) {
  const modelSet = new Set(selectedModels || []);
  const datasetSet = new Set(selectedDatasets || []);
  return (payload.records || [])
    .filter((row) => !modelSet.size || modelSet.has(row.model))
    .filter((row) => !datasetSet.size || datasetSet.has(row.dataset))
    .sort((a, b) => {
      const av = a.ms_per_1000_samples ?? Number.POSITIVE_INFINITY;
      const bv = b.ms_per_1000_samples ?? Number.POSITIVE_INFINITY;
      return av - bv || String(a.model).localeCompare(String(b.model));
    });
}

export function inferenceSummary(payload = {}) {
  const ok = (payload.records || []).filter((row) => row.status === "succeeded");
  const fastest = [...ok].sort((a, b) =>
    (a.ms_per_1000_samples ?? Infinity) - (b.ms_per_1000_samples ?? Infinity))[0] || null;
  const bestTransfer = ok
    .filter((row) => row.mae != null)
    .sort((a, b) => a.mae - b.mae)[0] || null;
  const highestThroughput = [...ok].sort((a, b) =>
    (b.samples_per_second ?? 0) - (a.samples_per_second ?? 0))[0] || null;
  return { count: ok.length, fastest, bestTransfer, highestThroughput };
}

export function inferenceInsightSummary(records = []) {
  const succeeded = (records || []).filter((row) => row.status === "succeeded");
  const failed = (records || []).filter((row) => row.status === "failed");
  const summary = inferenceSummary({ records });
  const bestName = summary.bestTransfer?.model || "暂无模型";
  const fastestName = summary.fastest?.model || "暂无模型";
  return {
    succeededCount: succeeded.length,
    failedCount: failed.length,
    bestTransfer: summary.bestTransfer,
    fastest: summary.fastest,
    highestThroughput: summary.highestThroughput,
    recommendation: summary.bestTransfer
      ? `${bestName} 当前迁移精度最好，可优先进入模型库复核。`
      : "暂无成功推理记录，先选择模型和数据集启动推理测评。",
    risk: failed.length
      ? `${failed.length} 个组合未完成，优先检查 Pickle 模型的特征列兼容性。`
      : (summary.fastest
        ? `${fastestName} 当前推理速度表现最好，仍需结合 MAE 判断是否可部署。`
        : "暂无成功推理记录，完成 inference-only 测评后再判断迁移风险。"),
  };
}

export function buildInferencePayload({ models = [], datasets = [] } = {}) {
  return {
    models: nonEmptyStrings(models),
    datasets: nonEmptyStrings(datasets),
  };
}

export function trainingArtifactSummary(config = {}, modelCatalog = {}) {
  const models = Array.isArray(modelCatalog.models) ? modelCatalog.models : [];
  const generatedCount = models.filter((item) => item.source !== "upload").length;
  const uploadedCount = models.filter((item) => item.source === "upload").length;
  const pickleCount = models.length;
  const datasetCount = Array.isArray(config.datasets) ? config.datasets.length : 0;
  return {
    pickleCount,
    datasetCount,
    generatedCount,
    uploadedCount,
    label: `${pickleCount} 个 Pickle 模型`,
  };
}

export function modelAssetRows(modelCatalog = {}, inferencePayload = {}) {
  const latestByModel = new Map();
  (inferencePayload.records || []).forEach((row) => {
    if (!row?.model || row.status !== "succeeded") return;
    const current = latestByModel.get(row.model);
    const currentMae = current?.mae ?? Infinity;
    const nextMae = row.mae ?? Infinity;
    if (!current || nextMae < currentMae) latestByModel.set(row.model, row);
  });

  return modelOptions(modelCatalog).map((model) => {
    const latest = latestByModel.get(model.name);
    return {
      ...model,
      trainedOn: model.trainedOn || [],
      featureColumnCount: (model.featureColumns || []).length,
      latestMae: latest?.mae ?? null,
      latestSpeed: latest?.ms_per_1000_samples ?? null,
      latestDataset: latest?.dataset || "",
    };
  }).sort((a, b) => {
    const am = a.latestMae ?? Infinity;
    const bm = b.latestMae ?? Infinity;
    return am - bm || a.name.localeCompare(b.name);
  });
}

export function buildEvaluationPayload({
  datasets = [],
  estimators = [],
  featuresets = [],
  protocols = ["loco"],
  seedsText = "0",
} = {}) {
  return {
    datasets: nonEmptyStrings(datasets),
    estimators: nonEmptyStrings(estimators),
    featuresets: nonEmptyStrings(featuresets),
    protocols: nonEmptyStrings(protocols),
    seeds: parseSeeds(seedsText),
  };
}

export function scopeSummary({
  datasetOptions: options = [],
  selectedDatasets = [],
  scoped = {},
  config = {},
  fallback = false,
} = {}) {
  const selectedSet = new Set(normalizeSelection(selectedDatasets, options.map((item) => item.name)));
  const selected = options.filter((item) => selectedSet.has(item.name));
  const totalRows = options.reduce((sum, item) => sum + Number(item.rows || 0), 0);
  const selectedRows = selected.reduce((sum, item) => sum + Number(item.rows || 0), 0);
  const coverageKind = selected.length === options.length ? "full" : "partial";
  const protocols = config.protocols?.length ? config.protocols : [];
  const seeds = config.seeds?.length ? config.seeds : config.stability_seeds || [];

  return {
    datasetLabel: `${selected.length}/${options.length}`,
    rowLabel: selectedRows.toLocaleString(),
    shareLabel: totalRows ? `${(100 * selectedRows / totalRows).toFixed(1)}%` : "无",
    selectedNames: selected.map((item) => item.name),
    totalRows,
    selectedRows,
    protocolLabel: protocols.length ? protocols.join(", ") : "未记录",
    seedLabel: seeds.length ? String(seeds.length) : "未记录",
    sourceLabel: fallback ? "示例" : "实测",
    coverageKind,
    metricMode: scoped.usedDatasetLevel ? "按选择聚合" : "全量聚合",
    scopeLabel: scoped.scopeLabel || "全部数据集",
    notice: scoped.usedDatasetLevel
      ? "当前结果按所选 held-out 数据集重新聚合。"
      : "当前展示全量聚合结果。",
  };
}

export function evaluationPlanSummary({
  datasets = [],
  estimators = [],
  featuresets = [],
  protocols = [],
  seeds = [],
} = {}) {
  const counts = {
    datasets: nonEmptyStrings(datasets).length,
    estimators: nonEmptyStrings(estimators).length,
    featuresets: nonEmptyStrings(featuresets).length,
    protocols: nonEmptyStrings(protocols).length,
    seeds: Array.isArray(seeds) ? seeds.length : parseSeeds(seeds).length,
  };
  const comboCount = counts.datasets * counts.estimators * counts.featuresets * counts.protocols * counts.seeds;
  const ready = Object.values(counts).every((count) => count > 0);
  return {
    ...counts,
    comboCount,
    comboLabel: ready ? `${comboCount.toLocaleString()} 个组合` : "选择不完整",
    ready,
    partsLabel: `${counts.datasets} 数据集 x ${counts.estimators} 算法 x ${counts.featuresets} 特征 x ${counts.protocols} 协议 x ${counts.seeds} seeds`,
  };
}

export function leaderboardCoverage(entry, selectedDatasets = [], datasetRows = []) {
  const selected = new Set((selectedDatasets || []).filter(Boolean));
  const sourceDatasets = entry?.source_datasets?.length
    ? entry.source_datasets
    : matchingDatasetRows(entry, datasetRows).map((row) => heldOutDatasetName(row.held_out_key)).filter(Boolean);
  const unique = [...new Set(sourceDatasets)];

  if (!unique.length) {
    return {
      kind: "assumed",
      rank: 0,
      label: "全量",
      count: selected.size,
      total: selected.size,
      detail: "未记录数据集级明细",
      warning: false,
    };
  }

  const covered = selected.size ? unique.filter((name) => selected.has(name)) : unique;
  const total = selected.size || unique.length;
  const count = covered.length;
  const kind = count >= total ? "full" : count <= 1 ? "local" : "partial";
  return {
    kind,
    rank: coverageRank(kind),
    label: `${count}/${total}`,
    count,
    total,
    detail: covered.join(" / ") || "未覆盖当前选择",
    warning: kind !== "full",
  };
}

export function bestComparableEntry(rows = [], selectedDatasets = [], datasetRows = []) {
  return rows
    .filter((entry) => metricMean(entry, "mae") != null)
    .map((entry) => ({
      entry,
      coverage: leaderboardCoverage(entry, selectedDatasets, datasetRows),
    }))
    .sort((a, b) =>
      (a.coverage.rank - b.coverage.rank) ||
      (b.coverage.count - a.coverage.count) ||
      (metricMean(a.entry, "mae") - metricMean(b.entry, "mae")))
    [0]?.entry || null;
}

export function algorithmInsights(rows = []) {
  const valid = rows.filter((entry) => metricMean(entry, "mae") != null);
  return {
    bestAccuracy: minBy(valid, (entry) => metricMean(entry, "mae")),
    bestCalibration: minBy(valid.filter((entry) => metricMean(entry, "picp") != null),
      (entry) => Math.abs(metricMean(entry, "picp") - 0.9)),
    bestPhysical: minBy(valid.filter((entry) => metricMean(entry, "monotonicity_violation_rate") != null),
      (entry) => metricMean(entry, "monotonicity_violation_rate")),
  };
}

export function stabilityRows(config = {}) {
  const seedCount = Array.isArray(config.stability_seeds) ? config.stability_seeds.length : 0;
  return (config.stability_summary || [])
    .filter((row) => row?.mae_mean != null)
    .map((row) => {
      const mean = Number(row.mae_mean);
      const std = Number(row.mae_std || 0);
      const min = Number(row.mae_min ?? mean);
      const max = Number(row.mae_max ?? mean);
      const cv = mean > 0 ? std / mean : 0;
      return {
        estimator: row.estimator,
        featureset: row.featureset,
        label: `${row.estimator} x ${row.featureset}`,
        seedCount,
        mean,
        std,
        min,
        max,
        cv,
        rangeText: `${min.toFixed(3)}-${max.toFixed(3)}`,
        stabilityLabel: stabilityLabel(cv),
      };
    })
    .sort((a, b) => (a.mean - b.mean) || (a.std - b.std));
}

function stabilityLabel(cv) {
  if (cv <= 0.08) return "较稳";
  if (cv <= 0.15) return "中等";
  return "波动较大";
}

function normalizeSelection(selectedDatasets, allDatasets) {
  const selected = new Set((selectedDatasets || []).filter(Boolean));
  if (!selected.size) {
    (allDatasets || []).forEach((name) => selected.add(name));
  }
  return selected;
}

function heldOutDatasetName(value) {
  if (!value) return null;
  return String(value).split("/", 1)[0];
}

function matchingDatasetRows(entry, rows) {
  if (!entry) return [];
  const key = comboKey(entry);
  return (rows || []).filter((row) => comboKey(row) === key);
}

function comboKey(row) {
  return [
    row.estimator,
    row.featureset || "full",
    row.protocol || "",
    row.track || "unified",
  ].join("|");
}

function coverageRank(kind) {
  if (kind === "full" || kind === "assumed") return 0;
  if (kind === "partial") return 1;
  return 2;
}

function normalizeCatalogItems(items, decorate) {
  return items
    .filter((item) => item?.name)
    .map((item) => ({ ...item, ...decorate(item), name: String(item.name) }));
}

function nonEmptyStrings(values) {
  return (values || []).map((value) => String(value).trim()).filter(Boolean);
}

function parseSeeds(value) {
  const seeds = String(value || "")
    .split(/[,\s]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number.parseInt(item, 10))
    .filter((item) => Number.isInteger(item));
  return seeds.length ? [...new Set(seeds)] : [0];
}

function rollupEntries(rows) {
  const groups = new Map();
  rows.forEach((row) => {
    const key = comboKey(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });

  return [...groups.values()].map((items) => mergeGroup(items));
}

function mergeGroup(items) {
  const base = items[0];
  const metricNames = new Set();
  items.forEach((item) => {
    Object.keys(item.metrics || {}).forEach((name) => metricNames.add(name));
  });

  const metrics = {};
  metricNames.forEach((name) => {
    const values = items.map((item) => metricMean(item, name)).filter((value) => value != null);
    metrics[name] = values.length
      ? { mean: average(values), std: stddev(values) }
      : { mean: null, std: 0 };
  });

  return {
    estimator: base.estimator,
    featureset: base.featureset || "full",
    protocol: base.protocol,
    track: base.track || "unified",
    backend: base.backend,
    ci_source: sameValue(items.map((item) => item.ci_source)) || "mixed",
    source_datasets: [...new Set(items.map((item) => heldOutDatasetName(item.held_out_key)).filter(Boolean))],
    metrics,
    verdicts: mergeVerdicts(items.map((item) => item.verdicts || {})),
  };
}

function mergeVerdicts(verdicts) {
  const names = new Set();
  verdicts.forEach((verdict) => Object.keys(verdict || {}).forEach((name) => names.add(name)));
  const merged = {};
  names.forEach((name) => {
    const values = verdicts.map((verdict) => verdict[name]).filter((value) => value != null);
    merged[name] = values.length ? values.every(Boolean) : null;
  });
  return merged;
}

function sameValue(values) {
  const unique = [...new Set(values.filter((value) => value != null))];
  return unique.length === 1 ? unique[0] : null;
}

function minBy(items, score) {
  if (!items.length) return null;
  return items.reduce((best, item) => score(item) < score(best) ? item : best, items[0]);
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stddev(values) {
  if (values.length < 2) return 0;
  const mean = average(values);
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}
