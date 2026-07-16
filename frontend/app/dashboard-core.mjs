"use strict";

export function metricMean(entry, name) {
  const value = entry?.metrics?.[name]?.mean;
  return value == null || Number.isNaN(value) ? null : value;
}

export function evaluationIdentity(entry = {}) {
  const candidate = String(entry?.candidate_id || "").trim();
  const estimator = String(entry?.estimator || "").trim();
  const protocol = String(entry?.protocol || "").trim();
  const track = String(entry?.track || "unified").trim() || "unified";
  return JSON.stringify([
    candidate ? "candidate" : "estimator",
    candidate,
    estimator,
    protocol,
    track,
  ]);
}

export function evaluationLabel(entry = {}) {
  const candidate = String(entry?.candidate_id || "").trim();
  const estimator = String(entry?.estimator || "").trim();
  const protocol = String(entry?.protocol || "").trim();
  const track = String(entry?.track || "unified").trim() || "unified";
  const base = candidate && estimator && candidate !== estimator
    ? `${candidate} (${estimator})`
    : candidate || estimator;
  const qualifiers = [];
  if (protocol && protocol !== "lodo") qualifiers.push(protocol);
  if (track !== "unified") qualifiers.push(track);
  return qualifiers.length ? `${base} [${qualifiers.join(" / ")}]` : base;
}

export function evaluationCellKey(entry = {}, featureset = entry?.featureset || "full") {
  return JSON.stringify([evaluationIdentity(entry), String(featureset || "full")]);
}

export function evaluationMatrixRows(entries = [], featureSets = []) {
  const rows = new Map();
  const cells = new Map();
  entries.forEach((entry) => {
    const identity = evaluationIdentity(entry);
    if (!rows.has(identity)) rows.set(identity, entry);
    cells.set(evaluationCellKey(entry), metricMean(entry, "mae"));
  });

  return [...rows.entries()].map(([identity, entry]) => ({
    identity,
    label: evaluationLabel(entry),
    cells: featureSets.map((featureset) => {
      const key = evaluationCellKey(entry, featureset);
      return {
        featureset,
        value: cells.has(key) ? cells.get(key) : null,
      };
    }),
  }));
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

export function inferenceDatasetOptions({ baseOptions = [], catalog = {}, inference = {} } = {}) {
  const byName = new Map();
  const inferenceNames = new Set(
    (inference.records || [])
      .map((row) => String(row?.dataset || ""))
      .filter(Boolean),
  );

  const add = (item = {}, source = "") => {
    const name = String(item.name || item.dataset || "");
    if (!name) return;
    const current = byName.get(name) || { name, rows: 0, share: 0, hasMetrics: false };
    byName.set(name, {
      ...current,
      ...item,
      name,
      rows: Math.max(Number(current.rows || 0), Number(item.rows || item.n_samples || 0)),
      hasMetrics: Boolean(current.hasMetrics || item.hasMetrics),
      hasInference: Boolean(current.hasInference || inferenceNames.has(name) || source === "inference"),
    });
  };

  (baseOptions || []).forEach((item) => add(item, "base"));
  (catalog.datasets || []).forEach((item) => add(item, "catalog"));
  (inference.records || []).forEach((row) => add({ name: row.dataset, n_samples: row.n_samples }, "inference"));

  return [...byName.values()].sort((a, b) => {
    const ai = a.hasInference ? 0 : 1;
    const bi = b.hasInference ? 0 : 1;
    return ai - bi || (b.rows - a.rows) || a.name.localeCompare(b.name);
  });
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
      badge: item.source === "upload" ? "上传模型" : "训练产物",
    }))
    .filter((item) => item.name)
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function preferredInferenceModels(models = [], inference = {}) {
  const recordedModels = new Set(
    (inference.records || [])
      .map((row) => String(row?.model || ""))
      .filter(Boolean),
  );
  const preferred = (models || []).find((item) => recordedModels.has(item?.name));
  if (preferred?.name) return [preferred.name];
  const first = (models || []).find((item) => item?.name);
  return first?.name ? [first.name] : [];
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

export function profileOptions(index = {}, activeProfile = "root") {
  const rows = [
    {
      name: "root",
      label: "当前根目录",
      kind: "root",
      datasets: [],
      active: activeProfile === "root",
    },
  ];
  const seen = new Set(["root"]);
  (index.profiles || []).forEach((item = {}) => {
    const name = String(item.name || "").trim();
    if (!/^[A-Za-z0-9_][A-Za-z0-9_-]*$/.test(name) || seen.has(name)) return;
    seen.add(name);
    rows.push({
      name,
      label: String(item.label || name),
      kind: String(item.kind || "evaluation_profile"),
      datasets: nonEmptyStrings(item.datasets || []),
      trainingJobId: String(item.training_job_id || ""),
      inferenceJobId: String(item.inference_job_id || ""),
      active: name === activeProfile,
    });
  });
  if (!rows.some((row) => row.active)) rows[0].active = true;
  return rows;
}

export function buildInferencePayload({ models = [], datasets = [] } = {}) {
  return {
    models: nonEmptyStrings(models),
    datasets: nonEmptyStrings(datasets),
  };
}

export function inferenceCompatibilitySummary({ models = [], datasets = [] } = {}) {
  const rows = [];
  (models || []).forEach((model) => {
    const required = nonEmptyStrings(model?.featureColumns || model?.feature_columns || []);
    (datasets || []).forEach((dataset) => {
      const available = nonEmptyStrings(dataset?.columns || []);
      if (!required.length || !available.length) {
        rows.push({
          model: model?.name || "",
          dataset: dataset?.name || "",
          status: "unknown",
          missingColumns: [],
          requiredCount: required.length,
          availableCount: available.length,
        });
        return;
      }
      const availableSet = new Set(available);
      const missingColumns = required.filter((column) => !availableSet.has(column));
      rows.push({
        model: model?.name || "",
        dataset: dataset?.name || "",
        status: missingColumns.length ? "warn" : "ok",
        missingColumns,
        requiredCount: required.length,
        availableCount: available.length,
      });
    });
  });

  const okCount = rows.filter((row) => row.status === "ok").length;
  const warnRows = rows.filter((row) => row.status === "warn");
  const unknownCount = rows.filter((row) => row.status === "unknown").length;
  const firstMissing = warnRows[0]?.missingColumns?.slice(0, 3).join(", ") || "";
  const status = warnRows.length ? "warn" : (unknownCount ? "unknown" : (rows.length ? "ok" : "idle"));
  const message = warnRows.length
    ? `${warnRows.length} 个模型-数据集组合缺少特征列：${firstMissing}`
    : (unknownCount
      ? `${unknownCount} 个组合缺少 manifest 或数据集列清单，需由后端运行时确认。`
      : (rows.length ? "已选模型和数据集的特征列兼容。" : "请选择模型和数据集后进行 schema 预检。"));

  return {
    rows,
    status,
    okCount,
    warnCount: warnRows.length,
    unknownCount,
    message,
  };
}

export function inferenceLaunchPlan({
  modelNames = [],
  datasetNames = [],
  models = [],
  datasets = [],
  strategy = "compatible_only",
} = {}) {
  const selectedModelNames = nonEmptyStrings(modelNames);
  const selectedDatasetNames = nonEmptyStrings(datasetNames);
  if (strategy === "include_risk") {
    return {
      strategy,
      payload: {
        ...buildInferencePayload({ models: selectedModelNames, datasets: selectedDatasetNames }),
        launch_strategy: "include_risk",
        selected_datasets: selectedDatasetNames,
        submitted_datasets: selectedDatasetNames,
        skipped_datasets: [],
      },
      skippedDatasets: [],
      skippedCount: 0,
      message: "已选择包含风险组合，后端会执行运行时 schema 校验。",
    };
  }

  const modelByName = new Map((models || []).map((item) => [item?.name, item]).filter(([name]) => name));
  const datasetByName = new Map((datasets || []).map((item) => [item?.name, item]).filter(([name]) => name));
  const selectedModels = selectedModelNames.map((name) => modelByName.get(name) || { name });
  const selectedDatasets = selectedDatasetNames.map((name) => datasetByName.get(name) || { name });
  const compatibility = inferenceCompatibilitySummary({ models: selectedModels, datasets: selectedDatasets });
  const riskyDatasets = new Set(
    compatibility.rows
      .filter((row) => row.status !== "ok")
      .map((row) => row.dataset)
      .filter(Boolean),
  );
  const allowedDatasets = selectedDatasetNames.filter((name) => !riskyDatasets.has(name));
  const skippedDatasets = selectedDatasetNames.filter((name) => riskyDatasets.has(name));

  return {
    strategy: "compatible_only",
    payload: {
      ...buildInferencePayload({ models: selectedModelNames, datasets: allowedDatasets }),
      launch_strategy: "compatible_only",
      selected_datasets: selectedDatasetNames,
      submitted_datasets: allowedDatasets,
      skipped_datasets: skippedDatasets,
    },
    skippedDatasets,
    skippedCount: skippedDatasets.length,
    compatibility,
    message: skippedDatasets.length
      ? `已跳过 ${skippedDatasets.length} 个存在 schema 风险的数据集：${skippedDatasets.slice(0, 3).join(", ")}`
      : "所有已选数据集通过 schema 预检。",
  };
}

export function inferenceAuditSummary({
  inferenceConfig = {},
  catalog = {},
  currentJob = null,
} = {}) {
  const liveJob = currentJob?.kind === "inference" && currentJob?.request ? currentJob : null;
  const historyJobs = catalog?.inference_jobs || [];
  const historyJob = historyJobs.find((job) => hasInferenceAuditFields(job?.request))
    || historyJobs.find((job) => job?.request)
    || null;
  const job = liveJob || historyJob;
  const request = job?.request || inferenceConfig || {};
  const strategy = String(request.launch_strategy || request.launchStrategy || "unspecified");
  const selectedDatasets = nonEmptyStrings(request.selected_datasets || request.selectedDatasets || request.datasets || []);
  const submittedDatasets = nonEmptyStrings(request.submitted_datasets || request.submittedDatasets || request.datasets || []);
  const skippedDatasets = nonEmptyStrings(request.skipped_datasets || request.skippedDatasets || []);
  const jobId = String(job?.id || request.job_id || request.jobId || "");
  const status = String(job?.status || request.status || "");
  const runDir = String(job?.run_dir || job?.raw_run_dir || request.run_dir || request.raw_run_dir || request.runDir || "");
  const strategyLabel = strategy === "compatible_only"
    ? "仅运行兼容组合"
    : (strategy === "include_risk" ? "包含风险组合" : "未记录");
  const skippedLabel = skippedDatasets.length ? skippedDatasets.slice(0, 4).join(" / ") : "无跳过数据集";
  const hasAudit = Boolean(
    strategy !== "unspecified"
    || selectedDatasets.length
    || submittedDatasets.length
    || skippedDatasets.length
    || jobId
    || runDir,
  );

  return {
    hasAudit,
    sourceLabel: job ? (liveJob ? "当前任务" : "后端任务记录") : (hasAudit ? "前端运行配置" : "暂无记录"),
    strategy,
    strategyLabel,
    selectedDatasets,
    submittedDatasets,
    skippedDatasets,
    selectedCount: selectedDatasets.length,
    submittedCount: submittedDatasets.length,
    skippedCount: skippedDatasets.length,
    skippedLabel,
    jobId,
    status,
    runDir,
    summary: hasAudit
      ? `${strategyLabel}；提交 ${submittedDatasets.length} / 选择 ${selectedDatasets.length} 个数据集；跳过 ${skippedDatasets.length} 个。`
      : "暂无推理策略审计记录。",
  };
}

function hasInferenceAuditFields(request = {}) {
  return Boolean(
    request?.launch_strategy
    || request?.launchStrategy
    || request?.selected_datasets
    || request?.submitted_datasets
    || request?.skipped_datasets,
  );
}

export function taskHistoryRows(catalog = {}, filters = {}) {
  const kindFilter = filters.kind || "all";
  const statusFilter = filters.status || "all";
  const strategyFilter = filters.strategy || "all";
  const datasetFilter = String(filters.dataset || "").trim().toLowerCase();
  const trainingRows = (catalog.jobs || []).map((job) => taskHistoryRow(job, "training"));
  const inferenceRows = (catalog.inference_jobs || []).map((job) => taskHistoryRow(job, "inference"));
  const allRows = [...trainingRows, ...inferenceRows]
    .sort((a, b) => b.sortKey.localeCompare(a.sortKey) || b.id.localeCompare(a.id));
  const statusOptions = [...new Set(allRows.map((row) => row.status).filter(Boolean))].sort();
  const rows = allRows.filter((row) => {
    if (kindFilter !== "all" && row.kind !== kindFilter) return false;
    if (statusFilter !== "all" && row.status !== statusFilter) return false;
    if (strategyFilter !== "all" && row.strategy !== strategyFilter) return false;
    if (datasetFilter && !row.datasetLabel.toLowerCase().includes(datasetFilter)) return false;
    return true;
  });
  return {
    rows,
    totalCount: allRows.length,
    filteredCount: rows.length,
    statusOptions,
  };
}

function taskHistoryRow(job = {}, kind = "training") {
  const request = job.request || {};
  const audit = kind === "inference" ? inferenceAuditSummary({ currentJob: { kind: "inference", ...job } }) : null;
  const datasets = kind === "inference"
    ? uniqueStrings([
        ...(request.selected_datasets || []),
        ...(request.submitted_datasets || []),
        ...(request.skipped_datasets || []),
        ...(request.datasets || []),
      ])
    : nonEmptyStrings(request.datasets || []);
  const detailParts = kind === "inference"
    ? nonEmptyStrings(request.models || [])
    : [
        ...nonEmptyStrings(request.estimators || []),
        ...nonEmptyStrings(request.featuresets || []),
      ];
  const strategy = kind === "inference" ? (audit?.strategy || "unspecified") : "training";
  const strategyLabel = kind === "inference" ? (audit?.strategyLabel || "未记录") : "训练任务";
  const createdAt = String(job.created_at || "");
  const finishedAt = String(job.finished_at || "");
  const runDir = String(job.run_dir || job.raw_run_dir || "");
  const requestCopy = cloneJson(request);
  const rerunEndpoint = kind === "inference" ? "/api/inference-jobs" : "/api/jobs";
  const rerunPayload = rerunPayloadForTask(kind, requestCopy);
  const auditJson = {
    audit_version: 1,
    id: String(job.id || ""),
    kind,
    status: String(job.status || "unknown"),
    created_at: createdAt,
    started_at: job.started_at || "",
    finished_at: finishedAt,
    run_dir: runDir,
    raw_run_dir: String(job.raw_run_dir || ""),
    request: requestCopy,
    error: job.error || "",
    traceback: job.traceback || "",
    saved_models: cloneJson(job.saved_models || []),
    model_warnings: cloneJson(job.model_warnings || []),
    rerun_endpoint: rerunEndpoint,
    rerun_payload: rerunPayload,
  };
  return {
    id: String(job.id || ""),
    kind,
    kindLabel: kind === "inference" ? "推理" : "训练",
    status: String(job.status || "unknown"),
    statusLabel: taskStatusLabel(job.status),
    strategy,
    strategyLabel,
    createdAt,
    finishedAt,
    runDir,
    rawRunDir: String(job.raw_run_dir || ""),
    error: String(job.error || ""),
    datasets,
    datasetLabel: datasets.length ? datasets.join(" / ") : "未记录",
    detail: detailParts.length ? detailParts.join(" / ") : (kind === "inference" ? "未记录模型" : "未记录资源"),
    skippedLabel: audit?.skippedLabel || "",
    skippedCount: audit?.skippedCount || 0,
    request: requestCopy,
    rerunEndpoint,
    rerunPayload,
    auditJson,
    detailJson: JSON.stringify(auditJson, null, 2),
    sortKey: finishedAt || createdAt || String(job.id || ""),
  };
}

function rerunPayloadForTask(kind, request = {}) {
  if (kind === "inference") {
    return {
      ...request,
      models: nonEmptyStrings(request.models || []),
      datasets: nonEmptyStrings(request.datasets || request.submitted_datasets || []),
      launch_strategy: String(request.launch_strategy || "unspecified"),
      selected_datasets: nonEmptyStrings(request.selected_datasets || request.datasets || []),
      submitted_datasets: nonEmptyStrings(request.submitted_datasets || request.datasets || []),
      skipped_datasets: nonEmptyStrings(request.skipped_datasets || []),
    };
  }
  return {
    ...request,
    datasets: nonEmptyStrings(request.datasets || []),
    estimators: nonEmptyStrings(request.estimators || []),
    featuresets: nonEmptyStrings(request.featuresets || []),
    protocols: nonEmptyStrings(request.protocols || ["loco"]),
    seeds: Array.isArray(request.seeds) ? request.seeds.map((seed) => Number(seed)) : [0],
  };
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value || {}));
}

function taskStatusLabel(status) {
  const value = String(status || "unknown");
  if (value === "succeeded") return "成功";
  if (value === "partial") return "部分完成";
  if (value === "failed") return "失败";
  if (value === "running") return "运行中";
  if (value === "queued") return "排队中";
  return "未知";
}

export function isTerminalJobStatus(status) {
  return ["succeeded", "partial", "failed"].includes(String(status || ""));
}

function uniqueStrings(values = []) {
  return [...new Set(nonEmptyStrings(values))];
}

export function inferenceActionState({
  backendConnected = false,
  selectedModelCount = 0,
  selectedDatasetCount = 0,
  currentJob = null,
} = {}) {
  if (!backendConnected) {
    return {
      disabled: true,
      tone: "fail",
      message: "后端未连接，无法启动推理测评。",
    };
  }

  if (currentJob?.kind === "inference") {
    if (currentJob.status === "queued" || currentJob.status === "running") {
      return {
        disabled: true,
        tone: "active",
        message: `推理任务 ${currentJob.id} 正在执行，完成后会刷新推理结果。`,
      };
    }
    if (currentJob.status === "succeeded") {
      return {
        disabled: false,
        tone: "ok",
        message: `推理任务 ${currentJob.id} 已完成，结果已发布到推理结果表。`,
      };
    }
    if (currentJob.status === "partial") {
      return {
        disabled: false,
        tone: "warn",
        message: `推理任务 ${currentJob.id} 部分完成，成功结果已发布；请检查失败组合。`,
      };
    }
    if (currentJob.status === "failed") {
      return {
        disabled: false,
        tone: "fail",
        message: `推理任务 ${currentJob.id} 失败：${currentJob.error || "请检查模型和特征兼容性"}`,
      };
    }
  }

  if (!selectedModelCount && !selectedDatasetCount) {
    return {
      disabled: true,
      tone: "warn",
      message: "请选择至少一个 Pickle 模型和一个目标数据集。",
    };
  }
  if (!selectedModelCount) {
    return {
      disabled: true,
      tone: "warn",
      message: "请选择至少一个 Pickle 模型，或先上传模型。",
    };
  }
  if (!selectedDatasetCount) {
    return {
      disabled: true,
      tone: "warn",
      message: "请选择至少一个目标数据集。",
    };
  }

  return {
    disabled: false,
    tone: "ready",
    message: `已选择 ${selectedModelCount} 个模型 / ${selectedDatasetCount} 个数据集，可启动 inference-only 推理测评。`,
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
    row.candidate_id || "",
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
    ...(base.candidate_id ? { candidate_id: base.candidate_id } : {}),
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
