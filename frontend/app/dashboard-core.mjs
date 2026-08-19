"use strict";

export function metricMean(entry, name) {
  const value = entry?.metrics?.[name]?.mean;
  return value == null || Number.isNaN(value) ? null : value;
}

export function trajectoryContextConsistency(payload = {}, expected = {}) {
  const fields = [
    "case_id",
    "inference_job_id",
    "dataset",
    "candidate_id",
    "model_artifact_id",
    "context_binding_id",
  ];
  const mismatches = fields.flatMap((field) => {
    const expectedValue = String(expected?.[field] ?? "");
    if (!expectedValue) return [];
    const actualValue = String(payload?.[field] ?? "");
    return actualValue === expectedValue ? [] : [{ field, expected: expectedValue, actual: actualValue }];
  });
  return { matches: mismatches.length === 0, mismatches };
}

export function auditCandidateViewProjection({
  candidate = null,
  caseVerdict = {},
  pinned = false,
} = {}) {
  const sourceVerdict = caseVerdict || {};
  if (pinned) {
    return {
      ...sourceVerdict,
      mode: "pinned",
      result: String(sourceVerdict.result || "pending"),
      observed: { ...(sourceVerdict.observed || {}) },
      thresholds: { ...(sourceVerdict.thresholds || {}) },
      reasons: [...(sourceVerdict.reasons || [])],
    };
  }

  const observed = {
    mae: metricMean(candidate || {}, "mae"),
    picp: metricMean(candidate || {}, "picp"),
    monotonicity: metricMean(candidate || {}, "monotonicity_violation_rate"),
    value_range: metricMean(candidate || {}, "value_range_violation"),
  };
  const thresholds = { ...(sourceVerdict.thresholds || {}) };
  const definitions = [
    ["generalization", "mae", "mae_max", "max"],
    ["calibration", "picp", "picp_min", "min"],
    ["monotonicity", "monotonicity", "monotonicity_max", "max"],
    ["value_range", "value_range", "value_range_max", "max"],
  ];
  const checks = definitions.map(([reason, observedKey, thresholdKey, direction]) => {
    const value = observed[observedKey];
    const threshold = thresholds[thresholdKey];
    const known = Number.isFinite(Number(value)) && Number.isFinite(Number(threshold));
    const passed = !known
      ? null
      : (direction === "max" ? Number(value) <= Number(threshold) : Number(value) >= Number(threshold));
    return { reason, passed };
  });
  const reasons = checks.filter((item) => item.passed === false).map((item) => item.reason);
  const knownCount = checks.filter((item) => item.passed !== null).length;
  const result = reasons.length ? "fail" : (knownCount === checks.length ? "pass" : "pending");
  return {
    mode: "preview",
    result,
    observed,
    thresholds,
    reasons,
    criterion_set_version: String(sourceVerdict.criterion_set_version || ""),
  };
}

export function casePipelineJobId(caseDocument = null, profileOption = {}, kind = "training") {
  const isInference = kind === "inference";
  const executionKey = isInference ? "inference_job_id" : "training_job_id";
  const profileKey = isInference ? "inferenceJobId" : "trainingJobId";
  if (caseDocument?.id) return String(caseDocument?.execution?.[executionKey] || "");
  return String(profileOption?.[profileKey] || "");
}

export function pipelineJobSequenceValue(job = {}) {
  const idMatch = String(job?.id || "").match(/^(\d{8}T\d{6})/);
  if (idMatch) return idMatch[1];
  return String(job?.created_at || job?.started_at || job?.finished_at || "")
    .replace(/[^0-9]/g, "")
    .slice(0, 14);
}

export function requiresFreshTrainingCase(caseDocument = null, reviewWorkspace = {}) {
  if (!caseDocument?.id) return false;
  if (["completed", "failed", "archived"].includes(caseDocument.lifecycle_state)) return true;
  const execution = caseDocument.execution || {};
  const selection = caseDocument.evidence_selection || {};
  const reviewState = String(caseDocument.review?.state || "not_started");
  const hasEvidenceSelection = [
    "candidate_id",
    "dataset",
    "dataset_snapshot_id",
    "split_assignment_id",
    "asset_id",
    "sample_id",
    "cycle_id",
  ].some((key) => selection[key] !== null && selection[key] !== "" && selection[key] !== undefined);
  const hasWorkflow = ["issues", "retests", "evidence_packs"].some(
    (key) => Array.isArray(reviewWorkspace?.[key]) && reviewWorkspace[key].length > 0,
  );
  return Boolean(
    execution.training_job_id
    || execution.inference_job_id
    || execution.run_id
    || (caseDocument.candidates || []).length
    || hasEvidenceSelection
    || reviewState !== "not_started"
    || hasWorkflow
  );
}

export function pipelineReconciliationTarget({
  caseDocument = null,
  trainingJob = null,
  inferenceJob = null,
} = {}) {
  const execution = caseDocument?.execution || {};
  const training = trainingJob?.id && String(trainingJob.id) === String(execution.training_job_id || "")
    ? trainingJob
    : null;
  const inference = inferenceJob?.id && String(inferenceJob.id) === String(execution.inference_job_id || "")
    ? inferenceJob
    : null;
  if (!training && !inference) {
    return {
      kind: "",
      jobId: "",
      status: "idle",
      expectedState: "idle",
      expectedRunId: "",
      clearInferenceJob: false,
      needsIdentitySync: false,
      needsEvidenceBinding: false,
    };
  }

  const trainingIsLatest = Boolean(
    training
    && (!inference || pipelineJobSequenceValue(training) > pipelineJobSequenceValue(inference)),
  );
  const kind = trainingIsLatest ? "training" : "inference";
  const job = trainingIsLatest ? training : inference;
  const status = String(job?.status || "idle");
  const expectedState = status === "succeeded" ? "completed" : status;
  const expectedRunId = String(job?.run_id || job?.id || "");
  const clearInferenceJob = Boolean(
    trainingIsLatest
    && execution.inference_job_id
    && pipelineJobSequenceValue(training) > pipelineJobSequenceValue({ id: execution.inference_job_id }),
  );
  const needsIdentitySync = Boolean(
    expectedState !== String(execution.state || "idle")
    || expectedRunId !== String(execution.run_id || "")
    || clearInferenceJob,
  );
  const sampleId = String(caseDocument?.evidence_selection?.sample_id || "");
  const needsEvidenceBinding = Boolean(
    kind === "inference"
    && status === "succeeded"
    && !sampleId.startsWith(`pred:${job.id}:`),
  );
  return {
    kind,
    jobId: String(job?.id || ""),
    status,
    expectedState,
    expectedRunId,
    clearInferenceJob,
    needsIdentitySync,
    needsEvidenceBinding,
  };
}

export function caseContextProjection(caseDocument = {}, reviewWorkspace = {}, fallbacks = {}) {
  const selection = caseDocument?.evidence_selection || {};
  const candidates = Array.isArray(caseDocument?.candidates) ? caseDocument.candidates : [];
  const fallbackCandidate = fallbacks?.candidate || {};
  const fallbackRunId = String(fallbacks?.runId || "");
  const selectedCandidate = candidates.find((item) => item?.candidate_id === selection?.candidate_id)
    || candidates.find((item) => fallbackCandidate?.id && item?.candidate_id === fallbackCandidate.id)
    || candidates.find((item) => fallbackCandidate?.artifactId && (
      item?.model_artifact_id === fallbackCandidate.artifactId
      || item?.artifact_id === fallbackCandidate.artifactId
    ))
    || candidates.find((item) => fallbackRunId && item?.run_id === fallbackRunId && item?.role === "champion")
    || candidates.find((item) => fallbackRunId && item?.run_id === fallbackRunId)
    || candidates.find((item) => item?.role === "champion")
    || candidates[0]
    || fallbackCandidate
    || {};
  const issues = Array.isArray(reviewWorkspace?.issues) ? reviewWorkspace.issues : [];
  const regressionSets = Array.isArray(reviewWorkspace?.regression_sets) ? reviewWorkspace.regression_sets : [];
  const retests = Array.isArray(reviewWorkspace?.retests) ? reviewWorkspace.retests : [];
  const packs = Array.isArray(reviewWorkspace?.evidence_packs) ? reviewWorkspace.evidence_packs : [];
  const caseVersion = Number(caseDocument?.version || 0);
  const latestRetest = retests[0] || null;
  const latestCompletedRetest = retests.find((item) => item?.status === "completed") || null;
  const historicalSignedPack = packs.find((item) => item?.state === "signed") || null;
  const latestSignedPack = packs.find((item) =>
    item?.state === "signed"
    && caseVersion > 0
    && Number(item?.case_version) === caseVersion
  ) || null;
  const openIssues = issues.filter((item) => !["resolved", "closed"].includes(item?.status));
  const dataset = String(selection?.dataset || fallbacks?.datasets?.[0] || "");
  const protocolRef = String(
    selection?.split_assignment_id
    || fallbacks?.protocolRef
    || fallbacks?.protocols?.[0]
    || "",
  );
  const sourceClass = String(caseDocument?.source_class || fallbacks?.sourceClass || "");
  const execution = caseDocument?.execution || {};
  const verdict = caseDocument?.verdict || {};
  const retestResult = latestCompletedRetest?.result || latestRetest?.result || {};
  const retestMemberCount = Number(
    retestResult?.regression_summary?.member_count
    ?? retestResult?.metrics?.evaluated_count
    ?? 0,
  );
  let closureState = "ready";
  if (latestRetest && ["requested", "running"].includes(latestRetest.status)) closureState = "running";
  else if (latestRetest?.status === "failed") closureState = "failed";
  else if (openIssues.length) closureState = "open";
  else if (latestSignedPack) closureState = "signed";
  else if (latestCompletedRetest) closureState = "retested";

  return {
    hasCase: Boolean(caseDocument?.id),
    caseId: String(caseDocument?.id || fallbacks?.caseId || "LIVE"),
    caseVersion,
    profileName: String(caseDocument?.profile_name || fallbacks?.profileName || "root"),
    sourceClass,
    execution: {
      state: String(execution?.state || fallbacks?.executionState || "idle"),
      runId: String(execution?.run_id || fallbacks?.runId || ""),
      trainingJobId: String(execution?.training_job_id || ""),
      inferenceJobId: String(execution?.inference_job_id || ""),
    },
    dataset: {
      name: dataset,
      snapshotId: String(selection?.dataset_snapshot_id || ""),
      assetId: String(selection?.asset_id || ""),
      cycleId: selection?.cycle_id ?? null,
      sampleId: String(selection?.sample_id || ""),
    },
    candidate: {
      id: String(selectedCandidate?.candidate_id || selection?.candidate_id || ""),
      model: String(selectedCandidate?.model || selectedCandidate?.estimator || fallbacks?.candidate?.model || ""),
      featureset: String(selectedCandidate?.featureset || fallbacks?.candidate?.featureset || "full"),
      artifactId: String(selectedCandidate?.model_artifact_id || selectedCandidate?.artifact_id || ""),
    },
    protocolRef,
    verdict: {
      result: String(verdict?.result || "pending"),
      criterionSetVersion: String(verdict?.criterion_set_version || fallbacks?.criterionSetVersion || ""),
      observed: { ...(verdict?.observed || {}) },
      thresholds: { ...(verdict?.thresholds || {}) },
    },
    reviewState: String(caseDocument?.review?.state || "not_started"),
    closure: {
      state: closureState,
      openIssueCount: openIssues.length,
      issueCount: issues.length,
      regressionSetCount: regressionSets.length,
      retestCount: retests.length,
      latestRetest,
      latestCompletedRetest,
      retestMemberCount: Number.isFinite(retestMemberCount) ? retestMemberCount : 0,
      latestSignedPack,
      historicalSignedPack,
    },
  };
}

export function casePipelineProgress({
  caseDocument = {},
  reviewWorkspace = {},
  trainingJob = null,
  inferenceJob = null,
} = {}) {
  const execution = caseDocument?.execution || {};
  const candidates = Array.isArray(caseDocument?.candidates) ? caseDocument.candidates : [];
  const selection = caseDocument?.evidence_selection || {};
  const packs = Array.isArray(reviewWorkspace?.evidence_packs) ? reviewWorkspace.evidence_packs : [];
  const caseVersion = Number(caseDocument?.version || 0);
  const trainingId = String(execution.training_job_id || trainingJob?.id || "");
  const inferenceId = String(execution.inference_job_id || inferenceJob?.id || "");
  const boundTrainingJob = trainingJob?.id && String(trainingJob.id) === trainingId ? trainingJob : null;
  const boundInferenceJob = inferenceJob?.id && String(inferenceJob.id) === inferenceId ? inferenceJob : null;
  const trainingStatus = String(boundTrainingJob?.status || "");
  const inferenceStatus = String(boundInferenceJob?.status || "");
  const hasArtifact = candidates.some((item) => Boolean(item?.model_artifact_id));
  const trainingComplete = Boolean(
    trainingId
    && (trainingStatus === "succeeded" || (!trainingStatus && hasArtifact))
  );
  const trainingSequence = pipelineJobSequenceValue(boundTrainingJob || { id: trainingId });
  const inferenceSequence = pipelineJobSequenceValue(boundInferenceJob || { id: inferenceId });
  const inferenceBelongsToTraining = Boolean(
    inferenceId
    && (!trainingSequence || !inferenceSequence || inferenceSequence >= trainingSequence)
  );
  const inferenceComplete = Boolean(
    trainingComplete && inferenceBelongsToTraining && (
      inferenceStatus === "succeeded"
      || (!inferenceStatus && selection.candidate_id && selection.dataset)
    )
  );
  const signed = packs.some((item) =>
    item?.state === "signed" && caseVersion > 0 && Number(item?.case_version) === caseVersion
  );
  const reviewState = String(caseDocument?.review?.state || "not_started");
  const rejectedComplete = reviewState === "rejected" && caseDocument?.lifecycle_state === "failed";
  const auditComplete = inferenceComplete && (signed || reviewState === "approved" || rejectedComplete);

  const stage = (id, label, complete, active, blocked, detail) => ({
    id,
    label,
    state: complete ? "complete" : (blocked ? "blocked" : (active ? "active" : "pending")),
    detail,
  });
  return {
    trainingJobId: trainingId,
    inferenceJobId: inferenceId,
    stages: [
      stage(
        "training",
        "设计与训练",
        trainingComplete,
        Boolean(trainingId) && !trainingComplete && trainingStatus !== "failed",
        trainingStatus === "failed",
        trainingComplete
          ? "模型产物已固定"
          : (trainingStatus === "failed"
            ? "训练失败"
            : (trainingStatus === "running"
              ? "训练任务正在运行"
              : (trainingStatus === "queued" ? "训练任务排队中" : "选择组合并运行"))),
      ),
      stage(
        "inference",
        "推理验证",
        inferenceComplete,
        trainingComplete && inferenceBelongsToTraining && !inferenceComplete && inferenceStatus !== "failed",
        trainingComplete && inferenceBelongsToTraining && inferenceStatus === "failed",
        inferenceComplete
          ? "预测结果已绑定"
          : (trainingComplete && inferenceBelongsToTraining && inferenceStatus === "failed"
            ? "推理失败"
            : (trainingComplete ? "等待启动当前训练产物的推理验证" : "等待训练产物")),
      ),
      stage(
        "audit",
        "评测与证据",
        auditComplete,
        inferenceComplete && !auditComplete,
        false,
        !inferenceComplete
          ? "等待当前训练代次的推理结果"
          : (signed
            ? "EvidencePack 已签署"
            : (reviewState === "approved"
              ? "评审已批准"
              : (rejectedComplete
                ? "不通过结论已确认，案例已结束"
                : "运行已完成，等待判定、评审与发布闭环"))),
      ),
    ],
  };
}

/**
 * Return the single next audit action for the current case state.
 *
 * The UI deliberately exposes one primary action instead of forcing reviewers
 * to discover the issue -> regression set -> controlled retest chain in four
 * separate panels.  Returned identifiers are immutable workflow references;
 * the caller still performs the mutation through the authoritative API.
 */
export function auditNextAction({
  caseDocument = null,
  reviewWorkspace = {},
  readiness = {},
} = {}) {
  if (!caseDocument?.id) {
    return { action: "", label: "案例未连接", disabled: true, hint: "请先连接服务端案例" };
  }

  const lifecycle = String(caseDocument.lifecycle_state || "draft");
  const review = caseDocument.review || { state: "not_started" };
  const reviewState = String(review.state || "not_started");
  const verdict = String(caseDocument.verdict?.result || "pending");
  const evidenceComplete = caseDocument.evidence_coverage?.state === "complete";
  const verdictEligible = ["pass", "conditional"].includes(verdict);
  const version = Number(caseDocument.version || 0);
  const issues = Array.isArray(reviewWorkspace?.issues) ? reviewWorkspace.issues : [];
  const regressionSets = Array.isArray(reviewWorkspace?.regression_sets)
    ? reviewWorkspace.regression_sets
    : [];
  const retests = Array.isArray(reviewWorkspace?.retests) ? reviewWorkspace.retests : [];
  const packs = Array.isArray(reviewWorkspace?.evidence_packs)
    ? reviewWorkspace.evidence_packs
    : [];
  const reconciliations = Array.isArray(reviewWorkspace?.reconciliations)
    ? reviewWorkspace.reconciliations
    : [];
  const currentPack = packs.find((item) => Number(item?.case_version) === version) || null;
  const currentReconciliation = reconciliations.find(
    (item) => Number(item?.case_version) === version,
  ) || null;
  const openIssue = issues.find((item) => !["resolved", "closed"].includes(item?.status)) || null;
  const completedFailureIssue = verdict === "fail"
    ? issues.find((item) => {
      if (!["resolved", "closed"].includes(item?.status)) return false;
      return retests.some((retest) => retest?.issue_id === item.id && retest?.status === "completed");
    }) || null
    : null;

  const closureAction = () => {
    if (!openIssue && completedFailureIssue) {
      return {
        action: "finalize-failed",
        issueId: completedFailureIssue.id,
        label: "确认不通过并结束",
        disabled: false,
        hint: "受控复测已闭环；固定不通过结论并结束案例",
      };
    }
    if (!openIssue) {
      return {
        action: "create-issue",
        label: verdict === "fail" ? "创建问题并复测" : "建立证据问题",
        disabled: false,
        auto: true,
        hint: verdict === "fail"
          ? "当前判定不通过；先建立可审计问题单"
          : "先记录证据缺口，再进入受控闭环",
      };
    }
    const regressionSet = regressionSets.find((item) => item?.issue_id === openIssue.id) || null;
    if (!regressionSet) {
      return {
        action: "create-regression",
        issueId: openIssue.id,
        label: "建立回归集",
        disabled: false,
        hint: "固定同一判据、模型与数据快照",
      };
    }
    const members = Array.isArray(regressionSet.members) ? regressionSet.members : [];
    if (!members.length) {
      const hasPinnedCycle = caseDocument.evidence_selection?.sample_id
        && caseDocument.evidence_selection?.cycle_id !== null
        && caseDocument.evidence_selection?.cycle_id !== "";
      return {
        action: "add-member",
        issueId: openIssue.id,
        setId: regressionSet.id,
        neighborRadius: 2,
        label: hasPinnedCycle ? "固定当前与邻近 Cycle" : "先固定 Cycle",
        disabled: !hasPinnedCycle,
        hint: hasPinnedCycle
          ? "一次固定当前 Cycle 及前后各 2 个邻近样本"
          : "请先在证据曲线上选择并固定精确 Cycle",
      };
    }
    const latestRetest = retests.find((item) => (
      item?.issue_id === openIssue.id || item?.regression_set_id === regressionSet.id
    )) || null;
    if (latestRetest && ["requested", "running"].includes(latestRetest.status)) {
      return {
        action: "focus-closure",
        issueId: openIssue.id,
        setId: regressionSet.id,
        label: "查看复测进度",
        disabled: false,
        hint: `${members.length} 条固定样本正在受控复测`,
      };
    }
    if (!latestRetest || latestRetest.status === "failed") {
      return {
        action: "request-retest",
        issueId: openIssue.id,
        setId: regressionSet.id,
        label: latestRetest?.status === "failed" ? "重新运行受控复测" : "运行受控复测",
        disabled: false,
        hint: `${members.length} 条固定样本将使用同一协议重新推理`,
      };
    }
    if (latestRetest.status === "completed" && openIssue.status === "in_retest") {
      return {
        action: "resolve-issue",
        issueId: openIssue.id,
        label: "确认复测并闭环",
        disabled: false,
        auto: true,
        hint: "复测结果已回填；确认后关闭当前问题",
      };
    }
    return {
      action: "focus-closure",
      issueId: openIssue.id,
      label: "查看闭环详情",
      disabled: false,
      hint: "查看复测对比与问题状态",
    };
  };

  if (lifecycle === "archived") {
    return { action: "", label: "案例已归档", disabled: true, hint: "归档案例不可修改" };
  }
  if (lifecycle === "completed") {
    return { action: "", label: "案例已完成", disabled: true, hint: "发布流程已完成" };
  }
  if (lifecycle === "failed") {
    return { action: "", label: "案例已结束（不通过）", disabled: true, hint: "不通过结论已固定" };
  }
  if (currentPack?.state === "signed") {
    return { action: "", label: "已签署发布", disabled: true, hint: "EvidencePack 已签署" };
  }
  if (reviewState === "not_started") {
    return {
      action: "review",
      target: "in_review",
      label: "提交评审",
      disabled: false,
      hint: "进入人工评审并读取服务端门禁",
    };
  }
  if (reviewState === "rejected") {
    return {
      action: "finalize-failed",
      label: "结束为不通过",
      disabled: false,
      hint: "固定驳回结论并结束当前案例",
    };
  }
  if (reviewState === "in_review") {
    if (openIssue || !evidenceComplete || !verdictEligible) return closureAction();
    return {
      action: "review",
      target: "approved",
      label: "批准评审",
      disabled: false,
      hint: "证据与判定均满足评审门禁",
    };
  }
  if (reviewState === "approved" && openIssue) return closureAction();
  if (!currentReconciliation || currentReconciliation.status !== "match") {
    return {
      action: "reconcile",
      label: "运行 KPI 对账",
      disabled: false,
      hint: "核对案例事实表与证据画布口径",
    };
  }
  if (!currentPack) {
    return {
      action: "create-pack",
      label: "生成 EvidencePack",
      disabled: false,
      hint: "生成当前案例版本的不可变证据草稿",
    };
  }
  if (currentPack.state === "draft") {
    return {
      action: readiness?.can_sign ? "sign-pack" : "focus-readiness",
      packId: currentPack.id,
      label: readiness?.can_sign ? "签署并发布" : "查看发布门禁",
      disabled: false,
      hint: readiness?.can_sign
        ? "签署当前不可变证据包"
        : "仍有门禁未满足；点击定位阻断原因",
    };
  }
  return { action: "", label: "流程已完成", disabled: true, hint: "当前流程没有待办动作" };
}

export function pipelineCombinationConsistency({
  caseDocument = {},
  trainingJob = null,
  inferenceJob = null,
  models = [],
} = {}) {
  const selection = caseDocument?.evidence_selection || {};
  const binding = caseDocument?.context_binding || {};
  const combination = {
    dataset: String(binding?.active_dataset || ""),
    estimator: String(binding?.model || ""),
    featureset: String(binding?.featureset || ""),
    artifactId: String(binding?.model_artifact_id || ""),
  };
  const trainingRequest = trainingJob?.request || {};
  const inferenceRequest = inferenceJob?.request || {};
  const trainingDatasets = nonEmptyStrings(trainingRequest.datasets || []);
  const trainingEstimators = nonEmptyStrings(trainingRequest.estimators || []);
  const trainingFeaturesets = nonEmptyStrings(trainingRequest.featuresets || []);
  const inferenceDatasets = nonEmptyStrings(
    inferenceRequest.submitted_datasets
    || inferenceRequest.selected_datasets
    || inferenceRequest.datasets
    || [],
  );
  const inferenceModelNames = nonEmptyStrings(inferenceRequest.models || []);
  const inferenceBindings = (Array.isArray(inferenceRequest.model_bindings)
    ? inferenceRequest.model_bindings
    : []).map((item) => ({
    name: String(item?.name || ""),
    estimator: String(item?.estimator || ""),
    featureset: String(item?.featureset || "full"),
    artifactId: String(item?.artifact_id || ""),
  }));
  const inferredModels = inferenceBindings;
  const expectedTrainingId = String(binding?.training_job_id || "");
  const expectedInferenceId = String(binding?.inference_job_id || "");
  const trainingId = String(trainingJob?.id || "");
  const inferenceId = String(inferenceJob?.id || "");
  const trainingAvailable = Boolean(
    trainingId
    && combination.estimator
    && combination.featureset
    && trainingDatasets.length
    && trainingEstimators.length
    && trainingFeaturesets.length,
  );
  const inferenceAvailable = Boolean(
    inferenceId
    && combination.dataset
    && combination.estimator
    && combination.featureset
    && inferenceDatasets.length
    && inferenceModelNames.length
    && inferenceBindings.length === inferenceModelNames.length,
  );
  const auditAvailable = Boolean(
    caseDocument?.id
    && binding?.binding_id
    && binding?.candidate_id
    && combination.dataset
    && combination.estimator
    && combination.featureset,
  );
  const trainingMatches = trainingAvailable
    && (!expectedTrainingId || expectedTrainingId === trainingId)
    && trainingEstimators.includes(combination.estimator)
    && trainingFeaturesets.includes(combination.featureset);
  const inferenceCandidateMatches = inferredModels.some((item) =>
    combination.artifactId && item.artifactId === combination.artifactId
  );
  const inferenceMatches = inferenceAvailable
    && (!expectedInferenceId || expectedInferenceId === inferenceId)
    && inferenceDatasets.includes(combination.dataset)
    && inferenceCandidateMatches;
  const auditMatches = auditAvailable
    && String(selection?.candidate_id || "") === String(binding?.candidate_id || "")
    && String(selection?.dataset || "") === String(binding?.active_dataset || "");
  const pageChecks = {
    training: { available: trainingAvailable, matches: trainingMatches, jobId: trainingId },
    inference: { available: inferenceAvailable, matches: inferenceMatches, jobId: inferenceId },
    audit: {
      available: auditAvailable,
      matches: auditMatches,
      caseId: String(caseDocument?.id || ""),
      bindingId: String(binding?.binding_id || ""),
      generation: Number(binding?.generation || 0),
    },
  };
  const available = Object.values(pageChecks).every((item) => item.available);
  const consistent = available && Object.values(pageChecks).every((item) => item.matches);
  const reasons = [];
  if (!trainingAvailable) reasons.push("缺少可核验的训练组合");
  else if (!trainingMatches) reasons.push("训练页组合与当前证据候选不一致");
  if (!inferenceAvailable) reasons.push("缺少可核验的推理组合");
  else if (!inferenceMatches) reasons.push("推理页模型或数据集与当前证据候选不一致");
  if (!auditAvailable) reasons.push("第 3 页尚未固定数据集、算法和特征");
  return {
    state: consistent ? "consistent" : (available ? "mismatch" : "incomplete"),
    consistent,
    combination,
    pageChecks,
    reasons,
  };
}

export function retestComparisonView(result = {}) {
  const summary = result?.regression_summary || {};
  const localization = result?.failure_localization || {};
  const records = Array.isArray(result?.record_deltas) ? result.record_deltas : [];
  const topRecords = [...records]
    .sort((a, b) => Number(b?.retest_absolute_error || 0) - Number(a?.retest_absolute_error || 0))
    .slice(0, 3)
    .map((item) => ({
      sampleId: String(item?.sample_id || ""),
      dataset: String(item?.dataset || ""),
      assetId: String(item?.asset_id || ""),
      cycleId: item?.cycle_id ?? null,
      baselineError: finiteOrNull(item?.baseline_absolute_error),
      retestError: finiteOrNull(item?.retest_absolute_error),
      errorDelta: finiteOrNull(item?.absolute_error_delta),
      contributionDelta: finiteOrNull(item?.subset_mae_contribution_delta),
      change: String(item?.change || "unchanged"),
    }));
  return {
    available: Boolean(result?.comparison_contract || records.length),
    contract: String(result?.comparison_contract || ""),
    verdict: String(result?.verdict || "not_evaluated"),
    memberCount: Number(summary?.member_count ?? result?.metrics?.evaluated_count ?? records.length) || 0,
    baselineMae: finiteOrNull(result?.baseline_metrics?.mae),
    retestMae: finiteOrNull(result?.metrics?.mae),
    maeDelta: finiteOrNull(result?.metric_deltas?.mae),
    baselinePicp: finiteOrNull(result?.baseline_metrics?.picp),
    retestPicp: finiteOrNull(result?.metrics?.picp),
    picpDelta: finiteOrNull(result?.metric_deltas?.picp),
    improvedCount: Number(summary?.improved_count || 0),
    unchangedCount: Number(summary?.unchanged_count || 0),
    regressedCount: Number(summary?.regressed_count || 0),
    topRecords,
    localization: {
      datasets: Array.isArray(localization?.datasets) ? localization.datasets.map(String) : [],
      assets: Array.isArray(localization?.assets) ? localization.assets.map(String) : [],
      protocolRef: String(localization?.protocol_ref || result?.protocol_ref || ""),
      splitAssignmentId: String(localization?.split_assignment_id || ""),
      candidateId: String(localization?.candidate_id || result?.candidate_id || ""),
      modelArtifactId: String(localization?.model_artifact_id || result?.model_artifact_id || ""),
      failedCriteria: Array.isArray(localization?.failed_criteria) ? localization.failed_criteria.map(String) : [],
      thresholdExceedingCount: Array.isArray(localization?.threshold_exceeding_records)
        ? localization.threshold_exceeding_records.length
        : 0,
    },
  };
}

export const ACCEPTANCE_PRESETS = Object.freeze({
  strict: Object.freeze({
    id: "strict",
    label: "严格",
    kpiLabel: "严格验收",
    description: "论文级验收口径，要求低误差、高覆盖和极低物理违例。",
    thresholds: Object.freeze({
      maeMax: 0.02,
      picpMin: 0.90,
      monotonicityMax: 0.001,
      rangeMax: 0,
    }),
  }),
  engineering: Object.freeze({
    id: "engineering",
    label: "工程可用",
    kpiLabel: "工程可用验收",
    description: "用于跨数据集筛选可继续复核的模型，允许少量迁移误差和单调违例。",
    thresholds: Object.freeze({
      maeMax: 0.08,
      picpMin: 0.85,
      monotonicityMax: 0.15,
      rangeMax: 0,
    }),
  }),
  exploratory: Object.freeze({
    id: "exploratory",
    label: "探索",
    kpiLabel: "探索验收",
    description: "用于早期算法发现，重点观察趋势和候选方向。",
    thresholds: Object.freeze({
      maeMax: 0.12,
      picpMin: 0.80,
      monotonicityMax: 0.25,
      rangeMax: 0,
    }),
  }),
});

export function acceptancePresetOptions() {
  return Object.values(ACCEPTANCE_PRESETS);
}

export function datasetOptions(config = {}, qc = {}, datasetRows = [], catalog = {}) {
  const names = new Set(config.datasets || []);
  Object.keys(qc?.per_dataset || {}).forEach((name) => names.add(name));
  (catalog?.datasets || []).forEach((item) => {
    const name = String(item?.name || "");
    if (name) names.add(name);
  });
  datasetRows.forEach((row) => {
    const name = heldOutDatasetName(row?.held_out_key);
    if (name) names.add(name);
  });

  const metricNames = new Set(datasetRows.map((row) => heldOutDatasetName(row.held_out_key)).filter(Boolean));
  const totalRows = Object.values(qc?.per_dataset || {})
    .reduce((sum, item) => sum + Number(item?.rows || 0), 0);

  const catalogByName = new Map((catalog?.datasets || []).map((item) => [String(item?.name || ""), item]));
  return [...names].map((name) => {
    const item = catalogByName.get(name) || {};
    const rows = Math.max(Number(qc?.per_dataset?.[name]?.rows || 0), Number(item.rows || item.record_count || 0));
    return {
      ...item,
      name,
      rows,
      recordCount: Number(item.record_count || rows),
      cycleCount: Number(item.cycle_count || 0),
      batteryCount: Number(item.battery_count || 0),
      chemistry: String(item.chemistry || "unknown"),
      maxCapacityAh: Number.isFinite(Number(item.max_capacity_ah)) ? Number(item.max_capacity_ah) : null,
      capacitySource: String(item.capacity_source || "unknown"),
      heldOutReady: Boolean(item.held_out_ready),
      rawLocal: Boolean(item.raw_local),
      curveAvailable: Boolean(item.curve_available),
      dataStatus: String(item.status || "unknown"),
      source: String(item.source || ""),
      license: String(item.license || "unknown"),
      roles: Array.isArray(item.roles) ? [...item.roles] : [],
      share: totalRows > 0 ? rows / totalRows : 0,
      hasMetrics: metricNames.has(name),
    };
  }).sort((a, b) => (b.rows - a.rows) || a.name.localeCompare(b.name));
}

export function auditEvidenceDatasetScope({ caseDocument = null, inferenceJob = null } = {}) {
  if (!caseDocument) return [];
  const saved = Array.isArray(caseDocument.saved_combinations)
    ? caseDocument.saved_combinations
    : [];
  const activeId = String(caseDocument.active_saved_combination_id || "");
  const active = saved.find((item) => String(item?.saved_combination_id || "") === activeId) || null;
  const jobId = String(
    (active?.source_stage === "inference" ? active?.inference_job_id : "")
    || caseDocument.execution?.inference_job_id
    || inferenceJob?.id
    || "",
  );
  const names = [];
  const add = (value) => {
    const name = String(value || "").trim();
    if (name && !names.includes(name)) names.push(name);
  };
  saved
    .filter((item) => item?.source_stage === "inference"
      && (!jobId || String(item?.inference_job_id || "") === jobId))
    .forEach((item) => add(item?.target_dataset));
  if (inferenceJob && (!jobId || String(inferenceJob.id || "") === jobId)) {
    const request = inferenceJob.request || {};
    [request.submitted_datasets, request.selected_datasets, request.datasets]
      .find((values) => Array.isArray(values) && values.length)
      ?.forEach(add);
  }
  const selected = String(caseDocument.evidence_selection?.dataset || "").trim();
  if (!names.length || names.includes(selected)) add(selected);
  return names;
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
  const methodCategoryByPairing = {
    free: "free_combination",
    recommended: "recommended_preset",
    method_package: "joint_method",
    embedded: "joint_method",
  };
  const methodCategoryBadge = {
    free_combination: "自由组合",
    recommended_preset: "推荐预设",
    joint_method: "联合方法",
  };
  const combinationPolicyByPairing = {
    free: "cartesian",
    recommended: "recommended_default",
    method_package: "joint_package",
    embedded: "embedded_pipeline",
  };
  return {
    datasets: normalizeCatalogItems(catalog.datasets || [], (item) => ({
      rows: Number(item.rows || 0),
      recordCount: Number(item.record_count || item.rows || 0),
      cycleCount: Number(item.cycle_count || 0),
      batteryCount: Number(item.battery_count || 0),
      chemistry: String(item.chemistry || "unknown"),
      maxCapacityAh: Number.isFinite(Number(item.max_capacity_ah)) ? Number(item.max_capacity_ah) : null,
      capacitySource: String(item.capacity_source || "unknown"),
      heldOutReady: Boolean(item.held_out_ready),
      columns: item.columns || [],
      badge: item.uploaded ? "上传" : "数据",
    })),
    estimators: normalizeCatalogItems(catalog.estimators || [], (item) => {
      const pairingMode = item.pairing_mode || (item.feature_mode === "byo" ? "embedded" : "free");
      const methodCategory = item.method_category || methodCategoryByPairing[pairingMode] || "free_combination";
      return {
        featureMode: item.feature_mode || "unified",
        backend: item.backend || "external",
        methodFamily: item.method_family || "data_driven",
        methodCategory,
        methodCategoryLabel: methodCategoryBadge[methodCategory] || "自由组合",
        pairingMode,
        combinationPolicy: item.combination_policy || combinationPolicyByPairing[pairingMode] || "cartesian",
        executionContract: item.execution_contract || (
          pairingMode === "embedded"
            ? "embedded_pipeline/v1"
            : (methodCategory === "joint_method" ? "joint_method/v1" : "estimator/v1")
        ),
        compatibleFeaturesets: item.compatible_featuresets || [],
        preferredFeatureset: item.preferred_featureset || "",
        dataGranularity: item.data_granularity || "cycle_table",
        requiredSignals: item.required_signals || [],
        physicsConstraints: item.physics_constraints || [],
        description: item.description || "",
        badge: item.uploaded ? "上传" : (methodCategoryBadge[methodCategory] || item.backend || "内置"),
      };
    }),
    featuresets: normalizeCatalogItems(catalog.featuresets || [], (item) => ({
      columns: item.columns || [],
      hasTransform: Boolean(item.has_transform),
      family: item.family || "data_driven",
      mechanismLevel: item.mechanism_level || "none",
      dataGranularity: item.data_granularity || "cycle_table",
      requiredSignals: item.required_signals || [],
      description: item.description || "",
      badge: item.family === "mechanism_derived"
        ? "机理"
        : (item.has_transform ? "变换" : (item.uploaded ? "上传" : "内置")),
    })),
  };
}

export function modelOptions(source = {}) {
  const models = Array.isArray(source.models) ? source.models : [];
  return models
    .filter((item) => item?.usable !== false && item?.status !== "invalidated")
    .map((item) => ({
      name: String(item.name || ""),
      source: item.source || "training",
      estimator: item.estimator || "",
      featureset: item.featureset || "",
      featureColumns: item.feature_columns || item.featureColumns || [],
      trainedOn: item.trained_on || item.trainedOn || [],
      artifactId: item.artifact_id || item.artifactId || "",
      contractVersion: item.contract_version || item.contractVersion || "legacy",
      manifestSha256: item.manifest_sha256 || item.manifestSha256 || "",
      modelSha256: item.model_sha256 || item.modelSha256 || "",
      protocol: item.protocol || "",
      heldOutKey: item.held_out_key || item.heldOutKey || "",
      seed: item.seed ?? null,
      createdByRun: item.created_by_run || item.createdByRun || "",
      validation: item.validation || {},
      status: item.status || "ready",
      usable: item.usable !== false,
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

export function pipelineModelGroups({ trainingJob = {}, inferenceJob = {}, models = [] } = {}) {
  const normalizedModels = (models || []).filter((item) => item?.name);
  const byName = new Map(normalizedModels.map((item) => [item.name, item]));
  const artifacts = (Array.isArray(trainingJob?.model_artifacts) ? trainingJob.model_artifacts : [])
    .filter((item) => item?.usable !== false && item?.status !== "invalidated");
  const savedNames = nonEmptyStrings(trainingJob?.saved_models || []);
  const inferenceNames = nonEmptyStrings(inferenceJob?.request?.models || []);
  const artifactByName = new Map(artifacts.map((item) => [String(item?.name || ""), item]));
  const requestedRunNames = savedNames.length
    ? savedNames
    : normalizedModels
      .filter((item) => item.createdByRun && item.createdByRun === trainingJob?.id)
      .map((item) => item.name);
  const runNames = requestedRunNames.filter((name) => byName.has(name) || artifactByName.has(name));
  const linkedNames = runNames.length ? runNames : inferenceNames;
  const summary = trainingJob?.result_summary || {};
  const groups = new Map();

  linkedNames.forEach((name) => {
    const model = byName.get(name) || {};
    const artifact = artifactByName.get(name) || {};
    const estimator = artifact.estimator || model.estimator || "未知算法";
    const featureset = artifact.featureset || model.featureset || "full";
    const protocol = artifact.protocol || model.protocol || trainingJob?.request?.protocols?.[0] || "";
    const key = `${estimator}|${featureset}|${protocol}`;
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        estimator,
        featureset,
        protocol,
        runId: String(trainingJob?.id || model.createdByRun || ""),
        models: [],
      });
    }
    groups.get(key).models.push({
      ...model,
      ...artifact,
      name,
      estimator,
      featureset,
      protocol,
      heldOutKey: artifact.held_out_key || artifact.heldOutKey || model.heldOutKey || "",
    });
  });

  const rows = [...groups.values()].map((group) => {
    const exactInference = group.models.find((item) => inferenceNames.includes(item.name));
    const best = group.models.find((item) =>
      item.estimator === summary.best_estimator
      && item.featureset === summary.best_featureset
      && (!summary.best_protocol || item.protocol === summary.best_protocol)
      && (!summary.best_held_out_key || item.heldOutKey === summary.best_held_out_key)
    );
    const recommended = exactInference || best || group.models[0];
    return {
      ...group,
      artifactCount: group.models.length,
      recommendedModelName: recommended?.name || "",
      recommendedArtifactId: recommended?.artifact_id || recommended?.artifactId || "",
      recommendedHeldOutKey: recommended?.heldOutKey || "",
      selectedCount: group.models.filter((item) => inferenceNames.includes(item.name)).length,
    };
  }).sort((a, b) =>
    a.estimator.localeCompare(b.estimator)
    || a.featureset.localeCompare(b.featureset)
    || a.protocol.localeCompare(b.protocol)
  );

  const recommendedModelNames = inferenceNames.length
    ? inferenceNames.filter((name) => byName.has(name) || artifactByName.has(name))
    : rows.map((item) => item.recommendedModelName).filter(Boolean);
  const runNameSet = new Set(runNames);
  return {
    source: inferenceNames.length ? "inference" : (runNames.length ? "training" : "catalog"),
    runId: String(inferenceJob?.id || trainingJob?.id || ""),
    groups: rows,
    runModelNames: runNames,
    recommendedModelNames,
    linked: rows.length > 0,
    globalModelCount: normalizedModels.length,
    otherModelCount: normalizedModels.filter((item) => !runNameSet.has(item.name)).length,
  };
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

export function nativeDashboardUrl({
  base = "./battery-dashboard/index.html",
  mode = "soh",
  embed = true,
  dataset = "",
  source = "",
  model = "",
  featureset = "",
  cell = "",
  cycle = "",
} = {}) {
  const rawBase = String(base || "./battery-dashboard/index.html");
  const queryIndex = rawBase.indexOf("?");
  const path = queryIndex >= 0 ? rawBase.slice(0, queryIndex) : rawBase;
  const params = new URLSearchParams(queryIndex >= 0 ? rawBase.slice(queryIndex + 1) : "");
  if (mode) params.set("mode", String(mode));
  if (embed) params.set("embed", "1");
  [
    ["dataset", dataset],
    ["source", source],
    ["model", model],
    ["featureset", featureset],
    ["cell", cell],
    ["cycle", cycle],
  ].forEach(([key, value]) => {
    const text = value == null ? "" : String(value).trim();
    if (text) params.set(key, text);
  });
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

export function trainingJobRuntimeState(job = null, now = new Date()) {
  const status = String(job?.status || "");
  if (!status) {
    return { tone: "ready", statusLabel: "未启动", elapsedLabel: "", hint: "等待新任务" };
  }
  if (status === "succeeded") {
    const target = job?.run_dir ? ` / ${job.run_dir}` : "";
    return { tone: "ok", statusLabel: status, elapsedLabel: "已完成", hint: `已完成${target}` };
  }
  if (status === "failed") {
    return { tone: "fail", statusLabel: status, elapsedLabel: "失败", hint: job?.error || "任务失败" };
  }

  const started = parseJobTimestamp(job?.started_at || job?.created_at);
  const nowDate = now instanceof Date ? now : new Date(now);
  const elapsedMinutes = started && Number.isFinite(nowDate.getTime())
    ? Math.max(0, Math.floor((nowDate.getTime() - started.getTime()) / 60000))
    : null;
  const elapsedLabel = elapsedMinutes == null ? "计时中" : durationLabel(elapsedMinutes);
  const matrix = trainingMatrixLabel(job?.request || {});
  const isFullLoco = matrix.includes("LOCO") && Number(job?.request?.datasets?.length || 0) >= 7;
  const stale = elapsedMinutes != null && elapsedMinutes >= 120;
  const fullHint = isFullLoco ? "，全量 LOCO 训练通常需要几十分钟" : "";
  const staleHint = stale ? "，超过 2 小时，建议检查任务历史或后端日志" : "";
  return {
    tone: stale ? "warn" : "active",
    statusLabel: status,
    elapsedLabel,
    hint: `${matrix || "训练任务"}，已运行 ${elapsedLabel}${fullHint}${staleHint}`,
  };
}

export function algorithmRecommendationView(summary = {}) {
  const scenarios = Array.isArray(summary?.scenarios) ? summary.scenarios : [];
  const defaultAlgorithm = summary?.overall?.recommended_default || "暂无";
  const scenarioCards = scenarios.map((scenario) => {
    const rawRanking = Array.isArray(scenario?.ranking) ? scenario.ranking : [];
    const ranking = [...rawRanking]
      .sort((a, b) => numberOrInfinity(a?.mean_mae) - numberOrInfinity(b?.mean_mae))
      .map((row, index) => ({
        ...row,
        rank: index + 1,
        maeLabel: numberLabel(row?.mean_mae),
        rmseLabel: numberLabel(row?.mean_rmse),
        speedLabel: row?.mean_ms_per_1000 == null ? "暂无" : `${numberLabel(row.mean_ms_per_1000)} ms/1000`,
      }));
    const speedWinner = scenario?.speed_winner
      || minBy(rawRanking.filter((row) => row?.mean_ms_per_1000 != null), (row) => Number(row.mean_ms_per_1000))?.algorithm
      || "暂无";
    const records = Number(scenario?.records ?? rawRanking.reduce((sum, row) => sum + Number(row?.records || 0), 0));
    const succeeded = Number(scenario?.succeeded ?? rawRanking.reduce((sum, row) => sum + Number(row?.succeeded || 0), 0));
    const successRate = Number.isFinite(Number(scenario?.success_rate))
      ? Number(scenario.success_rate)
      : records > 0 ? succeeded / records : null;
    return {
      ...scenario,
      title: scenario?.title || scenario?.protocol || "未命名场景",
      protocolLabel: String(scenario?.protocol || "").toUpperCase() || "N/A",
      recommendedAlgorithm: scenario?.recommended_algorithm || ranking[0]?.algorithm || "暂无",
      accuracyWinner: scenario?.accuracy_winner || ranking[0]?.algorithm || "暂无",
      speedWinner,
      trainingWinner: scenario?.training_winner || "暂无",
      recordLabel: records > 0 ? records.toLocaleString("en-US") : "暂无",
      successLabel: successRate == null ? "暂无" : `${(successRate * 100).toFixed(1)}%`,
      ranking,
    };
  });
  return {
    available: Boolean(scenarioCards.length || summary?.overall?.recommended_default),
    defaultAlgorithm,
    scenarioCount: scenarioCards.length,
    generatedAt: summary?.generated_at || "",
    scope: summary?.scope || {},
    why: Array.isArray(summary?.overall?.why) ? summary.overall.why : [],
    caveats: Array.isArray(summary?.overall?.caveats) ? summary.overall.caveats : [],
    scenarioCards,
  };
}

export function batteryDiagnosticSummary(diagnostics = {}) {
  const manifestScenarios = Array.isArray(diagnostics?.manifest?.scenarios)
    ? diagnostics.manifest.scenarios
    : [];
  const overviewScenarios = Array.isArray(diagnostics?.overview?.scenarios) && diagnostics.overview.scenarios.length
    ? diagnostics.overview.scenarios
    : null;
  const scenarios = (overviewScenarios || manifestScenarios)
    .filter((item) => String(item?.kind || "").toLowerCase() !== "detection");
  const transfers = Array.isArray(diagnostics?.overview?.transfers)
    ? diagnostics.overview.transfers
    : [];
  const available = scenarios.length > 0;

  return {
    available,
    scenarioCount: scenarios.length,
    labScenarioCount: scenarios.filter((item) => item?.kind !== "field").length,
    fieldScenarioCount: scenarios.filter((item) => item?.kind === "field").length,
    transferCount: transfers.length,
    passedCount: scenarios.filter((item) => item?.overall_pass === true).length,
    sourceLabel: available ? "battery_SOH dashboard" : "未导入",
  };
}

export function batteryVisualAssetSummary(diagnostics = {}) {
  const manifest = diagnostics?.manifest || {};
  const profileManifest = diagnostics?.profileManifest || {};
  const artifacts = [...new Set([
    ...(Array.isArray(manifest.diagnostic_artifacts) ? manifest.diagnostic_artifacts : []),
    ...(Array.isArray(profileManifest.diagnostic_artifacts) ? profileManifest.diagnostic_artifacts : []),
    ...(Array.isArray(profileManifest.artifact_files) ? profileManifest.artifact_files : []),
    ...Object.keys(manifest.source_artifacts || {}),
    ...Object.keys(profileManifest.source_artifacts || {}),
  ].map((item) => String(item || "")).filter(Boolean))]
    .filter((name) => !isExcludedBatteryDiagnosticAsset(name));
  const curveIndex = [
    ...(Array.isArray(manifest.curve_artifacts) ? manifest.curve_artifacts : []),
    ...(Array.isArray(profileManifest.curve_artifacts) ? profileManifest.curve_artifacts : []),
  ].filter((item) => !isExcludedBatteryDiagnosticAsset(item?.dataset || item?.filename || item?.name));
  const indexedCurveArtifacts = curveIndex
    .map((item) => String(item?.filename || item?.name || ""))
    .filter(Boolean);
  const curveArtifacts = [...new Set([
    ...artifacts.filter((name) => /_curves\.json$/i.test(name)),
    ...indexedCurveArtifacts,
  ])];
  const curveCopiedCount = curveIndex.filter((item) => item?.copied === true).length;
  const curveTotalBytes = curveIndex.reduce((sum, item) => sum + finiteCount(item?.bytes, 0), 0);
  const algorithmDesign = diagnostics?.algorithmDesign || diagnostics?.algorithm_details || {};
  const modelComparisons = diagnostics?.modelComparisons || diagnostics?.model_comparisons || {};
  const featureWindows = diagnostics?.featureWindows || diagnostics?.feature_windows || {};
  const algorithmDetailAvailable = Boolean(
    algorithmDesign.title
    || algorithmDesign.model_family
    || (Array.isArray(algorithmDesign.sections) && algorithmDesign.sections.length)
    || (Array.isArray(algorithmDesign.features) && algorithmDesign.features.length),
  );
  const comparisonCount = countArrayFields(modelComparisons, ["comparisons", "models", "reports"]);
  const featureWindowCount = countArrayFields(featureWindows, ["windows", "datasets", "features"]);
  const pieces = [];
  if (algorithmDetailAvailable) pieces.push("算法说明");
  if (comparisonCount) pieces.push(`${comparisonCount} 组模型对比`);
  if (featureWindowCount) pieces.push(`${featureWindowCount} 组特征窗口`);
  if (curveArtifacts.length) pieces.push(`${curveArtifacts.length} 个曲线资产`);
  return {
    available: Boolean(pieces.length),
    algorithmDetailAvailable,
    comparisonCount,
    featureWindowCount,
    curveArtifactCount: curveArtifacts.length,
    curveCopiedCount,
    curveTotalBytes,
    curveTotalLabel: byteSizeLabel(curveTotalBytes),
    curveArtifacts,
    statusLabel: pieces.length ? pieces.join(" / ") : "暂无算法详情或曲线资产",
  };
}

export function batteryVisualAssetDetail(diagnostics = {}) {
  const summary = batteryVisualAssetSummary(diagnostics);
  const algorithmDesign = diagnostics?.algorithmDesign || diagnostics?.algorithm_details || {};
  const modelComparisons = diagnostics?.modelComparisons || diagnostics?.model_comparisons || {};
  const featureWindows = diagnostics?.featureWindows || diagnostics?.feature_windows || {};
  const curveIndex = [
    ...(Array.isArray(diagnostics?.manifest?.curve_artifacts) ? diagnostics.manifest.curve_artifacts : []),
    ...(Array.isArray(diagnostics?.profileManifest?.curve_artifacts) ? diagnostics.profileManifest.curve_artifacts : []),
  ].filter((item) => !isExcludedBatteryDiagnosticAsset(item?.dataset || item?.filename || item?.name));
  const algorithmFeatures = Array.isArray(algorithmDesign.features) ? algorithmDesign.features : [];
  const forbidden = Array.isArray(algorithmDesign?.feature_policy?.forbidden)
    ? algorithmDesign.feature_policy.forbidden
    : [];
  return {
    available: summary.available,
    summary,
    algorithm: {
      title: algorithmDesign.title || "暂无算法说明",
      task: algorithmDesign.task || "",
      modelFamily: algorithmDesign.model_family || algorithmDesign.modelFamily || "",
      features: algorithmFeatures,
      featuresLabel: algorithmFeatures.length ? algorithmFeatures.join(" / ") : "暂无",
      forbidden,
      forbiddenLabel: forbidden.length ? forbidden.join(" / ") : "暂无",
      sourceFile: algorithmDesign.source_file || "",
    },
    comparisons: (Array.isArray(modelComparisons.comparisons) ? modelComparisons.comparisons : [])
      .map((item) => ({
        id: String(item?.id || ""),
        title: item?.title || item?.id || "未命名对比",
        scope: item?.scope || item?.goal || "",
        maeLabel: metricLabel(item?.metrics?.mae),
        rmseLabel: metricLabel(item?.metrics?.rmse),
        r2Label: metricLabel(item?.metrics?.r2 ?? item?.metrics?.mean_r2),
        sourceFile: item?.source_file || "",
      })),
    featureWindows: (Array.isArray(featureWindows.windows) ? featureWindows.windows : [])
      .map((item) => {
        const win = Array.isArray(item?.feature_voltage_window) ? item.feature_voltage_window : [];
        return {
          dataset: String(item?.dataset || ""),
          window: win,
          windowLabel: win.length >= 2 ? `${numberLabel(win[0])}-${numberLabel(win[1])} V` : "暂无",
          rows: finiteCount(item?.rows, 0),
          curves: finiteCount(item?.curves, 0),
          sourcePlot: item?.source_plot || "",
        };
      }),
    curves: curveIndex.map((item) => ({
      dataset: String(item?.dataset || ""),
      filename: String(item?.filename || item?.name || ""),
      sizeLabel: byteSizeLabel(item?.bytes || 0),
      copied: item?.copied === true,
      copiedLabel: item?.copied === true ? "已内嵌" : "索引",
      source: String(item?.source || ""),
    })),
  };
}

export function batteryCurvePreviewView(diagnostics = {}, selectedDataset = "") {
  const previews = diagnostics?.curvePreviews || diagnostics?.curve_previews || {};
  const datasets = (Array.isArray(previews.datasets) ? previews.datasets : [])
    .filter((item) => String(item?.kind || "charge_curve") === "charge_curve")
    .filter((item) => !isExcludedBatteryDiagnosticAsset(item?.dataset))
    .map(normalizeCurvePreviewDataset)
    .filter((item) => item.curves.length);
  const selected = datasets.find((item) => item.dataset === selectedDataset) || datasets[0] || null;
  return {
    available: Boolean(selected),
    options: datasets.map((item) => ({
      value: item.dataset,
      label: item.dataset,
      kindLabel: item.kindLabel,
      curveCountLabel: `${item.curveCount.toLocaleString("en-US")} 条曲线`,
    })),
    selectedDataset: selected?.dataset || "",
    datasetLabel: selected?.dataset || "暂无",
    kind: selected?.kind || "",
    kindLabel: selected?.kindLabel || "暂无曲线",
    totalCurves: selected?.curveCount || 0,
    totalCurvesLabel: selected ? `${selected.curveCount.toLocaleString("en-US")} 条原始曲线` : "暂无",
    sourceFile: selected?.sourceFile || "",
    curves: selected?.curves || [],
  };
}

function normalizeCurvePreviewDataset(item = {}) {
  const dataset = String(item?.dataset || "");
  const kind = String(item?.kind || "charge_curve");
  const curves = (Array.isArray(item?.preview_curves) ? item.preview_curves : [])
    .map(normalizeCurvePreviewCurve)
    .filter((curve) => curve.series.length);
  return {
    dataset,
    kind,
    kindLabel: "充电曲线",
    curveCount: finiteCount(item?.total_curves, curves.length),
    sourceFile: String(item?.source_file || ""),
    curves,
  };
}

function normalizeCurvePreviewCurve(curve = {}) {
  return {
    id: String(curve?.id || "未命名曲线"),
    xKey: String(curve?.x_key || "x"),
    pointCount: finiteCount(curve?.point_count, 0),
    sampledPointCount: finiteCount(curve?.sampled_point_count, 0),
    thresholdMv: curve?.threshold_mv,
    levelMv: Array.isArray(curve?.level_mv) ? curve.level_mv : [],
    series: (Array.isArray(curve?.series) ? curve.series : [])
      .map((series) => {
        const points = (Array.isArray(series?.points) ? series.points : [])
          .map((point) => Array.isArray(point) ? [Number(point[0]), Number(point[1])] : null)
          .filter((point) => point && Number.isFinite(point[0]) && Number.isFinite(point[1]));
        return {
          key: String(series?.key || ""),
          label: String(series?.label || series?.key || "series"),
          min: Number.isFinite(Number(series?.min)) ? Number(series.min) : null,
          max: Number.isFinite(Number(series?.max)) ? Number(series.max) : null,
          pointCount: points.length,
          points,
        };
      })
      .filter((series) => series.points.length),
  };
}

export function acceptanceDetailSummary({
  rows = [],
  datasetRows = [],
  preset = "strict",
} = {}) {
  const selectedPreset = ACCEPTANCE_PRESETS[preset] || ACCEPTANCE_PRESETS.strict;
  const aggregateRows = acceptanceDetailRows(rows, selectedPreset);
  const datasetDetailRows = acceptanceDetailRows(datasetRows, selectedPreset);
  const aggregate = summarizeAcceptanceRows(aggregateRows);
  const dataset = summarizeAcceptanceRows(datasetDetailRows);
  const failingRows = [...aggregateRows]
    .filter((row) => row.failed.length)
    .sort((a, b) =>
      b.failed.length - a.failed.length
      || numberOrInfinity(b.metrics.mae) - numberOrInfinity(a.metrics.mae)
      || a.name.localeCompare(b.name));

  return {
    preset: selectedPreset,
    thresholds: selectedPreset.thresholds,
    aggregate,
    dataset,
    rows: aggregateRows,
    datasetRows: datasetDetailRows,
    failingRows,
  };
}

export function acceptanceKpiSummary({
  mode = "training",
  rows = [],
  datasetRows = [],
  inference = {},
  preset = "strict",
} = {}) {
  if (mode === "inference") {
    const records = Array.isArray(inference.records) ? inference.records : [];
    const summary = inference.summary || {};
    const total = finiteCount(summary.n_records, records.length);
    const passed = finiteCount(
      summary.n_succeeded,
      records.filter((row) => row?.status === "succeeded").length,
    );
    return {
      label: "推理成功",
      value: `${passed}/${total}`,
      hint: "inference-only 成功记录",
      color: total > 0 && passed === total ? "var(--pass)" : (passed ? "var(--warn)" : "var(--fail)"),
    };
  }

  const detail = acceptanceDetailSummary({ rows, datasetRows, preset });
  const selectedPreset = detail.preset;
  return {
    label: selectedPreset.kpiLabel,
    value: detail.aggregate.label,
    hint: detail.dataset.total
      ? `聚合口径；数据集级 ${detail.dataset.label} 子项通过`
      : "泛化 / 校准 / 物理一致",
    color: detail.aggregate.total > 0 && detail.aggregate.pass === detail.aggregate.total
      ? "var(--pass)"
      : (detail.aggregate.pass ? "var(--warn)" : "var(--fail)"),
  };
}

export function batteryScenarioRows(scenarios = []) {
  return (Array.isArray(scenarios) ? scenarios : [])
    .filter((item) => String(item?.kind || "").toLowerCase() !== "detection")
    .map((item) => {
    const nestedMae = Array.isArray(item?.algorithms)
      ? item.algorithms.map((algo) => metricLabel(algo?.metrics?.mae ?? algo?.mae)).find(Boolean)
      : "";
    return {
      name: firstNonEmpty(item?.name, item?.title, item?.id, item?.scenario, item?.mode, "未命名场景"),
      kind: firstNonEmpty(item?.kind, "lab"),
      source: firstNonEmpty(item?.dataset, item?.source, item?.profile, item?.mode, item?.chemistry, "未记录"),
      verdict: item?.overall_pass === true ? "通过" : (item?.overall_pass === false ? "未通过" : "未记录"),
      metric: firstNonEmpty(metricLabel(item?.mae), metricLabel(item?.metric), nestedMae, "暂无"),
    };
    });
}

function isExcludedBatteryDiagnosticAsset(value) {
  return String(value || "").toLowerCase().includes("ess_detection");
}

function firstNonEmpty(...values) {
  const value = values.find((item) => item != null && String(item).trim() !== "");
  return value == null ? "" : String(value).trim();
}

function parseJobTimestamp(value) {
  if (!value) return null;
  const normalized = String(value).trim().replace(" ", "T");
  const parsed = new Date(normalized);
  return Number.isFinite(parsed.getTime()) ? parsed : null;
}

function durationLabel(totalMinutes) {
  const minutes = Math.max(0, Number(totalMinutes) || 0);
  if (minutes < 60) return `${minutes} 分钟`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} 小时 ${rest} 分钟` : `${hours} 小时`;
}

function trainingMatrixLabel(request = {}) {
  const datasets = nonEmptyStrings(request.datasets || []).length;
  const estimators = nonEmptyStrings(request.estimators || []).length;
  const featuresets = nonEmptyStrings(request.featuresets || []).length;
  const protocols = nonEmptyStrings(request.protocols || []).map((item) => item.toUpperCase()).join("/");
  const parts = [];
  if (datasets) parts.push(`${datasets} 数据集`);
  if (estimators) parts.push(`${estimators} 算法`);
  if (featuresets) parts.push(`${featuresets} 特征`);
  if (protocols) parts.push(protocols);
  return parts.join(" x ");
}

function metricLabel(value) {
  if (value == null || value === "") return "";
  if (typeof value === "object") {
    return metricLabel(value.mean ?? value.value ?? value.mae);
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  return numeric.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
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

export function inferenceCompatibilitySummary({ models = [], datasets = [], featureSets = [], featuresets = [] } = {}) {
  const featureSetOptions = featureSets.length ? featureSets : featuresets;
  const featureColumnsByName = new Map(
    (featureSetOptions || [])
      .map((item) => [String(item?.name || ""), nonEmptyStrings(item?.columns || item?.featureColumns || item?.feature_columns || [])])
      .filter(([name]) => name),
  );
  const rows = [];
  (models || []).forEach((model) => {
    const required = nonEmptyStrings(model?.featureColumns || model?.feature_columns || []);
    const generated = featureColumnsByName.get(String(model?.featureset || model?.featureSet || "")) || [];
    (datasets || []).forEach((dataset) => {
      const available = nonEmptyStrings(dataset?.columns || []);
      if (!required.length || (!available.length && !generated.length)) {
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
      const availableSet = new Set([...available, ...generated]);
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

export function inferenceEvaluationEligibility(model = {}, dataset = {}) {
  const datasetName = String(dataset?.name || dataset?.dataset || "");
  const trainedOn = nonEmptyStrings(model?.trainedOn || model?.trained_on || []);
  const heldOutKey = String(model?.heldOutKey || model?.held_out_key || "");
  const protocol = String(model?.protocol || "").toLowerCase();
  const targetChemistry = String(dataset?.chemistry || "unknown");
  const sourceChemistries = nonEmptyStrings(model?.sourceChemistries || model?.source_chemistries || []);
  const isKnownChemistry = (value) => value && value.toLowerCase() !== "unknown";

  if (!datasetName) return { eligible: false, scope: "unknown", reason: "缺少目标数据集" };
  if (protocol === "loco" && heldOutKey.startsWith(`${datasetName}/`)) {
    return { eligible: true, scope: "held_out_test", reason: `仅评测训练未使用的 ${heldOutKey}` };
  }
  if (protocol === "lodo" && heldOutKey === datasetName) {
    return { eligible: true, scope: "held_out_test", reason: "使用训练时完整留出的目标数据集" };
  }
  if (trainedOn.includes(datasetName)) {
    return { eligible: false, scope: "training_overlap", reason: "目标数据参与过训练且没有匹配的留出切分" };
  }
  if (!trainedOn.length) {
    return { eligible: false, scope: "provenance_unknown", reason: "模型未声明训练数据范围" };
  }
  if (!isKnownChemistry(targetChemistry)) {
    return { eligible: false, scope: "chemistry_unknown", reason: "目标数据化学体系待确认" };
  }
  if (!sourceChemistries.length) {
    return { eligible: true, scope: "runtime_chemistry_check", reason: "后端将根据训练数据元数据复核同化学体系" };
  }
  if (sourceChemistries.includes(targetChemistry)) {
    return { eligible: true, scope: "same_chemistry_transfer", reason: `同化学体系迁移（${targetChemistry}）` };
  }
  return { eligible: false, scope: "cross_chemistry", reason: `训练与目标化学体系不一致（${sourceChemistries.join("/")} → ${targetChemistry}）` };
}

export function inferenceLaunchPlan({
  modelNames = [],
  datasetNames = [],
  models = [],
  datasets = [],
  featureSets = [],
  featuresets = [],
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
        evaluation_policy: "research_any",
        selected_models: selectedModelNames,
        submitted_models: selectedModelNames,
        skipped_models: [],
        selected_datasets: selectedDatasetNames,
        submitted_datasets: selectedDatasetNames,
        skipped_datasets: [],
      },
      skippedModels: [],
      skippedDatasets: [],
      skippedCount: 0,
      message: "已选择包含风险组合，后端会执行运行时 schema 校验。",
    };
  }

  const modelByName = new Map((models || []).map((item) => [item?.name, item]).filter(([name]) => name));
  const datasetByName = new Map((datasets || []).map((item) => [item?.name, item]).filter(([name]) => name));
  const selectedModels = selectedModelNames.map((name) => modelByName.get(name) || { name });
  const selectedDatasets = selectedDatasetNames.map((name) => datasetByName.get(name) || { name });
  const compatibility = inferenceCompatibilitySummary({
    models: selectedModels,
    datasets: selectedDatasets,
    featureSets: featureSets.length ? featureSets : featuresets,
  });
  const eligibilityRows = selectedModels.flatMap((model) => selectedDatasets.map((dataset) => ({
    model: model.name,
    dataset: dataset.name,
    ...inferenceEvaluationEligibility(model, dataset),
  })));
  compatibility.rows = compatibility.rows.map((row) => {
    const eligibility = eligibilityRows.find((item) => item.model === row.model && item.dataset === row.dataset);
    return { ...row, evaluation: eligibility || null };
  });
  const compatibleModels = selectedModelNames.filter((modelName) =>
    selectedDatasetNames.some((datasetName) => compatibility.rows.some((row) =>
      row.model === modelName && row.dataset === datasetName && row.status === "ok" && row.evaluation?.eligible
    ))
  );
  const compatibleDatasets = selectedDatasetNames.filter((datasetName) =>
    compatibleModels.some((modelName) => compatibility.rows.some((row) =>
      row.model === modelName && row.dataset === datasetName && row.status === "ok" && row.evaluation?.eligible
    ))
  );
  const allowedModels = compatibleModels;
  const allowedDatasets = compatibleDatasets;
  const skippedModels = selectedModelNames.filter((name) => !allowedModels.includes(name));
  const skippedDatasets = selectedDatasetNames.filter((name) => !allowedDatasets.includes(name));
  const skippedParts = [];
  if (skippedModels.length) {
    skippedParts.push(`已跳过 ${skippedModels.length} 个无安全测试范围的模型：${skippedModels.slice(0, 3).join(", ")}`);
  }
  if (skippedDatasets.length) {
    skippedParts.push(`已跳过 ${skippedDatasets.length} 个无留出或同体系范围的数据集：${skippedDatasets.slice(0, 3).join(", ")}`);
  }

  return {
    strategy: "compatible_only",
    payload: {
      ...buildInferencePayload({ models: allowedModels, datasets: allowedDatasets }),
      launch_strategy: "compatible_only",
      evaluation_policy: "held_out_same_chemistry",
      selected_models: selectedModelNames,
      submitted_models: allowedModels,
      skipped_models: skippedModels,
      selected_datasets: selectedDatasetNames,
      submitted_datasets: allowedDatasets,
      skipped_datasets: skippedDatasets,
    },
    skippedModels,
    skippedDatasets,
    skippedCount: skippedModels.length + skippedDatasets.length,
    compatibility,
    message: skippedParts.length
      ? skippedParts.join("；")
      : "所有已选模型与数据集通过 schema 预检。",
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
  const selectedModels = nonEmptyStrings(request.selected_models || request.selectedModels || request.models || []);
  const submittedModels = nonEmptyStrings(request.submitted_models || request.submittedModels || request.models || []);
  const skippedModels = nonEmptyStrings(request.skipped_models || request.skippedModels || []);
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
    || selectedModels.length
    || submittedModels.length
    || skippedModels.length
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
    selectedModels,
    submittedModels,
    skippedModels,
    selectedModelCount: selectedModels.length,
    submittedModelCount: submittedModels.length,
    skippedModelCount: skippedModels.length,
    skippedModelLabel: skippedModels.length ? skippedModels.slice(0, 4).join(" / ") : "无跳过模型",
    selectedDatasets,
    submittedDatasets,
    skippedDatasets,
    selectedCount: selectedDatasets.length,
    submittedCount: submittedDatasets.length,
    skippedDatasetCount: skippedDatasets.length,
    skippedCount: skippedModels.length + skippedDatasets.length,
    skippedLabel,
    jobId,
    status,
    runDir,
    summary: hasAudit
      ? `${strategyLabel}；提交 ${submittedModels.length} / ${selectedModels.length} 个模型、${submittedDatasets.length} / ${selectedDatasets.length} 个数据集；共跳过 ${skippedModels.length + skippedDatasets.length} 项。`
      : "暂无推理策略审计记录。",
  };
}

function hasInferenceAuditFields(request = {}) {
  return Boolean(
    request?.launch_strategy
    || request?.launchStrategy
    || request?.selected_models
    || request?.submitted_models
    || request?.skipped_models
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
  if (value === "failed") return "失败";
  if (value === "running") return "运行中";
  if (value === "queued") return "排队中";
  return "未知";
}

function uniqueStrings(values = []) {
  return [...new Set(nonEmptyStrings(values))];
}

function countArrayFields(source = {}, names = []) {
  return names.reduce((count, name) => {
    const value = source?.[name];
    return Math.max(count, Array.isArray(value) ? value.length : 0);
  }, 0);
}

function acceptanceDetailRows(rows = [], preset = ACCEPTANCE_PRESETS.strict) {
  return (rows || []).map((row) => {
    const metrics = acceptanceMetrics(row);
    const verdicts = evaluateAcceptanceRow(row, preset, metrics);
    const failed = Object.entries(verdicts)
      .filter(([, passed]) => passed === false)
      .map(([name]) => name);
    return {
      estimator: String(row?.estimator || "未记录算法"),
      featureset: String(row?.featureset || "full"),
      protocol: String(row?.protocol || ""),
      heldOutKey: String(row?.held_out_key || ""),
      scope: heldOutDatasetName(row?.held_out_key) || row?.protocol || "aggregate",
      name: `${row?.estimator || "未记录算法"} x ${row?.featureset || "full"}`,
      verdicts,
      failed,
      reasons: acceptanceFailureReasons(metrics, verdicts, preset),
      metrics,
      ciSource: row?.ci_source || "",
    };
  });
}

function acceptanceMetrics(row = {}) {
  return {
    mae: metricMean(row, "mae"),
    picp: metricMean(row, "picp"),
    monotonicity: metricMean(row, "monotonicity_violation_rate"),
    valueRange: metricMean(row, "value_range_violation"),
  };
}

function evaluateAcceptanceRow(row = {}, preset = ACCEPTANCE_PRESETS.strict, metrics = acceptanceMetrics(row)) {
  const thresholds = preset.thresholds;
  const fallback = row?.verdicts || {};
  const hasMae = metrics.mae != null;
  const hasPicp = metrics.picp != null;
  const hasMono = metrics.monotonicity != null;
  const hasRange = metrics.valueRange != null;
  const generalization = hasMae
    ? metrics.mae <= thresholds.maeMax
    : fallback.generalization ?? null;
  const calibration = hasPicp
    ? metrics.picp >= thresholds.picpMin
    : fallback.calibration ?? null;
  const physical_consistency = (hasMono || hasRange)
    ? ((hasMono ? metrics.monotonicity <= thresholds.monotonicityMax : true)
      && (hasRange ? metrics.valueRange <= thresholds.rangeMax : true))
    : fallback.physical_consistency ?? null;

  return {
    generalization,
    calibration,
    physical_consistency,
  };
}

function acceptanceFailureReasons(metrics = {}, verdicts = {}, preset = ACCEPTANCE_PRESETS.strict) {
  const thresholds = preset.thresholds;
  const reasons = [];
  if (verdicts.generalization === false) {
    reasons.push(metrics.mae == null
      ? "泛化未通过"
      : `MAE ${numberLabel(metrics.mae)} > ${numberLabel(thresholds.maeMax)}`);
  }
  if (verdicts.calibration === false) {
    reasons.push(metrics.picp == null
      ? "校准未通过"
      : `PICP ${numberLabel(metrics.picp)} < ${numberLabel(thresholds.picpMin)}`);
  }
  if (verdicts.physical_consistency === false) {
    const parts = [];
    if (metrics.monotonicity != null && metrics.monotonicity > thresholds.monotonicityMax) {
      parts.push(`单调违例 ${numberLabel(metrics.monotonicity)} > ${numberLabel(thresholds.monotonicityMax)}`);
    }
    if (metrics.valueRange != null && metrics.valueRange > thresholds.rangeMax) {
      parts.push(`越界 ${numberLabel(metrics.valueRange)} > ${numberLabel(thresholds.rangeMax)}`);
    }
    reasons.push(parts.length ? parts.join("；") : "物理一致性未通过");
  }
  return reasons;
}

function summarizeAcceptanceRows(rows = []) {
  let pass = 0;
  let total = 0;
  (rows || []).forEach((row) => {
    Object.values(row?.verdicts || {}).forEach((value) => {
      if (value == null) return;
      total += 1;
      if (value === true) pass += 1;
    });
  });
  return {
    pass,
    total,
    fail: Math.max(0, total - pass),
    label: `${pass}/${total}`,
  };
}

function countVerdicts(rows = []) {
  let pass = 0;
  let total = 0;
  (rows || []).forEach((row) => {
    Object.values(row?.verdicts || {}).forEach((value) => {
      if (value == null) return;
      total += 1;
      if (value === true) pass += 1;
    });
  });
  return { pass, total };
}

function finiteCount(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

export function inferenceActionState({
  backendConnected = false,
  selectedModelCount = 0,
  selectedDatasetCount = 0,
  currentJob = null,
  currentJobMatchesSelection = true,
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
    if (currentJob.status === "succeeded" && currentJobMatchesSelection) {
      return {
        disabled: false,
        tone: "ok",
        message: `推理任务 ${currentJob.id} 已完成，结果已发布到推理结果表。`,
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
      message: "请选择至少一个候选产物和一个目标数据集。",
    };
  }
  if (!selectedModelCount) {
    return {
      disabled: true,
      tone: "warn",
      message: "请选择至少一个候选产物，或先完成训练。",
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
    tone: currentJob?.status === "succeeded" && !currentJobMatchesSelection ? "warn" : "ready",
    message: currentJob?.status === "succeeded" && !currentJobMatchesSelection
      ? `已完成任务 ${currentJob.id} 的结果仍保留；当前选择已变化，点击将创建新的推理任务。`
      : `已选择 ${selectedModelCount} 个候选 / ${selectedDatasetCount} 个数据集，可启动推理验证。`,
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
    const aContracted = Boolean(a.contractVersion && a.contractVersion !== "legacy");
    const bContracted = Boolean(b.contractVersion && b.contractVersion !== "legacy");
    return am - bm || Number(bContracted) - Number(aContracted) || a.name.localeCompare(b.name);
  });
}

export function modelAssetPage(assets = [], { limit = 120 } = {}) {
  const totalCount = Array.isArray(assets) ? assets.length : 0;
  const safeLimit = Math.max(0, Number(limit) || 0);
  const rows = (assets || []).slice(0, safeLimit);
  const hiddenCount = Math.max(0, totalCount - rows.length);
  return {
    rows,
    totalCount,
    shownCount: rows.length,
    hiddenCount,
    limited: hiddenCount > 0,
    note: hiddenCount > 0
      ? `显示 ${rows.length} / ${totalCount} 个模型，继续筛选或后续分页查看剩余 ${hiddenCount} 个。`
      : `显示 ${rows.length} / ${totalCount} 个模型。`,
  };
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

function numberOrInfinity(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : Infinity;
}

function finiteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function numberLabel(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "暂无";
  return numeric.toFixed(Math.abs(numeric) >= 10 ? 2 : 4).replace(/0+$/, "").replace(/\.$/, "");
}

function byteSizeLabel(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  const kib = bytes / 1024;
  if (kib < 1024) return `${kib.toFixed(1).replace(/\.0$/, "")} KiB`;
  const mib = kib / 1024;
  return `${mib.toFixed(1).replace(/\.0$/, "")} MiB`;
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
