import assert from "node:assert/strict";
import {
  summarizeBySelection,
  datasetOptions,
  algorithmInsights,
  auditNextAction,
  bestComparableEntry,
  buildEvaluationPayload,
  buildInferencePayload,
  batteryDiagnosticSummary,
  batteryCurvePreviewView,
  batteryVisualAssetDetail,
  batteryVisualAssetSummary,
  batteryScenarioRows,
  caseContextProjection,
  casePipelineJobId,
  casePipelineProgress,
  pipelineJobSequenceValue,
  pipelineReconciliationTarget,
  pipelineCombinationConsistency,
  auditEvidenceDatasetScope,
  requiresFreshTrainingCase,
  catalogOptions,
  evaluationPlanSummary,
  ACCEPTANCE_PRESETS,
  acceptanceDetailSummary,
  acceptanceKpiSummary,
  inferenceRows,
  inferenceAuditSummary,
  inferenceActionState,
  algorithmRecommendationView,
  inferenceCompatibilitySummary,
  inferenceEvaluationEligibility,
  inferenceDatasetOptions,
  inferenceLaunchPlan,
  inferenceInsightSummary,
  inferenceSummary,
  preferredInferenceModels,
  profileOptions,
  retestComparisonView,
  leaderboardCoverage,
  modelAssetPage,
  modelAssetRows,
  modelOptions,
  pipelineModelGroups,
  scopeSummary,
  stabilityRows,
  taskHistoryRows,
  trainingArtifactSummary,
  trainingJobRuntimeState,
  trajectoryContextConsistency,
  auditCandidateViewProjection,
  nextDatasetSelection,
  nativeDashboardUrl,
} from "./dashboard-core.mjs";

{
  const caseVerdict = {
    result: "pass",
    observed: { mae: 0.01, picp: 0.93, monotonicity: 0.02, value_range: 0 },
    thresholds: { mae_max: 0.08, picp_min: 0.85, monotonicity_max: 0.15, value_range_max: 0 },
    reasons: [],
    criterion_set_version: "engineering@custom-1",
  };
  const pinned = auditCandidateViewProjection({
    candidate: { metrics: { mae: { mean: 0.4 } } },
    caseVerdict,
    pinned: true,
  });
  assert.equal(pinned.mode, "pinned");
  assert.equal(pinned.observed.mae, 0.01);

  const preview = auditCandidateViewProjection({
    candidate: {
      metrics: {
        mae: { mean: 0.12 },
        picp: { mean: 0.9 },
        monotonicity_violation_rate: { mean: 0.2 },
        value_range_violation: { mean: 0 },
      },
    },
    caseVerdict,
    pinned: false,
  });
  assert.equal(preview.mode, "preview");
  assert.equal(preview.result, "fail");
  assert.equal(preview.observed.mae, 0.12);
  assert.deepEqual(preview.reasons, ["generalization", "monotonicity"]);
}

{
  const expected = {
    case_id: "EV-1",
    inference_job_id: "infer-1",
    dataset: "nasa",
    candidate_id: "candidate-a",
    model_artifact_id: "model@sha256:a",
    context_binding_id: "ctx@sha256:1",
  };
  assert.deepEqual(trajectoryContextConsistency({ ...expected }, expected), {
    matches: true,
    mismatches: [],
  });
  const mismatch = trajectoryContextConsistency({
    ...expected,
    candidate_id: "candidate-b",
    model_artifact_id: "model@sha256:b",
  }, expected);
  assert.equal(mismatch.matches, false);
  assert.deepEqual(mismatch.mismatches.map((item) => item.field), [
    "candidate_id",
    "model_artifact_id",
  ]);
}

{
  const scope = auditEvidenceDatasetScope({
    caseDocument: {
      active_saved_combination_id: "saved-nasa",
      execution: { inference_job_id: "infer-current" },
      evidence_selection: { dataset: "calce_cs2" },
      saved_combinations: [
        { saved_combination_id: "saved-nasa", source_stage: "inference", inference_job_id: "infer-current", target_dataset: "nasa" },
        { saved_combination_id: "saved-hust", source_stage: "inference", inference_job_id: "infer-current", target_dataset: "hust" },
        { saved_combination_id: "saved-old", source_stage: "inference", inference_job_id: "infer-old", target_dataset: "calce_cs2" },
      ],
    },
    inferenceJob: {
      id: "infer-current",
      request: { submitted_datasets: ["nasa", "hust"] },
    },
  });
  assert.deepEqual(scope, ["nasa", "hust"]);
}

{
  assert.equal(requiresFreshTrainingCase({ id: "EV-DRAFT", lifecycle_state: "draft" }), false);
  assert.equal(requiresFreshTrainingCase({
    id: "EV-USED",
    lifecycle_state: "active",
    execution: { training_job_id: "train-old" },
  }), true);
  assert.equal(requiresFreshTrainingCase({
    id: "EV-FAILED",
    lifecycle_state: "failed",
  }), true);
}

{
  assert.equal(pipelineJobSequenceValue({ id: "20260816T070748-9634676d" }), "20260816T070748");
  const staleInference = pipelineReconciliationTarget({
    caseDocument: {
      execution: {
        state: "queued",
        run_id: "20260816T070748-9634676d",
        training_job_id: "20260816T070748-9634676d",
        inference_job_id: "20260816T044224-a95a77a9",
      },
      evidence_selection: { sample_id: "pred:20260816T044224-a95a77a9:old" },
    },
    trainingJob: { id: "20260816T070748-9634676d", status: "succeeded" },
    inferenceJob: { id: "20260816T044224-a95a77a9", status: "succeeded" },
  });
  assert.equal(staleInference.kind, "training");
  assert.equal(staleInference.expectedState, "completed");
  assert.equal(staleInference.clearInferenceJob, true);
  assert.equal(staleInference.needsIdentitySync, true);

  const completedInference = pipelineReconciliationTarget({
    caseDocument: {
      execution: {
        state: "completed",
        run_id: "20260816T071500-newinfer",
        training_job_id: "20260816T070748-9634676d",
        inference_job_id: "20260816T071500-newinfer",
      },
      evidence_selection: { sample_id: "pred:20260816T044224-a95a77a9:old" },
    },
    trainingJob: { id: "20260816T070748-9634676d", status: "succeeded" },
    inferenceJob: { id: "20260816T071500-newinfer", status: "succeeded" },
  });
  assert.equal(completedInference.kind, "inference");
  assert.equal(completedInference.needsIdentitySync, false);
  assert.equal(completedInference.needsEvidenceBinding, true);
}

{
  const profile = { trainingJobId: "profile-train", inferenceJobId: "profile-infer" };
  assert.equal(casePipelineJobId(null, profile, "training"), "profile-train");
  assert.equal(casePipelineJobId(null, profile, "inference"), "profile-infer");
  assert.equal(casePipelineJobId({
    id: "EV-NEW",
    execution: { training_job_id: "case-train", inference_job_id: "" },
  }, profile, "training"), "case-train");
  assert.equal(casePipelineJobId({
    id: "EV-NEW",
    execution: { training_job_id: "case-train", inference_job_id: "" },
  }, profile, "inference"), "");
}

{
  const models = modelOptions({
    models: [
      {
        name: "ridge_full_loco_nasa_B0005_seed0",
        estimator: "ridge",
        featureset: "full",
        protocol: "loco",
        held_out_key: "nasa/B0005",
        created_by_run: "train-24",
      },
      {
        name: "ridge_full_loco_nasa_B0018_seed0",
        estimator: "ridge",
        featureset: "full",
        protocol: "loco",
        held_out_key: "nasa/B0018",
        created_by_run: "train-24",
      },
      { name: "global_old_model", estimator: "gbdt", featureset: "full" },
    ],
  });
  const grouped = pipelineModelGroups({
    trainingJob: {
      id: "train-24",
      saved_models: [
        "ridge_full_loco_nasa_B0005_seed0",
        "ridge_full_loco_nasa_B0018_seed0",
      ],
      model_artifacts: [
        { name: "ridge_full_loco_nasa_B0005_seed0", estimator: "ridge", featureset: "full", protocol: "loco", held_out_key: "nasa/B0005" },
        { name: "ridge_full_loco_nasa_B0018_seed0", estimator: "ridge", featureset: "full", protocol: "loco", held_out_key: "nasa/B0018" },
      ],
      result_summary: {
        best_estimator: "ridge",
        best_featureset: "full",
        best_protocol: "loco",
        best_held_out_key: "nasa/B0018",
      },
    },
    models,
  });
  assert.equal(grouped.groups.length, 1);
  assert.equal(grouped.groups[0].artifactCount, 2);
  assert.equal(grouped.groups[0].recommendedHeldOutKey, "nasa/B0018");
  assert.deepEqual(grouped.recommendedModelNames, ["ridge_full_loco_nasa_B0018_seed0"]);
  assert.equal(grouped.otherModelCount, 1);

  const inherited = pipelineModelGroups({
    trainingJob: { id: "train-24", saved_models: grouped.runModelNames },
    inferenceJob: { id: "infer-24", request: { models: ["ridge_full_loco_nasa_B0005_seed0"] } },
    models,
  });
  assert.equal(inherited.source, "inference");
  assert.deepEqual(inherited.recommendedModelNames, ["ridge_full_loco_nasa_B0005_seed0"]);
  assert.equal(inherited.groups[0].artifactCount, 2);
}

{
  const progress = casePipelineProgress({
    caseDocument: {
      version: 4,
      execution: { training_job_id: "train-01", inference_job_id: "infer-01" },
      candidates: [{ model_artifact_id: "model@sha256:abc" }],
      evidence_selection: { candidate_id: "artifact:ridge", dataset: "nasa" },
      review: { state: "in_review" },
    },
    trainingJob: { id: "train-01", status: "succeeded" },
    inferenceJob: { id: "infer-01", status: "succeeded" },
  });
  assert.deepEqual(progress.stages.map((item) => item.state), ["complete", "complete", "active"]);
  assert.equal(progress.trainingJobId, "train-01");
  assert.equal(progress.inferenceJobId, "infer-01");
}

{
  const staleDownstream = {
    caseDocument: {
      version: 11,
      lifecycle_state: "active",
      execution: {
        training_job_id: "20260816T120000-newtrain",
        inference_job_id: "20260816T110000-oldinfer",
      },
      candidates: [{ model_artifact_id: "model@sha256:old" }],
      evidence_selection: { candidate_id: "artifact:old", dataset: "nasa" },
      review: { state: "approved" },
    },
    reviewWorkspace: {
      evidence_packs: [{ state: "signed", case_version: 11 }],
    },
    inferenceJob: { id: "20260816T110000-oldinfer", status: "succeeded" },
  };
  const whileTraining = casePipelineProgress({
    ...staleDownstream,
    trainingJob: { id: "20260816T120000-newtrain", status: "running" },
  });
  assert.deepEqual(whileTraining.stages.map((item) => item.state), ["active", "pending", "pending"]);
  assert.equal(whileTraining.stages[0].detail, "训练任务正在运行");
  assert.equal(whileTraining.stages[1].detail, "等待训练产物");
  assert.match(whileTraining.stages[2].detail, /等待当前训练代次/);

  const afterTraining = casePipelineProgress({
    ...staleDownstream,
    trainingJob: { id: "20260816T120000-newtrain", status: "succeeded" },
  });
  assert.deepEqual(afterTraining.stages.map((item) => item.state), ["complete", "pending", "pending"]);
  assert.match(afterTraining.stages[1].detail, /等待启动/);
}

{
  const failedCase = {
    id: "EV-FAILED",
    version: 7,
    lifecycle_state: "active",
    review: { state: "in_review" },
    verdict: { result: "fail" },
    evidence_coverage: { state: "complete" },
    evidence_selection: { sample_id: "calce:CS2_38:277", cycle_id: 277 },
  };
  const createIssue = auditNextAction({ caseDocument: failedCase });
  assert.equal(createIssue.action, "create-issue");
  assert.equal(createIssue.label, "创建问题并复测");
  assert.equal(createIssue.disabled, false);

  const createRegression = auditNextAction({
    caseDocument: failedCase,
    reviewWorkspace: { issues: [{ id: "ISS-1", status: "open" }] },
  });
  assert.equal(createRegression.action, "create-regression");
  assert.equal(createRegression.issueId, "ISS-1");

  const addMembers = auditNextAction({
    caseDocument: failedCase,
    reviewWorkspace: {
      issues: [{ id: "ISS-1", status: "open" }],
      regression_sets: [{ id: "REG-1", issue_id: "ISS-1", members: [] }],
    },
  });
  assert.equal(addMembers.action, "add-member");
  assert.equal(addMembers.neighborRadius, 2);

  const runRetest = auditNextAction({
    caseDocument: failedCase,
    reviewWorkspace: {
      issues: [{ id: "ISS-1", status: "open" }],
      regression_sets: [{ id: "REG-1", issue_id: "ISS-1", members: [{ cycle_id: 277 }] }],
    },
  });
  assert.equal(runRetest.action, "request-retest");
  assert.equal(runRetest.setId, "REG-1");

  const closeIssue = auditNextAction({
    caseDocument: failedCase,
    reviewWorkspace: {
      issues: [{ id: "ISS-1", status: "in_retest" }],
      regression_sets: [{ id: "REG-1", issue_id: "ISS-1", members: [{ cycle_id: 277 }] }],
      retests: [{ id: "RET-1", issue_id: "ISS-1", regression_set_id: "REG-1", status: "completed" }],
    },
  });
  assert.equal(closeIssue.action, "resolve-issue");
  assert.equal(closeIssue.label, "确认复测并闭环");

  const finalizeClosedFailure = auditNextAction({
    caseDocument: failedCase,
    reviewWorkspace: {
      issues: [{ id: "ISS-1", status: "resolved" }],
      regression_sets: [{ id: "REG-1", issue_id: "ISS-1", members: [{ cycle_id: 277 }] }],
      retests: [{ id: "RET-1", issue_id: "ISS-1", regression_set_id: "REG-1", status: "completed" }],
    },
  });
  assert.equal(finalizeClosedFailure.action, "finalize-failed");
  assert.equal(finalizeClosedFailure.label, "确认不通过并结束");

  const rejected = auditNextAction({
    caseDocument: { ...failedCase, review: { state: "rejected" } },
  });
  assert.equal(rejected.action, "finalize-failed");
}

{
  const progress = casePipelineProgress({
    caseDocument: {
      version: 8,
      lifecycle_state: "failed",
      execution: { training_job_id: "train-01", inference_job_id: "infer-01" },
      candidates: [{ model_artifact_id: "model@sha256:abc" }],
      evidence_selection: { candidate_id: "artifact:ridge", dataset: "nasa" },
      review: { state: "rejected" },
    },
    trainingJob: { id: "train-01", status: "succeeded" },
    inferenceJob: { id: "infer-01", status: "succeeded" },
  });
  assert.equal(progress.stages[2].state, "complete");
  assert.match(progress.stages[2].detail, /不通过/);
}

{
  const caseDocument = {
    id: "EV-PIPELINE-01",
    execution: { training_job_id: "train-01", inference_job_id: "infer-01" },
    evidence_selection: { candidate_id: "artifact:ridge", dataset: "nasa" },
    context_binding: {
      binding_id: "ctx@sha256:pipe01",
      generation: 3,
      training_job_id: "train-01",
      inference_job_id: "infer-01",
      candidate_id: "artifact:ridge",
      active_dataset: "nasa",
      model: "ridge",
      featureset: "full",
      model_artifact_id: "model@sha256:ridge",
    },
    candidates: [{
      candidate_id: "artifact:ridge",
      model: "ridge",
      featureset: "full",
      model_artifact_id: "model@sha256:ridge",
    }],
  };
  const trainingJob = {
    id: "train-01",
    request: { datasets: ["nasa"], estimators: ["ridge"], featuresets: ["full"] },
    model_artifacts: [{
      name: "ridge_full_loco_nasa_B0018_seed0",
      estimator: "ridge",
      featureset: "full",
      artifact_id: "model@sha256:ridge",
    }],
  };
  const consistent = pipelineCombinationConsistency({
    caseDocument,
    trainingJob,
    inferenceJob: {
      id: "infer-01",
      request: {
        datasets: ["nasa"],
        models: ["ridge_full_loco_nasa_B0018_seed0"],
        model_bindings: [{
          name: "ridge_full_loco_nasa_B0018_seed0",
          estimator: "ridge",
          featureset: "full",
          artifact_id: "model@sha256:ridge",
        }],
      },
    },
  });
  assert.equal(consistent.state, "consistent");
  assert.equal(consistent.consistent, true);
  assert.deepEqual(consistent.combination, {
    dataset: "nasa",
    estimator: "ridge",
    featureset: "full",
    artifactId: "model@sha256:ridge",
  });
  assert.deepEqual(
    Object.values(consistent.pageChecks).map((item) => item.matches),
    [true, true, true],
  );

  const mismatch = pipelineCombinationConsistency({
    caseDocument,
    trainingJob,
    inferenceJob: {
      id: "infer-01",
      request: {
        datasets: ["calce_cs2"],
        models: ["gbdt_voltage"],
        model_bindings: [{
          name: "gbdt_voltage",
          estimator: "gbdt",
          featureset: "voltage",
          artifact_id: "model@sha256:gbdt",
        }],
      },
    },
  });
  assert.equal(mismatch.state, "mismatch");
  assert.equal(mismatch.pageChecks.training.matches, true);
  assert.equal(mismatch.pageChecks.inference.matches, false);
  assert.match(mismatch.reasons.join("；"), /推理页模型或数据集/);
}

{
  const projected = caseContextProjection(
    {
      id: "EV-20260813-7870FB08",
      version: 5,
      profile_name: "phase22-golden-nasa",
      source_class: "measured",
      execution: { state: "completed", run_id: "run-golden" },
      evidence_selection: {
        candidate_id: "artifact:ridge",
        dataset: "nasa",
        dataset_snapshot_id: "dataset-snapshot@sha256:data",
        split_assignment_id: "protocol@sha256:split",
        asset_id: "B0018",
        cycle_id: 77,
        sample_id: "nasa:B0018:77",
      },
      candidates: [{
        candidate_id: "artifact:ridge",
        model: "ridge",
        featureset: "full",
        model_artifact_id: "model@sha256:model",
      }],
      verdict: {
        result: "pass",
        criterion_set_version: "exploratory",
        observed: { mae: 0.04, picp: 0.99 },
      },
      review: { state: "in_review" },
    },
    {
      issues: [{ id: "ISS-1", status: "in_retest" }],
      regression_sets: [{ id: "REG-1", members: [{ sample_id: "nasa:B0018:77" }] }],
      retests: [{
        id: "RET-1",
        status: "completed",
        result: { regression_summary: { member_count: 5 } },
      }],
      evidence_packs: [{ id: "PACK-1", state: "signed", case_version: 5 }],
    },
  );
  assert.equal(projected.caseId, "EV-20260813-7870FB08");
  assert.equal(projected.dataset.name, "nasa");
  assert.equal(projected.candidate.model, "ridge");
  assert.equal(projected.candidate.artifactId, "model@sha256:model");
  assert.equal(projected.protocolRef, "protocol@sha256:split");
  assert.equal(projected.verdict.criterionSetVersion, "exploratory");
  assert.equal(projected.closure.state, "open");
  assert.equal(projected.closure.retestMemberCount, 5);
}

{
  const projected = caseContextProjection(
    {
      id: "EV-RETRAINED",
      evidence_selection: {},
      candidates: [
        {
          candidate_id: "artifact:old",
          model: "ridge",
          featureset: "full",
          role: "champion",
          run_id: "train-old",
        },
        {
          candidate_id: "artifact:new",
          model: "gbdt",
          featureset: "capacity_time",
          role: "champion",
          run_id: "train-new",
        },
      ],
    },
    {},
    {
      runId: "train-new",
      datasets: ["calce_cs2"],
      candidate: { id: "artifact:new" },
    },
  );
  assert.equal(projected.dataset.name, "calce_cs2");
  assert.equal(projected.candidate.model, "gbdt");
  assert.equal(projected.candidate.featureset, "capacity_time");
}

{
  const view = retestComparisonView({
    comparison_contract: "controlled-retest-comparison/1.0",
    verdict: "fail",
    baseline_metrics: { mae: 0.01, picp: 1 },
    metrics: { mae: 0.03, picp: 0.8, evaluated_count: 5 },
    metric_deltas: { mae: 0.02, picp: -0.2 },
    regression_summary: {
      member_count: 5,
      improved_count: 1,
      unchanged_count: 1,
      regressed_count: 3,
    },
    record_deltas: [
      { sample_id: "nasa:B0018:77", dataset: "nasa", asset_id: "B0018", cycle_id: 77, baseline_absolute_error: 0.01, retest_absolute_error: 0.04, absolute_error_delta: 0.03, subset_mae_contribution_delta: 0.006, change: "regressed" },
      { sample_id: "nasa:B0018:76", dataset: "nasa", asset_id: "B0018", cycle_id: 76, baseline_absolute_error: 0.01, retest_absolute_error: 0.02, absolute_error_delta: 0.01, subset_mae_contribution_delta: 0.002, change: "regressed" },
    ],
    failure_localization: {
      datasets: ["nasa"],
      assets: ["B0018"],
      protocol_ref: "protocol@sha256:split",
      model_artifact_id: "model@sha256:model",
      failed_criteria: ["mae_max"],
      threshold_exceeding_records: [{ sample_id: "nasa:B0018:77" }],
    },
  });
  assert.equal(view.available, true);
  assert.equal(view.memberCount, 5);
  assert.equal(view.maeDelta, 0.02);
  assert.deepEqual(
    [view.improvedCount, view.unchangedCount, view.regressedCount],
    [1, 1, 3],
  );
  assert.equal(view.topRecords[0].cycleId, 77);
  assert.equal(view.localization.failedCriteria[0], "mae_max");
  assert.equal(view.localization.thresholdExceedingCount, 1);
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
  const recent = trainingJobRuntimeState(
    {
      status: "running",
      started_at: "2026-07-01T23:03:39-07:00",
      request: {
        datasets: ["a", "b", "c", "d", "e", "f", "g"],
        estimators: ["battery_soh_tabular_ensemble", "gbdt"],
        featuresets: ["battery_soh_operational"],
        protocols: ["loco"],
      },
    },
    new Date("2026-07-01T23:10:39-07:00"),
  );
  assert.equal(recent.tone, "active");
  assert.equal(recent.elapsedLabel, "7 分钟");
  assert.match(recent.hint, /7 数据集 x 2 算法 x 1 特征 x LOCO/);
  assert.match(recent.hint, /全量 LOCO/);

  const stale = trainingJobRuntimeState(
    { status: "running", started_at: "2026-07-01T19:00:00-07:00", request: { datasets: ["a"], estimators: ["ridge"], protocols: ["loco"] } },
    new Date("2026-07-01T23:10:00-07:00"),
  );
  assert.equal(stale.tone, "warn");
  assert.match(stale.hint, /超过 2 小时/);

  const done = trainingJobRuntimeState({ status: "succeeded", finished_at: "2026-07-01 21:31:03" });
  assert.equal(done.tone, "ok");
  assert.match(done.hint, /已完成/);
}

{
  const url = nativeDashboardUrl({
    base: "./battery-dashboard/index.html",
    dataset: "rwth_isea_public",
    model: "ridge voltage",
    source: "inference",
    featureset: "voltage",
  });
  assert.equal(
    url,
    "./battery-dashboard/index.html?mode=soh&embed=1&dataset=rwth_isea_public&source=inference&model=ridge+voltage&featureset=voltage",
  );
  assert.equal(nativeDashboardUrl({ dataset: "" }), "./battery-dashboard/index.html?mode=soh&embed=1");
  assert.equal(
    nativeDashboardUrl({ dataset: "xjtu", cell: "batch-1:cell-1", cycle: 438 }),
    "./battery-dashboard/index.html?mode=soh&embed=1&dataset=xjtu&cell=batch-1%3Acell-1&cycle=438",
  );
  assert.equal(nativeDashboardUrl({ cycle: 0 }), "./battery-dashboard/index.html?mode=soh&embed=1&cycle=0");
}

{
  const view = algorithmRecommendationView({
    overall: {
      recommended_default: "gbdt",
      why: ["LODO mean MAE lowest", "LOCO mean MAE lowest"],
      caveats: ["Ridge is fastest on LODO"],
    },
    scenarios: [
      {
        id: "lodo_transfer",
        title: "跨数据集迁移能力",
        protocol: "lodo",
        recommended_algorithm: "gbdt",
        records: 147,
        success_rate: 1,
        ranking: [
          { algorithm: "ridge", mean_mae: 0.116, mean_ms_per_1000: 0.9 },
          { algorithm: "gbdt", mean_mae: 0.089, mean_ms_per_1000: 5.4 },
          { algorithm: "battery_soh_tabular_ensemble", mean_mae: 0.093, mean_ms_per_1000: 39.1 },
        ],
      },
    ],
  });
  assert.equal(view.available, true);
  assert.equal(view.defaultAlgorithm, "gbdt");
  assert.equal(view.scenarioCount, 1);
  assert.equal(view.scenarioCards[0].accuracyWinner, "gbdt");
  assert.equal(view.scenarioCards[0].speedWinner, "ridge");
  assert.equal(view.scenarioCards[0].successLabel, "100.0%");
  assert.deepEqual(view.scenarioCards[0].ranking.map((row) => row.algorithm), [
    "gbdt",
    "battery_soh_tabular_ensemble",
    "ridge",
  ]);
}

{
  const view = algorithmRecommendationView({});
  assert.equal(view.available, false);
  assert.equal(view.defaultAlgorithm, "暂无");
  assert.deepEqual(view.scenarioCards, []);
}

{
  const summary = batteryDiagnosticSummary({
    manifest: {
      scenarios: [
        { id: "legacy_lab", kind: "lab", overall_pass: true },
      ],
    },
    overview: {
      scenarios: [
        { id: "lab_a", kind: "lab", overall_pass: true },
        { id: "field_a", kind: "field", overall_pass: false },
        { id: "detect_a", kind: "detection", overall_pass: true },
      ],
      transfers: [{ id: "transfer_a" }, { id: "transfer_b" }],
    },
  });

  assert.equal(summary.available, true);
  assert.equal(summary.scenarioCount, 2);
  assert.equal(summary.labScenarioCount, 1);
  assert.equal(summary.fieldScenarioCount, 1);
  assert.equal(summary.transferCount, 2);
  assert.equal(summary.passedCount, 1);
  assert.equal(summary.sourceLabel, "battery_SOH dashboard");
}

{
  const summary = batteryDiagnosticSummary({
    essDetection: {
      kind: "detection",
      summary: {
        worst: {
          module: "battery_01",
          period: "real_world_02-03",
          level: "L3",
          peak_dev_mv: 175,
        },
      },
    },
  });

  assert.equal(summary.available, false);
  assert.equal(summary.scenarioCount, 0);
  assert.equal(summary.sourceLabel, "未导入");
}

{
  const summary = batteryDiagnosticSummary({});
  assert.equal(summary.available, false);
  assert.equal(summary.scenarioCount, 0);
  assert.equal(summary.labScenarioCount, 0);
  assert.equal(summary.fieldScenarioCount, 0);
  assert.equal(summary.transferCount, 0);
  assert.equal(summary.passedCount, 0);
  assert.equal(summary.sourceLabel, "未导入");
}

{
  const detail = batteryVisualAssetDetail({
    profileManifest: {
      curve_artifacts: [
        { dataset: "xjtu", filename: "battery_dashboard_xjtu_curves.json", bytes: 12287881, copied: false },
      ],
    },
    algorithmDesign: {
      title: "SOH 算法设计说明书",
      task: "SOH 点估计 + 95% CI",
      model_family: "TabularSOHEnsemble",
      features: ["q_win_ah", "t_win_s"],
      feature_policy: { forbidden: ["DCIR", "cycle_fraction"] },
    },
    modelComparisons: {
      comparisons: [
        { id: "phase2", title: "固定结构复现", metrics: { mae: 0.019, rmse: 0.02, r2: 0.83 } },
      ],
    },
    featureWindows: {
      windows: [
        { dataset: "xjtu", feature_voltage_window: [3.7, 4.1], rows: 768, curves: 6 },
      ],
    },
  });
  assert.equal(detail.available, true);
  assert.equal(detail.algorithm.modelFamily, "TabularSOHEnsemble");
  assert.equal(detail.algorithm.featuresLabel, "q_win_ah / t_win_s");
  assert.equal(detail.comparisons[0].maeLabel, "0.019");
  assert.equal(detail.featureWindows[0].windowLabel, "3.7-4.1 V");
  assert.equal(detail.curves[0].sizeLabel, "11.7 MiB");
  assert.equal(batteryCurvePreviewView({
    curvePreviews: {
      datasets: [
        {
          dataset: "xjtu",
          kind: "charge_curve",
          total_curves: 2,
          source_file: "D:\\battery_SOH\\dashboard\\data\\xjtu_curves.json",
          preview_curves: [
            {
              id: "cell-1|1",
              point_count: 120,
              sampled_point_count: 80,
              x_key: "t_s",
              series: [
                { key: "voltage_v", label: "Voltage", min: 3, max: 4.1, points: [[0, 3], [1, 3.1]] },
              ],
            },
          ],
        },
        {
          dataset: "ess_detection",
          kind: "ess_detection",
          total_curves: 1,
          preview_curves: [
            {
              id: "battery_01|real_world",
              point_count: 10,
              sampled_point_count: 10,
              x_key: "sample",
              series: [
                { key: "L1", label: "L1", min: 0, max: 35, points: [[0, 0], [1, 35]] },
              ],
            },
          ],
        },
      ],
    },
  }, "ess_detection").selectedDataset, "xjtu");
  assert.equal(detail.curves[0].copiedLabel, "索引");
}

{
  const summary = batteryVisualAssetSummary({
    manifest: {
      diagnostic_artifacts: [
        "battery_dashboard_xjtu.json",
        "battery_dashboard_xjtu_curves.json",
        "battery_dashboard_ess_detection_curves.json",
      ],
      source_artifacts: {
        "battery_dashboard_xjtu_curves.json": "D:\\battery_SOH\\dashboard\\data\\xjtu_curves.json",
      },
    },
    algorithmDesign: {
      title: "SOH 算法设计",
      model_family: "TabularSOHEnsemble",
    },
    modelComparisons: {
      comparisons: [{ id: "phase2" }, { id: "phase3" }],
    },
    featureWindows: {
      windows: [{ dataset: "xjtu" }],
    },
  });
  assert.equal(summary.available, true);
  assert.equal(summary.algorithmDetailAvailable, true);
  assert.equal(summary.curveArtifactCount, 1);
  assert.equal(summary.comparisonCount, 2);
  assert.equal(summary.featureWindowCount, 1);
  assert.match(summary.statusLabel, /算法说明/);
  assert.match(summary.statusLabel, /曲线/);
}

{
  const summary = batteryVisualAssetSummary({
    profileManifest: {
      curve_artifacts: [
        { dataset: "xjtu", filename: "battery_dashboard_xjtu_curves.json", bytes: 100, copied: false },
        { dataset: "mit", filename: "battery_dashboard_mit_curves.json", bytes: 200, copied: false },
      ],
    },
  });
  assert.equal(summary.available, true);
  assert.equal(summary.curveArtifactCount, 2);
  assert.equal(summary.curveCopiedCount, 0);
  assert.equal(summary.curveTotalBytes, 300);
  assert.equal(summary.curveTotalLabel, "300 B");
  assert.deepEqual(summary.curveArtifacts, [
    "battery_dashboard_xjtu_curves.json",
    "battery_dashboard_mit_curves.json",
  ]);
}

{
  const summary = batteryDiagnosticSummary({
    manifest: {
      scenarios: [
        { id: "manifest_only", kind: "lab", overall_pass: true },
      ],
    },
    overview: {
      scenarios: [],
    },
  });
  assert.equal(summary.available, true);
  assert.equal(summary.scenarioCount, 1);
  assert.equal(summary.passedCount, 1);
}

{
  const rows = batteryScenarioRows([
    {
      mode: "xjtu",
      kind: "lab",
      title: "XJTU 全充电段",
      chemistry: "NCM",
      overall_pass: true,
      algorithms: [
        {
          metrics: {
            mae: 0.011440533564562254,
          },
        },
      ],
    },
    {
      id: "legacy",
      dataset: "nasa",
      overall_pass: false,
      metric: "MAE 0.2",
    },
  ]);

  assert.deepEqual(rows, [
    {
      name: "XJTU 全充电段",
      kind: "lab",
      source: "xjtu",
      verdict: "通过",
      metric: "0.0114",
    },
    {
      name: "legacy",
      kind: "lab",
      source: "nasa",
      verdict: "未通过",
      metric: "MAE 0.2",
    },
  ]);
}

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
  assert.equal(ACCEPTANCE_PRESETS.strict.thresholds.maeMax, 0.02);
  assert.equal(ACCEPTANCE_PRESETS.engineering.thresholds.maeMax, 0.08);

  const rows = [
    entry("battery_soh_tabular_ensemble", "battery_soh_operational", 0.077, { picp: 0.871, mono: 0.077 }),
    entry("gbdt", "battery_soh_operational", 0.074, { picp: 0.883, mono: 0.121 }),
  ];
  const strict = acceptanceDetailSummary({ rows, datasetRows: rows, preset: "strict" });
  assert.equal(strict.aggregate.label, "0/6");
  assert.equal(strict.dataset.label, "0/6");
  assert.equal(strict.failingRows.length, 2);
  assert.deepEqual(strict.rows[0].failed, ["generalization", "calibration", "physical_consistency"]);
  assert.match(strict.rows[0].reasons.join(" "), /MAE/);
  assert.match(strict.rows[0].reasons.join(" "), /PICP/);
  assert.match(strict.rows[0].reasons.join(" "), /单调/);

  const engineering = acceptanceDetailSummary({ rows, datasetRows: rows, preset: "engineering" });
  assert.equal(engineering.aggregate.label, "6/6");
  assert.equal(engineering.dataset.label, "6/6");
  assert.deepEqual(engineering.failingRows, []);

  const trainingKpi = acceptanceKpiSummary({
    mode: "training",
    rows: [
      { verdicts: { generalization: false, calibration: false, physical_consistency: false } },
      { verdicts: { generalization: false, calibration: false, physical_consistency: false } },
    ],
    datasetRows: [
      { verdicts: { generalization: true, calibration: true, physical_consistency: false } },
      { verdicts: { generalization: false, calibration: true, physical_consistency: false } },
    ],
  });
  assert.equal(trainingKpi.label, "严格验收");
  assert.equal(trainingKpi.value, "0/6");
  assert.match(trainingKpi.hint, /数据集级 3\/6/);

  const engineeringKpi = acceptanceKpiSummary({
    mode: "training",
    rows,
    datasetRows: rows,
    preset: "engineering",
  });
  assert.equal(engineeringKpi.label, "工程可用验收");
  assert.equal(engineeringKpi.value, "6/6");

  const inferenceKpi = acceptanceKpiSummary({
    mode: "inference",
    inference: {
      summary: { n_records: 8, n_succeeded: 8 },
      records: [],
    },
  });
  assert.equal(inferenceKpi.label, "推理成功");
  assert.equal(inferenceKpi.value, "8/8");
  assert.match(inferenceKpi.hint, /inference-only/);
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
  const options = catalogOptions({
    estimators: [{
      name: "stress_resid",
      pairing_mode: "method_package",
      method_family: "gray_box_degradation",
      compatible_featuresets: ["operating_stress"],
      preferred_featureset: "operating_stress",
    }],
    featuresets: [{
      name: "operating_stress",
      family: "mechanism_derived",
      mechanism_level: "stress_proxy",
      has_transform: true,
    }],
  });
  assert.equal(options.estimators[0].badge, "联合方法");
  assert.equal(options.estimators[0].pairingMode, "method_package");
  assert.equal(options.estimators[0].methodCategory, "joint_method");
  assert.equal(options.estimators[0].methodCategoryLabel, "联合方法");
  assert.equal(options.estimators[0].executionContract, "joint_method/v1");
  assert.deepEqual(options.estimators[0].compatibleFeaturesets, ["operating_stress"]);
  assert.equal(options.featuresets[0].badge, "机理");
  assert.equal(options.featuresets[0].mechanismLevel, "stress_proxy");
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
      { name: "ridge_full", source: "training", estimator: "ridge", featureset: "full", artifact_id: "model@sha256:abc", contract_version: "1.0.0" },
      { name: "external_mlp", source: "upload", estimator: "mlp", featureset: "ic_dva" },
      { name: "leaking_model", status: "invalidated", usable: false, estimator: "ridge", featureset: "full" },
    ],
  });
  assert.deepEqual(rows.map((row) => row.name), ["external_mlp", "ridge_full"]);
  assert.equal(rows[0].badge, "上传模型");
  assert.equal(rows[1].badge, "训练产物");
  assert.equal(rows[1].artifactId, "model@sha256:abc");
  assert.equal(rows[1].contractVersion, "1.0.0");
}

{
  const models = modelOptions({
    models: [
      { name: "safe_model", estimator: "ridge", featureset: "voltage" },
      { name: "blocked_model", estimator: "ridge", featureset: "full", status: "invalidated", usable: false },
    ],
  });
  const grouped = pipelineModelGroups({
    trainingJob: {
      id: "train-label-safety",
      saved_models: ["safe_model", "blocked_model"],
      model_artifacts: [
        { name: "safe_model", estimator: "ridge", featureset: "voltage" },
        { name: "blocked_model", estimator: "ridge", featureset: "full", status: "invalidated", usable: false },
      ],
    },
    models,
  });
  assert.deepEqual(models.map((item) => item.name), ["safe_model"]);
  assert.deepEqual(grouped.runModelNames, ["safe_model"]);
  assert.deepEqual(grouped.recommendedModelNames, ["safe_model"]);
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
  const compatibility = inferenceCompatibilitySummary({
    models: [{
      name: "battery_operational",
      featureset: "battery_soh_operational",
      featureColumns: [
        "battery_soh_cycle_norm",
        "battery_soh_voltage_sag",
        "battery_soh_resistance_growth",
      ],
    }],
    datasets: [{
      name: "hust",
      columns: ["cycle_index", "voltage_v", "current_a", "capacity_ah"],
    }],
    featureSets: [{
      name: "battery_soh_operational",
      columns: [
        "battery_soh_cycle_norm",
        "battery_soh_voltage_sag",
        "battery_soh_resistance_growth",
      ],
    }],
  });
  assert.equal(compatibility.status, "ok");
  assert.equal(compatibility.okCount, 1);
  assert.equal(compatibility.warnCount, 0);
  assert.equal(compatibility.unknownCount, 0);
  assert.deepEqual(compatibility.rows.map((row) => [row.model, row.dataset, row.status, row.missingColumns]), [
    ["battery_operational", "hust", "ok", []],
  ]);
}

{
  const models = [
    { name: "ridge_voltage", featureColumns: ["voltage_mean"], trainedOn: ["calce_cs2"], sourceChemistries: ["LCO"] },
    { name: "ridge_full", featureColumns: ["voltage_mean", "ambient_temp_c"], trainedOn: ["calce_cs2"], sourceChemistries: ["LCO"] },
  ];
  const datasets = [
    { name: "nasa", chemistry: "LCO", columns: ["dataset", "soh", "voltage_mean", "ambient_temp_c"] },
    { name: "calce_cs2", chemistry: "LCO", columns: ["dataset", "soh", "voltage_mean"] },
    { name: "uconn", chemistry: "LFP" },
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
    evaluation_policy: "held_out_same_chemistry",
    selected_models: ["ridge_voltage", "ridge_full"],
    submitted_models: ["ridge_voltage", "ridge_full"],
    skipped_models: [],
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
  assert.equal(includeRisk.payload.evaluation_policy, "research_any");
  assert.deepEqual(includeRisk.payload.selected_models, ["ridge_voltage", "ridge_full"]);
  assert.deepEqual(includeRisk.payload.submitted_models, ["ridge_voltage", "ridge_full"]);
  assert.deepEqual(includeRisk.payload.skipped_models, []);
  assert.deepEqual(includeRisk.payload.selected_datasets, ["nasa", "calce_cs2", "uconn"]);
  assert.deepEqual(includeRisk.payload.submitted_datasets, ["nasa", "calce_cs2", "uconn"]);
  assert.deepEqual(includeRisk.payload.skipped_datasets, []);
  assert.deepEqual(includeRisk.skippedDatasets, []);
}

{
  const plan = inferenceLaunchPlan({
    modelNames: ["good", "needs_temp"],
    datasetNames: ["calce"],
    models: [
      { name: "good", featureColumns: ["voltage_mean"], trainedOn: ["source"], sourceChemistries: ["LCO"] },
      { name: "needs_temp", featureColumns: ["voltage_mean", "ambient_temp_c"], trainedOn: ["source"], sourceChemistries: ["LCO"] },
    ],
    datasets: [{ name: "calce", chemistry: "LCO", columns: ["dataset", "soh", "voltage_mean"] }],
    strategy: "compatible_only",
  });
  assert.deepEqual(plan.payload.models, ["good"]);
  assert.deepEqual(plan.payload.datasets, ["calce"]);
  assert.deepEqual(plan.payload.selected_models, ["good", "needs_temp"]);
  assert.deepEqual(plan.payload.submitted_models, ["good"]);
  assert.deepEqual(plan.payload.skipped_models, ["needs_temp"]);
  assert.deepEqual(plan.skippedModels, ["needs_temp"]);
  assert.deepEqual(plan.skippedDatasets, []);
  assert.equal(plan.skippedCount, 1);
}

{
  const plan = inferenceLaunchPlan({
    modelNames: ["battery_operational"],
    datasetNames: ["hust"],
    models: [{
      name: "battery_operational",
      featureset: "battery_soh_operational",
      featureColumns: [
        "battery_soh_cycle_norm",
        "battery_soh_voltage_sag",
        "battery_soh_resistance_growth",
      ],
      trainedOn: ["calce_cs2"],
      sourceChemistries: ["LCO"],
    }],
    datasets: [{ name: "hust", chemistry: "LCO", columns: ["cycle_index", "voltage_v", "current_a", "capacity_ah"] }],
    featureSets: [{
      name: "battery_soh_operational",
      columns: [
        "battery_soh_cycle_norm",
        "battery_soh_voltage_sag",
        "battery_soh_resistance_growth",
      ],
    }],
    strategy: "compatible_only",
  });
  assert.deepEqual(plan.payload.datasets, ["hust"]);
  assert.deepEqual(plan.skippedDatasets, []);
  assert.equal(plan.skippedCount, 0);
}

{
  const heldOut = inferenceEvaluationEligibility(
    { protocol: "loco", heldOutKey: "nasa/B0018", trainedOn: ["nasa"], sourceChemistries: ["LCO"] },
    { name: "nasa", chemistry: "LCO" },
  );
  assert.equal(heldOut.eligible, true);
  assert.equal(heldOut.scope, "held_out_test");
  const transfer = inferenceEvaluationEligibility(
    { trainedOn: ["nasa"], sourceChemistries: ["LCO"] },
    { name: "calce_cs2", chemistry: "LCO" },
  );
  assert.equal(transfer.scope, "same_chemistry_transfer");
  const mismatch = inferenceEvaluationEligibility(
    { trainedOn: ["nasa"], sourceChemistries: ["LCO"] },
    { name: "hust", chemistry: "LFP" },
  );
  assert.equal(mismatch.eligible, false);
  assert.equal(mismatch.scope, "cross_chemistry");
}

{
  const plan = inferenceLaunchPlan({
    modelNames: ["heldout_nasa", "heldout_calce"],
    datasetNames: ["nasa", "calce_cs2"],
    models: [
      { name: "heldout_nasa", protocol: "lodo", heldOutKey: "nasa", trainedOn: ["calce_cs2"], sourceChemistries: ["LCO"], featureColumns: ["voltage_mean"] },
      { name: "heldout_calce", protocol: "lodo", heldOutKey: "calce_cs2", trainedOn: ["nasa"], sourceChemistries: ["LCO"], featureColumns: ["voltage_mean"] },
    ],
    datasets: [
      { name: "nasa", chemistry: "LCO", columns: ["voltage_mean"] },
      { name: "calce_cs2", chemistry: "LCO", columns: ["voltage_mean"] },
    ],
    strategy: "compatible_only",
  });
  assert.deepEqual(plan.payload.models, ["heldout_nasa", "heldout_calce"]);
  assert.deepEqual(plan.payload.datasets, ["nasa", "calce_cs2"]);
  assert.equal(plan.skippedCount, 0);
}

{
  const staticSummary = inferenceAuditSummary({
    inferenceConfig: {
      job_id: "job-static",
      launch_strategy: "compatible_only",
      selected_models: ["ridge_voltage", "ridge_full"],
      submitted_models: ["ridge_voltage"],
      skipped_models: ["ridge_full"],
      selected_datasets: ["nasa", "rwth_isea_public"],
      submitted_datasets: ["nasa"],
      skipped_datasets: ["rwth_isea_public"],
      run_dir: "runs/20260630T120000",
    },
  });
  assert.equal(staticSummary.hasAudit, true);
  assert.equal(staticSummary.strategyLabel, "仅运行兼容组合");
  assert.equal(staticSummary.selectedModelCount, 2);
  assert.equal(staticSummary.submittedModelCount, 1);
  assert.equal(staticSummary.skippedModelCount, 1);
  assert.equal(staticSummary.skippedModelLabel, "ridge_full");
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
          models: ["ridge_voltage"],
          selected_models: ["ridge_voltage"],
          submitted_models: ["ridge_voltage"],
          skipped_models: [],
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
  assert.equal(liveSummary.selectedModelCount, 1);
  assert.equal(liveSummary.submittedModelCount, 1);
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
  assert.match(ready.message, /2 个候选 \/ 3 个数据集/);

  const running = inferenceActionState({
    backendConnected: true,
    selectedModelCount: 1,
    selectedDatasetCount: 1,
    currentJob: { kind: "inference", id: "job-1", status: "running" },
  });
  assert.equal(running.disabled, true);
  assert.match(running.message, /job-1/);
  assert.match(running.message, /正在执行/);

  const changed = inferenceActionState({
    backendConnected: true,
    selectedModelCount: 2,
    selectedDatasetCount: 1,
    currentJob: { kind: "inference", id: "job-old", status: "succeeded" },
    currentJobMatchesSelection: false,
  });
  assert.equal(changed.disabled, false);
  assert.equal(changed.tone, "warn");
  assert.match(changed.message, /选择已变化/);
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
  const assets = modelAssetRows({
    models: [
      { name: "legacy_alpha", source: "training", contract_version: "legacy" },
      { name: "versioned_zeta", source: "training", contract_version: "1.0.0", artifact_id: "model@sha256:zeta" },
    ],
  });
  assert.deepEqual(assets.map((row) => row.name), ["versioned_zeta", "legacy_alpha"]);
}

{
  const page = modelAssetPage(
    Array.from({ length: 5 }, (_, index) => ({ name: `model_${index}` })),
    { limit: 3 },
  );
  assert.deepEqual(page.rows.map((row) => row.name), ["model_0", "model_1", "model_2"]);
  assert.equal(page.totalCount, 5);
  assert.equal(page.shownCount, 3);
  assert.equal(page.hiddenCount, 2);
  assert.equal(page.limited, true);
  assert.match(page.note, /显示 3 \/ 5/);
}
