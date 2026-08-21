import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sohbench.evaluation_cases import (
    EvaluationCaseConflictError,
    EvaluationCaseStore,
    EvaluationCaseValidationError,
    normalize_evidence_coverage,
    normalize_evidence_selection,
    validate_case_document,
)
from sohbench.case_evidence import derive_verdict


def _case_payload():
    return {
        "profile_name": "battery-soh-diagnostics",
        "idempotency_key": "profile:battery-soh-diagnostics|run:run-01",
        "title": "电池 SOH 诊断案例",
        "source_class": "measured",
        "execution": {
            "state": "completed",
            "training_job_id": "train-01",
            "run_id": "run-01",
        },
        "candidates": [
            {
                "candidate_id": "legacy:ridge:voltage",
                "model": "ridge",
                "featureset": "voltage",
                "role": "candidate",
                "run_id": "run-01",
            }
        ],
    }


def _server_deriver(case, payload):
    if {"coverage", "verdict"}.intersection(payload):
        raise EvaluationCaseValidationError("server-derived fields are read-only")
    selection = normalize_evidence_selection(
        payload.get("selection") or {}, current=case["evidence_selection"]
    )
    if selection.get("candidate_id") not in {item["candidate_id"] for item in case["candidates"]}:
        raise EvaluationCaseValidationError("unknown candidate")
    coverage = normalize_evidence_coverage(
        {
            "items": [
                {"kind": "prediction_record", "required": True, "available": bool(selection.get("sample_id"))},
                {"kind": "raw_curve", "required": True, "available": selection.get("evidence_tab") == "raw"},
            ]
        }
    )
    verdict = derive_verdict(
        {"mae": 0.0027, "picp": 0.3333, "monotonicity": 0.0, "value_range": 0.0},
        payload.get("criterion_set_id") or "strict",
        payload.get("criterion_thresholds"),
    )
    return {"selection": selection, "coverage": coverage, "verdict": verdict}


def test_resolve_case_is_idempotent_and_persists_across_store_instances(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases")

    first, created = store.resolve(_case_payload())
    second, created_again = store.resolve(_case_payload())

    assert created is True
    assert created_again is False
    assert first["id"] == second["id"]
    assert first["schema_version"] == "1.3.0"
    assert first["context_binding"]["binding_id"].startswith("ctx@sha256:")
    assert first["idempotency_key"] == _case_payload()["idempotency_key"]
    assert first["execution"]["state"] == "completed"
    assert first["review"]["state"] == "not_started"
    assert first["evidence_coverage"]["state"] == "empty"

    reloaded = EvaluationCaseStore(tmp_path / "cases").get(first["id"])
    assert reloaded == second
    validate_case_document(reloaded)
    assert json.loads((tmp_path / "cases" / f"{first['id']}.json").read_text(encoding="utf-8")) == reloaded
    assert list((tmp_path / "cases").glob("*.tmp")) == []


def test_update_evidence_records_traceable_selection_coverage_and_verdict(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases", evidence_deriver=_server_deriver)
    case = store.create(_case_payload())

    updated = store.update_evidence(
        case["id"],
        {
            "expected_version": case["version"],
            "selection": {
                "comparison_id": "cmp-01",
                "candidate_id": "legacy:ridge:voltage",
                "dataset": "xjtu",
                "dataset_snapshot_id": "legacy-profile:battery-soh-diagnostics",
                "asset_id": "batch-1:cell-1",
                "sample_id": "batch-1:cell-1|236",
                "cycle_id": 236,
                "evidence_tab": "raw",
                "compare_cycle_ids": [120, 236],
                "curve_ref": {
                    "record_id": "batch-1:cell-1|236",
                    "source_uri": "battery-dashboard/data/xjtu_curves.json#batch-1:cell-1|236",
                    "signals": ["voltage", "current", "temperature", "ic"],
                    "availability": "available",
                },
            },
            "criterion_set_id": "strict",
        },
    )

    assert updated["version"] == 2
    assert updated["evidence_selection"]["asset_id"] == "batch-1:cell-1"
    assert updated["evidence_selection"]["cycle_id"] == 236
    assert updated["evidence_selection"]["curve_ref"]["availability"] == "available"
    assert updated["evidence_coverage"] == {
        "state": "complete",
        "available_count": 2,
        "required_count": 2,
        "items": updated["evidence_coverage"]["items"],
    }
    assert updated["verdict"]["result"] == "fail"
    assert updated["review"]["state"] == "not_started"

    workspace = store.evidence_workspace(updated["id"], {"cycle_id": "438", "evidence_tab": "icdv"})
    assert workspace["context"]["cycle_id"] == "438"
    assert workspace["context"]["evidence_tab"] == "icdv"
    assert workspace["status_axes"] == {
        "execution": "completed",
        "evidence": "complete",
        "verdict": "fail",
        "review": "not_started",
    }


def test_threshold_update_is_versioned_rejudged_and_audited(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases", evidence_deriver=_server_deriver)
    case = store.create(_case_payload())
    pinned = store.update_evidence(
        case["id"],
        {
            "expected_version": case["version"],
            "selection": {
                "candidate_id": "legacy:ridge:voltage",
                "dataset": "xjtu",
                "sample_id": "batch-1:cell-1|236",
                "cycle_id": 236,
                "evidence_tab": "raw",
            },
            "criterion_set_id": "strict",
        },
    )
    assert pinned["verdict"]["result"] == "fail"

    customized = store.update_evidence(
        case["id"],
        {
            "expected_version": pinned["version"],
            "criterion_set_id": "strict",
            "criterion_thresholds": {"picp_min": 0.30},
            "reason": "数据域校准阈值复核",
        },
    )

    assert customized["verdict"]["result"] == "pass"
    assert customized["verdict"]["criterion_set_version"].startswith("strict@custom-")
    assert customized["verdict"]["thresholds"]["picp_min"] == 0.30
    assert store.list_events(case["id"])[-1]["event_type"] == "criterion.updated"


def test_update_identity_keeps_one_case_across_training_and_inference(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases")
    case = store.create({
        "profile_name": "pipeline-profile",
        "idempotency_key": "pipeline:persistent-case",
        "source_class": "measured",
    })
    assert case["lifecycle_state"] == "draft"

    trained = store.update_identity(
        case["id"],
        {
            "expected_version": case["version"],
            "execution": {
                "state": "completed",
                "training_job_id": "train-01",
                "run_id": "train-01",
            },
            "candidates": [{
                "candidate_id": "artifact:ridge-voltage",
                "model": "ridge",
                "featureset": "voltage",
                "role": "champion",
                "model_artifact_id": "model@sha256:abc123",
                "run_id": "train-01",
            }],
            "reason": "training completed",
        },
    )
    inferred = store.update_identity(
        case["id"],
        {
            "expected_version": trained["version"],
            "execution": {
                "state": "completed",
                "inference_job_id": "infer-01",
                "run_id": "infer-01",
            },
            "reason": "inference completed",
        },
    )
    assert inferred["id"] == case["id"]
    assert inferred["idempotency_key"] == "pipeline:persistent-case"
    assert inferred["execution"] == {
        "state": "completed",
        "training_job_id": "train-01",
        "inference_job_id": "infer-01",
        "run_id": "infer-01",
        "failure_reason": "",
    }
    assert inferred["candidates"][0]["model_artifact_id"] == "model@sha256:abc123"
    assert inferred["lifecycle_state"] == "active"
    assert [event["event_type"] for event in store.list_events(case["id"])] == [
        "case.created",
        "case.identity_updated",
        "case.identity_updated",
    ]


def test_completed_combinations_are_curated_separately_and_inference_can_be_activated(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases")
    case = store.create(_case_payload())
    training = {
        "saved_combination_id": "combo:training-01",
        "source_stage": "training",
        "validation_kind": "training_internal",
        "label": "ridge × voltage · 训练内评估",
        "candidate": {
            "candidate_id": "artifact:model-training",
            "model": "ridge",
            "featureset": "voltage",
            "role": "champion",
            "model_artifact_id": "model@sha256:training",
            "run_id": "train-01",
        },
        "training_job_id": "train-01",
        "inference_job_id": "",
        "run_id": "train-01",
        "training_datasets": ["xjtu"],
        "target_dataset": "",
        "dataset_snapshot_id": "dataset@sha256:xjtu",
        "protocol": "loco",
        "criterion_set_version": "engineering",
        "metrics": {"mae": 0.01, "picp": 0.95},
        "verdict_result": "pass",
    }
    inference = {
        "saved_combination_id": "combo:inference-01",
        "source_stage": "inference",
        "validation_kind": "cross_dataset",
        "label": "ridge × voltage · nasa",
        "candidate": {
            "candidate_id": "artifact:model-inference",
            "model": "ridge",
            "featureset": "voltage",
            "role": "candidate",
            "model_artifact_id": "model@sha256:inference",
            "run_id": "train-01",
        },
        "training_job_id": "train-01",
        "inference_job_id": "infer-01",
        "run_id": "infer-01",
        "training_datasets": ["xjtu"],
        "target_dataset": "nasa",
        "dataset_snapshot_id": "dataset@sha256:nasa",
        "protocol": "loco",
        "criterion_set_version": "engineering",
        "metrics": {"mae": 0.03, "monotonicity": 0.1},
        "verdict_result": "pass",
    }

    curated = store.update_saved_combinations(
        case["id"],
        {
            "expected_version": case["version"],
            "action": "save",
            "items": [training, inference],
        },
    )
    assert len(curated["saved_combinations"]) == 2
    assert curated["candidates"] == case["candidates"]
    assert store.list_events(case["id"])[-1]["event_type"] == "combination.saved"

    activated = store.activate_saved_combination(
        case["id"],
        {
            "expected_version": curated["version"],
            "saved_combination_id": "combo:inference-01",
        },
    )
    assert activated["active_saved_combination_id"] == "combo:inference-01"
    assert activated["execution"]["inference_job_id"] == "infer-01"
    assert activated["evidence_selection"]["dataset"] == "nasa"
    assert activated["candidates"][0]["model_artifact_id"] == "model@sha256:inference"

    removed = store.update_saved_combinations(
        case["id"],
        {
            "expected_version": activated["version"],
            "action": "remove",
            "saved_combination_ids": ["combo:training-01"],
        },
    )
    assert [item["saved_combination_id"] for item in removed["saved_combinations"]] == [
        "combo:inference-01"
    ]

def test_update_identity_can_replace_pipeline_candidates(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases")
    case = store.create(_case_payload())
    replacement = {
        "candidate_id": "artifact:gpr-capacity",
        "model": "gpr",
        "featureset": "capacity_time",
        "role": "champion",
        "model_artifact_id": "model@sha256:gpr-capacity",
        "run_id": "train-gpr",
    }

    replaced = store.update_identity(
        case["id"],
        {
            "expected_version": case["version"],
            "candidates": [replacement],
            "replace_candidates": True,
        },
    )
    assert replaced["candidates"] == [replacement]

    cleared = store.update_identity(
        case["id"],
        {
            "expected_version": replaced["version"],
            "candidates": [],
            "replace_candidates": True,
        },
    )
    assert cleared["candidates"] == []


def test_case_updates_use_optimistic_version_and_reject_unsafe_ids(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases")
    case = store.create(_case_payload())
    store.update_evidence(case["id"], {"expected_version": 1, "selection": {"dataset": "xjtu"}})

    with pytest.raises(EvaluationCaseConflictError, match="version conflict"):
        store.update_evidence(case["id"], {"expected_version": 1, "selection": {"dataset": "mit"}})

    with pytest.raises(EvaluationCaseConflictError, match="expected None"):
        store.update_evidence(case["id"], {"selection": {"dataset": "mit"}})

    with pytest.raises(EvaluationCaseValidationError, match="invalid case id"):
        store.get("../../outside")

    with pytest.raises(EvaluationCaseValidationError, match="at most 3"):
        store.update_evidence(
            case["id"],
            {"expected_version": 2, "selection": {"compare_cycle_ids": [1, 2, 3, 4]}},
        )


def test_selection_must_reference_a_registered_candidate(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases", evidence_deriver=_server_deriver)
    case = store.create(_case_payload())

    with pytest.raises(EvaluationCaseValidationError, match="unknown candidate"):
        store.update_evidence(
            case["id"],
            {"expected_version": 1, "selection": {"candidate_id": "legacy:missing:full"}},
        )


def test_client_cannot_submit_coverage_or_verdict(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases", evidence_deriver=_server_deriver)
    case = store.create(_case_payload())

    with pytest.raises(EvaluationCaseValidationError, match="read-only"):
        store.update_evidence(
            case["id"],
            {"expected_version": 1, "coverage": {"items": []}, "verdict": {"result": "pass"}},
        )


def test_same_profile_supports_multiple_cases_by_idempotency_key(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases")
    first, _ = store.resolve(_case_payload())
    second_payload = {**_case_payload(), "idempotency_key": "profile:battery-soh-diagnostics|run:run-02"}
    second, created = store.resolve(second_payload)

    assert created is True
    assert first["id"] != second["id"]
    assert len(store.list(profile_name="battery-soh-diagnostics")) == 2


def test_sqlite_compare_and_swap_rejects_stale_writer_across_store_instances(tmp_path):
    first_store = EvaluationCaseStore(tmp_path / "cases")
    case = first_store.create(_case_payload())
    second_store = EvaluationCaseStore(tmp_path / "cases")

    first_store.update_evidence(
        case["id"],
        {"expected_version": case["version"], "selection": {"dataset": "xjtu"}},
    )

    with pytest.raises(EvaluationCaseConflictError, match="version conflict"):
        second_store.update_evidence(
            case["id"],
            {"expected_version": case["version"], "selection": {"dataset": "calce_cs2"}},
        )


def test_concurrent_resolve_returns_one_case_for_one_idempotency_key(tmp_path):
    stores = [EvaluationCaseStore(tmp_path / "cases") for _ in range(4)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda store: store.resolve(_case_payload()), stores))

    assert len({case["id"] for case, _ in results}) == 1
    assert sum(created for _, created in results) == 1


def test_noop_update_keeps_version_and_events_are_append_only(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases", evidence_deriver=_server_deriver)
    case = store.create(_case_payload())
    payload = {
        "expected_version": 1,
        "selection": {"candidate_id": "legacy:ridge:voltage", "dataset": "xjtu"},
    }
    changed = store.update_evidence(case["id"], payload)
    time.sleep(1.05)
    unchanged = store.update_evidence(case["id"], {**payload, "expected_version": changed["version"]})

    assert changed["version"] == 2
    assert unchanged["version"] == 2
    assert [event["event_type"] for event in store.list_events(case["id"])] == [
        "case.created", "evidence.pinned"
    ]


def test_completed_case_is_immutable_and_transition_is_recorded(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases", evidence_deriver=_server_deriver)
    case = store.create(_case_payload())
    completed = store.transition_lifecycle(
        case["id"],
        {"expected_version": 1, "lifecycle_state": "completed", "actor": "reviewer"},
    )

    with pytest.raises(EvaluationCaseConflictError, match="immutable"):
        store.update_evidence(
            case["id"],
            {"expected_version": completed["version"], "selection": {"dataset": "xjtu"}},
        )
    assert store.list_events(case["id"])[-1]["event_type"] == "case.completed"


def test_archived_cases_are_retained_but_hidden_from_default_lists(tmp_path):
    store = EvaluationCaseStore(tmp_path / "cases")
    draft = store.create({
        "profile_name": "draft-profile",
        "idempotency_key": "draft:archive-policy",
        "source_class": "measured",
    })

    archived = store.transition_lifecycle(
        draft["id"],
        {
            "expected_version": draft["version"],
            "lifecycle_state": "archived",
            "actor": "test",
            "reason": "discard incomplete draft",
        },
    )

    assert archived["lifecycle_state"] == "archived"
    assert store.list(profile_name="draft-profile") == []
    assert [item["id"] for item in store.list(
        profile_name="draft-profile", include_archived=True
    )] == [draft["id"]]
    assert [item["id"] for item in store.list(
        profile_name="draft-profile", lifecycle_state="archived"
    )] == [draft["id"]]
    assert store.get(draft["id"])["lifecycle_state"] == "archived"
    assert store.list_events(draft["id"])[-1]["event_type"] == "case.archived"


def test_legacy_schema_migrates_and_invalid_schema_is_reported(tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    legacy = EvaluationCaseStore(tmp_path / "seed_root" / "cases").create(_case_payload())
    legacy["schema_version"] = "1.0.0"
    legacy.pop("idempotency_key")
    (cases / f"{legacy['id']}.json").write_text(json.dumps(legacy), encoding="utf-8")
    broken = {**legacy, "id": "EV-BROKEN", "schema_version": "9.9.9"}
    (cases / "EV-BROKEN.json").write_text(json.dumps(broken), encoding="utf-8")

    store = EvaluationCaseStore(cases)

    migrated = store.get(legacy["id"])
    assert migrated["schema_version"] == "1.3.0"
    assert migrated["context_binding"]["generation"] >= 1
    assert migrated["idempotency_key"] == f"legacy:{legacy['id']}"
    assert migrated["saved_combinations"] == []
    assert migrated["active_saved_combination_id"] == ""
    assert store.migration_errors()[0]["source"].endswith("EV-BROKEN.json")


def test_legacy_schema_migration_clears_dangling_evidence_candidate(tmp_path):
    seed = EvaluationCaseStore(tmp_path / "seed_root" / "cases").create(_case_payload())
    seed["schema_version"] = "1.2.0"
    seed.pop("context_binding")
    seed["evidence_selection"] = normalize_evidence_selection(
        {
            "candidate_id": "legacy:removed:feature",
            "dataset": "demo",
            "evidence_tab": "overview",
        }
    )
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / f"{seed['id']}.json").write_text(json.dumps(seed), encoding="utf-8")

    migrated = EvaluationCaseStore(cases).get(seed["id"])

    assert migrated["schema_version"] == "1.3.0"
    assert migrated["evidence_selection"]["candidate_id"] == ""
    assert migrated["evidence_coverage"]["state"] == "empty"
    assert migrated["verdict"]["result"] == "pending"
    assert migrated["review"]["state"] == "not_started"
    assert migrated["context_binding"]["candidate_id"] == ""


def test_atomic_case_write_retries_transient_windows_sharing_error(tmp_path, monkeypatch):
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient sharing violation")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    case = EvaluationCaseStore(tmp_path / "cases").create(_case_payload())

    assert attempts >= 3
    assert (tmp_path / "cases" / f"{case['id']}.json").exists()


def test_production_store_requires_current_semantic_context_binding(tmp_path):
    store = EvaluationCaseStore(
        tmp_path / "cases",
        evidence_deriver=_server_deriver,
        require_context_binding=True,
    )
    case = store.create(_case_payload())

    with pytest.raises(EvaluationCaseConflictError, match="required"):
        store.update_evidence(
            case["id"],
            {
                "expected_version": case["version"],
                "selection": {"candidate_id": "legacy:ridge:voltage", "dataset": "xjtu"},
            },
        )

    pinned = store.update_evidence(
        case["id"],
        {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
            "selection": {"candidate_id": "legacy:ridge:voltage", "dataset": "xjtu"},
        },
    )
    assert pinned["context_binding"]["binding_id"] != case["context_binding"]["binding_id"]

    with pytest.raises(EvaluationCaseConflictError, match="case context conflict"):
        store.update_evidence(
            case["id"],
            {
                "expected_version": pinned["version"],
                "expected_context_binding_id": case["context_binding"]["binding_id"],
                "selection": {"dataset": "nasa"},
            },
        )
    with pytest.raises(EvaluationCaseConflictError, match="case context conflict"):
        store.assert_context_binding(case["id"], case["context_binding"]["binding_id"])


def test_new_training_generation_atomically_clears_downstream_context(tmp_path):
    store = EvaluationCaseStore(
        tmp_path / "cases",
        evidence_deriver=_server_deriver,
        require_context_binding=True,
    )
    case = store.create(_case_payload())
    pinned = store.update_evidence(
        case["id"],
        {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
            "selection": {
                "candidate_id": "legacy:ridge:voltage",
                "dataset": "xjtu",
                "sample_id": "xjtu:b2c9:438",
                "cycle_id": 438,
                "evidence_tab": "raw",
            },
            "criterion_set_id": "strict",
        },
    )

    retrained = store.update_identity(
        case["id"],
        {
            "expected_version": pinned["version"],
            "expected_context_binding_id": pinned["context_binding"]["binding_id"],
            "execution": {
                "state": "completed",
                "training_job_id": "train-02",
                "run_id": "train-02",
            },
            "candidates": [{
                "candidate_id": "artifact:gbdt-full",
                "model": "gbdt",
                "featureset": "full",
                "role": "champion",
                "model_artifact_id": "model@sha256:train02",
                "run_id": "train-02",
            }],
            "replace_candidates": True,
        },
    )

    assert retrained["execution"]["training_job_id"] == "train-02"
    assert retrained["execution"]["inference_job_id"] == ""
    assert retrained["active_saved_combination_id"] == ""
    assert retrained["evidence_selection"]["candidate_id"] == ""
    assert retrained["evidence_selection"]["dataset"] == ""
    assert retrained["evidence_coverage"]["state"] == "empty"
    assert retrained["verdict"]["result"] == "pending"
    assert retrained["review"]["state"] == "not_started"
    assert retrained["context_binding"]["stage"] == "training"
    assert retrained["context_binding"]["generation"] == pinned["context_binding"]["generation"] + 1


def test_queued_training_generation_does_not_retain_old_candidate_identity(tmp_path):
    store = EvaluationCaseStore(
        tmp_path / "cases",
        evidence_deriver=_server_deriver,
        require_context_binding=True,
    )
    case = store.create(_case_payload())

    queued = store.update_identity(
        case["id"],
        {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
            "execution": {
                "state": "queued",
                "training_job_id": "train-new",
                "run_id": "train-new",
            },
        },
    )

    assert queued["candidates"] == []
    assert queued["context_binding"]["candidate_id"] == ""
    assert queued["context_binding"]["model_artifact_id"] == ""
