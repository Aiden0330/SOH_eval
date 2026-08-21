import json

import pytest

from sohbench.case_evidence import EvidenceDerivationService
from sohbench.evaluation_cases import EvaluationCaseStore, EvaluationCaseValidationError
from sohbench.review_workflow import ReviewWorkflowError, ReviewWorkflowStore


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_case(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    _write_json(
        frontend_data / "run_config.json",
        {"datasets": ["xjtu"], "protocols": ["loco"], "seeds": [0]},
    )
    _write_json(
        frontend_data / "leaderboard.json",
        [
            {
                "estimator": "ridge",
                "featureset": "voltage",
                "metrics": {
                    "mae": {"mean": 0.01},
                    "picp": {"mean": 0.95},
                    "monotonicity_violation_rate": {"mean": 0.0},
                    "value_range_violation": {"mean": 0.0},
                },
            }
        ],
    )
    profile = frontend_data / "profiles" / "battery-profile"
    _write_json(profile / "manifest.json", {})
    _write_json(
        profile / "battery_dashboard_xjtu.json",
        {
            "config_brief": {"model": "ridge voltage"},
            "metrics": {"mae": 0.01, "coverage": 0.95},
            "points": [
                {
                    "cell_id": "b2c9",
                    "cycle_index": cycle,
                    "true_soh": 0.81 - (cycle - 438) * 0.001,
                    "pred_soh": 0.809 - (cycle - 438) * 0.001,
                }
                for cycle in range(436, 441)
            ],
        },
    )
    _write_json(
        frontend_data.parent / "battery-dashboard" / "data" / "xjtu_curves.json",
        {"b2c9|438": {"voltage_v": [3.4, 3.5], "current_a": [1.0, 1.0]}},
    )
    cases = EvaluationCaseStore(
        tmp_path / "work" / "cases",
        evidence_deriver=EvidenceDerivationService(frontend_data).derive,
    )
    case = cases.create(
        {
            "profile_name": "battery-profile",
            "idempotency_key": "phase2:workflow",
            "source_class": "measured",
            "execution": {"state": "completed", "run_id": "run-original"},
            "candidates": [
                {
                    "candidate_id": "legacy:ridge:voltage",
                    "model": "ridge",
                    "featureset": "voltage",
                    "model_artifact_id": "model@sha256:abc123",
                }
            ],
        }
    )
    case = cases.update_evidence(
        case["id"],
        {
            "expected_version": case["version"],
            "criterion_set_id": "strict",
            "selection": {
                "candidate_id": "legacy:ridge:voltage",
                "dataset": "xjtu",
                "asset_id": "b2c9",
                "cycle_id": 438,
                "evidence_tab": "raw",
            },
        },
    )
    workflow = ReviewWorkflowStore(
        tmp_path / "work" / "evaluation_cases.sqlite3",
        tmp_path / "work" / "artifacts",
        get_case=cases.get,
        list_case_events=cases.list_events,
        frontend_data_dir=frontend_data,
    )
    return cases, workflow, case


def test_batch_regression_members_expand_only_the_pinned_asset_and_cycle_window(tmp_path):
    _, workflow, case = _build_case(tmp_path)
    issue = workflow.create_issue(
        {
            "case_id": case["id"],
            "expected_case_version": case["version"],
            "title": "batch regression",
        }
    )
    regression_set = workflow.create_regression_set(
        {
            "issue_id": issue["id"],
            "expected_case_version": case["version"],
        }
    )

    batch = workflow.add_current_evidence_member(
        regression_set["id"],
        {"expected_case_version": case["version"], "neighbor_radius": 2},
    )

    assert batch["neighbor_radius"] == 2
    assert batch["requested_count"] == 5
    assert batch["added_count"] == 5
    assert [member["cycle_id"] for member in batch["members"]] == [436, 437, 438, 439, 440]
    assert {member["dataset"] for member in batch["members"]} == {"xjtu"}
    assert {member["asset_id"] for member in batch["members"]} == {"b2c9"}

    repeated = workflow.add_current_evidence_member(
        regression_set["id"],
        {"expected_case_version": case["version"], "neighbor_radius": 2},
    )
    assert repeated["added_count"] == 0
    assert repeated["existing_count"] == 5
    workspace = workflow.workspace(case["id"])
    assert len(workspace["regression_sets"][0]["members"]) == 5
    assert any(event["event_type"] == "regression_set.members_added" for event in workspace["events"])

    with pytest.raises(ReviewWorkflowError, match="between 0 and 20"):
        workflow.add_current_evidence_member(
            regression_set["id"],
            {"expected_case_version": case["version"], "neighbor_radius": 21},
        )


def test_issue_regression_retest_reconciliation_and_signing_loop(tmp_path):
    cases, workflow, case = _build_case(tmp_path)
    assert case["evidence_coverage"]["state"] == "complete"
    assert case["verdict"]["result"] == "pass"

    issue = workflow.create_issue(
        {
            "case_id": case["id"],
            "expected_case_version": case["version"],
            "title": "cycle 438 需要回归验证",
            "severity": "major",
            "actor": "reviewer-a",
        }
    )
    regression_set = workflow.create_regression_set(
        {
            "issue_id": issue["id"],
            "expected_case_version": case["version"],
            "actor": "reviewer-a",
        }
    )
    member = workflow.add_current_evidence_member(
        regression_set["id"],
        {"expected_case_version": case["version"], "actor": "reviewer-a"},
    )
    assert member["sample_id"] == "xjtu:b2c9:438"

    retest = workflow.create_retest(
        case["id"],
        {
            "expected_case_version": case["version"],
            "issue_id": issue["id"],
            "regression_set_id": regression_set["id"],
            "actor": "reviewer-a",
        },
    )
    with pytest.raises(ReviewWorkflowError, match="completed retest"):
        workflow.update_issue_status(
            issue["id"],
            {
                "expected_case_version": case["version"],
                "status": "resolved",
                "reason": "too early",
            },
        )
    workflow.complete_retest(
        retest["id"],
        {
            "expected_case_version": case["version"],
            "status": "completed",
            "run_id": "run-retest-01",
            "result": {
                "mae": 0.009,
                "delta_mae": -0.001,
                "prediction_recomputed": True,
            },
        },
    )
    resolved = workflow.update_issue_status(
        issue["id"],
        {
            "expected_case_version": case["version"],
            "status": "resolved",
            "reason": "同协议复测通过",
            "actor": "reviewer-a",
        },
    )
    assert resolved["status"] == "resolved"

    case = cases.update_review(
        case["id"],
        {"expected_version": case["version"], "state": "in_review", "reviewer": "reviewer-a"},
    )
    case = cases.update_review(
        case["id"],
        {
            "expected_version": case["version"],
            "state": "approved",
            "reviewer": "reviewer-a",
            "note": "复测和 KPI 对账均通过",
        },
    )
    reconciliation = workflow.reconcile(
        case["id"],
        {"expected_case_version": case["version"], "tolerance": 1e-9},
    )
    assert reconciliation["status"] == "match"
    assert reconciliation["deltas"] == {"mae": 0.0, "picp": 0.0}
    pack = workflow.create_evidence_pack(
        case["id"], {"expected_case_version": case["version"], "actor": "reviewer-a"}
    )
    signed = workflow.sign_evidence_pack(
        pack["id"],
        {
            "expected_case_version": case["version"],
            "signer": "reviewer-a",
            "reason": "批准发布基线",
        },
    )

    assert signed["state"] == "signed"
    assert signed["digest"].startswith("sha256:")
    assert signed["manifest"]["schema_version"] == "1.1.0"
    assert {item["kind"] for item in signed["manifest"]["artifact_refs"]} >= {
        "run_spec", "aggregate_metrics", "raw_curve_source", "model_artifact"
    }
    workspace = workflow.workspace(case["id"])
    assert workspace["signing_readiness"]["can_sign"] is True
    workflow_events = [
        event for event in reversed(workspace["events"])
        if event["event_source"] == "workflow"
    ]
    assert [event["event_type"] for event in workflow_events] == [
        "issue.created",
        "regression_set.created",
        "regression_set.member_added",
        "retest.requested",
        "retest.completed",
        "issue.resolved",
        "reconciliation.match",
        "evidence_pack.created",
        "evidence_pack.signed",
    ]
    assert {event["event_type"] for event in workspace["events"] if event["event_source"] == "case"} >= {
        "case.created", "evidence.pinned", "review.in_review", "review.approved"
    }


def test_evidence_pack_signing_is_blocked_until_all_gates_pass(tmp_path):
    _, workflow, case = _build_case(tmp_path)
    pack = workflow.create_evidence_pack(
        case["id"], {"expected_case_version": case["version"]}
    )

    with pytest.raises(ReviewWorkflowError, match="signing is blocked"):
        workflow.sign_evidence_pack(
            pack["id"],
            {
                "expected_case_version": case["version"],
                "signer": "reviewer-a",
                "reason": "premature",
            },
        )

    readiness = workflow.workspace(case["id"])["signing_readiness"]
    assert readiness["can_sign"] is False
    assert "review is not approved" in readiness["blockers"]
    assert "canonical and legacy KPI reconciliation has not matched the current case version" in readiness["blockers"]


def test_review_cannot_approve_incomplete_or_failed_case(tmp_path):
    cases = EvaluationCaseStore(tmp_path / "cases")
    case = cases.create(
        {
            "profile_name": "p",
            "idempotency_key": "review:block",
            "execution": {"state": "completed"},
        }
    )
    case = cases.update_review(
        case["id"],
        {"expected_version": 1, "state": "in_review", "reviewer": "r"},
    )

    with pytest.raises(EvaluationCaseValidationError, match="complete evidence"):
        cases.update_review(
            case["id"],
            {
                "expected_version": case["version"],
                "state": "approved",
                "reviewer": "r",
                "note": "approve",
            },
        )
