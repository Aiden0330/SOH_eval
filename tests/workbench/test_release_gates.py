"""Release-gate invariants of the frozen workbench (extraction plan Task 9).

These are the checks the phase 3.0 release-gate script used to run against a measured case:
the measured chain reaches a signed EvidencePack, the case survives a service reload and
resolves idempotently, the signed pack and the audit events are tamper-evident, and a case
whose scope is incompatible or whose required evidence is missing cannot be signed.  The
case itself is built with the workbench's measured-case pipeline on a fixture dataset.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohbench.evaluation_service import EvaluationService, _canonical_digest
from sohbench.phase22_golden import build_and_sign_phase22_golden
from sohbench.review_workflow import ReviewWorkflowError

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_EVENT_TYPES = {
    "case.created",
    "evidence.pinned",
    "review.approved",
    "retest.completed",
    "reconciliation.match",
    "evidence_pack.signed",
}


class FrozenLinearModel:
    def predict(self, X, meta=None):
        return np.asarray(X["voltage_mean"], dtype=float)

    def predict_interval(self, X, meta=None):
        center = self.predict(X, meta)
        return center - 0.03, center + 0.03


def _service(root: Path) -> EvaluationService:
    return EvaluationService(
        data_dir=root / "data",
        work_dir=root / "work",
        runs_dir=root / "runs",
        frontend_data_dir=root / "frontend" / "data",
        allow_unsafe_plugins=False,
    )


@pytest.fixture
def signed_case(tmp_path, monkeypatch):
    """A measured NASA/B0018 fixture case taken through retest, reconciliation and signing."""
    monkeypatch.chdir(REPO_ROOT)  # the pipeline checks that sohbench resolves to the active workspace
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    rows, curves = [], {}
    for cell, offset in (("TRAIN", 0.0), ("B0018", 0.002)):
        for cycle in range(1, 9):
            truth = 1.0 - cycle * 0.01
            rows.append({"dataset": "nasa", "cell_id": cell, "cycle_index": cycle, "soh": truth,
                         "voltage_mean": truth + offset})
            if cell == "B0018":
                curves[f"{cell}|{cycle}"] = {"voltage_v": [3.4, 3.5], "current_a": [1.0, 1.0],
                                             "temperature_c": [25.0, 25.1]}
    pd.DataFrame(rows).to_pickle(processed / "nasa_features.pkl")
    curve_source = tmp_path / "nasa_curves.json"
    curve_source.write_text(json.dumps(curves), encoding="utf-8")
    service = _service(tmp_path)
    model_name = "fixture_nasa_b0018"
    service.model_store.save_training_model(
        model_name, FrozenLinearModel(),
        {"model_name": model_name, "model_kind": "pickle", "estimator": "fixture_linear", "featureset": "full",
         "feature_columns": ["voltage_mean"],
         "input_columns": ["dataset", "cell_id", "cycle_index", "soh", "voltage_mean"],
         "target": "soh", "trained_on": ["nasa"], "created_by_run": "fixture-run",
         "predict_contract": "predict(X, meta=None)", "held_out_key": "nasa/B0018", "protocol": "loco", "seed": 0},
    )
    summary = build_and_sign_phase22_golden(
        service, profile_name="gate-fixture", model_name=model_name, curve_source=curve_source,
        reviewer="fixture-reviewer",
    )
    return tmp_path, service, summary


def test_measured_chain_reaches_a_signed_pack(signed_case):
    _root, _service_, summary = signed_case
    retest = summary["retest"]
    conditions = {
        "pipeline_completed": summary["status"] == "completed",
        "prediction_facts_present": int(summary["prediction_records"]) > 0,
        "evidence_complete": summary["evidence_coverage"]["state"] == "complete",
        "verdict_pass": summary["verdict"]["result"] == "pass",
        "review_approved": summary["review"]["state"] == "approved",
        "controlled_retest_pass": retest["status"] == "completed" and retest["execution_mode"] == "model_reinference"
        and retest["prediction_recomputed"] is True and retest["verdict"] == "pass",
        "kpi_reconciled": summary["reconciliation"]["status"] == "match",
        "evidence_pack_signed": summary["evidence_pack"]["state"] == "signed",
        "release_ready": summary["signing_readiness"]["can_sign"] is True,
        "immutable_lineage_present": all(
            bool(summary.get(key)) for key in ("comparison_id", "model_artifact_id", "dataset_snapshot_id", "split_assignment_id")
        ),
    }
    assert all(conditions.values()), conditions


def test_case_survives_a_service_reload_and_resolves_idempotently(signed_case):
    root, service, summary = signed_case
    before = service.get_evaluation_case(summary["case_id"])
    reloaded = _service(root)
    case = reloaded.get_evaluation_case(summary["case_id"])
    resolved = reloaded.resolve_evaluation_case({"profile_name": case["profile_name"], "idempotency_key": case["idempotency_key"]})
    workspace = reloaded.evaluation_case_review_workspace(case["id"])
    signed = next(item for item in workspace["evidence_packs"] if item.get("state") == "signed")
    assert case["id"] == summary["case_id"] == before["id"]
    assert case["version"] == summary["case_version"] == before["version"]
    assert case["profile_name"] == summary["profile_name"] == before["profile_name"]
    for key in ("execution", "evidence_selection", "candidates", "context_binding", "evidence_coverage"):
        assert case.get(key) == before.get(key), key
    assert signed["id"] == summary["evidence_pack"]["id"] and signed["digest"] == summary["evidence_pack"]["digest"]
    assert resolved["created"] is False and resolved["case"]["id"] == case["id"]


def _cas_path(root: Path, digest: str) -> Path:
    return root / "sha256" / digest[:2] / digest


def test_signed_pack_and_audit_events_are_tamper_evident(signed_case):
    root, service, summary = signed_case
    workspace = service.evaluation_case_review_workspace(summary["case_id"])
    signed = next(item for item in workspace["evidence_packs"] if item.get("state") == "signed")
    expected_digest = str(signed["digest"]).removeprefix("sha256:")
    cas_root = service.work_dir / "artifacts"
    assert _canonical_digest(signed["manifest"]) == expected_digest
    pack_file = _cas_path(cas_root, expected_digest)
    assert pack_file.is_file() and hashlib.sha256(pack_file.read_bytes()).hexdigest() == expected_digest
    tampered = json.loads(json.dumps(signed["manifest"]))
    tampered["case_id"] = f"{tampered.get('case_id') or 'case'}-tampered"
    assert _canonical_digest(tampered) != expected_digest
    events = workspace["events"]
    assert REQUIRED_EVENT_TYPES <= {str(item.get("event_type") or "") for item in events}
    for event in events:
        digest = str(event.get("payload_digest") or "").removeprefix("sha256:")
        path = _cas_path(cas_root, digest)
        assert digest and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    keys = [(str(item.get("event_source") or ""), int(item.get("sequence") or 0)) for item in events]
    assert len(keys) == len(set(keys))


def test_canonical_digest_is_order_independent_and_detects_tampering():
    manifest = {"case_id": "EV-30", "artifacts": ["a", "b"]}
    assert _canonical_digest(manifest) == _canonical_digest({"artifacts": ["a", "b"], "case_id": "EV-30"})
    assert _canonical_digest(manifest) != _canonical_digest({"case_id": "EV-30-tampered", "artifacts": ["a", "b"]})


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _negative_guard(service, baseline: dict, sandbox_root: Path, *, scenario: str, selected_dataset: str) -> dict:
    """The phase 3.0 negative guard: copy the baseline profile into a sandbox service, degrade it
    for the scenario, resolve a fresh case, pin evidence, and try to sign."""
    profile_name = baseline["profile_name"]
    source_profile = service.frontend_data_dir / "profiles" / profile_name
    frontend_data = sandbox_root / "frontend" / "app" / "data"
    target_profile = frontend_data / "profiles" / profile_name
    target_profile.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_profile, target_profile)
    if scenario == "missing_evidence":
        manifest_path = target_profile / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["curve_artifacts"] = [
            {**item, "source": str(sandbox_root / "intentionally-missing" / str(item.get("filename") or "curves.json")), "copied": False}
            for item in manifest.get("curve_artifacts") or []
        ]
        _write_json(manifest_path, manifest)
    isolated = EvaluationService(data_dir=service.data_dir, work_dir=sandbox_root / "work", runs_dir=sandbox_root / "runs",
                                 frontend_data_dir=frontend_data, allow_unsafe_plugins=False)
    baseline_case = service.get_evaluation_case(baseline["case_id"])
    candidate = next(
        (item for item in baseline_case.get("candidates") or [] if item.get("model_artifact_id") == baseline.get("model_artifact_id")),
        (baseline_case.get("candidates") or [{}])[0],
    )
    resolved = isolated.resolve_evaluation_case({
        "profile_name": profile_name, "idempotency_key": f"gate-negative:{scenario}:{uuid.uuid4().hex}",
        "title": f"negative gate · {scenario}", "source_class": "measured",
        "execution": {"state": "completed", "training_job_id": baseline.get("training_job_id") or "",
                      "inference_job_id": baseline.get("inference_job_id") or "", "run_id": baseline.get("inference_job_id") or ""},
        "candidates": [candidate],
    })
    case = resolved["case"]
    baseline_selection = baseline_case.get("evidence_selection") or {}
    selection = {"candidate_id": candidate.get("candidate_id") or "", "dataset": selected_dataset,
                 "asset_id": baseline_selection.get("asset_id") or "", "cycle_id": baseline_selection.get("cycle_id"),
                 "evidence_tab": "raw"}
    if scenario == "incompatible_scope":
        selection.update({"asset_id": "", "cycle_id": None})
    case = isolated.update_evaluation_case_evidence(case["id"], {
        "expected_version": case["version"], "expected_context_binding_id": case["context_binding"]["binding_id"],
        "criterion_set_id": "exploratory", "selection": selection, "actor": "negative-fixture",
        "reason": f"expected release block: {scenario}",
    })
    workspace = isolated.evaluation_case_review_workspace(case["id"])
    pack = isolated.create_case_evidence_pack(case["id"], {"expected_case_version": case["version"], "actor": "negative-fixture"})
    sign_error = ""
    try:
        isolated.sign_evidence_pack(pack["id"], {"expected_case_version": case["version"], "signer": "negative-fixture",
                                                 "reason": "this negative fixture must not be signable"})
    except ReviewWorkflowError as exc:
        sign_error = str(exc)
    missing_required = sorted(item["kind"] for item in case["evidence_coverage"].get("items") or []
                              if item.get("required") and not item.get("available"))
    return {
        "evidence_incomplete": case["evidence_coverage"].get("state") == "incomplete",
        "release_not_ready": workspace["signing_readiness"].get("can_sign") is False,
        "signing_rejected": "signing is blocked" in sign_error,
        "missing_required": missing_required,
    }


def test_incompatible_scope_and_missing_evidence_block_signing(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    profile_name = "gate-source"
    profile = frontend_data / "profiles" / profile_name
    curve_source = tmp_path / "xjtu_curves.json"
    _write_json(curve_source, {"b2c9|438": {"voltage_v": [3.5, 3.6], "current_a": [1.0, 1.0], "temperature_c": [25.0, 26.0]}})
    _write_json(profile / "run_config.json", {"datasets": ["xjtu"], "protocols": ["loco"], "seeds": [0]})
    _write_json(profile / "leaderboard.json", [{"estimator": "ridge", "featureset": "voltage", "metrics": {
        "mae": {"mean": 0.01}, "picp": {"mean": 0.95}, "monotonicity_violation_rate": {"mean": 0.0},
        "value_range_violation": {"mean": 0.0}}}])
    _write_json(profile / "manifest.json", {"curve_artifacts": [{"dataset": "xjtu", "filename": "xjtu_curves.json", "source": str(curve_source)}]})
    record = {"prediction_record_id": "pred:xjtu:b2c9:438", "sample_id": "xjtu:b2c9:438", "candidate_id": "artifact:ridge-xjtu",
              "model_artifact_id": "model@sha256:fixture", "dataset": "xjtu", "cell_id": "b2c9", "cycle_index": 438,
              "true_soh": 0.81, "pred_soh": 0.80, "absolute_error": 0.01}
    _write_json(profile / "prediction_records.json", {"records": [record]})
    _write_json(profile / "battery_dashboard_xjtu.json", {"points": [record]})
    service = EvaluationService(data_dir=tmp_path / "data", work_dir=tmp_path / "work", runs_dir=tmp_path / "runs",
                                frontend_data_dir=frontend_data, allow_unsafe_plugins=False)
    resolved = service.resolve_evaluation_case({
        "profile_name": profile_name, "idempotency_key": "gate:source",
        "execution": {"state": "completed", "training_job_id": "train-30", "inference_job_id": "infer-30", "run_id": "infer-30"},
        "candidates": [{"candidate_id": "artifact:ridge-xjtu", "model": "ridge", "featureset": "voltage",
                        "model_artifact_id": "model@sha256:fixture"}],
    })
    baseline = {"profile_name": profile_name, "case_id": resolved["case"]["id"], "model_artifact_id": "model@sha256:fixture",
                "training_job_id": "train-30", "inference_job_id": "infer-30"}
    incompatible = _negative_guard(service, baseline, tmp_path / "guard-incompatible", scenario="incompatible_scope", selected_dataset="outside-scope")
    missing = _negative_guard(service, baseline, tmp_path / "guard-missing", scenario="missing_evidence", selected_dataset="xjtu")
    for outcome in (incompatible, missing):
        assert outcome["evidence_incomplete"] and outcome["release_not_ready"] and outcome["signing_rejected"], outcome
    assert {"comparison_scope", "split_assignment", "prediction_record"} <= set(incompatible["missing_required"])
    assert "raw_curve" in missing["missing_required"]
