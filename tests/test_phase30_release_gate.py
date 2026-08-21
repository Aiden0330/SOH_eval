import json

from sohbench.evaluation_service import EvaluationService
from sohbench.phase30_release_gate import (
    Phase30ReleaseGateConfig,
    _digest_json,
    _negative_guard_check,
    _pipeline_check,
    render_phase30_release_gate_markdown,
    run_phase30_release_gate,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _passing_summary():
    return {
        "status": "completed",
        "profile_name": "phase30-fixture",
        "case_id": "EV-30-FIXTURE",
        "case_version": 4,
        "comparison_id": "comparison@sha256:a",
        "model_artifact_id": "model@sha256:b",
        "dataset_snapshot_id": "dataset-snapshot@sha256:c",
        "split_assignment_id": "protocol@sha256:d",
        "training_job_id": "train-30",
        "inference_job_id": "infer-30",
        "selected_sample_id": "xjtu:b2c9:438",
        "prediction_records": 10,
        "pipeline": {
            "preflight_ready": True,
            "training_status": "succeeded",
            "training_model_count": 3,
            "inference_status": "succeeded",
            "inference_records": 1,
        },
        "evidence_coverage": {"state": "complete"},
        "verdict": {"result": "pass"},
        "review": {"state": "approved"},
        "retest": {
            "status": "completed",
            "execution_mode": "model_reinference",
            "prediction_recomputed": True,
            "verdict": "pass",
        },
        "reconciliation": {"status": "match"},
        "evidence_pack": {"state": "signed", "id": "EPK-30", "digest": "sha256:e"},
        "signing_readiness": {"can_sign": True},
        "url": "http://127.0.0.1:8765/example",
    }


def test_pipeline_gate_requires_the_complete_measured_chain():
    summary = _passing_summary()
    assert _pipeline_check(summary)["passed"] is True

    summary["retest"]["prediction_recomputed"] = False
    check = _pipeline_check(summary)
    assert check["passed"] is False
    assert check["details"]["conditions"]["controlled_retest_pass"] is False


def test_manifest_digest_is_canonical_and_detects_tampering():
    manifest = {"case_id": "EV-30", "artifacts": ["a", "b"]}
    equivalent = {"artifacts": ["a", "b"], "case_id": "EV-30"}
    tampered = {"case_id": "EV-30-tampered", "artifacts": ["a", "b"]}

    assert _digest_json(manifest) == _digest_json(equivalent)
    assert _digest_json(manifest) != _digest_json(tampered)


def test_negative_fixtures_block_incompatible_scope_and_missing_curve(tmp_path):
    data_dir = tmp_path / "data"
    work_dir = tmp_path / "work"
    frontend_data = tmp_path / "frontend" / "app" / "data"
    profile_name = "phase30-source"
    profile = frontend_data / "profiles" / profile_name
    curve_source = tmp_path / "xjtu_curves.json"
    _write_json(
        curve_source,
        {
            "b2c9|438": {
                "voltage_v": [3.5, 3.6],
                "current_a": [1.0, 1.0],
                "temperature_c": [25.0, 26.0],
            }
        },
    )
    _write_json(profile / "run_config.json", {"datasets": ["xjtu"], "protocols": ["loco"], "seeds": [0]})
    _write_json(
        profile / "leaderboard.json",
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
    _write_json(
        profile / "manifest.json",
        {
            "curve_artifacts": [
                {
                    "dataset": "xjtu",
                    "filename": "xjtu_curves.json",
                    "source": str(curve_source),
                }
            ]
        },
    )
    record = {
        "prediction_record_id": "pred:xjtu:b2c9:438",
        "sample_id": "xjtu:b2c9:438",
        "candidate_id": "artifact:ridge-xjtu",
        "model_artifact_id": "model@sha256:fixture",
        "dataset": "xjtu",
        "cell_id": "b2c9",
        "cycle_index": 438,
        "true_soh": 0.81,
        "pred_soh": 0.80,
        "absolute_error": 0.01,
    }
    _write_json(profile / "prediction_records.json", {"records": [record]})
    _write_json(profile / "battery_dashboard_xjtu.json", {"points": [record]})

    service = EvaluationService(
        data_dir=data_dir,
        work_dir=work_dir,
        runs_dir=tmp_path / "runs",
        frontend_data_dir=frontend_data,
        allow_unsafe_plugins=False,
    )
    resolved = service.resolve_evaluation_case(
        {
            "profile_name": profile_name,
            "idempotency_key": "phase30:source",
            "execution": {
                "state": "completed",
                "training_job_id": "train-30",
                "inference_job_id": "infer-30",
                "run_id": "infer-30",
            },
            "candidates": [
                {
                    "candidate_id": "artifact:ridge-xjtu",
                    "model": "ridge",
                    "featureset": "voltage",
                    "model_artifact_id": "model@sha256:fixture",
                }
            ],
        }
    )
    baseline = {
        **_passing_summary(),
        "profile_name": profile_name,
        "case_id": resolved["case"]["id"],
        "model_artifact_id": "model@sha256:fixture",
        "selected_sample_id": "xjtu:b2c9:438",
    }

    incompatible = _negative_guard_check(
        service,
        baseline,
        tmp_path / "guard-incompatible",
        scenario="incompatible_scope",
        selected_dataset="outside-scope",
        criterion_set_id="exploratory",
    )
    missing = _negative_guard_check(
        service,
        baseline,
        tmp_path / "guard-missing",
        scenario="missing_evidence",
        selected_dataset="xjtu",
        criterion_set_id="exploratory",
    )

    assert incompatible["actual"] == "blocked"
    assert incompatible["passed"] is True
    assert "comparison_scope" in incompatible["details"]["missing_required"]
    assert missing["actual"] == "blocked"
    assert missing["passed"] is True
    assert "raw_curve" in missing["details"]["missing_required"]


def test_gate_returns_a_machine_readable_failed_report_when_pipeline_errors(tmp_path, monkeypatch):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend" / "app" / "data",
        allow_unsafe_plugins=False,
    )

    def fail_pipeline(*_args, **_kwargs):
        raise RuntimeError("expected fixture failure")

    monkeypatch.setattr("sohbench.phase30_release_gate.run_phase24_pipeline", fail_pipeline)
    report = run_phase30_release_gate(service, Phase30ReleaseGateConfig())
    markdown = render_phase30_release_gate_markdown(report)

    assert report["status"] == "failed"
    assert report["summary"] == {"total": 5, "passed": 0, "failed": 5}
    assert report["checks"][0]["details"]["error"] == "expected fixture failure"
    assert "阶段 3.0 自动回归门禁报告" in markdown
