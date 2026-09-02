import json
import sqlite3

import pytest

from sohbench.case_evidence import (
    EvidenceDerivationService,
    derive_verdict,
    write_prediction_trajectory_index,
)
from sohbench.evaluation_cases import EvaluationCaseValidationError, build_evaluation_case


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prediction_lookup_tolerates_duplicate_record_ids(tmp_path):
    run_dir = tmp_path / "run"
    records = [
        {
            "prediction_record_id": "pred:duplicate",
            "sample_id": "dataset:asset:1",
            "candidate_id": "candidate-a",
            "model_artifact_id": "model-a",
            "dataset": "dataset",
            "cell_id": "asset",
            "cycle_index": 1,
            "pred_soh": value,
        }
        for value in (0.9, 0.8)
    ]
    _write_json(run_dir / "prediction_records.json", {"records": records})

    write_prediction_trajectory_index(run_dir)

    with sqlite3.connect(run_dir / "prediction_evidence_lookup.sqlite3") as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        stored = json.loads(
            connection.execute(
                "SELECT record_json FROM records WHERE prediction_record_id = ?",
                ("pred:duplicate",),
            ).fetchone()[0]
        )
    assert metadata["duplicate_record_count"] == "1"
    assert stored["pred_soh"] == 0.8


def _fixture_case(frontend_data):
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
            "points": [
                {
                    "cell_id": "b2c9",
                    "cycle_index": 438,
                    "true_soh": 0.81,
                    "pred_soh": 0.809,
                }
            ]
        },
    )
    _write_json(
        frontend_data.parent / "battery-dashboard" / "data" / "xjtu_curves.json",
        {
            "b2c9|438": {
                "t_s": [0.0, 1.0],
                "voltage_v": [3.4, 3.5],
                "current_a": [1.0, 1.0],
                "temperature_c": [25.0, 26.0],
                "ic_ah_per_v": [0.2, 0.3],
            }
        },
    )
    return build_evaluation_case(
        {
            "profile_name": "battery-profile",
            "idempotency_key": "run:fixture",
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


def test_evidence_coverage_and_verdict_are_derived_from_server_artifacts(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    case = _fixture_case(frontend_data)
    result = EvidenceDerivationService(frontend_data).derive(
        case,
        {
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

    assert result["selection"]["sample_id"] == "xjtu:b2c9:438"
    assert result["selection"]["dataset_snapshot_id"].startswith("dataset-snapshot@sha256:")
    assert result["selection"]["split_assignment_id"].startswith("protocol@sha256:")
    assert result["selection"]["curve_ref"]["availability"] == "available"
    assert result["selection"]["curve_ref"]["signals"] == [
        "voltage", "current", "temperature", "ic"
    ]
    assert result["coverage"]["state"] == "complete"
    assert result["coverage"]["available_count"] == 8
    assert result["verdict"]["result"] == "pass"
    assert result["verdict"]["observed"] == {
        "mae": 0.01,
        "picp": 0.95,
        "monotonicity": 0.0,
        "value_range": 0.0,
    }


def test_case_scoped_thresholds_create_versioned_verdict_without_mutating_preset():
    metrics = {
        "mae": 0.0579,
        "picp": 0.95,
        "monotonicity": 0.2986,
        "value_range": 0.0,
    }

    baseline = derive_verdict(metrics, "engineering")
    customized = derive_verdict(
        metrics,
        "engineering",
        {"monotonicity_max": 0.30},
    )
    baseline_again = derive_verdict(metrics, "engineering")

    assert baseline["result"] == "fail"
    assert customized["result"] == "pass"
    assert customized["criterion_set_version"].startswith("engineering@custom-")
    assert customized["thresholds"]["monotonicity_max"] == 0.30
    assert baseline_again["criterion_set_version"] == "engineering"
    assert baseline_again["thresholds"]["monotonicity_max"] == 0.15


def test_case_scoped_thresholds_reject_unknown_or_out_of_range_values():
    with pytest.raises(EvaluationCaseValidationError, match="unknown criterion threshold"):
        derive_verdict({"mae": 0.01}, "engineering", {"unknown": 0.1})
    with pytest.raises(EvaluationCaseValidationError, match="between 0 and 1"):
        derive_verdict({"mae": 0.01}, "engineering", {"mae_max": 1.1})


def test_prediction_evidence_auto_selects_representative_cycle(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    case = _fixture_case(frontend_data)
    result = EvidenceDerivationService(frontend_data).derive(
        case,
        {
            "selection": {
                "candidate_id": "legacy:ridge:voltage",
                "dataset": "xjtu",
                "asset_id": "b2c9",
            }
        },
    )

    items = {item["kind"]: item for item in result["coverage"]["items"]}
    assert items["prediction_record"]["available"] is True
    assert items["raw_curve"]["available"] is True
    assert result["selection"]["asset_id"] == "b2c9"
    assert result["selection"]["cycle_id"] == 438
    assert result["selection"]["sample_id"] == "xjtu:b2c9:438"


def test_prediction_evidence_auto_selects_asset_and_highest_error(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    case = _fixture_case(frontend_data)
    profile = frontend_data / "profiles" / "battery-profile"
    _write_json(
        profile / "prediction_records.json",
        {
            "records": [
                {
                    "sample_id": "xjtu:b2c9:438",
                    "candidate_id": "legacy:ridge:voltage",
                    "dataset": "xjtu",
                    "cell_id": "b2c9",
                    "cycle_index": 438,
                    "true_soh": 0.81,
                    "pred_soh": 0.809,
                    "absolute_error": 0.001,
                },
                {
                    "sample_id": "xjtu:b2c9:439",
                    "candidate_id": "legacy:ridge:voltage",
                    "dataset": "xjtu",
                    "cell_id": "b2c9",
                    "cycle_index": 439,
                    "true_soh": 0.80,
                    "pred_soh": 0.795,
                    "absolute_error": 0.005,
                },
            ]
        },
    )

    result = EvidenceDerivationService(frontend_data).derive(
        case,
        {
            "selection": {
                "candidate_id": "legacy:ridge:voltage",
                "dataset": "xjtu",
                "asset_id": "",
                "cycle_id": None,
            }
        },
    )

    assert result["selection"]["asset_id"] == "b2c9"
    assert result["selection"]["cycle_id"] == 439
    assert result["selection"]["sample_id"] == "xjtu:b2c9:439"


def test_root_pipeline_case_reads_published_root_prediction_artifacts(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    _fixture_case(frontend_data)
    _write_json(
        frontend_data / "battery_dashboard_xjtu.json",
        {
            "points": [{
                "cell_id": "b2c9",
                "cycle_index": 438,
                "true_soh": 0.81,
                "pred_soh": 0.809,
            }]
        },
    )
    case = build_evaluation_case({
        "profile_name": "root",
        "idempotency_key": "pipeline:root-case",
        "candidates": [{
            "candidate_id": "legacy:ridge:voltage",
            "model": "ridge",
            "featureset": "voltage",
            "model_artifact_id": "model@sha256:abc123",
        }],
    })

    result = EvidenceDerivationService(frontend_data).derive(
        case,
        {
            "selection": {
                "candidate_id": "legacy:ridge:voltage",
                "dataset": "xjtu",
                "asset_id": "b2c9",
                "cycle_id": 438,
            }
        },
    )

    assert result["selection"]["sample_id"] == "xjtu:b2c9:438"
    assert result["selection"]["split_assignment_id"].startswith("protocol@sha256:")


def test_client_cannot_spoof_server_derived_evidence_fields(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    case = _fixture_case(frontend_data)
    service = EvidenceDerivationService(frontend_data)

    with pytest.raises(EvaluationCaseValidationError, match="read-only"):
        service.derive(
            case,
            {
                "coverage": {"state": "complete"},
                "verdict": {"result": "pass"},
            },
        )


def test_canonical_prediction_detail_joins_contribution_and_curve_ref(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    case = _fixture_case(frontend_data)
    profile = frontend_data / "profiles" / "battery-profile"
    record = {
        "prediction_record_id": "pred:xjtu:b2c9:438",
        "sample_id": "xjtu:b2c9:438",
        "cell_id": "b2c9",
        "cycle_index": 438,
        "true_soh": 0.81,
        "pred_soh": 0.809,
    }
    _write_json(profile / "prediction_records.json", {"run_id": "run-1", "records": [record]})
    _write_json(
        profile / "metric_contributions.json",
        {"records": [{"prediction_record_id": record["prediction_record_id"], "mae": 0.001}]},
    )
    _write_json(
        profile / "curve_segment_refs.json",
        {"records": [{"prediction_record_id": record["prediction_record_id"], "curve_key": "b2c9|438"}]},
    )
    service = EvidenceDerivationService(frontend_data)
    derived = service.derive(
        case,
        {
            "criterion_set_id": "strict",
            "selection": {
                "candidate_id": "legacy:ridge:voltage",
                "dataset": "xjtu",
                "asset_id": "b2c9",
                "cycle_id": 438,
            },
        },
    )
    updated = {**case, "evidence_selection": derived["selection"]}

    detail = service.prediction_detail(updated)

    assert detail["prediction_record"]["prediction_record_id"] == "pred:xjtu:b2c9:438"
    assert detail["metric_contribution"]["mae"] == 0.001
    assert detail["curve_segment_ref"]["curve_key"] == "b2c9|438"


def test_canonical_prediction_curves_return_measured_and_derived_signals(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    case = _fixture_case(frontend_data)
    profile = frontend_data / "profiles" / "battery-profile"
    record = {
        "prediction_record_id": "pred:xjtu:b2c9:438",
        "sample_id": "xjtu:b2c9:438",
        "dataset": "xjtu",
        "cell_id": "b2c9",
        "cycle_index": 438,
        "true_soh": 0.81,
        "pred_soh": 0.809,
        "featureset": "partial_window",
        "feature_values": {"q_win_ah": 0.2, "t_win_s": 720.0},
    }
    _write_json(profile / "prediction_records.json", {"records": [record]})
    _write_json(
        profile / "curve_segment_refs.json",
        {"records": [{"prediction_record_id": record["prediction_record_id"], "curve_key": "b2c9|438"}]},
    )
    _write_json(
        frontend_data.parent / "battery-dashboard" / "data" / "xjtu_curves.json",
        {
            "b2c9|438": {
                "t_s": [0.0, 360.0, 720.0, 1080.0, 1440.0],
                "voltage_v": [3.50, 3.70, 3.90, 4.10, 4.20],
                "current_a": [1.0, 1.0, 1.0, 1.0, 1.0],
                "temperature_c": [25.0, 25.5, 26.0, 26.4, 26.7],
            }
        },
    )
    _write_json(
        frontend_data.parent / "battery-dashboard" / "data" / "xjtu.json",
        {
            "feature_window": [3.7, 4.1],
            "feature_window_approx": True,
            "feature_columns": ["q_win_ah", "t_win_s"],
        },
    )
    service = EvidenceDerivationService(frontend_data)
    derived = service.derive(
        case,
        {
            "criterion_set_id": "strict",
            "selection": {
                "candidate_id": "legacy:ridge:voltage",
                "dataset": "xjtu",
                "asset_id": "b2c9",
                "cycle_id": 438,
            },
        },
    )
    updated = {**case, "evidence_selection": derived["selection"]}

    curves = service.prediction_curves(
        updated,
        record_id="pred:xjtu:b2c9:438",
        signals=["voltage", "current", "temperature", "ic", "dv"],
    )

    assert curves["prediction_record"]["sample_id"] == "xjtu:b2c9:438"
    assert curves["curve_segment_ref"]["curve_key"] == "b2c9|438"
    assert curves["available_signals"] == ["voltage", "current", "temperature", "ic", "dv"]
    assert curves["signals"]["voltage"]["derivation"] == "measured"
    assert curves["signals"]["ic"]["derivation"] == "derived from measured V/I/time"
    assert curves["signals"]["dv"]["x_name"] == "charge_ah"
    assert curves["feature_window"]["cycle_time_s"] == [360.0, 1080.0]
    assert curves["feature_window"]["duration_s"] == 720.0
    assert curves["feature_window"]["approximate"] is True
    assert [item["name"] for item in curves["feature_curves"]] == ["q_win_ah", "t_win_s"]
    assert all(
        item["highlight"]["cycle_time_s"] == [360.0, 1080.0]
        for item in curves["feature_curves"]
    )
    assert curves["feature_curves"][0]["source_signal"] == "cumulative_charge"
    assert curves["sampling"] == {
        "resolution": "viewport",
        "method": "none",
        "source_points": 5,
        "returned_points": 5,
        "max_points": 480,
    }


def test_case_uses_bound_inference_job_for_trajectory_and_raw_curves(tmp_path):
    frontend_data = tmp_path / "frontend" / "app" / "data"
    profile = frontend_data / "profiles" / "stale-profile"
    _write_json(profile / "manifest.json", {"curve_artifacts": []})
    _write_json(profile / "prediction_records.json", {"records": [{
        "prediction_record_id": "pred:stale",
        "dataset": "nasa",
        "cell_id": "B0018",
        "cycle_index": 21,
        "true_soh": 0.9,
        "pred_soh": 0.9,
    }]})

    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "inference-01"
    curve_source = tmp_path / "dashboard" / "data" / "calce_curves.json"
    _write_json(curve_source, {
        "CS2_38|21": {
            "t_s": [0.0, 360.0, 720.0],
            "voltage_v": [3.7, 3.9, 4.1],
            "current_a": [1.0, 1.0, 1.0],
        },
        "CS2_38|22": {
            "t_s": [0.0, 360.0, 720.0],
            "voltage_v": [4.15, 4.18, 4.2],
            "current_a": [1.0, 0.5, 0.0],
        }
    })
    record = {
        "prediction_record_id": "pred:infer-01:abc",
        "sample_id": "calce_cs2:CS2_38:21",
        "run_id": "infer-01",
        "candidate_id": "artifact:model-a",
        "model_artifact_id": "model@sha256:a",
        "featureset": "capacity_time",
        "dataset": "calce_cs2",
        "dataset_snapshot_id": "dataset-snapshot@sha256:data",
        "split_assignment_id": "protocol@sha256:split",
        "cell_id": "CS2_38",
        "cycle_index": 21,
        "true_soh": 0.97,
        "pred_soh": 0.96,
        "absolute_error": 0.01,
        "curve_segment_ref_id": "curve:calce_cs2:CS2_38:21",
        "feature_values": {
            "charge_capacity_max": 1.07,
            "time_s_max": 720.0,
            "cycle_index_feature": 21.0,
            "cycle_fraction": 0.5,
        },
    }
    raw_only_record = {
        **record,
        "prediction_record_id": "pred:infer-01:raw-only",
        "sample_id": "calce_cs2:CS2_38:22",
        "cycle_index": 22,
        "curve_segment_ref_id": "",
        "pred_soh": 0.8,
        "absolute_error": 0.17,
    }
    wrong_candidate_record = {
        **record,
        "prediction_record_id": "pred:infer-01:wrong-candidate",
        "sample_id": "calce_cs2:CS2_38:23:wrong",
        "candidate_id": "artifact:model-b",
        "model_artifact_id": "model@sha256:b",
        "cycle_index": 23,
        "pred_soh": 0.42,
        "absolute_error": 0.55,
        "curve_segment_ref_id": "",
    }
    _write_json(
        run_dir / "prediction_records.json",
        {"records": [record, raw_only_record, wrong_candidate_record]},
    )
    _write_json(run_dir / "metric_contributions.json", {"records": [{
        "prediction_record_id": record["prediction_record_id"],
        "mae_contribution": 0.01,
    }]})
    _write_json(run_dir / "curve_segment_refs.json", {"records": [{
        "prediction_record_id": record["prediction_record_id"],
        "curve_key": "CS2_38|21",
        "signals": ["voltage", "current", "ic", "dv"],
        "availability": "available",
    }, {
        "prediction_record_id": raw_only_record["prediction_record_id"],
        "curve_key": "CS2_38|22",
        "signals": ["voltage", "current"],
        "availability": "available",
    }]})
    _write_json(run_dir / "inference_evidence_manifest.json", {
        "curve_artifacts": [{
            "dataset": "calce_cs2",
            "source": str(curve_source),
        }]
    })
    _write_json(curve_source.with_name("calce.json"), {
        "feature_window": [3.6, 4.1],
        "feature_window_approx": False,
        "feature_columns": [
            "charge_capacity_max", "time_s_max", "cycle_index_feature", "cycle_fraction"
        ],
    })
    _write_json(run_dir / "inference_run_config.json", {
        "datasets": ["calce_cs2"],
        "protocols": ["loco"],
        "seeds": [0],
    })
    _write_json(run_dir / "inference_benchmark.json", {"records": [{
        "model": "model-a",
        "dataset": "calce_cs2",
        "estimator": "gbdt",
        "featureset": "capacity_time",
        "candidate_id": "artifact:model-a",
        "model_artifact_id": "model@sha256:a",
        "mae": 0.01,
        "picp": 0.823,
        "mean_interval_width": 0.03,
        "monotonicity_violation_rate": 0.0,
        "value_range_violation": 0.0,
    }]})
    jobs_dir = tmp_path / "work" / "inference_jobs"
    _write_json(jobs_dir / "infer-01.json", {
        "id": "infer-01",
        "status": "succeeded",
        "raw_run_dir": str(run_dir),
        "request": {"case_id": "EV-BOUND"},
    })
    data_dir = tmp_path / "data"
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "processed" / "calce_cs2_features.pkl").write_bytes(b"snapshot")
    case = build_evaluation_case({
        "id": "EV-BOUND",
        "profile_name": "stale-profile",
        "idempotency_key": "run:bound",
        "execution": {
            "state": "completed",
            "inference_job_id": "infer-01",
            "run_id": "infer-01",
        },
        "candidates": [{
            "candidate_id": "artifact:model-a",
            "model": "gbdt",
            "featureset": "capacity_time",
            "model_artifact_id": "model@sha256:a",
        }, {
            "candidate_id": "artifact:model-b",
            "model": "gbdt",
            "featureset": "voltage",
            "model_artifact_id": "model@sha256:b",
        }],
    })
    service = EvidenceDerivationService(
        frontend_data,
        data_dir,
        inference_jobs_dir=jobs_dir,
        runs_dir=runs_dir,
    )

    derived = service.derive(case, {
        "criterion_set_id": "strict",
        "selection": {
            "candidate_id": "artifact:model-a",
            "dataset": "calce_cs2",
        },
    })
    updated = {**case, "evidence_selection": derived["selection"]}
    trajectory = service.prediction_trajectory(
        updated,
        inference_job_id="infer-01",
        dataset="calce_cs2",
        asset_id="CS2_38",
    )
    compact_trajectory = service.prediction_trajectory(
        updated,
        inference_job_id="infer-01",
        dataset="calce_cs2",
        asset_id="CS2_38",
        compact=True,
        candidate_id="artifact:model-a",
        model_artifact_id="model@sha256:a",
    )
    with pytest.raises(EvaluationCaseValidationError, match="does not match"):
        service.prediction_trajectory(
            updated,
            inference_job_id="infer-01",
            dataset="calce_cs2",
            compact=True,
            candidate_id="artifact:model-a",
            model_artifact_id="model@sha256:b",
        )
    saved_history_case = {
        **updated,
        "execution": {"state": "completed", "inference_job_id": "infer-newer"},
        "active_saved_combination_id": "combo:model-a",
        "saved_combinations": [{
            "saved_combination_id": "combo:model-b",
            "source_stage": "inference",
            "inference_job_id": "infer-01",
            "target_dataset": "calce_cs2",
            "candidate": {
                "candidate_id": "artifact:model-b",
                "model_artifact_id": "model@sha256:b",
            },
        }, {
            "saved_combination_id": "combo:model-a",
            "source_stage": "inference",
            "inference_job_id": "infer-01",
            "target_dataset": "calce_cs2",
            "candidate": {
                "candidate_id": "artifact:model-a",
                "model_artifact_id": "model@sha256:a",
            },
        }],
    }
    saved_history_trajectory = service.prediction_trajectory(
        saved_history_case,
        dataset="calce_cs2",
        asset_id="CS2_38",
        compact=True,
    )
    curves = service.prediction_curves(
        updated,
        record_id="pred:infer-01:abc",
        signals=["voltage", "current", "ic", "dv"],
    )

    assert derived["selection"]["sample_id"] == "pred:infer-01:abc"
    assert derived["selection"]["cycle_id"] == 21
    # Job-scoped verdicts read the inference benchmark: the calibration gate
    # must see picp (0.823 < strict 0.90) and fail, not silently pass.
    assert derived["verdict"]["observed"]["picp"] == 0.823
    assert derived["verdict"]["result"] == "fail"
    assert any("calibration" in reason for reason in derived["verdict"]["reasons"])
    assert derived["selection"]["curve_ref"]["availability"] == "available"
    assert trajectory["inference_job_id"] == "infer-01"
    assert trajectory["points"][0]["feature_values"]["charge_capacity_max"] == 1.07
    assert compact_trajectory["compact"] is True
    assert compact_trajectory["candidate_id"] == "artifact:model-a"
    assert compact_trajectory["model_artifact_id"] == "model@sha256:a"
    assert compact_trajectory["points"][0]["prediction_record_id"] == "pred:infer-01:abc"
    assert "feature_values" not in compact_trajectory["points"][0]
    assert (run_dir / "prediction_trajectory_index.json").is_file()
    trajectory_index = json.loads(
        (run_dir / "prediction_trajectory_index.json").read_text(encoding="utf-8")
    )
    assert trajectory_index["schema_version"] == "prediction-trajectory-index/2.1"
    assert trajectory_index["scopes"][0]["raw_curve_records"] == 1
    assert trajectory_index["scopes"][0]["path"].startswith("prediction_trajectory_v2/")
    assert (run_dir / trajectory_index["scopes"][0]["path"]).is_file()
    assert (run_dir / "prediction_evidence_lookup.sqlite3").is_file()
    assert saved_history_trajectory["inference_job_id"] == "infer-01"
    assert saved_history_trajectory["candidate_id"] == "artifact:model-a"
    assert saved_history_trajectory["points"][0]["prediction_record_id"] == "pred:infer-01:abc"
    assert curves["available_signals"] == ["voltage", "current", "ic", "dv"]
    assert [item["name"] for item in curves["feature_curves"]] == [
        "charge_capacity_max", "time_s_max", "cycle_index_feature", "cycle_fraction"
    ]
    assert curves["feature_window"]["cycle_time_s"] == [0.0, 720.0]
    assert curves["feature_curves"][0]["scope"] == "full_cycle"
    assert curves["feature_curves"][2]["scope"] == "cycle_context"


def test_hust_raw_curve_fallback_materializes_missing_extract(tmp_path):
    """Pre-converted curve extracts cover only sampled cells; a prediction
    record outside the grid must fall back to the local raw acquisition file
    and materialize a stable per-record curve cache."""
    import pandas as pd
    import pickle as _pickle

    from sohbench.case_evidence import EvidenceDerivationService

    raw_dir = tmp_path / "data" / "raw" / "hust" / "our_data"
    raw_dir.mkdir(parents=True)
    frame = pd.DataFrame({
        "Status": ["Constant current charge"] * 4,
        "Cycle number": [7] * 4,
        "Current (mA)": [5500.0, 5500.0, 5500.0, 5500.0],
        "Voltage (V)": [3.1, 3.3, 3.45, 3.52],
        "Capacity (mAh)": [0.0, 300.0, 700.0, 900.0],
        "Time (s)": [0.0, 120.0, 240.0, 360.0],
    })
    with open(raw_dir / "9-9.pkl", "wb") as handle:
        _pickle.dump({"9-9": {"rul": 0, "dq": {}, "data": {7: frame}}}, handle)

    service = EvidenceDerivationService(
        tmp_path / "frontend_data",
        data_dir=tmp_path / "data",
    )
    path, available = service._curve_state(
        {"curve_artifacts": []},
        {"dataset": "hust", "asset_id": "9-9", "cycle_id": 7},
    )
    assert available and path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    curve = payload["9-9|7"]
    assert curve["voltage_v"] == [3.1, 3.3, 3.45, 3.52]
    assert curve["current_a"] == [5.5, 5.5, 5.5, 5.5]

    # Second resolution reuses the cache file (stable digest for refs).
    again, available_again = service._curve_state(
        {"curve_artifacts": []},
        {"dataset": "hust", "asset_id": "9-9", "cycle_id": 7},
    )
    assert available_again and again == path

    # Unknown cells/datasets still report unavailable.
    _, missing = service._curve_state(
        {"curve_artifacts": []},
        {"dataset": "hust", "asset_id": "no-such-cell", "cycle_id": 1},
    )
    assert not missing


def test_verdict_never_passes_with_an_unverifiable_criterion():
    metrics = {"mae": 0.01, "monotonicity": 0.0, "value_range": 0.0}  # no picp
    verdict = derive_verdict(metrics, "strict")
    assert verdict["result"] == "pending"
    assert "picp" not in verdict["observed"]
    assert any("no auditable picp" in reason for reason in verdict["reasons"])
    # A failing criterion still fails even when another one is unverifiable.
    failing = derive_verdict({**metrics, "mae": 0.5}, "strict")
    assert failing["result"] == "fail"
    # Nothing observed at all stays pending.
    assert derive_verdict({}, "strict")["result"] == "pending"


def test_candidate_metrics_follow_the_selected_dataset():
    from sohbench.case_evidence import _candidate_metrics, _inference_leaderboard

    rows = _inference_leaderboard({"records": [
        {"model": "m", "dataset": "calce_cs2", "estimator": "gbdt", "featureset": "voltage",
         "model_artifact_id": "model@sha256:a", "mae": 0.30, "picp": 0.5},
        {"model": "m", "dataset": "nasa", "estimator": "gbdt", "featureset": "voltage",
         "model_artifact_id": "model@sha256:a", "mae": 0.01, "picp": 0.95},
    ]})
    candidate = {"model": "gbdt", "featureset": "voltage", "model_artifact_id": "model@sha256:a"}
    assert _candidate_metrics(rows, candidate, "nasa")["mae"] == 0.01
    assert _candidate_metrics(rows, candidate, "calce_cs2")["picp"] == 0.5
    # No dataset pinned: first matching row (previous behaviour) still works.
    assert _candidate_metrics(rows, candidate)["mae"] == 0.30
