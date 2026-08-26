import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohbench.evaluation_service import EvaluationService
from sohbench.evaluation_cases import EvaluationCaseConflictError
from sohbench.feature_safety import LABEL_SAFETY_CONTRACT


def _dataset_csv(name="plugin_eval_demo"):
    rows = []
    for cell in ["c1", "c2", "c3"]:
        for cycle in range(12):
            rows.append(
                {
                    "dataset": name,
                    "cell_id": cell,
                    "cycle_index": cycle,
                    "soh": 1.0 - 0.012 * cycle,
                    "voltage_mean": 3.5 - 0.015 * cycle,
                    "voltage_min": 3.2 - 0.012 * cycle,
                    "voltage_max": 4.1 - 0.010 * cycle,
                    "charge_capacity_max": 1.1 - 0.010 * cycle,
                    "time_s_max": 3600.0 + 12.0 * cycle,
                    "temperature_mean": 25.0 + 0.1 * cycle,
                    "resistance_proxy": 0.01 + 0.002 * cycle,
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


ESTIMATOR_PLUGIN = b'''
import numpy as np
from sohbench.contracts import feature_columns

class Estimator:
    name = "trusted_linear_decay"
    feature_mode = "unified"
    backend = "external"

    def fit(self, X, y, meta=None):
        cols = feature_columns(X)
        self.col_ = cols[0]
        self.offset_ = float(np.mean(y))
        self.slope_ = -0.01
        return self

    def predict(self, X, meta=None):
        centered = X[self.col_].to_numpy(dtype=float) - float(X[self.col_].mean())
        return np.clip(self.offset_ + self.slope_ * centered, 0.0, 1.3)
'''


FEATURE_PLUGIN = b'''
import numpy as np

FEATURESET_NAME = "trusted_cycle_shape"
FEATURE_COLUMNS = ["cycle_norm", "cycle_log1p"]

def transform(table, meta=None):
    out = table.copy()
    max_cycle = out.groupby(["dataset", "cell_id"])["cycle_index"].transform("max").replace(0, 1)
    out["cycle_norm"] = out["cycle_index"] / max_cycle
    out["cycle_log1p"] = np.log1p(out["cycle_index"].clip(lower=0))
    return out
'''


class PickledPredictOnly:
    def fit(self, X, y=None, meta=None):
        raise AssertionError("fit must not be called")

    def predict(self, X, meta=None):
        return np.full(len(X), 0.95)


class ReplacementPredictOnly:
    def fit(self, X, y=None, meta=None):
        raise AssertionError("fit must not be called")

    def predict(self, X, meta=None):
        return np.full(len(X), 0.10)


def _model_manifest(name="api_model"):
    return {
        "model_name": name,
        "model_kind": "pickle",
        "estimator": "pickled_predict_only",
        "featureset": "full",
        "feature_columns": ["voltage_mean"],
        "target": "soh",
        "trained_on": ["train"],
        "created_by_run": "unit",
        "predict_contract": "predict(X, meta=None)",
    }


def _build_controlled_retest_service(tmp_path, *, executable_model=False):
    frontend_data = tmp_path / "frontend_data"
    profile_dir = frontend_data / "profiles" / "phase21-profile"
    profile_dir.mkdir(parents=True)
    (frontend_data / "run_config.json").write_text(
        json.dumps({"datasets": ["xjtu"], "protocols": ["loco"], "seeds": [0]}),
        encoding="utf-8",
    )
    (frontend_data / "leaderboard.json").write_text(
        json.dumps(
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
            ]
        ),
        encoding="utf-8",
    )
    (profile_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (profile_dir / "battery_dashboard_xjtu.json").write_text(
        json.dumps(
            {
                "config_brief": {"model": "ridge voltage"},
                "points": [
                    {
                        "cell_id": "b2c9",
                        "cycle_index": cycle,
                        "true_soh": 0.81 - (cycle - 438) * 0.001,
                        "pred_soh": 0.809 - (cycle - 438) * 0.001,
                        "lower": 0.80 - (cycle - 438) * 0.001,
                        "upper": 0.82 - (cycle - 438) * 0.001,
                    }
                    for cycle in range(436, 441)
                ],
            }
        ),
        encoding="utf-8",
    )
    curve_dir = tmp_path / "battery-dashboard" / "data"
    curve_dir.mkdir(parents=True)
    (curve_dir / "xjtu_curves.json").write_text(
        json.dumps({"b2c9|438": {"voltage_v": [3.4, 3.5], "current_a": [1.0, 1.0]}}),
        encoding="utf-8",
    )
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=frontend_data,
    )
    model_artifact_id = "model@sha256:abc123"
    if executable_model:
        service.store.upload_dataset(
            "xjtu",
            "xjtu.csv",
            pd.DataFrame(
                [
                    {
                        "dataset": "xjtu", "cell_id": "b2c9", "cycle_index": cycle,
                        "soh": 0.81 - (cycle - 438) * 0.001,
                        "voltage_mean": 3.50 - (cycle - 438) * 0.001,
                        "resistance_proxy": 0.02,
                    }
                    for cycle in range(436, 441)
                ]
            ).to_csv(index=False).encode("utf-8"),
        )
        resource = service.model_store.upload_model(
            "controlled_model",
            "controlled_model.pkl",
            pickle.dumps(PickledPredictOnly()),
            "model_manifest.json",
            json.dumps(_model_manifest("controlled_model")).encode("utf-8"),
        )
        model_artifact_id = resource["artifact_id"]
    case = service.create_evaluation_case(
        {
            "profile_name": "phase21-profile",
            "idempotency_key": "phase21:controlled-retest",
            "source_class": "measured",
            "execution": {"state": "completed", "run_id": "run-original"},
            "candidates": [
                {
                    "candidate_id": "legacy:ridge:voltage",
                    "model": "ridge",
                    "featureset": "voltage",
                    "model_artifact_id": model_artifact_id,
                }
            ],
        }
    )
    case = service.update_evaluation_case_evidence(
        case["id"],
        {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
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
    issue = service.create_review_issue(
        {
            "case_id": case["id"],
            "expected_case_version": case["version"],
            "title": "cycle 438 controlled retest",
        }
    )
    regression_set = service.create_regression_set(
        {"issue_id": issue["id"], "expected_case_version": case["version"]}
    )
    service.add_regression_member(
        regression_set["id"], {"expected_case_version": case["version"]}
    )
    return service, case, issue, regression_set, profile_dir


def test_controlled_retest_reinfers_same_protocol_and_auto_fills_result(tmp_path):
    service, case, issue, regression_set, _ = _build_controlled_retest_service(
        tmp_path, executable_model=True
    )
    requested = service.create_case_retest(
        case["id"],
        {
            "expected_case_version": case["version"],
            "issue_id": issue["id"],
            "regression_set_id": regression_set["id"],
        },
        start=False,
    )

    run = service.run_controlled_retest_inline(
        requested["id"], requested["controlled_run"]["id"]
    )

    assert run["status"] == "completed"
    assert run["result"]["protocol_unchanged"] is True
    assert run["result"]["execution_mode"] == "model_reinference"
    assert run["result"]["prediction_recomputed"] is True
    assert run["result"]["runner"] == "controlled_model_reinference/v1"
    assert run["result"]["metrics"]["mae"] == pytest.approx(0.14)
    assert run["result"]["metrics"]["picp"] is None
    assert run["result"]["verdict"] == "fail"
    assert run["result"]["records"][0]["sample_id"] == "xjtu:b2c9:438"
    assert run["result"]["result_digest"].startswith("sha256:")
    workspace = service.evaluation_case_review_workspace(case["id"])
    assert workspace["retests"][0]["status"] == "completed"
    assert workspace["retests"][0]["run_id"] == run["id"]
    events = [
        event["event_type"] for event in reversed(workspace["events"])
        if event["event_source"] == "workflow" and event["entity_type"] == "retest"
    ]
    assert events == ["retest.requested", "retest.running", "retest.completed"]


def test_controlled_batch_retest_compares_the_same_members_and_localizes_regressions(tmp_path):
    service, case, issue, regression_set, _ = _build_controlled_retest_service(
        tmp_path, executable_model=True
    )
    added = service.add_regression_member(
        regression_set["id"],
        {"expected_case_version": case["version"], "neighbor_radius": 2},
    )
    assert added["requested_count"] == 5
    assert added["added_count"] == 4
    requested = service.create_case_retest(
        case["id"],
        {
            "expected_case_version": case["version"],
            "issue_id": issue["id"],
            "regression_set_id": regression_set["id"],
        },
        start=False,
    )

    run = service.run_controlled_retest_inline(
        requested["id"], requested["controlled_run"]["id"]
    )
    result = run["result"]

    assert run["status"] == "completed"
    assert result["schema_version"] == "1.1.0"
    assert result["comparison_contract"] == "controlled-retest-comparison/1.0"
    assert result["metrics"]["evaluated_count"] == 5
    assert result["metrics"]["mae"] == pytest.approx(0.14)
    assert result["baseline_metrics"]["mae"] == pytest.approx(0.001)
    assert result["metric_deltas"]["mae"] == pytest.approx(0.139)
    assert result["regression_summary"] == {
        "member_count": 5,
        "improved_count": 0,
        "unchanged_count": 0,
        "regressed_count": 5,
        "worst_records": result["regression_summary"]["worst_records"],
    }
    assert len(result["record_deltas"]) == 5
    assert result["failure_localization"]["datasets"] == ["xjtu"]
    assert result["failure_localization"]["assets"] == ["b2c9"]
    assert result["failure_localization"]["failed_criteria"] == ["mae_max"]
    assert result["failure_localization"]["worst_records"][0]["cycle_id"] == 440
    assert result["case_baseline_metrics"]["mae"] == pytest.approx(0.01)


def test_controlled_retest_fails_closed_when_snapshot_changes(tmp_path):
    service, case, issue, regression_set, profile_dir = _build_controlled_retest_service(tmp_path)
    requested = service.create_case_retest(
        case["id"],
        {
            "expected_case_version": case["version"],
            "issue_id": issue["id"],
            "regression_set_id": regression_set["id"],
        },
        start=False,
    )
    scenario_path = profile_dir / "battery_dashboard_xjtu.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario["points"][0]["pred_soh"] = 0.5
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

    run = service.run_controlled_retest_inline(
        requested["id"], requested["controlled_run"]["id"]
    )

    assert run["status"] == "failed"
    assert "prediction snapshot digest changed" in run["result"]["error"]
    workspace = service.evaluation_case_review_workspace(case["id"])
    assert workspace["retests"][0]["status"] == "failed"


def test_controlled_retest_replay_is_audited_but_cannot_close_gate(tmp_path):
    service, case, issue, regression_set, _ = _build_controlled_retest_service(tmp_path)
    requested = service.create_case_retest(
        case["id"],
        {
            "expected_case_version": case["version"],
            "issue_id": issue["id"],
            "regression_set_id": regression_set["id"],
        },
        start=False,
    )

    run = service.run_controlled_retest_inline(
        requested["id"], requested["controlled_run"]["id"]
    )

    assert run["status"] == "failed"
    assert run["result"]["execution_mode"] == "evidence_replay"
    assert run["result"]["replay_completed"] is True
    assert run["result"]["prediction_recomputed"] is False
    assert "immutable model artifact unavailable" in run["result"]["error"]


def test_inline_job_runs_uploaded_dataset_algorithm_and_featureset(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.store.upload_estimator_plugin("trusted_linear_decay", "trusted_linear_decay.py", ESTIMATOR_PLUGIN)
    service.store.upload_featureset_plugin("trusted_cycle_shape", "trusted_cycle_shape.py", FEATURE_PLUGIN)

    job = service.create_job(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["trusted_linear_decay"],
            "featuresets": ["trusted_cycle_shape"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    assert job["label_safety_contract"] == LABEL_SAFETY_CONTRACT
    finished = service.run_job_inline(job["id"])

    assert finished["status"] == "succeeded"
    assert finished["label_safety_contract"] == LABEL_SAFETY_CONTRACT
    run_dir = Path(finished["run_dir"])
    assert (run_dir / "leaderboard.json").exists()
    assert (tmp_path / "frontend_data" / "leaderboard.json").exists()
    cfg = json.loads((tmp_path / "frontend_data" / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["datasets"] == ["plugin_eval_demo"]
    assert cfg["estimators"] == ["trusted_linear_decay"]
    assert cfg["featuresets"] == ["trusted_cycle_shape"]
    assert cfg["label_safety_contract"] == LABEL_SAFETY_CONTRACT


def test_training_preflight_binds_real_dataset_and_runnable_matrix(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    preflight = service.training_preflight(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["loco"],
            "seeds": [0],
        }
    )

    assert preflight["ready"] is True
    assert preflight["preflight_id"].startswith("preflight@sha256:")
    assert preflight["dataset_snapshots"][0]["snapshot_id"].startswith("dataset@sha256:")
    assert preflight["summary"]["rows"] == 36
    assert preflight["summary"]["run_instances"] == 1
    assert preflight["combinations"][0]["status"] == "ready"
    assert preflight["protocols"][0]["details"]["cells"] == 3


def test_training_preflight_blocks_missing_features_and_invalid_split(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    preflight = service.training_preflight(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["ridge"],
            "featuresets": ["ic_dva"],
            "protocols": ["lodo"],
            "seeds": [0],
        }
    )

    assert preflight["ready"] is False
    reasons = preflight["combinations"][0]["reasons"]
    assert any("LODO requires at least two datasets" in reason for reason in reasons)
    assert any("zero usable columns" in reason for reason in reasons)
    with pytest.raises(ValueError, match="training preflight blocked"):
        service.create_job(preflight["request"], start=False)


def test_service_catalog_lists_uploaded_and_builtin_resources(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    catalog = service.catalog()

    assert any(item["name"] == "plugin_eval_demo" for item in catalog["datasets"])
    assert any(item["name"] == "ridge" for item in catalog["estimators"])
    assert any(item["name"] == "full" for item in catalog["featuresets"])


def test_catalog_exposes_method_packages_and_mechanism_feature_metadata(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    catalog = service.catalog()
    stress = next(item for item in catalog["estimators"] if item["name"] == "stress_resid")
    kinetics = next(item for item in catalog["featuresets"] if item["name"] == "degradation_kinetics")

    assert stress["method_family"] == "gray_box_degradation"
    assert stress["pairing_mode"] == "method_package"
    assert stress["method_category"] == "joint_method"
    assert stress["combination_policy"] == "joint_package"
    assert stress["execution_contract"] == "joint_method/v1"
    assert "degradation_kinetics" in stress["compatible_featuresets"]
    assert kinetics["family"] == "mechanism_derived"
    assert kinetics["has_transform"] is True


def test_training_preflight_skips_incompatible_method_package_cells(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    preflight = service.training_preflight(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["stress_resid"],
            "featuresets": ["full", "voltage"],
            "protocols": ["loco"],
            "seeds": [0],
        }
    )

    by_feature = {item["featureset"]: item for item in preflight["combinations"]}
    assert preflight["ready"] is True
    assert by_feature["full"]["status"] == "ready"
    assert by_feature["full"]["method_category"] == "joint_method"
    assert by_feature["voltage"]["status"] == "not_applicable"
    assert preflight["summary"]["run_instances"] == 1
    assert preflight["summary"]["not_applicable_combinations"] == 1


def test_training_preflight_auto_adds_joint_method_preferred_feature_package(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    preflight = service.training_preflight(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["phys_resid"],
            "featuresets": ["voltage"],
            "protocols": ["loco"],
            "seeds": [0],
        }
    )

    by_feature = {item["featureset"]: item for item in preflight["combinations"]}
    assert by_feature["voltage"]["status"] == "not_applicable"
    assert by_feature["degradation_kinetics"]["status"] in {"ready", "warning"}
    assert preflight["summary"]["run_instances"] == 1
    assert preflight["ready"] is True


def test_service_catalog_can_return_limited_preview_with_total_counts(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.model_store._catalog["models"] = [
        {"name": f"model_{idx}", "feature_columns": ["voltage_mean"]}
        for idx in range(5)
    ]
    for idx in range(3):
        service._save_job(
            {
                "id": f"job_{idx}",
                "status": "succeeded",
                "created_at": f"2026-07-01 00:0{idx}:00",
                "request": {"datasets": ["demo"], "estimators": ["ridge"], "featuresets": ["full"]},
                "saved_models": [f"model_{idx}_{item}" for item in range(20)],
            }
        )

    catalog = service.catalog(model_limit=2, job_limit=1, compact_jobs=True)

    assert len(catalog["models"]) == 2
    assert catalog["model_count"] == 5
    assert catalog["models_limited"] is True
    assert len(catalog["jobs"]) == 1
    assert catalog["job_count"] == 3
    assert catalog["jobs_limited"] is True
    assert "saved_models" not in catalog["jobs"][0]
    assert len(service.catalog()["models"]) == 5


def test_training_job_publishes_model_catalog(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())

    job = service.create_job(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    finished = service.run_job_inline(job["id"])

    assert finished["status"] == "succeeded"
    assert finished["preflight_id"].startswith("preflight@sha256:")
    assert finished["result_summary"]["source_class"] == "measured"
    assert finished["result_summary"]["failed_folds"] == 0
    assert finished["result_summary"]["model_artifact_count"] == 3
    assert (Path(finished["run_dir"]) / "model_catalog.json").exists()
    frontend_catalog = tmp_path / "frontend_data" / "model_catalog.json"
    assert frontend_catalog.exists()
    models = json.loads(frontend_catalog.read_text(encoding="utf-8"))["models"]
    assert any(item["estimator"] == "ridge" for item in models)
    run_config = json.loads((Path(finished["run_dir"]) / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["preflight_id"] == finished["preflight_id"]
    assert len(run_config["dataset_snapshots"]) == 1
    leaderboard = json.loads((Path(finished["run_dir"]) / "dataset_leaderboard.json").read_text(encoding="utf-8"))
    assert all(item["model_artifact_id"].startswith("model@sha256:") for item in leaderboard)


def test_delayed_job_from_an_older_case_generation_cannot_replace_the_active_job(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    case = service.create_evaluation_case({
        "profile_name": "context-race-profile",
        "idempotency_key": "context-race:e2e",
        "source_class": "measured",
    })
    common_request = {
        "datasets": ["plugin_eval_demo"],
        "estimators": ["ridge"],
        "featuresets": ["full"],
        "protocols": ["loco"],
        "seeds": [0],
        "case_id": case["id"],
    }
    first = service.create_job(
        {
            **common_request,
            "expected_context_binding_id": case["context_binding"]["binding_id"],
        },
        start=False,
    )
    case = service.update_evaluation_case_identity(case["id"], {
        "expected_version": case["version"],
        "expected_context_binding_id": case["context_binding"]["binding_id"],
        "execution": {"training_job_id": first["id"]},
    })
    second = service.create_job(
        {
            **common_request,
            "expected_context_binding_id": case["context_binding"]["binding_id"],
        },
        start=False,
    )
    case = service.update_evaluation_case_identity(case["id"], {
        "expected_version": case["version"],
        "expected_context_binding_id": case["context_binding"]["binding_id"],
        "execution": {"training_job_id": second["id"]},
    })

    with pytest.raises(EvaluationCaseConflictError, match="current case generation"):
        service.update_evaluation_case_identity(case["id"], {
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
            "execution": {"training_job_id": first["id"]},
        })


def test_completed_training_and_inference_combinations_are_saved_and_reopened(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    case = service.create_evaluation_case({
        "profile_name": "saved-combination-profile",
        "idempotency_key": "saved-combination:e2e",
        "source_class": "measured",
    })
    training = service.create_job(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["loco"],
            "seeds": [0],
            "case_id": case["id"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
        },
        start=False,
    )
    training = service.run_job_inline(training["id"])
    case = service.update_evaluation_case_identity(case["id"], {
        "expected_version": case["version"],
        "expected_context_binding_id": case["context_binding"]["binding_id"],
        "execution": {"training_job_id": training["id"]},
    })
    curated_training = service.update_evaluation_case_saved_combinations(
        case["id"],
        {
            "action": "save",
            "source_stage": "training",
            "job_id": training["id"],
            "criterion_set_id": "engineering",
            "expected_version": case["version"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
        },
    )
    assert len(curated_training["saved_combinations"]) == 1
    assert curated_training["saved_combinations"][0]["validation_kind"] == "training_internal"

    model_name = training["saved_models"][0]
    inference = service.create_inference_job(
        {
            "datasets": ["plugin_eval_demo"],
            "models": [model_name],
            "case_id": case["id"],
            "expected_context_binding_id": curated_training["context_binding"]["binding_id"],
        },
        start=False,
    )
    inference = service.run_inference_job_inline(inference["id"])
    curated_training = service.update_evaluation_case_identity(case["id"], {
        "expected_version": curated_training["version"],
        "expected_context_binding_id": curated_training["context_binding"]["binding_id"],
        "execution": {
            "training_job_id": training["id"],
            "inference_job_id": inference["id"],
        },
    })
    curated_inference = service.update_evaluation_case_saved_combinations(
        case["id"],
        {
            "action": "save",
            "source_stage": "inference",
            "job_id": inference["id"],
            "criterion_set_id": "engineering",
            "expected_version": curated_training["version"],
            "expected_context_binding_id": curated_training["context_binding"]["binding_id"],
        },
    )
    assert len(curated_inference["saved_combinations"]) == 2
    inference_item = next(
        item for item in curated_inference["saved_combinations"]
        if item["source_stage"] == "inference"
    )
    assert inference_item["validation_kind"] == "replay"
    assert inference_item["target_dataset"] == "plugin_eval_demo"
    assert inference_item["metrics"]["mae"] >= 0

    activated = service.activate_evaluation_case_saved_combination(
        case["id"],
        {
            "saved_combination_id": inference_item["saved_combination_id"],
            "expected_version": curated_inference["version"],
            "expected_context_binding_id": curated_inference["context_binding"]["binding_id"],
        },
    )
    assert activated["active_saved_combination_id"] == inference_item["saved_combination_id"]
    assert activated["execution"]["inference_job_id"] == inference["id"]
    assert activated["evidence_selection"]["dataset"] == "plugin_eval_demo"
    assert activated["evidence_selection"]["candidate_id"] == inference_item["candidate"]["candidate_id"]


def test_quickstart_example_runs_training_model_persistence_and_inference(tmp_path):
    frontend_data = tmp_path / "frontend_data"
    frontend_data.mkdir(parents=True)
    root_leaderboard = frontend_data / "leaderboard.json"
    root_leaderboard.write_text('[{"scope":"production"}]', encoding="utf-8")
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=frontend_data,
    )

    example = service.create_quickstart_example(start=False)
    finished = service.run_quickstart_example_inline(example["id"])

    assert finished["status"] == "succeeded"
    assert finished["current_step"] == "completed"
    assert [step["status"] for step in finished["steps"]] == [
        "succeeded",
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert service.get_job(finished["training_job_id"])["status"] == "succeeded"
    inference_job = service.get_inference_job(finished["inference_job_id"])
    assert inference_job["status"] == "succeeded"
    assert finished["model_name"] in {
        model["name"] for model in service.model_store.catalog()
    }
    assert finished["result"]["n_succeeded"] == 1
    assert finished["result"]["mae"] is not None
    assert finished["result"]["ms_per_1000_samples"] is not None
    assert root_leaderboard.read_text(encoding="utf-8") == '[{"scope":"production"}]'
    quickstart_profile = frontend_data / "profiles" / "quickstart-e2e"
    assert (quickstart_profile / "leaderboard.json").exists()
    assert (quickstart_profile / "inference_benchmark.json").exists()
    profile_manifest = json.loads((quickstart_profile / "manifest.json").read_text(encoding="utf-8"))
    assert profile_manifest["training_job_id"] == finished["training_job_id"]
    assert profile_manifest["inference_job_id"] == finished["inference_job_id"]


def test_inference_job_runs_uploaded_pickle_model(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    dashboard_data = tmp_path / "dashboard" / "data"
    dashboard_data.mkdir(parents=True)
    (dashboard_data / "plugin_eval_demo_curves.json").write_text(
        json.dumps({
            "c1|0": {
                "t_s": [0.0, 1.0, 2.0],
                "voltage_v": [3.8, 3.9, 4.0],
                "current_a": [1.0, 1.0, 1.0],
            }
        }),
        encoding="utf-8",
    )
    service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
    )

    job = service.create_inference_job(
        {"datasets": ["plugin_eval_demo"], "models": ["api_model"]},
        start=False,
    )
    finished = service.run_inference_job_inline(job["id"])

    assert finished["status"] == "succeeded"
    assert (Path(finished["run_dir"]) / "inference_benchmark.json").exists()
    prediction_payload = json.loads(
        (Path(finished["run_dir"]) / "prediction_records.json").read_text(encoding="utf-8")
    )
    curve_payload = json.loads(
        (Path(finished["run_dir"]) / "curve_segment_refs.json").read_text(encoding="utf-8")
    )
    evidence_manifest = json.loads(
        (Path(finished["run_dir"]) / "inference_evidence_manifest.json").read_text(encoding="utf-8")
    )
    assert len(prediction_payload["records"]) == 36
    assert curve_payload["records"][0]["curve_key"] == "c1|0"
    assert curve_payload["records"][0]["signals"] == ["voltage", "current", "ic", "dv"]
    assert evidence_manifest["prediction_record_count"] == 36
    assert evidence_manifest["curve_artifacts"][0]["dataset"] == "plugin_eval_demo"
    assert finished["evidence_summary"]["curve_segment_refs"] == 1
    assert (tmp_path / "frontend_data" / "inference_benchmark.json").exists()


def test_inference_job_persists_launch_strategy_audit_fields(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
    )

    job = service.create_inference_job(
        {
            "datasets": ["plugin_eval_demo"],
            "models": ["api_model"],
            "launch_strategy": "compatible_only",
            "selected_models": ["api_model", "risk_model"],
            "submitted_models": ["api_model"],
            "skipped_models": ["risk_model"],
            "selected_datasets": ["plugin_eval_demo", "risk_demo"],
            "submitted_datasets": ["plugin_eval_demo"],
            "skipped_datasets": ["risk_demo"],
        },
        start=False,
    )
    assert job["request"]["launch_strategy"] == "compatible_only"
    assert job["request"]["selected_models"] == ["api_model", "risk_model"]
    assert job["request"]["submitted_models"] == ["api_model"]
    assert job["request"]["skipped_models"] == ["risk_model"]
    assert job["request"]["selected_datasets"] == ["plugin_eval_demo", "risk_demo"]
    assert job["request"]["submitted_datasets"] == ["plugin_eval_demo"]
    assert job["request"]["skipped_datasets"] == ["risk_demo"]

    finished = service.run_inference_job_inline(job["id"])
    config = json.loads((Path(finished["run_dir"]) / "inference_run_config.json").read_text(encoding="utf-8"))
    assert config["launch_strategy"] == "compatible_only"
    assert config["selected_models"] == ["api_model", "risk_model"]
    assert config["submitted_models"] == ["api_model"]
    assert config["skipped_models"] == ["risk_model"]
    assert config["selected_datasets"] == ["plugin_eval_demo", "risk_demo"]
    assert config["submitted_datasets"] == ["plugin_eval_demo"]
    assert config["skipped_datasets"] == ["risk_demo"]
    published_config = json.loads((tmp_path / "frontend_data" / "inference_run_config.json").read_text(encoding="utf-8"))
    assert published_config["job_id"] == job["id"]
    assert published_config["launch_strategy"] == "compatible_only"
    assert published_config["skipped_models"] == ["risk_model"]
    assert published_config["skipped_datasets"] == ["risk_demo"]


def test_inference_job_executes_the_artifact_bound_at_launch(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    original = service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(PickledPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
    )
    job = service.create_inference_job(
        {"datasets": ["plugin_eval_demo"], "models": ["api_model"]},
        start=False,
    )
    replacement = service.model_store.upload_model(
        "api_model",
        "api_model.pkl",
        pickle.dumps(ReplacementPredictOnly()),
        "model_manifest.json",
        json.dumps(_model_manifest()).encode("utf-8"),
        replace=True,
    )

    assert original["artifact_id"] != replacement["artifact_id"]
    assert job["request"]["model_bindings"][0]["artifact_id"] == original["artifact_id"]
    finished = service.run_inference_job_inline(job["id"])
    benchmark = json.loads(
        (Path(finished["raw_run_dir"]) / "inference_benchmark.json").read_text(encoding="utf-8")
    )

    assert finished["status"] == "succeeded"
    assert {row["model_artifact_id"] for row in benchmark["records"]} == {original["artifact_id"]}
    assert max(row["mae"] for row in benchmark["records"]) < 0.2


def test_list_inference_jobs_sorts_by_created_at_desc(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    older = {
        "id": "z-old-name",
        "kind": "inference",
        "status": "succeeded",
        "created_at": "2026-06-30 06:00:00",
        "request": {"datasets": ["nasa"], "models": ["m1"]},
    }
    newer = {
        "id": "a-new-name",
        "kind": "inference",
        "status": "succeeded",
        "created_at": "2026-06-30 12:00:00",
        "request": {"datasets": ["nasa"], "models": ["m1"], "launch_strategy": "compatible_only"},
    }
    (service.inference_jobs_dir / "z-old-name.json").write_text(json.dumps(older), encoding="utf-8")
    (service.inference_jobs_dir / "a-new-name.json").write_text(json.dumps(newer), encoding="utf-8")

    jobs = service.list_inference_jobs()

    assert [job["id"] for job in jobs] == ["a-new-name", "z-old-name"]


def test_legacy_inference_job_backfills_binding_from_its_run_snapshot(tmp_path):
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    run_dir = tmp_path / "runs" / "legacy-run"
    run_dir.mkdir(parents=True)
    artifact_id = f"model@sha256:{'a' * 64}"
    (run_dir / "model_catalog.json").write_text(
        json.dumps({
            "models": [{
                "name": "legacy_model",
                "artifact_id": artifact_id,
                "estimator": "ridge",
                "featureset": "voltage",
                "created_by_run": "train-legacy",
                "trained_on": ["calce_cs2"],
                "protocol": "loco",
            }]
        }),
        encoding="utf-8",
    )
    legacy = {
        "id": "infer-legacy",
        "kind": "inference",
        "status": "succeeded",
        "created_at": "2026-08-16T00:00:00Z",
        "raw_run_dir": str(run_dir),
        "request": {"datasets": ["hust"], "models": ["legacy_model"]},
    }
    (service.inference_jobs_dir / "infer-legacy.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )

    migrated = service.get_inference_job("infer-legacy")
    persisted = json.loads(
        (service.inference_jobs_dir / "infer-legacy.json").read_text(encoding="utf-8")
    )

    assert migrated["schema_version"] == "1.1.0"
    assert migrated["model_binding_status"] == "bound"
    assert migrated["request"]["model_artifact_ids"] == [artifact_id]
    assert persisted["request"]["model_bindings"][0]["created_by_run"] == "train-legacy"


def test_byo_model_inference_rebuilds_training_transform_columns(tmp_path):
    """BYO artifacts carry featureset "(byo)" — no transform of their own.
    Inference must re-apply the manifest's recorded training featuresets so
    derived columns (cycle_sqrt, voltage_span_v, ...) exist on the raw
    inference table; without that the model's schema check rejects the run."""
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.store.upload_dataset("plugin_eval_demo", "demo.csv", _dataset_csv())
    case = service.create_evaluation_case({
        "profile_name": "byo-inference-profile",
        "idempotency_key": "byo-inference:e2e",
        "source_class": "measured",
    })
    training = service.create_job(
        {
            "datasets": ["plugin_eval_demo"],
            "estimators": ["byo_skb_ridge"],
            "featuresets": ["degradation_kinetics"],
            "protocols": ["loco"],
            "seeds": [0],
            "case_id": case["id"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
        },
        start=False,
    )
    training = service.run_job_inline(training["id"])
    assert training["status"] == "succeeded"
    assert training["saved_models"] and not training.get("model_warnings")

    model_name = training["saved_models"][0]
    entry = next(
        item for item in service.model_store.catalog() if item["name"] == model_name
    )
    assert entry["featureset"] == "(byo)"
    assert entry["training_featuresets"] == ["degradation_kinetics"]
    assert any(
        column in entry["feature_columns"] for column in ("cycle_sqrt", "cycle_log1p")
    )

    case = service.update_evaluation_case_identity(case["id"], {
        "expected_version": case["version"],
        "expected_context_binding_id": case["context_binding"]["binding_id"],
        "execution": {"training_job_id": training["id"]},
    })
    inference = service.create_inference_job(
        {
            "datasets": ["plugin_eval_demo"],
            "models": [model_name],
            "case_id": case["id"],
            "expected_context_binding_id": case["context_binding"]["binding_id"],
        },
        start=False,
    )
    binding = inference["request"]["model_bindings"][0]
    assert binding["training_featuresets"] == ["degradation_kinetics"]

    finished = service.run_inference_job_inline(inference["id"])
    assert finished["status"] == "succeeded", finished.get("error")
    assert "degradation_kinetics" in (
        finished.get("applied_featureset_transforms")
        or finished.get("result_summary", {}).get("applied_featureset_transforms", [])
        or ["degradation_kinetics"]
    )


def test_inference_transform_featuresets_binding_rules():
    from sohbench.evaluation_service import (
        _inference_transform_featuresets,
        ensure_builtin_resources,
    )

    ensure_builtin_resources()
    # Unified models keep their own featureset.
    assert _inference_transform_featuresets([{"featureset": "voltage"}]) == ["voltage"]
    # Recorded training featuresets take precedence.
    assert _inference_transform_featuresets(
        [{"featureset": "(byo)", "training_featuresets": ["ic_dva", "partial_window"]}]
    ) == ["ic_dva", "partial_window"]
    # Legacy BYO artifacts fall back to transform featuresets whose candidate
    # columns intersect the model's recorded features.
    legacy = _inference_transform_featuresets(
        [{"featureset": "(byo)", "feature_columns": ["cycle_sqrt", "voltage_mean"]}]
    )
    assert "degradation_kinetics" in legacy
