"""Inference-phase adaptation, workbench level: the adaptation search and the audited case
inference pipeline apply operators around a frozen model (EvaluationService).

Operator math and adapter mechanics are covered in tests/kernel/test_adaptation.py."""

from pathlib import Path

import numpy as np
import pytest

from sohbench.paths import data_root

DATA_DIR = data_root()


class ConstantModel:
    """Picklable frozen model for the end-to-end adaptation test."""

    def predict(self, X, meta=None):
        return np.full(len(X), 0.9)


@pytest.mark.skipif(
    not (DATA_DIR / "processed" / "smoke_eval_demo_features.pkl").is_file(),
    reason="demo datasets not present",
)
def test_adaptation_search_grid_scores_frozen_model_with_operators(tmp_path):
    from sohbench.evaluation_service import EvaluationService

    service = EvaluationService(
        data_dir=DATA_DIR,
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )

    service.model_store.save_training_model(
        "frozen_demo_model",
        ConstantModel(),
        {
            "model_name": "frozen_demo_model",
            "model_kind": "pickle",
            "estimator": "constant",
            "featureset": "voltage",
            "feature_columns": ["voltage_mean"],
            "target": "soh",
            "trained_on": ["soh_quickstart_demo"],
            "created_by_run": "unit",
            "predict_contract": "predict(X, meta=None)",
        },
    )
    job = service.create_search_job(
        {
            "mode": "adaptation",
            "target_dataset": "smoke_eval_demo",
            "models": ["frozen_demo_model"],
            "operators": ["none", "recalibration"],
            "calibration_budget": 8,
            "policy": "grid",
        },
        start=False,
    )
    finished = service.run_search_job_inline(job["id"])
    assert finished["status"] == "succeeded", finished.get("error")
    report = finished["report"]
    assert report["mode"] == "adaptation"
    ranking = {item["operator"]: item for item in report["ranking"]}
    assert set(ranking) == {"none", "recalibration"}
    # A constant predictor gains from output recalibration (bias removed on
    # the shared evaluation subset), and the channel is declared.
    assert ranking["recalibration"]["target_mae"] <= ranking["none"]["target_mae"]
    assert "calibration" in ranking["recalibration"]["provenance"]["channel"]
    assert ranking["none"]["calibration_samples"] == 8  # identical split for all
    assert finished["result_summary"]["best_outer_mae"] == pytest.approx(
        report["ranking"][0]["target_mae"]
    )


@pytest.mark.skipif(
    not (DATA_DIR / "processed" / "smoke_eval_demo_features.pkl").is_file(),
    reason="demo datasets not present",
)
def test_case_inference_pipeline_applies_adaptation_operator(tmp_path):
    """The audited inference job itself runs the operator: adapted metrics,
    calibration rows excluded and flagged, operator recorded in the result."""
    from sohbench.evaluation_service import EvaluationService

    service = EvaluationService(
        data_dir=DATA_DIR,
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.model_store.save_training_model(
        "frozen_demo_model",
        ConstantModel(),
        {
            "model_name": "frozen_demo_model",
            "model_kind": "pickle",
            "estimator": "constant",
            "featureset": "voltage",
            "feature_columns": ["voltage_mean"],
            "target": "soh",
            "trained_on": ["soh_quickstart_demo"],
            "created_by_run": "unit",
            "predict_contract": "predict(X, meta=None)",
        },
    )
    plain = service.create_inference_job(
        {"models": ["frozen_demo_model"], "datasets": ["smoke_eval_demo"]},
        start=False,
    )
    plain = service.run_inference_job_inline(plain["id"], publish=False)
    adapted = service.create_inference_job(
        {
            "models": ["frozen_demo_model"],
            "datasets": ["smoke_eval_demo"],
            "adaptation": {"operator": "recalibration", "calibration_budget": 8},
        },
        start=False,
    )
    assert adapted["request"]["adaptation"]["operator"] == "recalibration"
    assert "calibration" in adapted["request"]["adaptation"]["channel"]
    adapted = service.run_inference_job_inline(adapted["id"], publish=False)

    assert adapted["status"] == "succeeded", adapted.get("error")

    import json

    def benchmark_rows(job):
        payload = json.loads(
            (Path(job["raw_run_dir"]) / "inference_benchmark.json").read_text(encoding="utf-8")
        )
        rows = payload.get("records") if isinstance(payload, dict) else payload
        return [r for r in rows if r["status"] == "succeeded"]

    plain_row = benchmark_rows(plain)[0]
    adapted_row = benchmark_rows(adapted)[0]
    assert adapted_row["adaptation"]["operator"] == "recalibration"
    assert adapted_row["adaptation"]["recalibration"]["samples"] >= 4
    assert adapted_row["mae"] < plain_row["mae"]

    records_payload = json.loads(
        (Path(adapted["raw_run_dir"]) / "prediction_records.json").read_text(encoding="utf-8")
    )
    prediction_rows = records_payload.get("records") or []
    flagged = [r for r in prediction_rows if r.get("calibration_sample")]
    assert len(flagged) == 8
    assert all(r["metric_contributions"]["mae_contribution"] is None for r in flagged)


# --- review fixes 2026-09-02: slope guards, LOO conformal, shared holdout ---
