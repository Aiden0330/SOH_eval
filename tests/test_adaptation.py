"""Inference-phase adaptation: operator math, channels, end-to-end search."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohbench.adaptation import (
    ADAPTATION_OPERATORS,
    OPERATOR_CHANNELS,
    apply_coral,
    apply_recalibration,
    calibration_split,
    fit_coral,
    fit_recalibration,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class ConstantModel:
    """Picklable frozen model for the end-to-end adaptation test."""

    def predict(self, X, meta=None):
        return np.full(len(X), 0.9)


def test_coral_aligns_target_statistics_to_source():
    rng = np.random.default_rng(0)
    source = rng.normal([0.0, 1.0], [1.0, 0.5], size=(400, 2))
    target = rng.normal([5.0, -2.0], [3.0, 0.1], size=(400, 2))
    coral = fit_coral(source, target, ["a", "b"])
    table = pd.DataFrame(target, columns=["a", "b"])
    aligned = apply_coral(table, coral)[["a", "b"]].to_numpy()
    assert np.allclose(aligned.mean(axis=0), source.mean(axis=0), atol=0.15)
    assert np.allclose(aligned.std(axis=0), source.std(axis=0), atol=0.15)


def test_recalibration_removes_scale_and_bias():
    truth = np.linspace(1.0, 0.7, 50)
    biased = truth * 1.3 + 0.2
    recal = fit_recalibration(truth, biased)
    corrected = apply_recalibration(biased, recal)
    assert np.abs(corrected - truth).max() < 1e-9


def test_calibration_split_takes_earliest_cycles_round_robin():
    table = pd.DataFrame({
        "cell_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
        "cycle_index": [1, 2, 3, 1, 2, 3],
    })
    calibration, evaluation = calibration_split(table, 2)
    # One earliest cycle per cell, not two from the same cell.
    chosen = table[calibration]
    assert sorted(chosen["cell_id"]) == ["c1", "c2"]
    assert (chosen["cycle_index"] == 1).all()
    assert evaluation.sum() == 4


def test_every_operator_declares_its_channel():
    assert set(OPERATOR_CHANNELS) == set(ADAPTATION_OPERATORS)


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
