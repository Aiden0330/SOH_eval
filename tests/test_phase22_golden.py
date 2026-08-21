import json
from pathlib import Path

import numpy as np
import pandas as pd

from sohbench.evaluation_service import EvaluationService
from sohbench.phase22_golden import build_and_sign_phase22_golden


class FrozenLinearModel:
    def predict(self, X, meta=None):
        return np.asarray(X["voltage_mean"], dtype=float)

    def predict_interval(self, X, meta=None):
        center = self.predict(X, meta)
        return center - 0.03, center + 0.03


def test_phase22_golden_builds_real_retest_and_signed_pack(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)
    rows = []
    curves = {}
    for cell, offset in (("TRAIN", 0.0), ("B0018", 0.002)):
        for cycle in range(1, 9):
            truth = 1.0 - cycle * 0.01
            rows.append(
                {
                    "dataset": "nasa",
                    "cell_id": cell,
                    "cycle_index": cycle,
                    "soh": truth,
                    "voltage_mean": truth + offset,
                }
            )
            if cell == "B0018":
                curves[f"{cell}|{cycle}"] = {
                    "voltage_v": [3.4, 3.5],
                    "current_a": [1.0, 1.0],
                    "temperature_c": [25.0, 25.1],
                }
    pd.DataFrame(rows).to_pickle(processed / "nasa_features.pkl")
    curve_source = tmp_path / "nasa_curves.json"
    curve_source.write_text(json.dumps(curves), encoding="utf-8")

    service = EvaluationService(
        data_dir=data_dir,
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend" / "data",
    )
    model_name = "fixture_nasa_b0018"
    service.model_store.save_training_model(
        model_name,
        FrozenLinearModel(),
        {
            "model_name": model_name,
            "model_kind": "pickle",
            "estimator": "fixture_linear",
            "featureset": "full",
            "feature_columns": ["voltage_mean"],
            "input_columns": [
                "dataset", "cell_id", "cycle_index", "soh", "voltage_mean"
            ],
            "target": "soh",
            "trained_on": ["nasa"],
            "created_by_run": "fixture-run",
            "predict_contract": "predict(X, meta=None)",
            "held_out_key": "nasa/B0018",
            "protocol": "loco",
            "seed": 0,
        },
    )

    summary = build_and_sign_phase22_golden(
        service,
        profile_name="phase22-fixture",
        model_name=model_name,
        curve_source=curve_source,
        reviewer="fixture-reviewer",
    )

    assert summary["status"] == "completed"
    assert summary["prediction_records"] == 8
    assert summary["evidence_coverage"]["state"] == "complete"
    assert summary["retest"] == {
        **summary["retest"],
        "status": "completed",
        "execution_mode": "model_reinference",
        "prediction_recomputed": True,
        "verdict": "pass",
    }
    assert summary["reconciliation"]["status"] == "match"
    assert summary["evidence_pack"]["state"] == "signed"
    assert summary["signing_readiness"]["can_sign"] is True
    assert summary["issue_statuses"] == ["resolved"]

    profile = service.frontend_data_dir / "profiles" / "phase22-fixture"
    prediction_records = json.loads(
        (profile / "prediction_records.json").read_text(encoding="utf-8")
    )["records"]
    assert len(prediction_records) == 8
    assert all(record["model_artifact_id"].startswith("model@sha256:") for record in prediction_records)
    assert all(record["curve_segment_ref_id"] for record in prediction_records)
