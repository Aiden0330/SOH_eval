"""Golden numeric invariants on the real NASA data (extraction plan Task 9; skips without data).

What the phase 2.2 golden case pinned by hand, stated as kernel invariants: the LOCO fold for
nasa/B0018 scores exactly that cell's QC-passed cycles, the label never reaches the estimator,
predictions stay in range, and the per-cycle contributions reconstruct the fold MAE.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sohkernel import api
from sohkernel.estimators.baselines import RidgeEstimator
from sohkernel.records import candidate_id, prediction_evidence_rows, validate_prediction_record

DATA_DIR = Path(os.environ.get("SOHBENCH_DATA_DIR") or Path(__file__).resolve().parents[2] / "data")
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "processed" / "nasa_features.pkl").is_file(),
    reason="real NASA data absent (set SOHBENCH_DATA_DIR)",
)


class SpyRidge(RidgeEstimator):
    seen_columns: list[list[str]] = []

    def fit(self, X, y, meta=None):
        SpyRidge.seen_columns.append(list(X.columns))
        return super().fit(X, y, meta)


def test_nasa_b0018_loco_golden_invariants():
    table, meta, qc = api.load(["nasa"], data_dir=DATA_DIR)
    assert qc.drop_reasons["duplicate_cycle_identity"] == 0 and qc.drop_reasons["soh_out_of_range"] == 0
    b0018_rows = int((table["cell_id"].astype(str) == "B0018").sum())
    assert b0018_rows >= 100

    captured = {}

    def on_model_fitted(*, estimator, record, train_table, test_table, meta):
        if record["held_out_key"] == "nasa/B0018":
            captured.update(estimator=estimator, test_table=test_table)

    SpyRidge.seen_columns.clear()
    results = api.score(None, table, meta, ["loco"], [SpyRidge], ["full"], seeds=[0], on_model_fitted=on_model_fitted)
    assert results["status"].eq("ok").all()
    assert SpyRidge.seen_columns and all("soh" not in cols for cols in SpyRidge.seen_columns)  # target excluded from the predictor
    fold = results[(results["held_out_key"] == "nasa/B0018") & (results["metric"] == "mae")]
    assert len(fold) == 1 and np.isfinite(fold["value"].iloc[0]) and fold["value"].iloc[0] < 0.05

    test_table = captured["test_table"]
    assert len(test_table) == b0018_rows                       # the fold is exactly the cell's QC-passed cycles
    estimator = captured["estimator"]
    pred = estimator.predict(test_table.drop(columns=["soh"]), meta)
    assert np.all((pred >= 0.0) & (pred < 1.3))
    lower, upper = estimator.predict_interval(test_table.drop(columns=["soh"]), meta)
    rows, contributions = prediction_evidence_rows(
        test_table, pred, (lower, upper), job_id="golden", model_name="ridge", candidate_id=candidate_id("ridge"),
        model_artifact_id="", featureset="full", feature_columns=[c for c in test_table.columns if c not in ("dataset", "cell_id", "cycle_index", "soh")],
        dataset="nasa", dataset_snapshot_id="", split_assignment_id="", evaluation_scope="research_any",
    )
    for record in rows:
        validate_prediction_record(record)
    mae = float(np.mean(np.abs(pred - test_table["soh"].to_numpy())))
    assert sum(r["metric_contributions"]["mae_contribution"] for r in rows) == pytest.approx(mae)
    assert abs(mae - float(fold["value"].iloc[0])) < 1e-9
