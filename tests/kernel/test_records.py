"""Prediction records (Task 8): identity derivations, the record shape and its validation."""
import numpy as np
import pandas as pd
import pytest

from sohkernel.records import (
    METRIC_CONTRIBUTION_FIELDS,
    PREDICTION_RECORD_FIELDS,
    candidate_id,
    failed_record,
    id_token,
    json_number,
    prediction_evidence_rows,
    split_assignment_id,
    validate_prediction_record,
)


def test_identity_derivations_are_deterministic_and_sanitised():
    assert candidate_id("model@sha256:abc") == "artifact:model-sha256-abc"
    assert candidate_id("") == "artifact:unknown" and candidate_id(None) == "artifact:unknown"  # the service maps "" -> "model" before deriving
    assert id_token("a/b c", 3) == "a-b" and id_token("", 8) == "unknown"
    sid = split_assignment_id({"protocol": "loco", "held_out_key": "hust/c1", "seed": 0})
    assert sid.startswith("protocol@sha256:") and sid == split_assignment_id({"seed": 0, "held_out_key": "hust/c1", "protocol": "loco"})
    assert split_assignment_id({}) == "" and split_assignment_id({"protocol": ""}) == ""
    assert json_number("3") == 3 and json_number(2.5) == 2.5 and json_number(float("nan")) is None and json_number("x") is None


def test_prediction_rows_carry_the_full_record_shape_and_contributions():
    group = pd.DataFrame({"dataset": ["d"] * 4, "cell_id": ["c1"] * 4, "cycle_index": [1, 2, 3, 4],
                          "soh": [1.0, 0.99, 0.98, 0.97], "f1": [0.1, 0.2, 0.3, 0.4]})
    pred = np.array([1.0, 0.98, 0.98, 0.96])
    interval = (pred - 0.05, pred + 0.05)
    rows, contributions = prediction_evidence_rows(
        group, pred, interval, job_id="job-1", model_name="m", candidate_id="artifact:m",
        model_artifact_id="model@sha256:x", featureset="voltage", feature_columns=["f1"],
        dataset="d", dataset_snapshot_id="dataset@sha256:y", split_assignment_id="", evaluation_scope="research_any",
        calibration_mask=np.array([True, False, False, False]),
    )
    assert len(rows) == len(contributions) == 4
    for record in rows:
        validate_prediction_record(record)
        assert set(record) >= set(PREDICTION_RECORD_FIELDS)
        assert set(record["metric_contributions"]) == set(METRIC_CONTRIBUTION_FIELDS)
    assert rows[0]["calibration_sample"] is True and rows[0]["metric_contributions"]["mae_contribution"] is None
    assert "calibration_sample" not in rows[1]
    # contributions normalise over the three evaluated rows and sum to the MAE
    mae = np.mean(np.abs(pred[1:] - group["soh"].to_numpy()[1:]))
    assert sum(r["metric_contributions"]["mae_contribution"] for r in rows[1:]) == pytest.approx(mae)
    assert rows[1]["interval_covered"] is True and rows[1]["feature_values"] == {"f1": 0.2}
    assert rows[1]["prediction_record_id"].startswith("pred:job-1:")


def test_validate_prediction_record_rejects_missing_or_foreign_fields():
    with pytest.raises(ValueError, match="missing fields"):
        validate_prediction_record({"schema_version": "1.0.0"})
    group = pd.DataFrame({"dataset": ["d"], "cell_id": ["c"], "cycle_index": [1], "soh": [1.0]})
    (record,), _ = prediction_evidence_rows(group, np.array([1.0]), None, job_id="", model_name="m", candidate_id="artifact:m",
                                            model_artifact_id="", featureset="full", feature_columns=[], dataset="d",
                                            dataset_snapshot_id="", split_assignment_id="", evaluation_scope="research_any")
    assert record["lower"] is None and record["interval_covered"] is None
    bad = dict(record); bad["metric_contributions"] = {"picp_contribution": None, "bogus": 1.0}
    with pytest.raises(ValueError, match="unknown metric contributions"):
        validate_prediction_record(bad)


def test_failed_record_shape():
    record = failed_record("m", "d", 12, 3.5, "boom")
    assert record["status"] == "failed" and record["error"] == "boom" and record["mae"] is None
    assert record["n_samples"] == 12 and record["load_ms"] == 3.5
