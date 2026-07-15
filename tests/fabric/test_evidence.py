import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohbench.fabric.evidence import (
    validate_metric_records,
    validate_prediction_records,
    verify_evidence_pack,
    write_evidence_pack,
)


PREDICTION_COLUMNS = [
    "run_id",
    "candidate_id",
    "featureset",
    "protocol",
    "fold",
    "seed",
    "dataset",
    "cell_id",
    "cycle_index",
    "sample_id",
    "y_pred",
]

METRIC_COLUMNS = [
    "run_id",
    "candidate_id",
    "featureset",
    "protocol",
    "fold",
    "seed",
    "dataset",
    "metric",
    "metric_version",
    "value",
    "unit",
    "direction",
    "aggregation_scope",
    "sample_ids",
    "cell_ids",
]


def prediction_frame(**overrides) -> pd.DataFrame:
    row = {
        "run_id": "RUN-one",
        "candidate_id": "ridge",
        "featureset": "full",
        "protocol": "loco",
        "fold": np.int64(0),
        "seed": np.int64(7),
        "dataset": "nasa",
        "cell_id": "B0005",
        "cycle_index": np.int64(12),
        "sample_id": "nasa:B0005:12",
        "y_pred": np.float64(0.91),
    }
    row.update(overrides)
    return pd.DataFrame([row])


def metric_frame(**overrides) -> pd.DataFrame:
    row = {
        "run_id": "RUN-one",
        "candidate_id": "ridge",
        "featureset": "full",
        "protocol": "loco",
        "fold": np.int64(0),
        "seed": np.int64(7),
        "dataset": "nasa",
        "metric": "mae",
        "metric_version": "1.0.0",
        "value": np.float64(0.03),
        "unit": "soh_fraction",
        "direction": "minimize",
        "aggregation_scope": "asset",
        "sample_ids": ["nasa:B0005:12"],
        "cell_ids": ["B0005"],
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_evidence_pack_verifies_and_detects_tamper(tmp_path):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    result = run_dir / "results.json"
    result.write_text('{"ok":true}', encoding="utf-8")

    path = write_evidence_pack(run_dir, {"run_id": "RUN-one"}, [result])

    verified = verify_evidence_pack(path)
    assert verified["ok"] is True
    assert verified["schema_version"] == "1.0.0"
    assert verified["run_id"] == "RUN-one"
    assert verified["artifacts"] == [
        {
            "path": "results.json",
            "size": len(b'{"ok":true}'),
            "sha256": "4062edaf750fb8074e7e83e0c9028c94e32468a8b6f1614774328ef045150f93",
        }
    ]

    result.write_text('{"ok":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|digest mismatch"):
        verify_evidence_pack(path)


def test_evidence_artifacts_are_sorted_by_relative_posix_path(tmp_path):
    run_dir = tmp_path / "RUN-one"
    nested = run_dir / "nested"
    nested.mkdir(parents=True)
    second = run_dir / "z.txt"
    first = nested / "a.txt"
    second.write_bytes(b"z")
    first.write_bytes(b"a")

    pack = write_evidence_pack(run_dir, {"run_id": "RUN-one"}, [second, first])

    document = json.loads(pack.read_text(encoding="utf-8"))
    assert [entry["path"] for entry in document["artifacts"]] == [
        "nested/a.txt",
        "z.txt",
    ]


@pytest.mark.parametrize(
    "artifact_factory, message",
    [
        (lambda run_dir, tmp_path: tmp_path / "outside.txt", "outside run_dir"),
        (lambda run_dir, tmp_path: run_dir / "missing.txt", "missing"),
        (lambda run_dir, tmp_path: run_dir / "directory", "regular file"),
        (lambda run_dir, tmp_path: run_dir / "evidence-pack.json", "reserved"),
        (lambda run_dir, tmp_path: run_dir / "evidence-pack.json.tmp", "reserved"),
    ],
)
def test_write_evidence_pack_rejects_invalid_artifacts(
    tmp_path, artifact_factory, message,
):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    (run_dir / "directory").mkdir()
    (run_dir / "evidence-pack.json").write_text("reserved", encoding="utf-8")
    (run_dir / "evidence-pack.json.tmp").write_text("reserved", encoding="utf-8")

    artifact = artifact_factory(run_dir, tmp_path)
    with pytest.raises(ValueError, match=message):
        write_evidence_pack(run_dir, {"run_id": "RUN-one"}, [artifact])


def test_write_evidence_pack_rejects_duplicate_paths(tmp_path):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    artifact = run_dir / "result.txt"
    artifact.write_text("result", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate artifact path"):
        write_evidence_pack(
            run_dir,
            {"run_id": "RUN-one"},
            [artifact, Path("result.txt")],
        )


@pytest.mark.parametrize("reserved", ["schema_version", "artifacts"])
def test_manifest_cannot_override_reserved_pack_fields(tmp_path, reserved):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    artifact = run_dir / "result.txt"
    artifact.write_text("result", encoding="utf-8")

    with pytest.raises(ValueError, match=reserved):
        write_evidence_pack(
            run_dir,
            {"run_id": "RUN-one", reserved: "override"},
            [artifact],
        )


@pytest.mark.parametrize(
    "artifact_path",
    ["../outside.txt", "/absolute.txt", "C:/absolute.txt", "nested/../../outside.txt"],
)
def test_verification_schema_rejects_absolute_and_traversal_paths(
    tmp_path, artifact_path,
):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    pack = run_dir / "evidence-pack.json"
    pack.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "RUN-one",
                "artifacts": [
                    {"path": artifact_path, "size": 0, "sha256": "a" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="path"):
        verify_evidence_pack(pack)


def test_verification_rejects_duplicate_paths_before_reading_artifacts(tmp_path):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    pack = run_dir / "evidence-pack.json"
    entry = {"path": "missing.txt", "size": 0, "sha256": "a" * 64}
    pack.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "RUN-one",
                "artifacts": [entry, {**entry, "sha256": "b" * 64}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate artifact path"):
        verify_evidence_pack(pack)


def test_verification_rejects_duplicate_normalized_paths(tmp_path):
    run_dir = tmp_path / "RUN-one"
    nested = run_dir / "nested"
    nested.mkdir(parents=True)
    artifact = nested / "result.txt"
    artifact.write_text("result", encoding="utf-8")
    digest = "f6a214f7a35e5168e42162a3da6b0d5b5f77d59f65e2c0f5d16f49a016b7b7b5"
    pack = run_dir / "evidence-pack.json"
    pack.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "RUN-one",
                "artifacts": [
                    {"path": "nested/result.txt", "size": 6, "sha256": digest},
                    {"path": "nested//result.txt", "size": 6, "sha256": digest},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate artifact path"):
        verify_evidence_pack(pack)


def test_verification_reports_size_and_digest_mismatches_separately(tmp_path):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    artifact = run_dir / "result.txt"
    artifact.write_text("result", encoding="utf-8")
    pack = write_evidence_pack(run_dir, {"run_id": "RUN-one"}, [artifact])
    document = json.loads(pack.read_text(encoding="utf-8"))

    document["artifacts"][0]["size"] += 1
    pack.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_evidence_pack(pack)

    document["artifacts"][0]["size"] -= 1
    document["artifacts"][0]["sha256"] = "a" * 64
    pack.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_evidence_pack(pack)


def test_prediction_records_require_exact_core_columns():
    assert list(prediction_frame().columns) == PREDICTION_COLUMNS
    records = prediction_frame().drop(columns=["sample_id"])

    with pytest.raises(ValueError, match="sample_id"):
        validate_prediction_records(records)


def test_prediction_duplicate_identity_includes_execution_and_sample():
    first = prediction_frame()
    same_sample_other_execution = prediction_frame(run_id="RUN-two")
    validate_prediction_records(pd.concat([first, same_sample_other_execution]))

    with pytest.raises(ValueError, match="duplicate prediction execution/sample key"):
        validate_prediction_records(pd.concat([first, first], ignore_index=True))


@pytest.mark.parametrize(
    "field, invalid",
    [
        ("fold", True),
        ("seed", "7"),
        ("cycle_index", np.datetime64("2026-01-01")),
        ("y_pred", np.nan),
        ("y_pred", np.inf),
        ("y_pred", 1 + 2j),
        ("y_pred", "0.91"),
    ],
)
def test_prediction_required_numeric_fields_are_finite_real_scalars(field, invalid):
    with pytest.raises(ValueError, match=field):
        validate_prediction_records(prediction_frame(**{field: invalid}))


@pytest.mark.parametrize(
    "field, invalid",
    [
        ("y_true", False),
        ("y_true", np.nan),
        ("interval_low", -np.inf),
        ("interval_high", 1 + 0j),
        ("predict_ms", "13.4"),
    ],
)
def test_prediction_optional_numeric_fields_are_finite_when_non_null(field, invalid):
    with pytest.raises(ValueError, match=field):
        validate_prediction_records(prediction_frame(**{field: invalid}))

    validate_prediction_records(prediction_frame(**{field: None}))


def test_prediction_intervals_are_ordered_and_optional_metadata_is_extensible():
    validate_prediction_records(
        prediction_frame(
            interval_low=0.8,
            interval_high=0.9,
            abstained=False,
            status="ok",
            future_context={"worker": "metric-1"},
        )
    )

    with pytest.raises(ValueError, match="interval_low.*interval_high"):
        validate_prediction_records(
            prediction_frame(interval_low=0.92, interval_high=0.90)
        )


def test_metric_records_require_exact_semantic_and_reference_columns():
    assert list(metric_frame().columns) == METRIC_COLUMNS
    for missing in ("metric_version", "aggregation_scope", "sample_ids", "cell_ids"):
        with pytest.raises(ValueError, match=missing):
            validate_metric_records(metric_frame().drop(columns=[missing]))


def test_metric_records_validate_identity_duplicates_and_extensible_context():
    first = metric_frame(extra_context={"bootstrap": "asset"})
    other_metric = metric_frame(metric="rmse")
    validate_metric_records(pd.concat([first, other_metric], ignore_index=True))

    with pytest.raises(ValueError, match="duplicate metric execution key"):
        validate_metric_records(pd.concat([first, first], ignore_index=True))


@pytest.mark.parametrize(
    "field, invalid",
    [
        ("fold", False),
        ("seed", np.datetime64("2026-01-01")),
        ("value", np.nan),
        ("value", -np.inf),
        ("value", 2 + 0j),
        ("value", "0.03"),
    ],
)
def test_metric_required_numeric_fields_are_finite_real_scalars(field, invalid):
    with pytest.raises(ValueError, match=field):
        validate_metric_records(metric_frame(**{field: invalid}))


def test_metric_optional_intervals_are_finite_ordered_or_null():
    validate_metric_records(metric_frame(interval_low=None, interval_high=None))
    with pytest.raises(ValueError, match="interval_low"):
        validate_metric_records(metric_frame(interval_low=np.nan))
    with pytest.raises(ValueError, match="interval_low"):
        validate_metric_records(metric_frame(interval_low=np.inf))
    with pytest.raises(ValueError, match="interval_low.*interval_high"):
        validate_metric_records(metric_frame(interval_low=0.04, interval_high=0.03))


def test_record_schemas_are_packaged_and_keep_future_context_extensible():
    for name in (
        "evidence-pack.schema.json",
        "prediction-record.schema.json",
        "metric-record.schema.json",
    ):
        schema = files("sohbench.fabric.schemas").joinpath(name)
        assert schema.is_file()
        assert '"additionalProperties": true' in schema.read_text(encoding="utf-8")
