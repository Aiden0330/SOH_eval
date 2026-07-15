import json
import os
from importlib.resources import files
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest

import sohbench.fabric.evidence as evidence_module
from sohbench.fabric.evidence import (
    validate_metric_records,
    validate_prediction_records,
    verify_evidence_pack,
    write_evidence_pack,
)
from sohbench.fabric.ids import sha256_file


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
    assert verify_evidence_pack(pack)["ok"] is True


@pytest.mark.skipif(
    os.path.normcase("A") != os.path.normcase("a"),
    reason="reserved aliases collide only on case-insensitive filesystems",
)
@pytest.mark.parametrize("reserved_alias", ["EVIDENCE-PACK.JSON", "Evidence-Pack.Json.Tmp"])
def test_writer_rejects_case_variant_reserved_artifact_paths(
    tmp_path, reserved_alias,
):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    artifact = run_dir / reserved_alias
    artifact.write_text("sentinel", encoding="utf-8")

    with pytest.raises(ValueError, match="reserved artifact path"):
        write_evidence_pack(run_dir, {"run_id": "RUN-one"}, [artifact])

    assert artifact.read_text(encoding="utf-8") == "sentinel"


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


def test_write_evidence_pack_refuses_preexisting_temp_hard_link(tmp_path):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    artifact = run_dir / "result.txt"
    artifact.write_text("result", encoding="utf-8")
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_bytes(b"external sentinel")
    temporary = run_dir / "evidence-pack.json.tmp"
    os.link(sentinel, temporary)

    with pytest.raises(FileExistsError):
        write_evidence_pack(run_dir, {"run_id": "RUN-one"}, [artifact])

    assert sentinel.read_bytes() == b"external sentinel"
    assert temporary.read_bytes() == b"external sentinel"
    assert os.path.samefile(sentinel, temporary)


def test_write_evidence_pack_refuses_preexisting_temp_symlink(tmp_path):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    artifact = run_dir / "result.txt"
    artifact.write_text("result", encoding="utf-8")
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_bytes(b"external sentinel")
    temporary = run_dir / "evidence-pack.json.tmp"
    try:
        temporary.symlink_to(sentinel)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(FileExistsError):
        write_evidence_pack(run_dir, {"run_id": "RUN-one"}, [artifact])

    assert sentinel.read_bytes() == b"external sentinel"
    assert temporary.is_symlink()


def test_write_evidence_pack_removes_only_its_temp_after_write_failure(
    tmp_path, monkeypatch,
):
    run_dir = tmp_path / "RUN-one"
    run_dir.mkdir()
    artifact = run_dir / "result.txt"
    artifact.write_text("result", encoding="utf-8")
    temporary = run_dir / "evidence-pack.json.tmp"

    def fail_dump(*args, **kwargs):
        raise RuntimeError("injected dump failure")

    monkeypatch.setattr(evidence_module.json, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="injected dump failure"):
        write_evidence_pack(run_dir, {"run_id": "RUN-one"}, [artifact])

    assert not temporary.exists()
    assert not (run_dir / "evidence-pack.json").exists()


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

    with pytest.raises(ValueError, match="path"):
        verify_evidence_pack(pack)


def test_verification_rejects_standalone_noncanonical_logical_path(
    tmp_path, monkeypatch,
):
    run_dir = tmp_path / "RUN-one"
    nested = run_dir / "nested"
    nested.mkdir(parents=True)
    artifact = nested / "result.txt"
    artifact.write_text("result", encoding="utf-8")
    pack = run_dir / "evidence-pack.json"
    pack.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "RUN-one",
                "artifacts": [
                    {
                        "path": "nested//result.txt",
                        "size": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evidence_module,
        "sha256_file",
        lambda path: pytest.fail("verifier read bytes before rejecting path"),
    )
    with pytest.raises(ValueError, match="path"):
        verify_evidence_pack(pack)


@pytest.mark.parametrize(
    "reserved_alias",
    ["EVIDENCE-PACK.JSON", "Evidence-Pack.Json.Tmp"],
)
def test_verification_rejects_reserved_case_aliases_before_artifact_reads(
    tmp_path, monkeypatch, reserved_alias,
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
                    {"path": reserved_alias, "size": 0, "sha256": "a" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evidence_module,
        "sha256_file",
        lambda path: pytest.fail("verifier read bytes before rejecting path"),
    )
    with pytest.raises(ValueError, match="path|reserved"):
        verify_evidence_pack(pack)


def test_evidence_schema_rejects_noncanonical_and_reserved_alias_paths():
    schema = json.loads(
        files("sohbench.fabric.schemas")
        .joinpath("evidence-pack.schema.json")
        .read_text(encoding="utf-8")
    )
    for invalid_path in (
        "nested//result.txt",
        "EVIDENCE-PACK.JSON",
        "Evidence-Pack.Json.Tmp",
    ):
        document = {
            "schema_version": "1.0.0",
            "run_id": "RUN-one",
            "artifacts": [
                {"path": invalid_path, "size": 0, "sha256": "a" * 64}
            ],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(document, schema)


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

    nullable = (
        {"interval_low": None, "interval_high": None}
        if field in ("interval_low", "interval_high")
        else {field: None}
    )
    validate_prediction_records(prediction_frame(**nullable))


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


@pytest.mark.parametrize(
    "validator, frame_factory",
    [
        (validate_prediction_records, prediction_frame),
        (validate_metric_records, metric_frame),
    ],
)
@pytest.mark.parametrize(
    "overrides",
    [
        {"interval_low": 0.8},
        {"interval_high": 0.9},
        {"interval_low": None, "interval_high": 0.9},
        {"interval_low": 0.8, "interval_high": None},
    ],
)
def test_interval_bounds_must_be_complete_pairs(validator, frame_factory, overrides):
    with pytest.raises(ValueError, match="interval_low.*interval_high|interval pair"):
        validator(frame_factory(**overrides))


@pytest.mark.parametrize(
    "validator, frame_factory",
    [
        (validate_prediction_records, prediction_frame),
        (validate_metric_records, metric_frame),
    ],
)
def test_intervals_allow_both_absent_both_null_or_ordered_finite_pair(
    validator, frame_factory,
):
    validator(frame_factory())
    validator(frame_factory(interval_low=None, interval_high=None))
    validator(frame_factory(interval_low=0.8, interval_high=0.9))


@pytest.mark.parametrize(
    "schema_name, frame_factory",
    [
        ("prediction-record.schema.json", prediction_frame),
        ("metric-record.schema.json", metric_frame),
    ],
)
@pytest.mark.parametrize(
    "overrides",
    [
        {"interval_low": 0.8},
        {"interval_high": 0.9},
        {"interval_low": None, "interval_high": 0.9},
        {"interval_low": 0.8, "interval_high": None},
    ],
)
def test_record_schemas_reject_half_defined_interval_pairs(
    schema_name, frame_factory, overrides,
):
    schema = json.loads(
        files("sohbench.fabric.schemas")
        .joinpath(schema_name)
        .read_text(encoding="utf-8")
    )
    record = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in frame_factory(**overrides).iloc[0].items()
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(record, schema)


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


def test_metric_identity_treats_contributing_references_as_sets():
    first = metric_frame(
        value=0.03,
        sample_ids=["nasa:B0005:12", "nasa:B0006:12"],
        cell_ids=["B0005", "B0006"],
    )
    reordered = metric_frame(
        value=0.04,
        sample_ids=["nasa:B0006:12", "nasa:B0005:12"],
        cell_ids=["B0006", "B0005"],
    )

    with pytest.raises(ValueError, match="duplicate metric execution key"):
        validate_metric_records(pd.concat([first, reordered], ignore_index=True))


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


def test_row_validation_builds_one_validator_for_the_frame(monkeypatch):
    records = pd.concat(
        [prediction_frame(), prediction_frame(sample_id="nasa:B0005:13", cycle_index=13)],
        ignore_index=True,
    )
    constructed = []
    validated = []

    class SpyValidator:
        def __init__(self, schema):
            constructed.append(schema)

        def validate(self, row):
            validated.append(row)

    monkeypatch.setattr(
        evidence_module.jsonschema.validators,
        "validator_for",
        lambda schema: SpyValidator,
    )
    validate_prediction_records(records)

    assert len(constructed) == 1
    assert len(validated) == 2


def test_row_validation_streams_rows_without_materializing_records(monkeypatch):
    def fail_to_dict(*args, **kwargs):
        raise AssertionError("frame.to_dict must not materialize all records")

    monkeypatch.setattr(pd.DataFrame, "to_dict", fail_to_dict)
    validate_prediction_records(prediction_frame())


def test_row_validation_keeps_row_number_in_schema_errors():
    records = pd.concat(
        [prediction_frame(), prediction_frame(sample_id="", cycle_index=13)],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="row 1 sample_id"):
        validate_prediction_records(records)


def test_record_schemas_are_packaged_and_keep_future_context_extensible():
    for name in (
        "evidence-pack.schema.json",
        "prediction-record.schema.json",
        "metric-record.schema.json",
    ):
        schema = files("sohbench.fabric.schemas").joinpath(name)
        assert schema.is_file()
        assert '"additionalProperties": true' in schema.read_text(encoding="utf-8")
