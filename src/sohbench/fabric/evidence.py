from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pandas as pd

from .ids import sha256_file
from .outcomes import is_finite_real_scalar


EVIDENCE_SCHEMA_VERSION = "1.0.0"
EVIDENCE_PACK_NAME = "evidence-pack.json"
EVIDENCE_TEMP_NAME = "evidence-pack.json.tmp"

PREDICTION_REQUIRED_COLUMNS = (
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
)
PREDICTION_IDENTITY_COLUMNS = PREDICTION_REQUIRED_COLUMNS[:-1]
PREDICTION_REQUIRED_NUMERIC_COLUMNS = ("fold", "seed", "cycle_index", "y_pred")
PREDICTION_OPTIONAL_NUMERIC_COLUMNS = (
    "y_true",
    "interval_low",
    "interval_high",
    "predict_ms",
)

METRIC_REQUIRED_COLUMNS = (
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
)
METRIC_REQUIRED_NUMERIC_COLUMNS = ("fold", "seed", "value")
METRIC_OPTIONAL_NUMERIC_COLUMNS = (
    "interval_low",
    "interval_high",
    "standard_error",
    "confidence",
    "n_samples",
    "n_assets",
)
METRIC_IDENTITY_COLUMNS = (
    "run_id",
    "candidate_id",
    "featureset",
    "protocol",
    "fold",
    "seed",
    "dataset",
    "metric",
    "metric_version",
    "unit",
    "direction",
    "aggregation_scope",
    "sample_ids",
    "cell_ids",
)

_RESERVED_ARTIFACT_PATHS = frozenset((EVIDENCE_PACK_NAME, EVIDENCE_TEMP_NAME))
_RESERVED_MANIFEST_FIELDS = frozenset(("schema_version", "artifacts"))


def _load_schema(name: str) -> dict:
    schema_path = files("sohbench.fabric.schemas").joinpath(name)
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _validate_document(document: dict, schema_name: str, root: str) -> None:
    try:
        jsonschema.validate(document, _load_schema(schema_name))
    except jsonschema.ValidationError as exc:
        location = ".".join(map(str, exc.absolute_path)) or root
        raise ValueError(f"{location}: {exc.message}") from exc


def validate_evidence_pack_document(document: dict) -> None:
    _validate_document(document, "evidence-pack.schema.json", "evidence_pack")


def _relative_artifact(run_root: Path, artifact_path: str | Path) -> tuple[Path, str]:
    path = Path(artifact_path)
    candidate = path if path.is_absolute() else run_root / path
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(run_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact path is outside run_dir: {artifact_path}") from exc
    if relative in _RESERVED_ARTIFACT_PATHS:
        raise ValueError(f"reserved artifact path: {relative}")
    if not resolved.exists():
        raise ValueError(f"artifact is missing: {relative}")
    if not resolved.is_file():
        raise ValueError(f"artifact is not a regular file: {relative}")
    return resolved, relative


def _artifact_entries(run_dir: Path, artifact_paths: list[Path]) -> list[dict]:
    resolved_entries: list[tuple[Path, str]] = []
    seen_paths: set[str] = set()
    seen_files: set[str] = set()
    for artifact_path in artifact_paths:
        resolved, relative = _relative_artifact(run_dir, artifact_path)
        path_key = os.path.normcase(relative)
        file_key = os.path.normcase(str(resolved))
        if path_key in seen_paths or file_key in seen_files:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen_paths.add(path_key)
        seen_files.add(file_key)
        resolved_entries.append((resolved, relative))

    return [
        {
            "path": relative,
            "size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
        for resolved, relative in sorted(resolved_entries, key=lambda item: item[1])
    ]


def write_evidence_pack(
    run_dir: Path, manifest: dict, artifact_paths: list[Path],
) -> Path:
    run_root = Path(run_dir).resolve()
    if not run_root.is_dir():
        raise ValueError(f"run_dir is not a directory: {run_dir}")
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dict")
    overridden = sorted(_RESERVED_MANIFEST_FIELDS.intersection(manifest))
    if overridden:
        raise ValueError(f"manifest cannot override reserved field: {overridden[0]}")

    document = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        **manifest,
        "artifacts": _artifact_entries(run_root, artifact_paths),
    }
    validate_evidence_pack_document(document)

    target = run_root / EVIDENCE_PACK_NAME
    temporary = run_root / EVIDENCE_TEMP_NAME
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            document,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    return target


def _verified_relative_path(run_root: Path, relative: str) -> Path:
    if relative in _RESERVED_ARTIFACT_PATHS:
        raise ValueError(f"reserved artifact path: {relative}")
    path = Path(relative)
    if path.is_absolute() or "\\" in relative or any(
        part in (".", "..") for part in path.parts
    ):
        raise ValueError(f"invalid artifact path: {relative}")
    resolved = (run_root / path).resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"artifact path is outside run_dir: {relative}") from exc
    return resolved


def verify_evidence_pack(path: str | Path) -> dict:
    pack_path = Path(path)
    try:
        document = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evidence pack: {pack_path}") from exc

    validate_evidence_pack_document(document)
    run_root = pack_path.resolve().parent
    seen_paths: set[str] = set()
    seen_files: set[str] = set()
    resolved_artifacts: list[tuple[dict, Path]] = []
    for entry in document["artifacts"]:
        relative = entry["path"]
        path_key = os.path.normcase(relative)
        if path_key in seen_paths:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen_paths.add(path_key)
        artifact = _verified_relative_path(run_root, relative)
        file_key = os.path.normcase(str(artifact))
        if file_key in seen_files:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen_files.add(file_key)
        resolved_artifacts.append((entry, artifact))

    for entry, artifact in resolved_artifacts:
        relative = entry["path"]
        if not artifact.exists():
            raise ValueError(f"artifact is missing: {relative}")
        if not artifact.is_file():
            raise ValueError(f"artifact is not a regular file: {relative}")
        actual_size = artifact.stat().st_size
        if actual_size != entry["size"]:
            raise ValueError(
                f"artifact size mismatch for {relative}: "
                f"expected {entry['size']}, got {actual_size}"
            )
        actual_digest = sha256_file(artifact)
        if actual_digest != entry["sha256"]:
            raise ValueError(f"artifact digest mismatch for {relative}")
    return {**document, "ok": True}


def _is_null(value: object) -> bool:
    return value is None or value is pd.NA


def _json_value(value: Any) -> Any:
    if _is_null(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _validate_numeric_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    optional: tuple[str, ...],
) -> None:
    for column in required:
        for value in frame[column]:
            if not is_finite_real_scalar(value):
                raise ValueError(f"{column} must contain finite real scalars")
    for column in optional:
        if column not in frame.columns:
            continue
        for value in frame[column]:
            if not _is_null(value) and not is_finite_real_scalar(value):
                raise ValueError(
                    f"{column} must contain null or finite real scalars"
                )


def _validate_intervals(frame: pd.DataFrame) -> None:
    if not {"interval_low", "interval_high"}.issubset(frame.columns):
        return
    for low, high in zip(frame["interval_low"], frame["interval_high"]):
        if _is_null(low) or _is_null(high):
            continue
        if low > high:
            raise ValueError("interval_low must be less than or equal to interval_high")


def _validate_rows(frame: pd.DataFrame, schema_name: str, root: str) -> None:
    schema = _load_schema(schema_name)
    for row_number, row in enumerate(frame.to_dict(orient="records")):
        try:
            jsonschema.validate(_json_value(row), schema)
        except jsonschema.ValidationError as exc:
            location = ".".join(map(str, exc.absolute_path)) or root
            raise ValueError(f"row {row_number} {location}: {exc.message}") from exc


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], kind: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{kind} records must be a pandas DataFrame")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"missing {kind} fields: {', '.join(missing)}")


def validate_prediction_records(frame: pd.DataFrame) -> None:
    _require_columns(frame, PREDICTION_REQUIRED_COLUMNS, "prediction")
    _validate_numeric_columns(
        frame,
        PREDICTION_REQUIRED_NUMERIC_COLUMNS,
        PREDICTION_OPTIONAL_NUMERIC_COLUMNS,
    )
    _validate_intervals(frame)
    _validate_rows(frame, "prediction-record.schema.json", "prediction_record")
    if frame.duplicated(list(PREDICTION_IDENTITY_COLUMNS)).any():
        raise ValueError("duplicate prediction execution/sample key")


def _metric_identity(row: dict) -> tuple:
    return tuple(
        tuple(row[column]) if column in ("sample_ids", "cell_ids") else row[column]
        for column in METRIC_IDENTITY_COLUMNS
    )


def validate_metric_records(frame: pd.DataFrame) -> None:
    _require_columns(frame, METRIC_REQUIRED_COLUMNS, "metric")
    _validate_numeric_columns(
        frame,
        METRIC_REQUIRED_NUMERIC_COLUMNS,
        METRIC_OPTIONAL_NUMERIC_COLUMNS,
    )
    _validate_intervals(frame)
    _validate_rows(frame, "metric-record.schema.json", "metric_record")

    seen: set[tuple] = set()
    for row in frame.to_dict(orient="records"):
        identity = _metric_identity(row)
        if identity in seen:
            raise ValueError("duplicate metric execution key")
        seen.add(identity)
