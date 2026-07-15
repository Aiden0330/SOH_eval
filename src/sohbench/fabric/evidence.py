from __future__ import annotations

import json
import math
import os
import re
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
_RESERVED_ARTIFACT_KEYS = frozenset(
    os.path.normcase(path) for path in _RESERVED_ARTIFACT_PATHS
)
_RESERVED_MANIFEST_FIELDS = frozenset(("schema_version", "artifacts"))
_WINDOWS_INVALID_PATH_CHARS = frozenset('<>:"|?*')
_DOS_DEVICE_NAME = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9\u00b9\u00b2\u00b3]|LPT[1-9\u00b9\u00b2\u00b3])"
    r"(?:\..*)?$",
    re.IGNORECASE,
)


def _load_schema(name: str) -> dict:
    schema_path = files("sohbench.fabric.schemas").joinpath(name)
    return _strict_json_loads(schema_path.read_text(encoding="utf-8"))


def _reject_json_constant(constant: str):
    raise ValueError(f"nonfinite JSON constant is not permitted: {constant}")


def _strict_json_loads(serialized: str):
    return json.loads(serialized, parse_constant=_reject_json_constant)


_JSON_MAX_DEPTH = 100
_JSON_MAX_NODES = 100_000


def _json_compatible_value(
    value: Any,
    path: str = "document",
    *,
    _state: dict | None = None,
    _depth: int = 0,
) -> Any:
    state = {"nodes": 0, "active": set()} if _state is None else _state
    state["nodes"] += 1
    if state["nodes"] > _JSON_MAX_NODES:
        raise ValueError(f"{path} exceeds the JSON container size limit")
    if _depth > _JSON_MAX_DEPTH:
        raise ValueError(f"{path} exceeds the JSON container depth limit")

    if _is_null(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite NumPy floating value")
        binary64_max = np.longdouble(np.finfo(np.float64).max)
        if value < -binary64_max or value > binary64_max:
            raise ValueError(f"{path} exceeds the finite JSON binary64 range")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{path} exceeds the finite JSON binary64 range")
        return converted
    if isinstance(value, np.generic):
        item = value.item()
        if type(item) is type(value):
            raise ValueError(
                f"{path} cannot normalize NumPy scalar type {type(value).__name__}"
            )
        return _json_compatible_value(
            item, path, _state=state, _depth=_depth + 1
        )
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite JSON number")
        return value
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        identity = id(value)
        if identity in state["active"]:
            raise ValueError(f"{path} contains a JSON container cycle")
        state["active"].add(identity)
        try:
            normalized = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{path} contains a non-string JSON object key")
                normalized[key] = _json_compatible_value(
                    item,
                    f"{path}.{key}",
                    _state=state,
                    _depth=_depth + 1,
                )
            return normalized
        finally:
            state["active"].remove(identity)

    if isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        identity = id(value)
        if identity in state["active"]:
            raise ValueError(f"{path} contains a JSON container cycle")
        state["active"].add(identity)
        try:
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return _json_compatible_value(
                        value.item(),
                        path,
                        _state=state,
                        _depth=_depth + 1,
                    )
                items = value.tolist()
            elif isinstance(value, (pd.Series, pd.Index)):
                items = value.tolist()
            else:
                items = value
            return [
                _json_compatible_value(
                    item,
                    f"{path}.{index}",
                    _state=state,
                    _depth=_depth + 1,
                )
                for index, item in enumerate(items)
            ]
        finally:
            state["active"].remove(identity)

    raise ValueError(
        f"{path} contains unsupported non-JSON value type "
        f"{type(value).__name__}"
    )


def _validate_document(document: dict, schema_name: str, root: str) -> dict:
    normalized = _json_compatible_value(document, root)
    try:
        jsonschema.validate(normalized, _load_schema(schema_name))
    except jsonschema.ValidationError as exc:
        location = ".".join(map(str, exc.absolute_path)) or root
        raise ValueError(f"{location}: {exc.message}") from exc
    return normalized


def validate_evidence_pack_document(document: dict) -> None:
    _validate_document(document, "evidence-pack.schema.json", "evidence_pack")


def _validate_portable_path_segment(segment: str, field: str) -> None:
    if (
        not segment
        or segment in (".", "..")
        or segment.endswith((".", " "))
        or any(
            ord(character) <= 0x1F or character in _WINDOWS_INVALID_PATH_CHARS
            for character in segment
        )
        or _DOS_DEVICE_NAME.fullmatch(segment)
    ):
        raise ValueError(f"invalid {field}: {segment}")


def _validate_logical_artifact_path(relative: str) -> None:
    if os.path.normcase(relative) in _RESERVED_ARTIFACT_KEYS:
        raise ValueError(f"reserved artifact path: {relative}")
    path = Path(relative)
    segments = relative.split("/")
    if (
        path.is_absolute()
        or re.match(r"^[A-Za-z]:", relative)
        or "\\" in relative
        or any(segment == "" for segment in segments)
    ):
        raise ValueError(f"invalid artifact path: {relative}")
    for segment in segments:
        _validate_portable_path_segment(segment, "artifact path")


def _relative_artifact(run_root: Path, artifact_path: str | Path) -> tuple[Path, str]:
    path = Path(artifact_path)
    candidate = path if path.is_absolute() else run_root / path
    try:
        lexical_relative = candidate.relative_to(run_root).as_posix()
    except ValueError:
        lexical_relative = None
    if lexical_relative is not None:
        _validate_logical_artifact_path(lexical_relative)
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(run_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact path is outside run_dir: {artifact_path}") from exc
    _validate_logical_artifact_path(relative)
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


def _remove_temporary_if_owned(temporary: Path, identity: tuple | None) -> None:
    try:
        current = temporary.lstat()
    except FileNotFoundError:
        return
    current_identity = (current.st_dev, current.st_ino)
    if identity is None or current_identity == identity:
        temporary.unlink()


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
    document = _validate_document(
        document, "evidence-pack.schema.json", "evidence_pack"
    )

    target = run_root / EVIDENCE_PACK_NAME
    temporary = run_root / EVIDENCE_TEMP_NAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(temporary, flags, 0o600)
    descriptor_owned = True
    identity = None
    try:
        descriptor_stat = os.fstat(descriptor)
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        stream = os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        )
        descriptor_owned = False
        with stream:
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        current = temporary.lstat()
        if (current.st_dev, current.st_ino) != identity:
            raise RuntimeError("evidence pack temporary path changed during write")
        temporary.replace(target)
    except BaseException:
        if descriptor_owned:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _remove_temporary_if_owned(temporary, identity)
        raise
    return target


def _verified_relative_path(run_root: Path, relative: str) -> Path:
    _validate_logical_artifact_path(relative)
    path = Path(relative)
    resolved = (run_root / path).resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"artifact path is outside run_dir: {relative}") from exc
    return resolved


def verify_evidence_pack(path: str | Path) -> dict:
    pack_path = Path(path)
    try:
        document = _strict_json_loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
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
    has_low = "interval_low" in frame.columns
    has_high = "interval_high" in frame.columns
    if has_low != has_high:
        raise ValueError("interval_low and interval_high must be present together")
    if not has_low:
        return
    for low, high in zip(frame["interval_low"], frame["interval_high"]):
        low_is_null = _is_null(low)
        high_is_null = _is_null(high)
        if low_is_null != high_is_null:
            raise ValueError("interval_low and interval_high must form an interval pair")
        if low_is_null:
            continue
        if low > high:
            raise ValueError("interval_low must be less than or equal to interval_high")


def _validate_rows(frame: pd.DataFrame, schema_name: str, root: str) -> None:
    schema = _load_schema(schema_name)
    validator_class = jsonschema.validators.validator_for(schema)
    validator = validator_class(schema)
    columns = tuple(frame.columns)
    for row_number, values in enumerate(frame.itertuples(index=False, name=None)):
        try:
            row = _json_compatible_value(
                dict(zip(columns, values)), root
            )
        except ValueError as exc:
            raise ValueError(f"row {row_number} {exc}") from exc
        try:
            validator.validate(row)
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
        tuple(sorted(row[column]))
        if column in ("sample_ids", "cell_ids")
        else row[column]
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
    columns = tuple(frame.columns)
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(columns, values))
        identity = _metric_identity(row)
        if identity in seen:
            raise ValueError("duplicate metric execution key")
        seen.add(identity)
