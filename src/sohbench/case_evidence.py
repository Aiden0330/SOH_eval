from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from .evaluation_cases import (
    EvaluationCaseValidationError,
    normalize_evidence_coverage,
    normalize_evidence_selection,
    normalize_verdict,
)


CRITERION_SETS = {
    "strict": {
        "mae_max": 0.02,
        "picp_min": 0.90,
        "monotonicity_max": 0.001,
        "value_range_max": 0.0,
    },
    "engineering": {
        "mae_max": 0.08,
        "picp_min": 0.85,
        "monotonicity_max": 0.15,
        "value_range_max": 0.0,
    },
    "exploratory": {
        "mae_max": 0.12,
        "picp_min": 0.80,
        "monotonicity_max": 0.25,
        "value_range_max": 0.0,
    },
}

CRITERION_THRESHOLD_KEYS = (
    "mae_max",
    "picp_min",
    "monotonicity_max",
    "value_range_max",
)


def criterion_base_id(criterion_set_id: str) -> str:
    """Return the preset id behind an immutable custom criterion version."""
    return str(criterion_set_id or "strict").split("@custom-", 1)[0]


def resolve_criterion_thresholds(
    criterion_set_id: str,
    threshold_overrides: dict | None = None,
) -> tuple[str, dict, str]:
    """Validate case-scoped overrides and return base, thresholds and version."""
    base_id = criterion_base_id(criterion_set_id)
    if base_id not in CRITERION_SETS:
        raise EvaluationCaseValidationError(f"unknown criterion_set_id: {criterion_set_id}")
    thresholds = dict(CRITERION_SETS[base_id])
    if threshold_overrides is not None:
        if not isinstance(threshold_overrides, dict):
            raise EvaluationCaseValidationError("criterion_thresholds must be an object or null")
        unknown = sorted(set(threshold_overrides) - set(CRITERION_THRESHOLD_KEYS))
        if unknown:
            raise EvaluationCaseValidationError(
                f"unknown criterion threshold fields: {unknown}"
            )
        for key, value in threshold_overrides.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise EvaluationCaseValidationError(f"criterion threshold {key} must be finite")
            numeric = float(value)
            if numeric < 0 or numeric > 1:
                raise EvaluationCaseValidationError(
                    f"criterion threshold {key} must be between 0 and 1"
                )
            thresholds[key] = numeric
    if thresholds == CRITERION_SETS[base_id]:
        return base_id, thresholds, base_id
    digest = hashlib.sha256(
        json.dumps(thresholds, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return base_id, thresholds, f"{base_id}@custom-{digest}"

TRAJECTORY_INDEX_FILENAME = "prediction_trajectory_index.json"
TRAJECTORY_INDEX_SCHEMA = "prediction-trajectory-index/2.1"
TRAJECTORY_SHARD_DIRNAME = "prediction_trajectory_v2"
PREDICTION_LOOKUP_FILENAME = "prediction_evidence_lookup.sqlite3"
PREDICTION_LOOKUP_SCHEMA = "prediction-evidence-lookup/1.0"
_TRAJECTORY_INDEX_FIELDS = (
    "prediction_record_id", "sample_id", "dataset", "cell_id", "asset_id",
    "candidate_id", "model_artifact_id", "cycle_index", "cycle_id", "true_soh",
    "pred_soh", "lower", "upper", "absolute_error", "mae_contribution",
    "raw_curve_available",
)
_TRAJECTORY_PUBLIC_FIELDS = (
    "prediction_record_id", "sample_id", "cell_id", "asset_id", "cycle_index",
    "cycle_id", "true_soh", "pred_soh", "lower", "upper", "absolute_error",
    "mae_contribution", "raw_curve_available",
)


def _compact_trajectory_point(record: dict, *, public: bool = False) -> dict:
    fields = _TRAJECTORY_PUBLIC_FIELDS if public else _TRAJECTORY_INDEX_FIELDS
    point = {key: record[key] for key in fields if key in record}
    contribution = record.get("mae_contribution")
    if contribution is None:
        contribution = (record.get("metric_contributions") or {}).get("mae_contribution")
    if contribution is not None:
        point["mae_contribution"] = contribution
    if "raw_curve_available" not in point:
        point["raw_curve_available"] = bool(record.get("curve_segment_ref_id"))
    if public and point.get("prediction_record_id"):
        point.pop("sample_id", None)
    return point


def write_prediction_trajectory_index(
    artifact_dir: str | Path,
    records: list[dict] | None = None,
) -> Path:
    """Persist the small immutable projection used by the Page 3 trajectory view."""
    root = Path(artifact_dir)
    source_path = root / "prediction_records.json"
    if records is None:
        records = (_read_json(source_path, {}) or {}).get("records") or []
    source_stat = source_path.stat()
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    raw_curve_counts: dict[tuple[str, str, str, str], int] = {}
    for item in records:
        if str(item.get("validity_status") or "valid") == "invalidated":
            continue
        candidate_id = str(item.get("candidate_id") or "")
        artifact_id = str(item.get("model_artifact_id") or "")
        dataset = str(item.get("dataset") or "")
        asset_id = str(item.get("cell_id") or item.get("asset_id") or "")
        scope_key = (candidate_id, artifact_id, dataset, asset_id)
        grouped.setdefault(scope_key, []).append(_compact_trajectory_point(item))
        if item.get("curve_segment_ref_id"):
            raw_curve_counts[scope_key] = raw_curve_counts.get(scope_key, 0) + 1

    shard_root = root / TRAJECTORY_SHARD_DIRNAME
    shard_root.mkdir(parents=True, exist_ok=True)
    scopes = []
    for (candidate_id, artifact_id, dataset, asset_id), points in sorted(grouped.items()):
        points.sort(
            key=lambda item: _sortable_cycle(
                item.get("cycle_index", item.get("cycle_id"))
            )
        )
        token = hashlib.sha256(
            "\0".join((candidate_id, artifact_id, dataset, asset_id)).encode("utf-8")
        ).hexdigest()[:24]
        relative = Path(TRAJECTORY_SHARD_DIRNAME) / f"{token}.json"
        target = root / relative
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps({"records": points}, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(target)
        scopes.append({
            "candidate_id": candidate_id,
            "model_artifact_id": artifact_id,
            "dataset": dataset,
            "asset_id": asset_id,
            "records": len(points),
            "raw_curve_records": raw_curve_counts.get(
                (candidate_id, artifact_id, dataset, asset_id), 0
            ),
            "path": str(relative).replace("\\", "/"),
        })

    payload = {
        "schema_version": TRAJECTORY_INDEX_SCHEMA,
        "source": {
            "size": int(source_stat.st_size),
            "mtime_ns": int(source_stat.st_mtime_ns),
        },
        "scopes": scopes,
    }
    target = root / TRAJECTORY_INDEX_FILENAME
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(target)
    if not _prediction_lookup_is_current(root, source_stat):
        _write_prediction_evidence_lookup(root, records, source_stat)
    return target


def _prediction_lookup_is_current(root: Path, source_stat) -> bool:
    lookup_path = root / PREDICTION_LOOKUP_FILENAME
    if not lookup_path.is_file():
        return False
    try:
        connection = sqlite3.connect(f"file:{lookup_path.as_posix()}?mode=ro", uri=True)
        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        return (
            metadata.get("schema_version") == PREDICTION_LOOKUP_SCHEMA
            and int(metadata.get("source_size", -1)) == int(source_stat.st_size)
            and int(metadata.get("source_mtime_ns", -1)) == int(source_stat.st_mtime_ns)
        )
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return False
    finally:
        if "connection" in locals():
            connection.close()


def _write_prediction_evidence_lookup(
    root: Path,
    records: list[dict],
    source_stat,
) -> Path:
    target = root / PREDICTION_LOOKUP_FILENAME
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", PREDICTION_LOOKUP_SCHEMA),
                ("source_size", str(int(source_stat.st_size))),
                ("source_mtime_ns", str(int(source_stat.st_mtime_ns))),
            ],
        )
        connection.execute(
            "CREATE TABLE records ("
            "prediction_record_id TEXT PRIMARY KEY, "
            "sample_id TEXT NOT NULL, "
            "record_json TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT OR REPLACE INTO records(prediction_record_id, sample_id, record_json) "
            "VALUES (?, ?, ?)",
            (
                (
                    str(item.get("prediction_record_id") or ""),
                    str(item.get("sample_id") or ""),
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                )
                for item in records
                if item.get("prediction_record_id")
                and str(item.get("validity_status") or "valid") != "invalidated"
            ),
        )
        indexed_count = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        source_record_count = sum(
            1 for item in records if item.get("prediction_record_id")
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("source_record_count", str(source_record_count)),
                ("indexed_record_count", str(indexed_count)),
                ("duplicate_record_count", str(source_record_count - indexed_count)),
            ],
        )
        connection.execute("CREATE INDEX records_sample_id_idx ON records(sample_id)")
        connection.commit()
    finally:
        connection.close()
    temporary.replace(target)
    return target


def _indexed_prediction_record(root: Path, requested: str) -> dict | None:
    if not requested:
        return None
    lookup_path = root / PREDICTION_LOOKUP_FILENAME
    source_path = root / "prediction_records.json"
    if not lookup_path.is_file() or not source_path.is_file():
        return None
    source_stat = source_path.stat()
    try:
        connection = sqlite3.connect(f"file:{lookup_path.as_posix()}?mode=ro", uri=True)
        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        if (
            metadata.get("schema_version") != PREDICTION_LOOKUP_SCHEMA
            or int(metadata.get("source_size", -1)) != int(source_stat.st_size)
            or int(metadata.get("source_mtime_ns", -1)) != int(source_stat.st_mtime_ns)
        ):
            return None
        row = connection.execute(
            "SELECT record_json FROM records "
            "WHERE prediction_record_id = ? OR sample_id = ? LIMIT 1",
            (requested, requested),
        ).fetchone()
        record = json.loads(row[0]) if row else None
        if record and str(record.get("validity_status") or "valid") == "invalidated":
            return None
        return record
    except (OSError, sqlite3.Error, ValueError, TypeError, json.JSONDecodeError):
        return None
    finally:
        if "connection" in locals():
            connection.close()

CURVE_SIGNAL_FIELDS = {
    "voltage": ("voltage_v", "V"),
    "current": ("current_a", "A"),
    "temperature": ("temperature_c", "°C"),
}
CURVE_DERIVED_SIGNALS = {"ic", "dv"}
CURVE_RESOLUTIONS = {"thumbnail": 160, "viewport": 480, "full": 5000}

DATASET_FEATURE_ALIASES = {
    "calce_cs2": "calce",
    "nasa_b0018": "nasa",
    "oxford_40c": "oxford",
}
DEFAULT_FEATURE_WINDOWS = {
    "xjtu": ([3.7, 4.1], True),
    "mit": ([3.2, 3.4], True),
    "nasa": ([3.7, 4.1], False),
    "calce": ([3.6, 4.1], False),
    "oxford": ([3.7, 4.1], False),
    "hust": ([3.0, 3.59], False),
    "snl_lfp": ([3.0, 3.59], False),
    "uconn": ([3.0, 3.59], False),
    "rwth_isea_public": ([3.0, 3.59], False),
}
WINDOW_FEATURES = {
    "q_win_ah",
    "t_win_s",
    "ic_peak_ah_per_v",
    "ic_peak_voltage_v",
}
CONTEXT_FEATURES = {
    "cycle_index_feature",
    "cycle_fraction",
    "battery_soh_cycle_norm",
}


class EvidenceDerivationService:
    """Derive evidence coverage and verdicts from server-side artifacts only."""

    def __init__(
        self,
        frontend_data_dir: str | Path,
        data_dir: str | Path | None = None,
        *,
        inference_jobs_dir: str | Path | None = None,
        runs_dir: str | Path | None = None,
    ):
        self.frontend_data_dir = Path(frontend_data_dir)
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.inference_jobs_dir = Path(inference_jobs_dir) if inference_jobs_dir is not None else None
        self.runs_dir = Path(runs_dir).resolve() if runs_dir is not None else None

    @staticmethod
    def _active_saved_inference_job_id(case: dict) -> str:
        active_saved_id = str(case.get("active_saved_combination_id") or "")
        return next((
            str(item.get("inference_job_id") or "")
            for item in (case.get("saved_combinations") or [])
            if str(item.get("saved_combination_id") or "") == active_saved_id
            and str(item.get("source_stage") or "") == "inference"
        ), "")

    def _job_artifact_dir(
        self,
        case: dict,
        inference_job_id: str | None = None,
    ) -> Path | None:
        execution_job_id = str((case.get("execution") or {}).get("inference_job_id") or "")
        saved_combinations = case.get("saved_combinations") or []
        active_saved_job_id = self._active_saved_inference_job_id(case)
        expected_job_id = active_saved_job_id or execution_job_id
        allowed_job_ids = {execution_job_id} if execution_job_id else set()
        allowed_job_ids.update(
            str(item.get("inference_job_id") or "")
            for item in saved_combinations
            if str(item.get("source_stage") or "") == "inference"
            and item.get("inference_job_id")
        )
        requested_job_id = str(inference_job_id or expected_job_id or "")
        if inference_job_id and requested_job_id not in allowed_job_ids:
            raise EvaluationCaseValidationError(
                "inference job does not match the evaluation case"
            )
        if (
            not requested_job_id
            or self.inference_jobs_dir is None
            or not all(ch.isalnum() or ch in "._-" for ch in requested_job_id)
        ):
            return None
        job = _read_json(self.inference_jobs_dir / f"{requested_job_id}.json", {})
        if str(job.get("id") or "") != requested_job_id or job.get("status") != "succeeded":
            return None
        request_case_id = str((job.get("request") or {}).get("case_id") or "")
        if request_case_id and request_case_id != str(case.get("id") or ""):
            raise EvaluationCaseValidationError(
                "inference evidence belongs to a different evaluation case"
            )
        run_value = str(job.get("raw_run_dir") or job.get("run_dir") or "").strip()
        if not run_value:
            return None
        run_dir = Path(run_value).resolve()
        if self.runs_dir is not None and not run_dir.is_relative_to(self.runs_dir):
            raise EvaluationCaseValidationError("inference evidence directory is outside the runs root")
        if not run_dir.is_dir() or not (run_dir / "prediction_records.json").is_file():
            return None
        return run_dir

    def _artifact_dir(self, case: dict, inference_job_id: str | None = None) -> Path:
        job_dir = self._job_artifact_dir(case, inference_job_id)
        if job_dir is not None:
            return job_dir
        profile_dir = self.frontend_data_dir / "profiles" / str(case.get("profile_name") or "")
        return profile_dir if profile_dir.is_dir() else self.frontend_data_dir

    def _manifest(self, case: dict, artifact_dir: Path) -> tuple[dict, Path]:
        profile_dir = self.frontend_data_dir / "profiles" / str(case.get("profile_name") or "")
        profile_manifest_path = profile_dir / "manifest.json"
        profile_manifest = _read_json(profile_manifest_path, {})
        evidence_manifest_path = artifact_dir / "inference_evidence_manifest.json"
        evidence_manifest = _read_json(evidence_manifest_path, {})
        if not evidence_manifest:
            return profile_manifest, profile_manifest_path
        evidence_datasets = {
            str(item.get("dataset") or "")
            for item in evidence_manifest.get("curve_artifacts") or []
        }
        merged_curves = list(evidence_manifest.get("curve_artifacts") or []) + [
            item
            for item in profile_manifest.get("curve_artifacts") or []
            if str(item.get("dataset") or "") not in evidence_datasets
        ]
        return {
            **profile_manifest,
            **evidence_manifest,
            "curve_artifacts": merged_curves,
        }, evidence_manifest_path

    def derive(self, case: dict, payload: dict) -> dict:
        forbidden = {"coverage", "verdict"}.intersection(payload)
        if forbidden:
            raise EvaluationCaseValidationError(
                f"server-derived fields are read-only: {', '.join(sorted(forbidden))}"
            )
        selection = normalize_evidence_selection(
            payload.get("selection") or {}, current=case["evidence_selection"]
        )
        candidate_id = selection.get("candidate_id")
        candidate = next(
            (item for item in case["candidates"] if item["candidate_id"] == candidate_id), None
        )
        if candidate_id and candidate is None:
            raise EvaluationCaseValidationError(
                f"evidence selection references unknown candidate: {candidate_id}"
        )
        profile_dir = self.frontend_data_dir / "profiles" / case["profile_name"]
        artifact_dir = self._artifact_dir(case)
        is_job_scope = artifact_dir != profile_dir and artifact_dir != self.frontend_data_dir
        benchmark_dir = artifact_dir if is_job_scope else (
            profile_dir if (profile_dir / "run_config.json").is_file() else self.frontend_data_dir
        )
        manifest, manifest_path = self._manifest(case, artifact_dir)
        config_path = benchmark_dir / (
            "inference_run_config.json"
            if (benchmark_dir / "inference_run_config.json").is_file()
            else "run_config.json"
        )
        leaderboard_path = benchmark_dir / "leaderboard.json"
        config = _read_json(config_path, {})
        leaderboard = _read_json(leaderboard_path, [])
        if not leaderboard and (benchmark_dir / "inference_benchmark.json").is_file():
            leaderboard = _inference_leaderboard(
                _read_json(benchmark_dir / "inference_benchmark.json", {})
            )

        scenario_path = _scenario_path(artifact_dir, selection.get("dataset") or "")
        scenario = _read_json(scenario_path, {}) if scenario_path else {}
        dataset_path = self._dataset_path(selection.get("dataset") or "")
        selection["dataset_snapshot_id"] = _digest_ref(
            "dataset-snapshot",
            dataset_path or scenario_path or (manifest_path if manifest_path.exists() else config_path),
        )
        compatible_scope = not selection.get("dataset") or selection.get("dataset") in (config.get("datasets") or [])
        selection["split_assignment_id"] = _protocol_ref(config) if compatible_scope else ""
        if not selection.get("asset_id") or selection.get("cycle_id") in (None, ""):
            representative = _representative_prediction_record(
                artifact_dir,
                scenario,
                selection,
                candidate,
            )
            if representative:
                selection["asset_id"] = str(
                    representative.get("cell_id")
                    or representative.get("asset_id")
                    or ""
                )
                selection["cycle_id"] = representative.get(
                    "cycle_index", representative.get("cycle_id")
                )
        record = _prediction_record(artifact_dir, scenario, selection)
        if record:
            selection["sample_id"] = str(
                record.get("prediction_record_id")
                or record.get("sample_id")
                or _sample_id(
                selection.get("dataset") or "", record.get("cell_id"), record.get("cycle_index")
                )
            )
            selection["cycle_id"] = record.get("cycle_index")
            selection["dataset_snapshot_id"] = str(
                record.get("dataset_snapshot_id") or selection["dataset_snapshot_id"]
            )
            selection["split_assignment_id"] = str(
                record.get("split_assignment_id") or selection["split_assignment_id"]
            )

        curve_path, curve_available = self._curve_state(manifest, selection)
        curve_key = _curve_record_key(selection)
        signals = _curve_signals(curve_path, curve_key) if curve_available else []
        selection["curve_ref"] = {
            "record_id": selection.get("sample_id") or "",
            "source_uri": _digest_ref("curve", curve_path) if curve_available else "",
            "signals": signals,
            "availability": "available" if curve_available else (
                "missing" if selection.get("sample_id") else "unknown"
            ),
            "missing_reason": "" if curve_available else (
                "prediction record has no matching raw curve" if selection.get("sample_id")
                else "prediction record is not selected"
            ),
        }

        metric_row = _candidate_metrics(leaderboard, candidate) if compatible_scope else {}
        criterion_set_id = str(
            payload.get("criterion_set_id")
            or criterion_base_id((case.get("verdict") or {}).get("criterion_set_version"))
            or "strict"
        )
        threshold_overrides = payload.get("criterion_thresholds")
        if "criterion_thresholds" not in payload:
            current_verdict = case.get("verdict") or {}
            current_version = str(current_verdict.get("criterion_set_version") or "")
            if (
                "@custom-" in current_version
                and criterion_base_id(current_version) == criterion_base_id(criterion_set_id)
            ):
                threshold_overrides = current_verdict.get("thresholds") or None
        verdict = derive_verdict(metric_row, criterion_set_id, threshold_overrides)
        coverage = normalize_evidence_coverage(
            {
                "items": [
                    _coverage("run_spec", config_path.exists(), "run config is unavailable"),
                    _coverage("dataset_snapshot", bool(selection["dataset_snapshot_id"]), "dataset snapshot is unavailable"),
                    _coverage("split_assignment", bool(selection["split_assignment_id"]), "split assignment is unavailable"),
                    _coverage("comparison_scope", compatible_scope, "selected evidence dataset is outside the benchmark comparison scope"),
                    _coverage("model_artifact", bool(candidate and candidate.get("model_artifact_id")), "candidate has no immutable model artifact"),
                    _coverage("prediction_record", bool(record), "prediction record is unavailable"),
                    _coverage("raw_curve", curve_available, "raw curve is unavailable"),
                    _coverage("criterion_set", True, ""),
                ]
            }
        )
        return {"selection": selection, "coverage": coverage, "verdict": verdict}

    def prediction_detail(self, case: dict, sample_id: str | None = None) -> dict:
        """Join one canonical prediction fact to its metric and raw-curve references."""
        selection = dict(case.get("evidence_selection") or {})
        requested_sample = str(sample_id or selection.get("sample_id") or "")
        artifact_dir = self._artifact_dir(case)
        record = _indexed_prediction_record(artifact_dir, requested_sample)
        records_payload = {}
        if record is not None and not _record_matches_selection(record, selection, case):
            record = None
        if record is None:
            records_payload = _read_json(artifact_dir / "prediction_records.json", {})
            records = records_payload.get("records") or []
            record = next(
                (
                    item for item in records
                    if (
                        str(item.get("sample_id") or "") == requested_sample
                        or str(item.get("prediction_record_id") or "") == requested_sample
                    )
                    and _record_matches_selection(item, selection, case)
                ),
                None,
            )
        if record is None and not requested_sample:
            scenario_path = _scenario_path(artifact_dir, str(selection.get("dataset") or ""))
            scenario = _read_json(scenario_path, {}) if scenario_path else {}
            record = _prediction_record(artifact_dir, scenario, selection)
        if record is None:
            raise EvaluationCaseValidationError("prediction record is unavailable")

        record_id = str(record.get("prediction_record_id") or "")
        contribution = dict(record.get("metric_contributions") or {})
        curve_ref = {}
        if record.get("curve_segment_ref_id"):
            curve_selection = {
                "dataset": record.get("dataset") or "",
                "asset_id": record.get("cell_id") or record.get("asset_id") or "",
                "cycle_id": record.get("cycle_index", record.get("cycle_id")),
            }
            manifest, _manifest_path = self._manifest(case, artifact_dir)
            curve_path, curve_available = self._curve_state(manifest, curve_selection)
            if curve_available and curve_path is not None:
                curve_ref = {
                    "curve_segment_ref_id": record.get("curve_segment_ref_id") or "",
                    "prediction_record_id": record_id,
                    "sample_id": record.get("sample_id") or "",
                    "candidate_id": record.get("candidate_id") or "",
                    "dataset": curve_selection["dataset"],
                    "asset_id": curve_selection["asset_id"],
                    "cycle_id": curve_selection["cycle_id"],
                    "curve_key": _curve_record_key(curve_selection),
                    "source_uri": _digest_ref("curve", curve_path),
                    "signals": _curve_signals(
                        curve_path, _curve_record_key(curve_selection)
                    ),
                    "availability": "available",
                }
        if not contribution:
            contribution_payload = _read_json(artifact_dir / "metric_contributions.json", {})
            contribution = next(
                (
                    item for item in contribution_payload.get("records") or []
                    if str(item.get("prediction_record_id") or "") == record_id
                    or str(item.get("sample_id") or "") == str(record.get("sample_id") or "")
                ),
                {},
            )
        if not curve_ref:
            curve_payload = _read_json(artifact_dir / "curve_segment_refs.json", {})
            curve_ref = next(
                (
                    item for item in curve_payload.get("records") or []
                    if str(item.get("prediction_record_id") or "") == record_id
                    or str(item.get("sample_id") or "") == str(record.get("sample_id") or "")
                ),
                {},
            )
        return {
            "schema_version": "1.0.0",
            "case_id": case.get("id") or "",
            "case_version": case.get("version"),
            "prediction_record": record,
            "metric_contribution": contribution,
            "curve_segment_ref": curve_ref,
            "lineage": {
                "comparison_id": records_payload.get("comparison_id") or selection.get("comparison_id") or "",
                "run_id": record.get("run_id") or records_payload.get("run_id") or case.get("execution", {}).get("run_id") or "",
                "dataset_snapshot_id": record.get("dataset_snapshot_id") or records_payload.get("dataset_snapshot_id") or selection.get("dataset_snapshot_id") or "",
                "split_assignment_id": record.get("split_assignment_id") or records_payload.get("split_assignment_id") or selection.get("split_assignment_id") or "",
                "model_artifact_id": record.get("model_artifact_id") or records_payload.get("model_artifact_id") or "",
            },
        }

    def prediction_curves(
        self,
        case: dict,
        *,
        record_id: str | None = None,
        signals: list[str] | None = None,
        resolution: str = "viewport",
        max_points: int | None = None,
    ) -> dict:
        """Resolve canonical raw and derived signals for one PredictionRecord."""
        detail = self.prediction_detail(case, sample_id=record_id)
        record = detail["prediction_record"]
        artifact_dir = self._artifact_dir(case)
        manifest, _manifest_path = self._manifest(case, artifact_dir)
        curve_selection = {
            "dataset": record.get("dataset") or case.get("evidence_selection", {}).get("dataset") or "",
            "asset_id": record.get("cell_id") or record.get("asset_id") or "",
            "cycle_id": record.get("cycle_index", record.get("cycle_id")),
        }
        curve_path, curve_available = self._curve_state(manifest, curve_selection)
        curve_ref = detail.get("curve_segment_ref") or {}
        curve_key = str(curve_ref.get("curve_key") or _curve_record_key(curve_selection))
        if not curve_available or curve_path is None or not curve_key:
            raise EvaluationCaseValidationError("raw curve is unavailable for prediction record")
        curve = (_read_curve_json(curve_path) or {}).get(curve_key)
        if not isinstance(curve, dict):
            raise EvaluationCaseValidationError("raw curve payload is unavailable for prediction record")

        requested = _normalize_curve_signals(signals)
        if resolution not in CURVE_RESOLUTIONS:
            raise EvaluationCaseValidationError(f"unknown curve resolution: {resolution}")
        point_limit = max_points or CURVE_RESOLUTIONS[resolution]
        point_limit = min(max(int(point_limit), 32), CURVE_RESOLUTIONS["full"])
        feature_contract = _feature_contract(curve_path, curve_selection["dataset"])
        feature_window = _feature_window_payload(
            curve,
            feature_contract.get("voltage_v"),
            source=str(feature_contract.get("source") or ""),
            approximate=bool(feature_contract.get("approximate")),
            feature_columns=feature_contract.get("feature_columns") or [],
        )
        signal_payload = _curve_signal_payload(
            curve,
            requested,
            point_limit,
            voltage_window=feature_window.get("voltage_v"),
        )
        feature_curves = _feature_curve_payload(
            curve,
            dict(record.get("feature_values") or {}),
            feature_window,
            point_limit,
        )
        source_points = max((item["source_points"] for item in signal_payload.values()), default=0)
        returned_points = max((item["returned_points"] for item in signal_payload.values()), default=0)
        source_digest = _digest_ref("curve", curve_path)
        recorded_digest = str(curve_ref.get("source_uri") or "")
        if recorded_digest and recorded_digest != source_digest:
            raise EvaluationCaseValidationError("raw curve digest does not match CurveSegmentRef")

        return {
            "schema_version": "1.1.0",
            "case_id": case.get("id") or "",
            "case_version": case.get("version"),
            "prediction_record": {
                "prediction_record_id": record.get("prediction_record_id") or "",
                "sample_id": record.get("sample_id") or "",
                "dataset": curve_selection["dataset"],
                "asset_id": curve_selection["asset_id"],
                "cycle_id": curve_selection["cycle_id"],
                "true_soh": record.get("true_soh"),
                "pred_soh": record.get("pred_soh"),
                "lower": record.get("lower"),
                "upper": record.get("upper"),
                "featureset": record.get("featureset") or "",
                "feature_values": dict(record.get("feature_values") or {}),
            },
            "curve_segment_ref": {
                **curve_ref,
                "curve_key": curve_key,
                "source_uri": source_digest,
                "availability": "available",
            },
            "sampling": {
                "resolution": resolution,
                "method": "none" if source_points <= point_limit else "uniform-index/v1",
                "source_points": source_points,
                "returned_points": returned_points,
                "max_points": point_limit,
            },
            "requested_signals": requested,
            "available_signals": list(signal_payload),
            "missing_signals": [name for name in requested if name not in signal_payload],
            "signals": signal_payload,
            "feature_window": feature_window,
            "feature_curves": feature_curves,
        }

    def prediction_trajectory(
        self,
        case: dict,
        *,
        inference_job_id: str | None = None,
        dataset: str | None = None,
        asset_id: str | None = None,
        compact: bool = False,
        candidate_id: str | None = None,
        model_artifact_id: str | None = None,
    ) -> dict:
        """Return one case/job/candidate trajectory from canonical PredictionRecords."""
        resolved_inference_job_id = str(
            inference_job_id
            or self._active_saved_inference_job_id(case)
            or (case.get("execution") or {}).get("inference_job_id")
            or ""
        )
        artifact_dir = self._artifact_dir(case, resolved_inference_job_id or None)
        source_path = artifact_dir / "prediction_records.json"
        selection = dict(case.get("evidence_selection") or {})
        selected_dataset = str(dataset or selection.get("dataset") or "")
        requested_candidate_id = str(candidate_id or "")
        requested_artifact_id = str(model_artifact_id or "")
        explicit_identity = bool(requested_candidate_id or requested_artifact_id)
        active_saved_id = str(case.get("active_saved_combination_id") or "")
        active_saved = next((
            item for item in (case.get("saved_combinations") or [])
            if active_saved_id
            and str(item.get("saved_combination_id") or "") == active_saved_id
            and str(item.get("source_stage") or "") == "inference"
            and str(item.get("inference_job_id") or "") == resolved_inference_job_id
            and (
                not selected_dataset
                or not item.get("target_dataset")
                or str(item.get("target_dataset")) == selected_dataset
            )
        ), None)
        active_saved_candidate = dict((active_saved or {}).get("candidate") or {})
        if active_saved_candidate and not explicit_identity:
            selection["candidate_id"] = str(active_saved_candidate.get("candidate_id") or selection.get("candidate_id") or "")

        candidate_catalog = list(case.get("candidates") or []) + [
            dict(item.get("candidate") or {})
            for item in (case.get("saved_combinations") or [])
            if item.get("candidate")
        ]
        selected_candidate_id = str(requested_candidate_id or selection.get("candidate_id") or "")
        selected_artifact_id = str(
            requested_artifact_id
            or active_saved_candidate.get("model_artifact_id")
            or ""
        )
        candidate_by_id = next((
            item for item in candidate_catalog
            if selected_candidate_id and str(item.get("candidate_id") or "") == selected_candidate_id
        ), None)
        candidate_by_artifact = next((
            item for item in candidate_catalog
            if selected_artifact_id and str(item.get("model_artifact_id") or "") == selected_artifact_id
        ), None)
        resolved_candidate = candidate_by_artifact or candidate_by_id or {}
        if explicit_identity and not resolved_candidate:
            raise EvaluationCaseValidationError("requested trajectory candidate is not registered in this case")
        resolved_candidate_id = str(resolved_candidate.get("candidate_id") or selected_candidate_id)
        resolved_artifact_id = str(resolved_candidate.get("model_artifact_id") or selected_artifact_id)
        if requested_candidate_id and resolved_candidate_id != requested_candidate_id:
            raise EvaluationCaseValidationError("trajectory candidate_id does not match model_artifact_id")
        if requested_artifact_id and resolved_artifact_id != requested_artifact_id:
            raise EvaluationCaseValidationError("trajectory model_artifact_id does not match candidate_id")
        selection["candidate_id"] = resolved_candidate_id
        selection["_model_artifact_id"] = resolved_artifact_id
        selection["_strict_identity"] = explicit_identity
        requested_asset = str(
            asset_id
            or (
                selection.get("asset_id")
                if selected_dataset == str(selection.get("dataset") or "")
                else ""
            )
            or ""
        )

        payload: dict = {}
        index_payload: dict = {}
        if compact and source_path.is_file():
            source_stat = source_path.stat()
            index_path = artifact_dir / TRAJECTORY_INDEX_FILENAME
            index_payload = _read_json(index_path, {})
            source = index_payload.get("source") or {}
            if (
                index_payload.get("schema_version") != TRAJECTORY_INDEX_SCHEMA
                or int(source.get("size") or -1) != int(source_stat.st_size)
                or int(source.get("mtime_ns") or -1) != int(source_stat.st_mtime_ns)
            ):
                write_prediction_trajectory_index(
                    artifact_dir,
                )
                index_payload = _read_json(index_path, {})
            scopes = [
                item for item in index_payload.get("scopes") or []
                if (not selected_dataset or str(item.get("dataset") or "") == selected_dataset)
                and _identity_matches(
                    item,
                    resolved_candidate_id,
                    resolved_artifact_id,
                    strict=explicit_identity,
                )
            ]
            selected_scope = next((
                item for item in scopes
                if requested_asset and str(item.get("asset_id") or "") == requested_asset
            ), None) or next((
                item for item in scopes
                if int(item.get("raw_curve_records") or 0) > 0
            ), None) or (scopes[0] if scopes else None)
            if selected_scope:
                relative = Path(str(selected_scope.get("path") or ""))
                shard_path = (artifact_dir / relative).resolve()
                if artifact_dir.resolve() not in shard_path.parents:
                    raise EvaluationCaseValidationError("invalid trajectory shard path")
                payload = _read_json(shard_path, {})
                requested_asset = str(selected_scope.get("asset_id") or requested_asset)
            else:
                payload = {"records": []}
        else:
            payload = _read_json(source_path, {})
        records = [
            item
            for item in payload.get("records") or []
            if (not selected_dataset or str(item.get("dataset") or "") == selected_dataset)
            and _record_matches_selection(item, selection, case)
        ]
        if requested_asset:
            records = [
                item for item in records
                if str(item.get("cell_id") or item.get("asset_id") or "") == requested_asset
            ]
        if not records:
            raise EvaluationCaseValidationError(
                "trajectory prediction records are unavailable for the selected case scope"
            )
        records.sort(
            key=lambda item: (
                str(item.get("cell_id") or item.get("asset_id") or ""),
                _sortable_cycle(item.get("cycle_index", item.get("cycle_id"))),
            )
        )
        available_assets = sorted({
            str(item.get("asset_id") or "")
            for item in (index_payload.get("scopes") or [])
            if compact
            and (not selected_dataset or str(item.get("dataset") or "") == selected_dataset)
            and _identity_matches(
                item,
                resolved_candidate_id,
                resolved_artifact_id,
                strict=explicit_identity,
            )
            and item.get("asset_id")
        }) if compact else sorted({
            str(item.get("cell_id") or item.get("asset_id") or "")
            for item in records
            if item.get("cell_id") or item.get("asset_id")
        })
        points = [_compact_trajectory_point(record, public=True) for record in records] if compact else records
        return {
            "schema_version": "1.0.0",
            "case_id": str(case.get("id") or ""),
            "case_version": case.get("version"),
            "inference_job_id": str(
                resolved_inference_job_id
            ),
            "dataset": selected_dataset,
            "candidate_id": resolved_candidate_id,
            "model_artifact_id": resolved_artifact_id,
            "context_binding_id": str((case.get("context_binding") or {}).get("binding_id") or ""),
            "asset_id": requested_asset or (available_assets[0] if available_assets else ""),
            "assets": available_assets,
            "compact": bool(compact),
            "points": points,
        }

    def _dataset_path(self, dataset: str) -> Path | None:
        if self.data_dir is None or not dataset:
            return None
        path = self.data_dir / "processed" / f"{dataset}_features.pkl"
        return path if path.is_file() else None

    def _curve_state(self, manifest: dict, selection: dict) -> tuple[Path | None, bool]:
        dataset = str(selection.get("dataset") or "")
        curve_key = _curve_record_key(selection)
        found: tuple[Path | None, bool] = (None, False)
        for item in manifest.get("curve_artifacts") or []:
            if str(item.get("dataset") or "") != dataset:
                continue
            source = Path(str(item.get("source") or ""))
            if source.is_file() and curve_key:
                found = (source, _json_object_has_key(source, curve_key))
                break
        if not found[1]:
            local = self.frontend_data_dir.parent / "battery-dashboard" / "data" / f"{dataset}_curves.json"
            if local.is_file() and curve_key:
                candidate = (local, _json_object_has_key(local, curve_key))
                found = candidate if candidate[1] or found[0] is None else found
        if not found[1]:
            # The pre-converted extracts cover only sampled cells/cycles;
            # extract the requested cycle from the local raw acquisition
            # files when the dataset supports it.
            generated = self._materialize_raw_curve(dataset, curve_key)
            if generated is not None:
                return generated, True
        return found

    def _materialize_raw_curve(self, dataset: str, curve_key: str) -> Path | None:
        """On-demand raw-curve extraction into a per-record cache file.

        Each extracted record gets its own file so its content digest — and
        therefore every recorded ``CurveSegmentRef.source_uri`` — stays
        stable regardless of later extractions.
        """
        if self.data_dir is None or not curve_key:
            return None
        extractor = _RAW_CURVE_EXTRACTORS.get(dataset)
        if extractor is None:
            return None
        safe_key = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in curve_key
        )
        out_path = self.data_dir / "curves" / "generated" / dataset / f"{safe_key}.json"
        if out_path.is_file():
            return out_path
        try:
            curve = extractor(self.data_dir, curve_key)
        except Exception:
            return None
        if not curve:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({curve_key: curve}, ensure_ascii=False), encoding="utf-8"
        )
        return out_path


def derive_verdict(
    metrics: dict,
    criterion_set_id: str,
    threshold_overrides: dict | None = None,
) -> dict:
    _, thresholds, criterion_version = resolve_criterion_thresholds(
        criterion_set_id,
        threshold_overrides,
    )
    observed = {
        key: value for key, value in {
            "mae": metrics.get("mae"),
            "picp": metrics.get("picp"),
            "monotonicity": metrics.get("monotonicity"),
            "value_range": metrics.get("value_range"),
        }.items() if isinstance(value, (int, float))
    }
    checks = []
    reasons = []
    for name, observed_key, threshold_key, direction in (
        ("generalization", "mae", "mae_max", "max"),
        ("calibration", "picp", "picp_min", "min"),
        ("monotonicity", "monotonicity", "monotonicity_max", "max"),
        ("value_range", "value_range", "value_range_max", "max"),
    ):
        value = observed.get(observed_key)
        if value is None:
            continue
        limit = thresholds[threshold_key]
        passed = value <= limit if direction == "max" else value >= limit
        checks.append(passed)
        if not passed:
            reasons.append(f"{name}: observed {value} violates {threshold_key}={limit}")
    result = "pending" if not checks else ("pass" if all(checks) else "fail")
    return normalize_verdict(
        {
            "criterion_set_version": criterion_version,
            "result": result,
            "scope": "candidate",
            "observed": observed,
            "thresholds": thresholds,
            "reasons": reasons,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )


def _record_matches_selection(record: dict, selection: dict, case: dict) -> bool:
    if str(record.get("validity_status") or "valid") == "invalidated":
        return False
    selected_candidate_id = str(selection.get("candidate_id") or "")
    record_candidate_id = str(record.get("candidate_id") or "")
    strict_identity = bool(selection.get("_strict_identity"))
    if selected_candidate_id and (
        (strict_identity and selected_candidate_id != record_candidate_id)
        or (not strict_identity and record_candidate_id and selected_candidate_id != record_candidate_id)
    ):
        return False
    candidate = next(
        (
            item for item in case.get("candidates") or []
            if str(item.get("candidate_id") or "") == selected_candidate_id
        ),
        {},
    )
    selected_artifact = str(selection.get("_model_artifact_id") or candidate.get("model_artifact_id") or "")
    record_artifact = str(record.get("model_artifact_id") or "")
    if selected_artifact and (
        (strict_identity and selected_artifact != record_artifact)
        or (not strict_identity and record_artifact and selected_artifact != record_artifact)
    ):
        return False
    return True


def _identity_matches(
    record: dict,
    candidate_id: str,
    model_artifact_id: str,
    *,
    strict: bool,
) -> bool:
    record_candidate_id = str(record.get("candidate_id") or "")
    record_artifact_id = str(record.get("model_artifact_id") or "")
    if candidate_id and (
        (strict and record_candidate_id != candidate_id)
        or (not strict and record_candidate_id and record_candidate_id != candidate_id)
    ):
        return False
    if model_artifact_id and (
        (strict and record_artifact_id != model_artifact_id)
        or (not strict and record_artifact_id and record_artifact_id != model_artifact_id)
    ):
        return False
    return True


def _sortable_cycle(value) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value or "")


def _candidate_metrics(rows: list, candidate: dict | None) -> dict:
    if not candidate:
        return {}
    artifact_id = str(candidate.get("model_artifact_id") or "")
    row = next(
        (
            item for item in rows
            if artifact_id
            and str(item.get("model_artifact_id") or item.get("artifact_id") or "") == artifact_id
        ),
        None,
    ) or next(
        (
            item for item in rows
            if item.get("estimator") == candidate.get("model")
            and (item.get("featureset") or "full") == (candidate.get("featureset") or "full")
        ),
        {},
    )
    metrics = row.get("metrics") or {}
    return {
        "mae": _mean(metrics, "mae"),
        "picp": _mean(metrics, "picp"),
        "monotonicity": _mean(metrics, "monotonicity_violation_rate"),
        "value_range": _mean(metrics, "value_range_violation"),
    }


def _inference_leaderboard(payload: dict) -> list[dict]:
    rows = []
    for record in payload.get("records") or []:
        metric = lambda name: (
            {"mean": record.get(name), "std": 0.0}
            if isinstance(record.get(name), (int, float))
            else None
        )
        rows.append(
            {
                "estimator": record.get("estimator") or record.get("model") or "",
                "featureset": record.get("featureset") or "full",
                "candidate_id": record.get("candidate_id") or "",
                "model_artifact_id": record.get("model_artifact_id") or "",
                "metrics": {
                    "mae": metric("mae"),
                    "rmse": metric("rmse"),
                    "mean_bias_error": metric("mean_bias_error"),
                    "monotonicity_violation_rate": metric("monotonicity_violation_rate"),
                    "value_range_violation": metric("value_range_violation"),
                },
            }
        )
    return rows


def _mean(metrics: dict, name: str):
    value = metrics.get(name)
    return value.get("mean") if isinstance(value, dict) else value


def _scenario_path(profile_dir: Path, dataset: str) -> Path | None:
    candidates = [
        profile_dir / f"battery_dashboard_{dataset}.json",
        profile_dir / f"battery_dashboard_{'calce' if dataset == 'calce_cs2' else dataset}.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _prediction_record(profile_dir: Path, scenario: dict, selection: dict) -> dict | None:
    asset = str(selection.get("asset_id") or "")
    cycle = selection.get("cycle_id")
    candidate_id = str(selection.get("candidate_id") or "")
    if not asset or cycle in (None, ""):
        return None
    canonical = _read_json(profile_dir / "prediction_records.json", {})
    for record in canonical.get("records") or []:
        record_candidate = str(record.get("candidate_id") or "")
        if candidate_id and record_candidate and record_candidate != candidate_id:
            continue
        if str(record.get("cell_id") or record.get("asset_id") or "") != asset:
            continue
        if str(record.get("cycle_index") or record.get("cycle_id")) != str(cycle):
            continue
        return record
    for point in scenario.get("points") or []:
        if str(point.get("cell_id") or "") != asset:
            continue
        if str(point.get("cycle_index")) != str(cycle):
            continue
        return point
    return None


def _representative_prediction_record(
    profile_dir: Path,
    scenario: dict,
    selection: dict,
    candidate: dict | None,
) -> dict | None:
    """Choose a stable, high-information record when the UI only pins a run scope."""
    canonical = _read_json(profile_dir / "prediction_records.json", {})
    records = list(canonical.get("records") or scenario.get("points") or [])
    curve_payload = _read_json(profile_dir / "curve_segment_refs.json", {})
    curve_refs_by_record = {
        str(item.get("prediction_record_id") or ""): item
        for item in curve_payload.get("records") or []
        if item.get("availability", "available") == "available"
    }
    dataset = str(selection.get("dataset") or "")
    asset = str(selection.get("asset_id") or "")
    candidate_id = str(selection.get("candidate_id") or "")
    artifact_id = str((candidate or {}).get("model_artifact_id") or "")

    def compatible(record: dict) -> bool:
        record_dataset = str(record.get("dataset") or "")
        record_asset = str(record.get("cell_id") or record.get("asset_id") or "")
        record_candidate = str(record.get("candidate_id") or "")
        record_artifact = str(record.get("model_artifact_id") or "")
        if dataset and record_dataset and record_dataset != dataset:
            return False
        if asset and record_asset != asset:
            return False
        if candidate_id and record_candidate and record_candidate != candidate_id:
            return False
        if artifact_id and record_artifact and record_artifact != artifact_id:
            return False
        return bool(record_asset) and record.get("cycle_index", record.get("cycle_id")) not in (None, "")

    matches = [record for record in records if compatible(record)]
    if not matches:
        return None
    raw_backed = [
        record for record in matches
        if str(record.get("prediction_record_id") or "") in curve_refs_by_record
    ]
    if raw_backed:
        feature_backed = [
            record for record in raw_backed
            if {"ic", "dv"}.issubset(
                set(
                    curve_refs_by_record[
                        str(record.get("prediction_record_id") or "")
                    ].get("signals")
                    or []
                )
            )
        ]
        matches = feature_backed or raw_backed

    def information_key(record: dict) -> tuple[float, str, str]:
        error = record.get("absolute_error")
        if not isinstance(error, (int, float)):
            truth = record.get("true_soh")
            prediction = record.get("pred_soh")
            error = abs(float(prediction) - float(truth)) if all(
                isinstance(value, (int, float)) for value in (truth, prediction)
            ) else -1.0
        return (
            float(error),
            str(record.get("cell_id") or record.get("asset_id") or ""),
            str(record.get("cycle_index", record.get("cycle_id"))),
        )

    return max(matches, key=information_key)


def _sample_id(dataset: str, asset: object, cycle: object) -> str:
    return f"{dataset}:{asset}:{cycle}"


@lru_cache(maxsize=4)
def _load_hust_raw_cell(path_value: str, _mtime_ns: int) -> dict:
    import pickle

    with open(path_value, "rb") as handle:
        payload = pickle.load(handle)
    cell_name = Path(path_value).stem
    return (payload.get(cell_name) or {}).get("data") or {}


def _extract_hust_raw_curve(data_dir: Path, curve_key: str) -> dict | None:
    """Extract one cycle's t/V/I series from the HUST per-cell raw pickle."""
    cell, _, cycle_text = curve_key.partition("|")
    try:
        cycle_number = int(float(cycle_text))
    except (TypeError, ValueError):
        return None
    path = data_dir / "raw" / "hust" / "our_data" / f"{cell}.pkl"
    if not path.is_file():
        return None
    cycles = _load_hust_raw_cell(str(path), path.stat().st_mtime_ns)
    frame = cycles.get(cycle_number)
    if frame is None or not len(frame):
        return None
    ordered = frame.sort_values("Time (s)")
    return {
        "t_s": [float(v) for v in ordered["Time (s)"]],
        "voltage_v": [float(v) for v in ordered["Voltage (V)"]],
        "current_a": [float(v) / 1000.0 for v in ordered["Current (mA)"]],
    }


# Datasets whose raw acquisition files are local and per-cycle extractable.
_RAW_CURVE_EXTRACTORS = {
    "hust": _extract_hust_raw_curve,
}


def _curve_record_key(selection: dict) -> str:
    asset = selection.get("asset_id")
    cycle = selection.get("cycle_id")
    return f"{asset}|{cycle}" if asset and cycle not in (None, "") else ""


def _protocol_ref(config: dict) -> str:
    protocols = config.get("protocols") or []
    if not protocols:
        return ""
    descriptor = config.get("split_assignment") or {
        "protocols": protocols,
        "held_out_key": config.get("held_out_key") or "",
        "seeds": config.get("seeds") or [],
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"protocol@sha256:{hashlib.sha256(encoded).hexdigest()}"


def _digest_ref(kind: str, path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    stat = path.stat()
    return _digest_ref_versioned(
        kind,
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


@lru_cache(maxsize=32)
def _digest_ref_versioned(
    kind: str,
    path_value: str,
    _size: int,
    _mtime_ns: int,
) -> str:
    digest = hashlib.sha256(Path(path_value).read_bytes()).hexdigest()
    return f"{kind}@sha256:{digest}"


def _coverage(kind: str, available: bool, reason: str) -> dict:
    return {
        "kind": kind,
        "required": True,
        "available": available,
        "missing_reason": "" if available else reason,
    }


def _read_json(path: Path, fallback):
    if not path.is_file():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _json_object_has_key(path: Path, key: str) -> bool:
    return key in _read_curve_json(path)


def _read_curve_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    stat = path.stat()
    return _read_curve_json_versioned(
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


@lru_cache(maxsize=3)
def _read_curve_json_versioned(
    path_value: str,
    _size: int,
    _mtime_ns: int,
) -> dict:
    return _read_json(Path(path_value), {})


def _curve_signals(path: Path | None, key: str) -> list[str]:
    if path is None:
        return []
    payload = _read_curve_json(path)
    curve = payload.get(key) or {}
    return curve_signal_names(curve)


def _normalize_curve_signals(signals: list[str] | None) -> list[str]:
    requested = signals or ["voltage", "current", "temperature", "ic", "dv"]
    normalized: list[str] = []
    allowed = set(CURVE_SIGNAL_FIELDS) | CURVE_DERIVED_SIGNALS
    for value in requested:
        name = str(value or "").strip().lower()
        if not name or name in normalized:
            continue
        if name not in allowed:
            raise EvaluationCaseValidationError(f"unknown curve signal: {name}")
        normalized.append(name)
    if not normalized:
        raise EvaluationCaseValidationError("at least one curve signal is required")
    return normalized


def _curve_signal_payload(
    curve: dict,
    requested: list[str],
    max_points: int,
    *,
    voltage_window: list[float] | tuple[float, float] | None = None,
) -> dict:
    time_values = _finite_series(curve.get("t_s"))
    result: dict[str, dict] = {}
    for name, (field, unit) in CURVE_SIGNAL_FIELDS.items():
        if name not in requested:
            continue
        values = _finite_series(curve.get(field))
        if not values:
            continue
        x_values = time_values[:len(values)] if len(time_values) >= len(values) else list(range(len(values)))
        x_name = "time_s" if len(time_values) >= len(values) else "sample_index"
        x_unit = "s" if x_name == "time_s" else "index"
        result[name] = _sampled_signal(
            x_values,
            values,
            max_points,
            x_name=x_name,
            x_unit=x_unit,
            y_name=field,
            y_unit=unit,
            derivation="measured",
        )

    voltage = _finite_series(curve.get("voltage_v"))
    current = _finite_series(curve.get("current_a"))
    direct_ic = _finite_series(curve.get("ic_ah_per_v"))
    direct_dv = _finite_series(curve.get("dva_v_per_ah"))
    charge = _integrated_charge(time_values, current)
    derived_ic, derived_dv = _derive_ic_dv(
        voltage,
        charge,
        voltage_window=voltage_window,
    )

    if "ic" in requested:
        if direct_ic and voltage:
            count = min(len(direct_ic), len(voltage))
            result["ic"] = _sampled_signal(
                voltage[:count], direct_ic[:count], max_points,
                x_name="voltage_v", x_unit="V", y_name="ic_ah_per_v", y_unit="Ah/V",
                derivation="source",
            )
        elif derived_ic:
            result["ic"] = _sampled_signal(
                derived_ic[0], derived_ic[1], max_points,
                x_name="voltage_v", x_unit="V", y_name="ic_ah_per_v", y_unit="Ah/V",
                derivation="derived from measured V/I/time",
            )
    if "dv" in requested:
        if direct_dv and charge:
            count = min(len(direct_dv), len(charge))
            result["dv"] = _sampled_signal(
                charge[:count], direct_dv[:count], max_points,
                x_name="charge_ah", x_unit="Ah", y_name="dva_v_per_ah", y_unit="V/Ah",
                derivation="source",
            )
        elif derived_dv:
            result["dv"] = _sampled_signal(
                derived_dv[0], derived_dv[1], max_points,
                x_name="charge_ah", x_unit="Ah", y_name="dva_v_per_ah", y_unit="V/Ah",
                derivation="derived from measured V/I/time",
            )
    return result


def curve_signal_names(curve: dict) -> list[str]:
    """List measured and genuinely derivable signals for one raw curve."""
    return list(
        _curve_signal_payload(
            curve,
            ["voltage", "current", "temperature", "ic", "dv"],
            CURVE_RESOLUTIONS["thumbnail"],
        )
    )


def _feature_contract(curve_path: Path, dataset: str) -> dict:
    """Resolve the feature extraction window recorded beside a curve artifact."""
    dataset_name = str(dataset or "").strip().lower()
    alias = DATASET_FEATURE_ALIASES.get(dataset_name, dataset_name)
    curve_stem = curve_path.stem
    curve_alias = curve_stem[:-7] if curve_stem.endswith("_curves") else curve_stem
    candidates: list[Path] = []
    for name in (curve_alias, alias, dataset_name):
        candidate = curve_path.with_name(f"{name}.json")
        if candidate != curve_path and candidate not in candidates:
            candidates.append(candidate)
    # Generated per-record curves live in curves/generated/<dataset>/ and
    # inherit the dataset contract from the parent curve store.
    for name in (alias, dataset_name):
        for ancestor in (curve_path.parent.parent, curve_path.parent.parent.parent):
            candidate = ancestor / f"{name}.json"
            if candidate != curve_path and candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        payload = _read_json(candidate, {})
        window = _finite_window(payload.get("feature_window"))
        if window:
            return {
                "voltage_v": window,
                "approximate": bool(payload.get("feature_window_approx")),
                "feature_columns": [
                    str(value) for value in payload.get("feature_columns") or [] if value
                ],
                "source": f"dataset contract · {candidate.name}",
            }
    fallback = DEFAULT_FEATURE_WINDOWS.get(alias) or DEFAULT_FEATURE_WINDOWS.get(dataset_name)
    if fallback:
        window, approximate = fallback
        return {
            "voltage_v": list(window),
            "approximate": approximate,
            "feature_columns": [],
            "source": f"dataset contract fallback · {alias or dataset_name}",
        }
    return {
        "voltage_v": [],
        "approximate": False,
        "feature_columns": [],
        "source": "feature window unavailable",
    }


def _finite_window(values) -> list[float]:
    numbers = _finite_series(values)
    if len(numbers) != 2 or numbers[1] <= numbers[0]:
        return []
    return numbers


def _feature_window_payload(
    curve: dict,
    voltage_window,
    *,
    source: str,
    approximate: bool,
    feature_columns: list[str],
) -> dict:
    voltage = _finite_series(curve.get("voltage_v"))
    time_values = _finite_series(curve.get("t_s"))
    current = _finite_series(curve.get("current_a"))
    window = _finite_window(voltage_window)
    payload = {
        "voltage_v": window,
        "source": source,
        "approximate": approximate,
        "feature_columns": list(feature_columns),
        "available": False,
        "time_basis": "time_s" if time_values else "sample_index",
    }
    if not window or not voltage:
        return payload
    low, high = window
    indices = [index for index, value in enumerate(voltage) if low <= value <= high]
    if not indices:
        return payload
    start_index = min(indices)
    end_index = max(indices)
    payload.update({
        "available": True,
        "sample_index": [start_index, end_index],
        "point_count": len(indices),
    })
    if len(time_values) > end_index:
        origin = time_values[0]
        start_time = time_values[start_index]
        end_time = time_values[end_index]
        payload.update({
            "time_s": [start_time, end_time],
            "cycle_time_s": [start_time - origin, end_time - origin],
            "duration_s": max(0.0, end_time - start_time),
            "cycle_duration_s": max(
                0.0,
                time_values[min(len(time_values), len(voltage)) - 1] - origin,
            ),
        })
        charge = _integrated_charge(time_values, current)
        if len(charge) > end_index:
            payload["charge_ah"] = [charge[start_index], charge[end_index]]
    return payload


def _feature_curve_payload(
    curve: dict,
    feature_values: dict,
    feature_window: dict,
    max_points: int,
) -> list[dict]:
    """Build one trace for every feature actually consumed by the model."""
    if not feature_values:
        return []
    time_values = _finite_series(curve.get("t_s"))
    voltage = _finite_series(curve.get("voltage_v"))
    current = _finite_series(curve.get("current_a"))
    temperature = _finite_series(curve.get("temperature_c"))
    source_count = max(len(time_values), len(voltage), len(current), len(temperature), 2)
    if time_values:
        origin = time_values[0]
        cycle_time = [value - origin for value in time_values]
        x_name, x_unit = "cycle_time_s", "s"
    else:
        cycle_time = [float(index) for index in range(source_count)]
        x_name, x_unit = "sample_index", "index"
    charge = _integrated_charge(time_values, current)
    ic_time, ic_values = _derived_ic_over_time(
        time_values,
        voltage,
        charge,
        feature_window.get("voltage_v"),
    )
    full_highlight = [cycle_time[0], cycle_time[-1]] if len(cycle_time) >= 2 else []
    window_highlight = (
        feature_window.get("cycle_time_s")
        or feature_window.get("sample_index")
        or []
    )
    result: list[dict] = []
    for name, recorded_value in feature_values.items():
        feature_name = str(name)
        scope = "feature_window" if feature_name in WINDOW_FEATURES else (
            "cycle_context" if feature_name in CONTEXT_FEATURES else "full_cycle"
        )
        source_signal = "recorded_scalar"
        y_name = feature_name
        y_unit = _feature_unit(feature_name)
        derivation = "recorded feature value"
        x_values = list(cycle_time)
        y_values: list[float] = []

        if feature_name in {"charge_capacity_max", "q_win_ah"} and charge:
            x_values = cycle_time[:len(charge)]
            y_values = charge
            source_signal = "cumulative_charge"
            y_name = "charge_ah"
            y_unit = "Ah"
            derivation = "integrated from measured current/time"
        elif feature_name in {"time_s_max", "t_win_s"} and cycle_time:
            y_values = list(cycle_time)
            source_signal = "elapsed_time"
            y_name = "elapsed_time_s"
            y_unit = "s"
            derivation = "derived from measured time"
        elif feature_name.startswith("voltage_") and voltage:
            x_values = cycle_time[:len(voltage)]
            y_values = voltage[:len(x_values)]
            source_signal = "voltage"
            y_name = "voltage_v"
            y_unit = "V"
            derivation = "measured voltage underlying statistic"
        elif (
            feature_name.startswith("current_")
            or feature_name == "abs_current_mean"
        ) and current:
            x_values = cycle_time[:len(current)]
            y_values = current[:len(x_values)]
            if feature_name == "abs_current_mean":
                y_values = [abs(value) for value in y_values]
            source_signal = "current"
            y_name = "current_a"
            y_unit = "A"
            derivation = "measured current underlying statistic"
        elif feature_name.startswith("temperature_") and temperature:
            x_values = cycle_time[:len(temperature)]
            y_values = temperature[:len(x_values)]
            source_signal = "temperature"
            y_name = "temperature_c"
            y_unit = "°C"
            derivation = "measured temperature underlying statistic"
        elif feature_name == "ic_peak_ah_per_v" and ic_values:
            x_values = ic_time
            y_values = ic_values
            source_signal = "ic"
            y_name = "ic_ah_per_v"
            y_unit = "Ah/V"
            derivation = "derived from measured V/I/time inside feature window"
        elif feature_name == "ic_peak_voltage_v" and voltage:
            x_values = cycle_time[:len(voltage)]
            y_values = voltage[:len(x_values)]
            source_signal = "voltage"
            y_name = "voltage_v"
            y_unit = "V"
            derivation = "measured voltage inside feature window"
        elif "resistance" in feature_name and voltage and current:
            count = min(len(cycle_time), len(voltage), len(current))
            x_values = cycle_time[:count]
            y_values = [
                voltage[index] / max(abs(current[index]), 1e-6)
                for index in range(count)
            ]
            source_signal = "voltage/current"
            y_name = "resistance_proxy"
            y_unit = "Ω"
            derivation = "derived voltage/current proxy"
        elif "voltage" in feature_name and voltage:
            x_values = cycle_time[:len(voltage)]
            y_values = voltage[:len(x_values)]
            source_signal = "voltage"
            y_name = "voltage_v"
            y_unit = "V"
            derivation = "measured voltage proxy"

        if not y_values:
            scalar = _finite_number(recorded_value)
            if scalar is not None:
                if len(x_values) < 2:
                    x_values = [0.0, 1.0]
                y_values = [scalar for _ in x_values]
            else:
                x_values = []
        signal = _sampled_signal(
            x_values,
            y_values,
            max_points,
            x_name=x_name,
            x_unit=x_unit,
            y_name=y_name,
            y_unit=y_unit,
            derivation=derivation,
        ) if x_values and y_values else {
            "x_name": x_name,
            "x_unit": x_unit,
            "y_name": y_name,
            "y_unit": y_unit,
            "x": [],
            "y": [],
            "source_points": 0,
            "returned_points": 0,
            "derivation": derivation,
        }
        highlight = window_highlight if scope == "feature_window" else (
            full_highlight if scope == "full_cycle" else []
        )
        signal.update({
            "name": feature_name,
            "value": recorded_value,
            "scope": scope,
            "source_signal": source_signal,
            "highlight": {
                "cycle_time_s": list(highlight),
                "voltage_v": list(feature_window.get("voltage_v") or []),
                "label": (
                    "feature extraction window" if scope == "feature_window"
                    else "full cycle" if scope == "full_cycle"
                    else "cross-cycle context"
                ),
            },
        })
        result.append(signal)
    return result


def _derived_ic_over_time(
    time_values: list[float],
    voltage_values: list[float],
    charge_values: list[float],
    voltage_window,
) -> tuple[list[float], list[float]]:
    count = min(len(time_values), len(voltage_values), len(charge_values))
    window = _finite_window(voltage_window) or [3.7, 4.1]
    if count < 3:
        return [], []
    origin = time_values[0]
    x_values: list[float] = []
    y_values: list[float] = []
    for index in range(1, count):
        voltage_mid = (voltage_values[index] + voltage_values[index - 1]) / 2.0
        delta_voltage = voltage_values[index] - voltage_values[index - 1]
        delta_charge = charge_values[index] - charge_values[index - 1]
        if not window[0] <= voltage_mid <= window[1] or delta_voltage <= 1e-5 or delta_charge <= 1e-8:
            continue
        x_values.append((time_values[index] + time_values[index - 1]) / 2.0 - origin)
        y_values.append(delta_charge / delta_voltage)
    return x_values, y_values


def _feature_unit(name: str) -> str:
    if name in {"q_win_ah", "charge_capacity_max"}:
        return "Ah"
    if name in {"t_win_s", "time_s_max"}:
        return "s"
    if name == "ic_peak_ah_per_v":
        return "Ah/V"
    if name == "ic_peak_voltage_v" or "voltage" in name:
        return "V"
    if "temperature" in name or "temp" in name:
        return "°C"
    if "current" in name:
        return "A"
    if "resistance" in name:
        return "Ω"
    return ""


def _finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_series(values) -> list[float]:
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        result.append(number)
    return result


def _integrated_charge(time_values: list[float], current_values: list[float]) -> list[float]:
    count = min(len(time_values), len(current_values))
    if count < 2:
        return []
    charge = [0.0]
    for index in range(1, count):
        dt = max(0.0, time_values[index] - time_values[index - 1])
        mean_current = max(0.0, (current_values[index] + current_values[index - 1]) / 2.0)
        charge.append(charge[-1] + mean_current * dt / 3600.0)
    return charge


def _derive_ic_dv(
    voltage_values: list[float],
    charge_values: list[float],
    *,
    voltage_window: list[float] | tuple[float, float] | None = None,
) -> tuple[tuple[list[float], list[float]] | None, tuple[list[float], list[float]] | None]:
    count = min(len(voltage_values), len(charge_values))
    if count < 3:
        return None, None
    ic_x: list[float] = []
    ic_y: list[float] = []
    dv_x: list[float] = []
    dv_y: list[float] = []
    window = _finite_window(voltage_window) or [3.7, 4.1]
    for index in range(1, count):
        voltage_mid = (voltage_values[index] + voltage_values[index - 1]) / 2.0
        charge_mid = (charge_values[index] + charge_values[index - 1]) / 2.0
        delta_voltage = voltage_values[index] - voltage_values[index - 1]
        delta_charge = charge_values[index] - charge_values[index - 1]
        if not window[0] <= voltage_mid <= window[1] or delta_voltage <= 1e-5 or delta_charge <= 1e-8:
            continue
        ic_x.append(voltage_mid)
        ic_y.append(delta_charge / delta_voltage)
        dv_x.append(charge_mid)
        dv_y.append(delta_voltage / delta_charge)
    if len(ic_x) < 2:
        return None, None
    return (ic_x, ic_y), (dv_x, dv_y)


def _sampled_signal(
    x_values: list[float],
    y_values: list[float],
    max_points: int,
    *,
    x_name: str,
    x_unit: str,
    y_name: str,
    y_unit: str,
    derivation: str,
) -> dict:
    count = min(len(x_values), len(y_values))
    indices = _uniform_indices(count, max_points)
    return {
        "x_name": x_name,
        "x_unit": x_unit,
        "y_name": y_name,
        "y_unit": y_unit,
        "x": [x_values[index] for index in indices],
        "y": [y_values[index] for index in indices],
        "source_points": count,
        "returned_points": len(indices),
        "derivation": derivation,
    }


def _uniform_indices(count: int, max_points: int) -> list[int]:
    if count <= max_points:
        return list(range(count))
    if max_points <= 1:
        return [0]
    return sorted({round(index * (count - 1) / (max_points - 1)) for index in range(max_points)})
