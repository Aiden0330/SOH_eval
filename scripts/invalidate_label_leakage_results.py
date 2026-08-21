from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sohbench.case_evidence import write_prediction_trajectory_index  # noqa: E402
from sohbench.evaluation_service import EvaluationService  # noqa: E402
from sohbench.feature_safety import (  # noqa: E402
    LABEL_SAFETY_CONTRACT,
    suspicious_feature_names,
)
from sohbench.model_store import ModelStore  # noqa: E402
from sohbench.paths import DEFAULT_DATA_DIR  # noqa: E402


METRIC_FIELDS = (
    "mae",
    "rmse",
    "mean_bias_error",
    "picp",
    "monotonicity_violation_rate",
    "value_range_violation",
)
INVALIDATION_REASON = (
    "标签泄漏：模型输入包含真实 SOH 或其确定性派生字段；该模型及其既有测评指标不再可用"
)
LEGACY_LEAKING_FEATURESETS = {"smoke_cycle_shape"}


def _write_json(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _contaminated_models(store: ModelStore) -> list[dict]:
    candidates: list[dict] = []
    for manifest_path in (store.base_dir / "_artifacts").glob("*/model_manifest.json"):
        try:
            item = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidates.append({
            **item,
            "name": str(item.get("name") or item.get("model_name") or ""),
            "artifact_id": f"model@sha256:{manifest_path.parent.name}",
        })
    candidates.extend(store._catalog.get("models", []))

    contaminated: dict[str, dict] = {}
    for item in candidates:
        columns = list(item.get("feature_columns") or [])
        suspicious = suspicious_feature_names(columns, target=str(item.get("target") or "soh"))
        if not suspicious:
            continue
        name = str(item.get("name") or item.get("model_name") or "")
        artifact_id = str(item.get("artifact_id") or store.artifact_id(name))
        contaminated[artifact_id] = {
            "name": str(item.get("name") or ""),
            "artifact_id": artifact_id,
            "estimator": str(item.get("estimator") or ""),
            "featureset": str(item.get("featureset") or "full"),
            "protocol": str(item.get("protocol") or ""),
            "held_out_key": str(item.get("held_out_key") or ""),
            "seed": int(item.get("seed") or 0),
            "columns": suspicious,
            "created_by_run": str(item.get("created_by_run") or ""),
        }
    return list(contaminated.values())


def _recompute_summary(records: list[dict]) -> dict:
    valid = [row for row in records if row.get("status") == "succeeded"]
    fastest = min(
        (
            row for row in valid
            if isinstance(row.get("ms_per_1000_samples"), (int, float))
            and math.isfinite(float(row["ms_per_1000_samples"]))
        ),
        key=lambda row: float(row["ms_per_1000_samples"]),
        default=None,
    )
    best = min(
        (
            row for row in valid
            if isinstance(row.get("mae"), (int, float))
            and math.isfinite(float(row["mae"]))
        ),
        key=lambda row: float(row["mae"]),
        default=None,
    )
    return {
        "n_records": len(records),
        "n_succeeded": len(valid),
        "n_invalidated": sum(row.get("status") == "invalidated" for row in records),
        "fastest_model": fastest.get("model") if fastest else None,
        "fastest_ms_per_1000_samples": fastest.get("ms_per_1000_samples") if fastest else None,
        "best_transfer_model": best.get("model") if best else None,
        "best_transfer_mae": best.get("mae") if best else None,
        "label_safety_enforced": True,
    }


def _invalidate_benchmark(path: Path, bad_ids: set[str], bad_names: set[str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    changed = 0
    bad_candidates: set[str] = set()
    for row in records:
        if (
            str(row.get("model_artifact_id") or "") not in bad_ids
            and str(row.get("model") or "") not in bad_names
        ):
            continue
        bad_candidates.add(str(row.get("candidate_id") or ""))
        if row.get("status") == "invalidated":
            continue
        row["invalidated_result"] = {
            "status": row.get("status"),
            "error": row.get("error"),
            **{key: row.get(key) for key in METRIC_FIELDS},
        }
        for key in METRIC_FIELDS:
            if key in row:
                row[key] = None
        row["status"] = "invalidated"
        row["error"] = INVALIDATION_REASON
        row["validity_status"] = "invalidated"
        row["invalidation_reason"] = INVALIDATION_REASON
        changed += 1
    if changed:
        payload["records"] = records
        payload["summary"] = _recompute_summary(records)
        _write_json(path, payload)
        csv_path = path.with_name("inference_results.csv")
        parquet_path = path.with_name("inference_results.parquet")
        frame = pd.DataFrame(records)
        if csv_path.is_file():
            frame.to_csv(csv_path, index=False)
        if parquet_path.is_file():
            frame.to_parquet(parquet_path, index=False)
    return {"changed": changed, "bad_candidates": {value for value in bad_candidates if value}}


def _invalidate_prediction_evidence(root: Path, bad_ids: set[str], bad_candidates: set[str]) -> int:
    path = root / "prediction_records.json"
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    changed = 0
    for row in records:
        if (
            str(row.get("model_artifact_id") or "") not in bad_ids
            and str(row.get("candidate_id") or "") not in bad_candidates
        ):
            continue
        if row.get("validity_status") == "invalidated":
            continue
        row["validity_status"] = "invalidated"
        row["invalidation_reason"] = INVALIDATION_REASON
        changed += 1
    if changed:
        payload["records"] = records
        payload["valid_records"] = sum(
            str(row.get("validity_status") or "valid") != "invalidated" for row in records
        )
        payload["invalidated_records"] = sum(
            str(row.get("validity_status") or "valid") == "invalidated" for row in records
        )
        _write_json(path, payload)
        write_prediction_trajectory_index(root, records)

        contributions_path = root / "metric_contributions.json"
        if contributions_path.is_file():
            contributions = json.loads(contributions_path.read_text(encoding="utf-8"))
            for row in contributions.get("records") or []:
                if str(row.get("candidate_id") or "") in bad_candidates:
                    row["validity_status"] = "invalidated"
                    row["invalidation_reason"] = INVALIDATION_REASON
            _write_json(contributions_path, contributions)

        manifest_path = root / "inference_evidence_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["prediction_record_count"] = payload["valid_records"]
            manifest["invalidated_prediction_record_count"] = payload["invalidated_records"]
            manifest["invalidated_candidate_ids"] = sorted(bad_candidates)
            manifest["label_safety_enforced"] = True
            _write_json(manifest_path, manifest)
    return changed


def _update_run_catalog(path: Path, invalidations: dict[str, dict]) -> int:
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for item in payload.get("models") or []:
        invalidation = invalidations.get(str(item.get("artifact_id") or ""))
        if not invalidation:
            continue
        item["status"] = "invalidated"
        item["usable"] = False
        validation = dict(item.get("validation") or {})
        validation.update({"label_safety": "blocked", "usable": False, "invalidation": invalidation})
        item["validation"] = validation
        changed += 1
    if changed:
        _write_json(path, payload)
    return changed


def _historical_invalidation_totals(
    artifact_roots: set[Path],
    bad_ids: set[str],
    bad_names: set[str],
) -> dict:
    affected_roots = 0
    benchmark_rows = 0
    prediction_records = 0
    for root in artifact_roots:
        benchmark_path = root / "inference_benchmark.json"
        if not benchmark_path.is_file():
            continue
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        rows = [
            row for row in benchmark.get("records") or []
            if row.get("status") == "invalidated"
            and (
                str(row.get("model_artifact_id") or "") in bad_ids
                or str(row.get("model") or "") in bad_names
            )
        ]
        if not rows:
            continue
        affected_roots += 1
        benchmark_rows += len(rows)
        bad_candidates = {str(row.get("candidate_id") or "") for row in rows}
        records_path = root / "prediction_records.json"
        if not records_path.is_file():
            continue
        records = json.loads(records_path.read_text(encoding="utf-8"))
        prediction_records += sum(
            str(row.get("validity_status") or "") == "invalidated"
            and (
                str(row.get("model_artifact_id") or "") in bad_ids
                or str(row.get("candidate_id") or "") in bad_candidates
            )
            for row in records.get("records") or []
        )
    return {
        "affected_artifact_roots": affected_roots,
        "invalidated_benchmark_rows": benchmark_rows,
        "invalidated_prediction_records": prediction_records,
    }


def _merge_audit_rows(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in [*existing, *incoming]:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        merged[key] = row
    return list(merged.values())


def _legacy_unsafe_featuresets(payload: dict) -> set[str]:
    if payload.get("label_safety_contract") == LABEL_SAFETY_CONTRACT:
        return set()
    requested = payload.get("featuresets") or (payload.get("request") or {}).get("featuresets") or []
    return {str(name) for name in requested}.intersection(LEGACY_LEAKING_FEATURESETS)


def _name_matches_featuresets(name: str, featuresets: set[str]) -> bool:
    return any(f"_{featureset}_" in f"_{name}_" for featureset in featuresets)


def _invalidate_training_artifacts(
    root: Path,
    contaminated: list[dict],
    bad_ids: set[str],
    bad_names: set[str],
) -> dict:
    training_roots = {path.parent for path in (root / "runs").glob("*/leaderboard.json")}
    training_roots.update(
        path.parent for path in (root / "frontend" / "app" / "data").rglob("leaderboard.json")
    )
    affected_roots = 0
    leaderboard_rows = 0
    metric_rows = 0
    newly_archived = 0
    for artifact_root in sorted(training_roots):
        archive_path = artifact_root / "invalidated_training_results.json"
        archive = (
            json.loads(archive_path.read_text(encoding="utf-8"))
            if archive_path.is_file()
            else {
                "contract": "sohbench.label-safety.v1",
                "reason": INVALIDATION_REASON,
                "artifacts": {},
            }
        )
        config_path = artifact_root / "run_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        legacy_featuresets = _legacy_unsafe_featuresets(config)
        changed = False
        for filename in ("leaderboard.json", "dataset_leaderboard.json"):
            path = artifact_root / filename
            if not path.is_file():
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                continue
            invalid = [
                row for row in rows
                if bad_ids.intersection(str(value) for value in row.get("model_artifact_ids") or [])
                or str(row.get("model_artifact_id") or "") in bad_ids
                or str(row.get("featureset") or "") in legacy_featuresets
            ]
            if invalid:
                valid = [row for row in rows if row not in invalid]
                existing = list((archive.get("artifacts") or {}).get(filename) or [])
                archive.setdefault("artifacts", {})[filename] = _merge_audit_rows(existing, invalid)
                _write_json(path, valid)
                newly_archived += len(invalid)
                changed = True

        job_id = str(config.get("job_id") or config.get("run_id") or "")
        specs = [item for item in contaminated if item.get("created_by_run") == job_id]
        results_path = artifact_root / "results.parquet"
        csv_path = artifact_root / "results.csv"
        if specs and (results_path.is_file() or csv_path.is_file()):
            frame = pd.read_parquet(results_path) if results_path.is_file() else pd.read_csv(csv_path)
            mask = pd.Series(False, index=frame.index)
            for spec in specs:
                current = pd.Series(True, index=frame.index)
                for column in ("estimator", "featureset", "protocol", "held_out_key"):
                    if column in frame.columns:
                        current &= frame[column].astype(str) == str(spec.get(column) or "")
                if "seed" in frame.columns:
                    current &= pd.to_numeric(frame["seed"], errors="coerce") == int(spec.get("seed") or 0)
                mask |= current
            if "featureset" in frame.columns and legacy_featuresets:
                mask |= frame["featureset"].astype(str).isin(legacy_featuresets)
            if bool(mask.any()):
                invalid_frame = frame.loc[mask]
                valid_frame = frame.loc[~mask]
                invalid_rows = json.loads(invalid_frame.to_json(orient="records"))
                archive["results_records"] = _merge_audit_rows(
                    list(archive.get("results_records") or []), invalid_rows
                )
                if results_path.is_file():
                    valid_frame.to_parquet(results_path, index=False)
                if csv_path.is_file():
                    valid_frame.to_csv(csv_path, index=False)
                newly_archived += len(invalid_rows)
                changed = True

        if config:
            invalidated_names = {
                str(item.get("name") or "")
                for item in config.get("model_artifacts") or []
                if str(item.get("artifact_id") or "") in bad_ids
                or str(item.get("featureset") or "") in legacy_featuresets
            }
            invalidated_names.update(
                str(name) for name in config.get("saved_models") or []
                if str(name) in bad_names
                or _name_matches_featuresets(str(name), legacy_featuresets)
            )
            if invalidated_names:
                for item in config.get("model_artifacts") or []:
                    if (
                        str(item.get("artifact_id") or "") in bad_ids
                        or str(item.get("featureset") or "") in legacy_featuresets
                    ):
                        item["status"] = "invalidated"
                        item["usable"] = False
                        item["invalidation_reason"] = INVALIDATION_REASON
                config["invalidated_saved_models"] = sorted(
                    set(config.get("invalidated_saved_models") or []).union(invalidated_names)
                )
                config["saved_models"] = [
                    name for name in config.get("saved_models") or []
                    if str(name) not in invalidated_names
                ]
                config["result_validity"] = {
                    "status": "partially_invalidated",
                    "contract": "sohbench.label-safety.v1",
                    "reason": INVALIDATION_REASON,
                }
                _write_json(config_path, config)
                changed = True

        archived_artifacts = archive.get("artifacts") or {}
        current_leaderboard_rows = sum(len(rows) for rows in archived_artifacts.values())
        current_metric_rows = len(archive.get("results_records") or [])
        if current_leaderboard_rows or current_metric_rows:
            affected_roots += 1
            leaderboard_rows += current_leaderboard_rows
            metric_rows += current_metric_rows
            archive["invalidated_model_artifact_ids"] = sorted(bad_ids)
            archive["invalidated_model_names"] = sorted(bad_names)
            _write_json(archive_path, archive)
        elif changed:
            _write_json(archive_path, archive)
    return {
        "affected_training_roots": affected_roots,
        "invalidated_training_leaderboard_rows": leaderboard_rows,
        "invalidated_training_metric_rows": metric_rows,
        "newly_archived_training_rows": newly_archived,
    }


def _update_job_documents(work_dir: Path, bad_ids: set[str], bad_names: set[str]) -> dict:
    summary = {"training_jobs": 0, "inference_jobs": 0}
    for kind, directory in (
        ("training_jobs", work_dir / "jobs"),
        ("inference_jobs", work_dir / "inference_jobs"),
    ):
        for path in directory.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            legacy_featuresets = (
                _legacy_unsafe_featuresets(payload)
                if kind == "training_jobs"
                else set()
            )
            artifacts = (
                payload.get("model_artifacts") or []
                if kind == "training_jobs"
                else (payload.get("request") or {}).get("model_bindings") or []
            )
            affected = [
                str(item.get("artifact_id") or item.get("model_artifact_id") or "")
                for item in artifacts
                if (
                    str(item.get("artifact_id") or item.get("model_artifact_id") or "") in bad_ids
                    or str(item.get("featureset") or "") in legacy_featuresets
                )
            ]
            affected_names = {
                str(name) for name in payload.get("saved_models") or []
                if str(name) in bad_names
                or _name_matches_featuresets(str(name), legacy_featuresets)
            }
            if not affected and not affected_names:
                continue
            payload["result_validity"] = {
                "status": "partially_invalidated",
                "contract": "sohbench.label-safety.v1",
                "invalidated_model_artifact_ids": sorted(set(affected)),
                "reason": INVALIDATION_REASON,
            }
            if kind == "training_jobs":
                invalidated_names = set(affected_names)
                for item in payload.get("model_artifacts") or []:
                    if (
                        str(item.get("artifact_id") or "") not in bad_ids
                        and str(item.get("featureset") or "") not in legacy_featuresets
                    ):
                        continue
                    item["status"] = "invalidated"
                    item["usable"] = False
                    item["invalidation_reason"] = INVALIDATION_REASON
                    invalidated_names.add(str(item.get("name") or ""))
                payload["invalidated_saved_models"] = sorted(
                    set(payload.get("invalidated_saved_models") or []).union(invalidated_names)
                )
                payload["saved_models"] = [
                    name for name in payload.get("saved_models") or []
                    if str(name) not in invalidated_names
                ]
                summary_payload = dict(payload.get("result_summary") or {})
                summary_payload["model_artifact_count"] = sum(
                    item.get("usable", True) is not False
                    for item in payload.get("model_artifacts") or []
                )
                summary_payload["model_artifact_ids"] = [
                    item.get("artifact_id") for item in payload.get("model_artifacts") or []
                    if item.get("artifact_id") and item.get("usable", True) is not False
                ]
                summary_payload["invalidated_model_artifact_count"] = len(
                    [
                        item for item in payload.get("model_artifacts") or []
                        if item.get("usable", True) is False
                    ]
                )
                payload["result_summary"] = summary_payload
            _write_json(path, payload)
            summary[kind] += 1
    return summary


def _refresh_cases(service: EvaluationService, bad_ids: set[str]) -> dict:
    summary = {"cases_updated": 0, "saved_combinations_removed": 0, "cases_skipped": []}
    for case in service.list_evaluation_cases(include_archived=True):
        candidates = list(case.get("candidates") or [])
        saved = list(case.get("saved_combinations") or [])
        invalid_saved = [
            item["saved_combination_id"]
            for item in saved
            if str((item.get("candidate") or {}).get("model_artifact_id") or "") in bad_ids
        ]
        invalid_candidates = [
            item for item in candidates
            if str(item.get("model_artifact_id") or "") in bad_ids
        ]
        if not invalid_saved and not invalid_candidates:
            continue
        if case.get("lifecycle_state") in {"completed", "archived"}:
            summary["cases_skipped"].append(str(case.get("id") or ""))
            continue
        if invalid_saved:
            case = service.evaluation_cases.update_saved_combinations(
                case["id"],
                {
                    "action": "remove",
                    "saved_combination_ids": invalid_saved,
                    "expected_version": case["version"],
                    "expected_context_binding_id": (case.get("context_binding") or {}).get("binding_id"),
                    "actor": "label-safety-migration",
                    "reason": INVALIDATION_REASON,
                },
            )
            summary["saved_combinations_removed"] += len(invalid_saved)
        execution = dict(case.get("execution") or {})
        case = service.update_evaluation_case_identity(
            case["id"],
            {
                "execution": execution,
                "expected_version": case["version"],
                "expected_context_binding_id": (case.get("context_binding") or {}).get("binding_id"),
                "actor": "label-safety-migration",
                "reason": INVALIDATION_REASON,
            },
        )
        summary["cases_updated"] += 1
    return summary


def migrate(root: Path) -> dict:
    work_dir = root / "work" / "trusted_plugins"
    model_store = ModelStore(work_dir / "models")
    contaminated = _contaminated_models(model_store)
    invalidations: dict[str, dict] = {}
    for item in contaminated:
        existing = model_store.invalidation(item["artifact_id"])
        invalidations[item["artifact_id"]] = existing or model_store.invalidate_artifact(
                item["artifact_id"],
                reason=INVALIDATION_REASON,
                details={
                    "featureset": item["featureset"],
                    "feature_columns": item["columns"],
                    "created_by_run": item["created_by_run"],
                },
            )
    bad_ids = set(invalidations)
    bad_names = {item["name"] for item in contaminated}

    training = _invalidate_training_artifacts(
        root,
        contaminated,
        bad_ids,
        bad_names,
    )

    artifact_roots = {path.parent for path in (root / "runs").glob("*/inference_benchmark.json")}
    frontend_data = root / "frontend" / "app" / "data"
    artifact_roots.update(path.parent for path in frontend_data.rglob("inference_benchmark.json"))
    updated_runs = 0
    newly_invalidated_rows = 0
    newly_invalidated_predictions = 0
    for artifact_root in sorted(artifact_roots):
        result = _invalidate_benchmark(artifact_root / "inference_benchmark.json", bad_ids, bad_names)
        if result["changed"]:
            updated_runs += 1
            newly_invalidated_rows += int(result["changed"])
            newly_invalidated_predictions += _invalidate_prediction_evidence(
                artifact_root,
                bad_ids,
                result["bad_candidates"],
            )
        _update_run_catalog(artifact_root / "model_catalog.json", invalidations)

    historical = _historical_invalidation_totals(artifact_roots, bad_ids, bad_names)
    jobs = _update_job_documents(work_dir, bad_ids, bad_names)
    service = EvaluationService(
        data_dir=DEFAULT_DATA_DIR,
        work_dir=work_dir,
        runs_dir=root / "runs",
        frontend_data_dir=frontend_data,
    )
    cases = _refresh_cases(service, bad_ids)
    report = {
        "contract": "sohbench.label-safety.v1",
        "status": "completed",
        "invalidated_models": len(contaminated),
        **historical,
        **training,
        "newly_updated_artifact_roots": updated_runs,
        "newly_invalidated_benchmark_rows": newly_invalidated_rows,
        "newly_invalidated_prediction_records": newly_invalidated_predictions,
        **jobs,
        **cases,
        "reason": INVALIDATION_REASON,
        "models": contaminated,
    }
    _write_json(work_dir / "label_safety_migration_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Invalidate historical SOH results contaminated by target leakage.")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = migrate(args.root.resolve())
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps({key: value for key, value in report.items() if key != "models"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
