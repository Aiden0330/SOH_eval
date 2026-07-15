from __future__ import annotations
import hashlib, json, math, os, stat, unicodedata
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from .acceptance import evaluate_verdicts
from .fabric.evidence import (
    _validate_portable_path_segment,
    validate_prediction_records,
    write_evidence_pack,
)
from .fabric.ids import new_id
from .fabric.outcomes import (
    RunOutcome,
    is_finite_real_scalar,
    iter_execution_units,
    successful_result_rows,
)
from .tasks import SOHEstimationTask


_RUN_RESERVATION_PREFIX = ".sohbench-run-reservation-"


def _canonical_run_name(run_id: str) -> str:
    normalized = unicodedata.normalize("NFC", run_id)
    return unicodedata.normalize("NFC", normalized.casefold())


def _reservation_key(run_id: str) -> str:
    canonical = _canonical_run_name(run_id)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reservation_name(run_id: str) -> str:
    return _RUN_RESERVATION_PREFIX + _reservation_key(run_id)


def _json_safe(o):
    """Recursively replace non-finite floats (NaN/inf) with None so the output is
    valid JSON (browsers/JSON.parse reject bare NaN). NaN appears for failed folds
    or estimators that are N/A on a feature set (e.g. phys_resid without throughput)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def _dump(obj, path):
    path.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False, default=str))


def _successful_units_have_complete_metrics(
    evidence: pd.DataFrame, required_metrics: set[str],
) -> bool:
    for unit in iter_execution_units(evidence):
        finite_mask = unit["value"].map(is_finite_real_scalar).to_numpy(dtype=bool)
        finite_unit = unit.loc[finite_mask]
        if not required_metrics.issubset(set(finite_unit["metric"].dropna())):
            return False
    return True


def _leaderboard_entries(results: pd.DataFrame, task, group_cols: list[str]) -> list[dict]:
    entries = []
    has_track = "track" in results.columns

    for group_key, g in results.groupby(group_cols):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        keys = dict(zip(group_cols, group_key))

        est = keys["estimator"]
        fs_name = keys.get("featureset", "full")
        proto = keys["protocol"]
        track = keys.get("track")
        if track is None:
            if has_track:
                track = g["track"].iloc[0]
            else:
                track = results["feature_mode"].iloc[0] if "feature_mode" in results.columns else "unified"

        execution = RunOutcome.from_results(g)
        evidence = successful_result_rows(g)
        finite_mask = evidence["value"].map(is_finite_real_scalar).to_numpy(dtype=bool)
        finite_evidence = evidence.loc[finite_mask].copy()
        finite_evidence["value"] = [float(value) for value in finite_evidence["value"]]
        agg = finite_evidence.groupby("metric")["value"].agg(["mean", "std"])
        metric_names = g["metric"].dropna().drop_duplicates()
        metrics = {}
        for metric in metric_names:
            if metric in agg.index:
                row = agg.loc[metric]
                metrics[metric] = {
                    "mean": float(row["mean"]),
                    "std": float(row["std"] if math.isfinite(row["std"]) else 0.0),
                }
            else:
                metrics[metric] = {"mean": float("nan"), "std": 0.0}
        flat = {
            metric: values["mean"]
            for metric, values in metrics.items()
            if math.isfinite(values["mean"])
        }
        coverage = execution.succeeded / execution.attempted if execution.attempted else 0.0
        required_acceptance_metrics = {
            "mae", "picp", "monotonicity_violation_rate", "value_range_violation",
        }
        verdicts = (
            {v.name: v.passed for v in evaluate_verdicts(flat, task)}
            if (
                required_acceptance_metrics.issubset(flat)
                and _successful_units_have_complete_metrics(
                    evidence, required_acceptance_metrics,
                )
            )
            else {}
        )
        ci_source = g["ci_source"].iloc[0] if "ci_source" in g.columns else "conformal"

        entry = {
            "estimator": est,
            "featureset": fs_name,
            "track": track,
            "protocol": proto,
            "metrics": metrics,
            "verdicts": verdicts,
            "coverage": coverage,
            "execution": asdict(execution),
            "ci_source": ci_source,
        }
        if "held_out_key" in keys:
            entry["held_out_key"] = str(keys["held_out_key"])
        if "backend" in g.columns:
            entry["backend"] = g["backend"].iloc[0]
        entries.append(entry)

    return entries


def _run_identity(run_id):
    selected = new_id("RUN") if run_id is None else run_id
    if not isinstance(selected, str) or not selected:
        raise ValueError("run_id must be a non-empty directory name")
    if "/" in selected or "\\" in selected:
        raise ValueError("run_id must be a single relative directory name")
    try:
        _validate_portable_path_segment(selected, "run_id")
    except ValueError as exc:
        raise ValueError(f"invalid run_id: {selected}") from exc
    if _canonical_run_name(selected).startswith(
        _canonical_run_name(_RUN_RESERVATION_PREFIX)
    ):
        raise ValueError(f"run_id is reserved: {selected}")
    return selected


def _evidence_manifest(run_id, evidence_context):
    if evidence_context is None:
        context = {}
    elif isinstance(evidence_context, dict):
        context = dict(evidence_context)
    else:
        raise TypeError("evidence_context must be a dict or None")
    for reserved in ("schema_version", "artifacts"):
        if reserved in context:
            raise ValueError(f"evidence_context cannot override {reserved}")
    if "run_id" in context and context["run_id"] != run_id:
        raise ValueError("evidence_context run_id does not match explicit run_id")
    context.pop("run_id", None)
    return {"run_id": run_id, **context}


def _is_directory_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _require_within_root(target: Path, root: Path) -> None:
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path resolves outside out_dir: {target}") from exc


def _prepare_run_root(out_dir) -> Path:
    root_path = Path(out_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    if _is_directory_link(root_path):
        raise ValueError(f"out_dir cannot be a link or reparse point: {root_path}")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"out_dir is not an ordinary directory: {root_path}")
    return root


def _canonical_entries(root: Path, name: str) -> list[Path]:
    canonical_name = _canonical_run_name(name)
    return [
        entry
        for entry in root.iterdir()
        if _canonical_run_name(entry.name) == canonical_name
    ]


def _directory_identity(snapshot) -> tuple[int, int]:
    return snapshot.st_dev, snapshot.st_ino


def _require_ordinary_directory_snapshot(snapshot, path: Path, label: str) -> None:
    attributes = getattr(snapshot, "st_file_attributes", 0)
    is_reparse = bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    if (
        not stat.S_ISDIR(snapshot.st_mode)
        or stat.S_ISLNK(snapshot.st_mode)
        or is_reparse
    ):
        raise ValueError(
            f"{label} is not an ordinary non-reparse directory: {path}"
        )


def _validate_direct_directory_snapshot(
    path: Path, root: Path, label: str, snapshot,
) -> None:
    if path.parent != root:
        raise ValueError(f"{label} must be directly under out_dir")
    _require_ordinary_directory_snapshot(snapshot, path, label)
    resolved = path.resolve(strict=True)
    _require_within_root(resolved, root)
    if resolved.parent != root:
        raise ValueError(f"{label} must resolve directly under out_dir")


def _capture_created_directory_identity(
    path: Path, root: Path, label: str,
) -> tuple[int, int]:
    try:
        snapshot = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect newly created {label}: {path}") from exc
    _validate_direct_directory_snapshot(path, root, label, snapshot)
    return _directory_identity(snapshot)


def _validate_directory_handoff(
    path: Path,
    root: Path,
    label: str,
    captured_identity: tuple[int, int],
) -> Path:
    try:
        snapshot = path.lstat()
        _validate_direct_directory_snapshot(path, root, label, snapshot)
    except OSError as exc:
        raise ValueError(f"cannot validate {label} handoff: {path}") from exc
    if _directory_identity(snapshot) != captured_identity:
        raise ValueError(f"{label} identity changed before handoff: {path}")

    try:
        with os.scandir(path) as entries:
            if next(entries, None) is not None:
                raise ValueError(f"{label} must be empty at handoff: {path}")
    except OSError as exc:
        raise ValueError(f"cannot inspect {label} contents at handoff: {path}") from exc

    # This final snapshot catches replacement during resolution or emptiness checks.
    try:
        final_snapshot = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot finalize {label} handoff: {path}") from exc
    _require_ordinary_directory_snapshot(final_snapshot, path, label)
    if _directory_identity(final_snapshot) != captured_identity:
        raise ValueError(f"{label} identity changed before handoff: {path}")
    return path


def _require_sole_canonical_entry(
    root: Path, expected: Path, label: str,
) -> None:
    matches = _canonical_entries(root, expected.name)
    if len(matches) != 1 or matches[0].name != expected.name:
        rendered = ", ".join(entry.name for entry in matches) or "none"
        raise FileExistsError(
            f"{label} canonical collision for {expected.name}: {rendered}"
        )


def _acquire_run_reservation(root: Path, run_id: str) -> Path:
    reservation = root / _reservation_name(run_id)
    existing = _canonical_entries(root, reservation.name)
    if existing:
        raise FileExistsError(
            f"run reservation already exists for run_id {run_id}: {existing[0]}"
        )
    try:
        reservation.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"run reservation already exists for run_id: {run_id}"
        ) from exc
    identity = _capture_created_directory_identity(
        reservation, root, "run reservation"
    )
    _require_sole_canonical_entry(root, reservation, "run reservation")
    return _validate_directory_handoff(
        reservation, root, "run reservation", identity
    )


def _create_reserved_run_directory(root: Path, run_id: str) -> Path:
    out = root / run_id
    existing = _canonical_entries(root, out.name)
    if existing:
        raise FileExistsError(
            f"target run directory already exists: {existing[0]}"
        )
    try:
        out.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"target run directory already exists: {out}") from exc
    identity = _capture_created_directory_identity(
        out, root, "target run directory"
    )
    _require_sole_canonical_entry(root, out, "target run directory")
    return _validate_directory_handoff(
        out, root, "target run directory", identity
    )


@contextmanager
def _run_directory_transaction(out_dir, run_id):
    root = _prepare_run_root(out_dir)
    _acquire_run_reservation(root, run_id)
    out = _create_reserved_run_directory(root, run_id)
    yield out


def write_artifacts(
    results: pd.DataFrame,
    qc_report,
    task,
    config,
    out_dir,
    run_id=None,
    evidence_context=None,
    predictions=None,
):
    selected_run_id = _run_identity(run_id)
    manifest = _evidence_manifest(selected_run_id, evidence_context)
    if predictions is not None:
        validate_prediction_records(predictions)
        if not predictions["run_id"].eq(selected_run_id).all():
            raise ValueError("prediction run_id does not match artifact run_id")
    with _run_directory_transaction(out_dir, selected_run_id) as out:
        artifact_paths = []
        results.to_parquet(out / "results.parquet")
        artifact_paths.append(out / "results.parquet")
        results.to_csv(out / "results.csv", index=False)
        artifact_paths.append(out / "results.csv")

        # Group by featureset when the column is present; fall back for backward compat.
        has_featureset = "featureset" in results.columns
        has_track = "track" in results.columns

        if has_featureset:
            group_cols = ["estimator", "featureset", "protocol"]
        else:
            group_cols = ["estimator", "protocol"]
        if has_track:
            group_cols.append("track")

        leaderboard = _leaderboard_entries(results, task, group_cols)

        dataset_group_cols = group_cols + ["held_out_key"] if "held_out_key" in results.columns else []
        dataset_leaderboard = (
            _leaderboard_entries(results, task, dataset_group_cols)
            if dataset_group_cols else []
        )

        _dump(leaderboard, out / "leaderboard.json")
        artifact_paths.append(out / "leaderboard.json")
        _dump(dataset_leaderboard, out / "dataset_leaderboard.json")
        artifact_paths.append(out / "dataset_leaderboard.json")
        if qc_report is not None:
            _dump(qc_report.__dict__, out / "qc_report.json")
            artifact_paths.append(out / "qc_report.json")
        _dump(config, out / "run_config.json")
        artifact_paths.append(out / "run_config.json")
        if predictions is not None:
            predictions.to_parquet(out / "predictions.parquet", index=False)
            artifact_paths.append(out / "predictions.parquet")
        write_evidence_pack(out, manifest, artifact_paths)
    return str(out)


def write_inference_artifacts(
    records: list[dict],
    config: dict,
    out_dir,
    run_id=None,
    evidence_context=None,
):
    selected_run_id = _run_identity(run_id)
    manifest = _evidence_manifest(selected_run_id, evidence_context)
    with _run_directory_transaction(out_dir, selected_run_id) as out:
        artifact_paths = []
        frame = pd.DataFrame.from_records(records)
        frame.to_parquet(out / "inference_results.parquet")
        artifact_paths.append(out / "inference_results.parquet")
        frame.to_csv(out / "inference_results.csv", index=False)
        artifact_paths.append(out / "inference_results.csv")
        _dump(
            {
                "records": records,
                "summary": _inference_summary(records),
            },
            out / "inference_benchmark.json",
        )
        artifact_paths.append(out / "inference_benchmark.json")
        _dump(config, out / "inference_run_config.json")
        artifact_paths.append(out / "inference_run_config.json")
        write_evidence_pack(out, manifest, artifact_paths)
    return str(out)


def _inference_summary(records: list[dict]) -> dict:
    succeeded = [row for row in records if row.get("status") == "succeeded"]
    fastest = min(
        succeeded,
        key=lambda row: row.get("ms_per_1000_samples") or float("inf"),
        default=None,
    )
    best = min(
        [row for row in succeeded if row.get("mae") is not None],
        key=lambda row: row.get("mae"),
        default=None,
    )
    return {
        "n_records": len(records),
        "n_succeeded": len(succeeded),
        "fastest_model": fastest.get("model") if fastest else None,
        "fastest_ms_per_1000_samples": fastest.get("ms_per_1000_samples") if fastest else None,
        "best_transfer_model": best.get("model") if best else None,
        "best_transfer_mae": best.get("mae") if best else None,
    }
