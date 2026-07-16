from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import pandas as pd

from sohbench.report import write_artifacts
from sohbench.tasks import SOHEstimationTask


FRONTEND_FILES = (
    "leaderboard.json",
    "dataset_leaderboard.json",
    "qc_report.json",
    "run_config.json",
)

OPTIONAL_FRONTEND_FILES = (
    "inference_benchmark.json",
    "inference_run_config.json",
    "model_catalog.json",
)

CORE_CONFIG_KEYS = {
    "phase",
    "source_runs",
    "datasets",
    "estimators",
    "protocols",
    "featuresets",
    "seeds",
    "source_seeds",
    "stability_seeds",
    "stability_replaced_cells",
    "stability_summary",
}


@dataclass(frozen=True)
class BuildFrontendArtifactsConfig:
    source_runs: Sequence[Path]
    archive_dir: Path = Path("runs")
    frontend_data_dir: Path | None = Path("frontend/app/data")
    phase: str = "frontend_combined"
    replace_cells: Sequence[tuple[str, str]] = field(default_factory=tuple)
    auto_replace_from_stability: bool = True


def build_frontend_artifacts(config: BuildFrontendArtifactsConfig) -> Path:
    source_runs = [Path(run) for run in config.source_runs]
    if not source_runs:
        raise ValueError("at least one source run is required")

    loaded = [_load_run(run) for run in source_runs]
    results = _merge_results(loaded, config.replace_cells, config.auto_replace_from_stability)
    run_config = _merge_configs(loaded, config)
    qc_report = SimpleNamespace(**_merge_qc_reports([item["qc"] for item in loaded]))

    artifact_dir = Path(
        write_artifacts(
            results,
            qc_report=qc_report,
            task=SOHEstimationTask(),
            config=run_config,
            out_dir=config.archive_dir,
        )
    )
    _copy_optional_frontend_files(source_runs, artifact_dir)

    if config.frontend_data_dir is not None:
        _sync_frontend_files(artifact_dir, Path(config.frontend_data_dir))

    return artifact_dir


def _load_run(run_dir: Path) -> dict:
    result_path = run_dir / "results.parquet"
    if not result_path.exists():
        raise FileNotFoundError(f"missing results.parquet in {run_dir}")

    return {
        "path": run_dir,
        "results": pd.read_parquet(result_path),
        "config": _read_json(run_dir / "run_config.json", {}),
        "qc": _read_json(run_dir / "qc_report.json", {}),
        "stability": _read_stability_summary(run_dir / "seed_stability_summary.csv"),
    }


def _merge_results(
    loaded_runs: Sequence[dict],
    explicit_replace_cells: Sequence[tuple[str, str]],
    auto_replace_from_stability: bool,
) -> pd.DataFrame:
    combined = pd.DataFrame()
    explicit_cells = set(explicit_replace_cells)
    candidate_aware = any("candidate_id" in item["results"].columns for item in loaded_runs)

    for item in loaded_runs:
        frame = _normalize_results_frame(item["results"])
        if candidate_aware and "candidate_id" not in frame.columns:
            frame["candidate_id"] = frame["estimator"]
        replacement_cells = set(explicit_cells)
        if auto_replace_from_stability and not item["stability"].empty:
            replacement_cells.update(_config_cells(item["config"].get("configs", [])))

        if not combined.empty and replacement_cells:
            cells_present = replacement_cells.intersection(_frame_cells(frame))
            if cells_present:
                combined = _drop_cells(combined, cells_present)

        combined = pd.concat([combined, frame], ignore_index=True)
        combined = _drop_duplicate_metric_rows(combined)

    return combined


def _merge_configs(loaded_runs: Sequence[dict], config: BuildFrontendArtifactsConfig) -> dict:
    results = pd.concat([item["results"] for item in loaded_runs], ignore_index=True)
    configs = [item["config"] for item in loaded_runs]
    base_results = loaded_runs[0]["results"]
    base_config = configs[0]
    merged = {}

    for run_config in configs:
        for key, value in run_config.items():
            if key not in CORE_CONFIG_KEYS:
                merged[key] = value

    merged.update(
        {
            "phase": config.phase,
            "source_runs": [str(item["path"].resolve()) for item in loaded_runs],
            "datasets": _sorted_union(
                *(run_config.get("datasets", []) for run_config in configs),
                _dataset_names_from_held_out_keys(
                    results.get("held_out_key", pd.Series(dtype=object)).dropna().astype(str).tolist()
                ),
            ),
            "estimators": _sorted_union(
                *(run_config.get("estimators", []) for run_config in configs),
                results.get("estimator", pd.Series(dtype=object)).dropna().astype(str).tolist(),
            ),
            "protocols": _sorted_union(
                *(run_config.get("protocols", []) for run_config in configs),
                results.get("protocol", pd.Series(dtype=object)).dropna().astype(str).tolist(),
            ),
            "featuresets": _sorted_union(
                *(run_config.get("featuresets", []) for run_config in configs),
                results.get("featureset", pd.Series(dtype=object)).dropna().astype(str).tolist(),
            ),
            "seeds": _sorted_union(
                base_config.get("seeds", []),
                base_results.get("seed", pd.Series(dtype=object)).dropna().tolist(),
            ),
            "source_seeds": _sorted_union(
                *(run_config.get("seeds", []) for run_config in configs),
                results.get("seed", pd.Series(dtype=object)).dropna().tolist(),
            ),
        }
    )

    stability_rows = []
    stability_seeds = []
    stability_cells = set(config.replace_cells)
    for item in loaded_runs:
        if item["stability"].empty:
            continue
        stability_rows.extend(_stability_records(item["stability"]))
        stability_seeds.extend(item["config"].get("seeds", []))
        if config.auto_replace_from_stability:
            stability_cells.update(_config_cells(item["config"].get("configs", [])))

    if stability_rows:
        merged["stability_seeds"] = _sorted_union(stability_seeds)
        merged["stability_replaced_cells"] = [
            {"estimator": estimator, "featureset": featureset}
            for estimator, featureset in sorted(stability_cells)
        ]
        merged["stability_summary"] = stability_rows
    elif config.replace_cells:
        merged["stability_replaced_cells"] = [
            {"estimator": estimator, "featureset": featureset}
            for estimator, featureset in sorted(config.replace_cells)
        ]

    return merged


def _merge_qc_reports(qc_reports: Sequence[dict]) -> dict:
    merged = dict(qc_reports[0]) if qc_reports else {}
    per_dataset = {}
    dropped_columns = set(merged.get("dropped_columns", []))

    for report in qc_reports:
        per_dataset.update(report.get("per_dataset", {}))
        dropped_columns.update(report.get("dropped_columns", []))
        if "pass_rate" not in merged and "pass_rate" in report:
            merged["pass_rate"] = report["pass_rate"]
        if "n_in" not in merged and "n_in" in report:
            merged["n_in"] = report["n_in"]
        if "n_out" not in merged and "n_out" in report:
            merged["n_out"] = report["n_out"]

    merged["per_dataset"] = dict(sorted(per_dataset.items()))
    if dropped_columns:
        merged["dropped_columns"] = sorted(dropped_columns)
    merged.setdefault("drop_reasons", {})
    return merged


def _sync_frontend_files(artifact_dir: Path, frontend_data_dir: Path) -> None:
    frontend_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in FRONTEND_FILES:
        shutil.copy2(artifact_dir / filename, frontend_data_dir / filename)
    for filename in OPTIONAL_FRONTEND_FILES:
        source = artifact_dir / filename
        if source.exists():
            shutil.copy2(source, frontend_data_dir / filename)


def _copy_optional_frontend_files(source_runs: Sequence[Path], artifact_dir: Path) -> None:
    for run_dir in source_runs:
        for filename in OPTIONAL_FRONTEND_FILES:
            source = run_dir / filename
            if source.exists():
                shutil.copy2(source, artifact_dir / filename)


def _normalize_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "featureset" not in normalized.columns:
        normalized["featureset"] = normalized.get("feature_mode", "full")
    if "track" not in normalized.columns:
        normalized["track"] = normalized.get("feature_mode", "unified")
    if "ci_source" not in normalized.columns:
        normalized["ci_source"] = "conformal"
    return normalized


def _drop_cells(frame: pd.DataFrame, cells: Iterable[tuple[str, str]]) -> pd.DataFrame:
    keep = pd.Series(True, index=frame.index)
    for estimator, featureset in cells:
        keep &= ~(frame["estimator"].eq(estimator) & frame["featureset"].eq(featureset))
    return frame.loc[keep].copy()


def _drop_duplicate_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = [
        column
        for column in [
            "candidate_id",
            "estimator",
            "featureset",
            "protocol",
            "track",
            "held_out_key",
            "fold",
            "seed",
            "metric",
        ]
        if column in frame.columns
    ]
    return frame.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)


def _frame_cells(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if "estimator" not in frame.columns or "featureset" not in frame.columns:
        return set()
    return set(zip(frame["estimator"].astype(str), frame["featureset"].astype(str)))


def _config_cells(configs: object) -> set[tuple[str, str]]:
    cells = set()
    if not isinstance(configs, list):
        return cells
    for item in configs:
        if not isinstance(item, dict):
            continue
        estimator = item.get("estimator")
        featureset = item.get("featureset")
        if estimator and featureset:
            cells.add((str(estimator), str(featureset)))
    return cells


def _read_stability_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _stability_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in frame.iterrows():
        records.append(
            {
                "estimator": str(row["estimator"]),
                "featureset": str(row["featureset"]),
                "mae_mean": _float_or_none(row.get("mae_mean", row.get("mean"))),
                "mae_std": _float_or_none(row.get("mae_std", row.get("std"))),
                "mae_min": _float_or_none(row.get("mae_min", row.get("min"))),
                "mae_max": _float_or_none(row.get("mae_max", row.get("max"))),
            }
        )
    return records


def _read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_union(*groups: Iterable) -> list:
    values = set()
    for group in groups:
        for value in group:
            if pd.isna(value):
                continue
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            values.add(value)
    return sorted(values, key=lambda item: str(item))


def _dataset_names_from_held_out_keys(keys: Iterable) -> list[str]:
    names = []
    for key in keys:
        value = str(key)
        names.append(value.split("/", 1)[0])
    return names


def _float_or_none(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _parse_cell(value: str) -> tuple[str, str]:
    if ":" in value:
        estimator, featureset = value.split(":", 1)
    elif "/" in value:
        estimator, featureset = value.split("/", 1)
    else:
        raise argparse.ArgumentTypeError("replace cell must be estimator:featureset")
    if not estimator or not featureset:
        raise argparse.ArgumentTypeError("replace cell must be estimator:featureset")
    return estimator, featureset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge benchmark runs and publish frontend data artifacts."
    )
    parser.add_argument("runs", nargs="*", help="Source run directories, in precedence order.")
    parser.add_argument(
        "--source-run",
        action="append",
        default=[],
        help="Source run directory. Can be provided multiple times.",
    )
    parser.add_argument("--archive-dir", default="runs", help="Directory for the generated archive run.")
    parser.add_argument(
        "--frontend-data-dir",
        default="frontend/app/data",
        help="Directory that receives frontend JSON artifacts. Use empty string to skip sync.",
    )
    parser.add_argument("--phase", default="frontend_combined")
    parser.add_argument(
        "--replace-cell",
        action="append",
        default=[],
        type=_parse_cell,
        help="Replace an existing estimator/featureset cell before appending later runs, e.g. mlp:partial_window.",
    )
    parser.add_argument(
        "--no-auto-replace-from-stability",
        action="store_true",
        help="Do not infer replacement cells from runs that include seed_stability_summary.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_runs = [Path(run) for run in args.source_run + args.runs]
    frontend_data_dir = Path(args.frontend_data_dir) if args.frontend_data_dir else None
    out = build_frontend_artifacts(
        BuildFrontendArtifactsConfig(
            source_runs=source_runs,
            archive_dir=Path(args.archive_dir),
            frontend_data_dir=frontend_data_dir,
            phase=args.phase,
            replace_cells=args.replace_cell,
            auto_replace_from_stability=not args.no_auto_replace_from_stability,
        )
    )
    print(f"frontend artifacts: {out}", flush=True)
    if frontend_data_dir is not None:
        print(f"synced frontend data: {frontend_data_dir}", flush=True)


if __name__ == "__main__":
    main()
