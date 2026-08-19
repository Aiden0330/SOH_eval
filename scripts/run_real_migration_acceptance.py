from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sohbench.evaluation_service import EvaluationService
from sohbench.paths import DEFAULT_DATA_DIR

try:
    from scripts.run_smoke_acceptance import (
        _job_summary,
        _require_files,
        _require_frontend_files,
        _require_inference_success,
        _require_succeeded,
        _select_model,
    )
except ModuleNotFoundError:
    from run_smoke_acceptance import (
        _job_summary,
        _require_files,
        _require_frontend_files,
        _require_inference_success,
        _require_succeeded,
        _select_model,
    )


@dataclass
class RealMigrationAcceptanceConfig:
    data_dir: str | Path = DEFAULT_DATA_DIR
    work_dir: str | Path = Path("work/migration_acceptance")
    runs_dir: str | Path = Path("runs")
    frontend_data_dir: str | Path = Path("frontend/app/data")
    train_dataset: str = "nasa"
    inference_datasets: tuple[str, ...] = ("nasa", "calce_cs2")
    estimator: str = "ridge"
    featureset: str = "voltage"
    protocol: str = "loco"
    seeds: tuple[int, ...] = (0,)
    model_name: str = ""
    profile_name: str = "migration-smoke"
    profile_label: str = "真实数据小迁移验收"
    summary_json: str | Path | None = None


def run_real_migration_acceptance(config: RealMigrationAcceptanceConfig) -> dict:
    data_dir = Path(config.data_dir)
    frontend_data_dir = Path(config.frontend_data_dir)
    inference_datasets = list(config.inference_datasets)
    _require_dataset_files(data_dir, [config.train_dataset, *inference_datasets])

    service = EvaluationService(
        data_dir=data_dir,
        work_dir=Path(config.work_dir),
        runs_dir=Path(config.runs_dir),
        frontend_data_dir=frontend_data_dir,
    )
    training_job = service.create_job(
        {
            "datasets": [config.train_dataset],
            "estimators": [config.estimator],
            "featuresets": [config.featureset],
            "protocols": [config.protocol],
            "seeds": list(config.seeds),
        },
        start=False,
    )
    training = service.run_job_inline(training_job["id"])
    _require_succeeded(training, "training")
    _require_files(
        Path(training["run_dir"]),
        [
            "results.parquet",
            "results.csv",
            "leaderboard.json",
            "dataset_leaderboard.json",
            "qc_report.json",
            "run_config.json",
            "model_catalog.json",
        ],
    )

    model = _select_model(service, training, config.model_name)
    inference_job = service.create_inference_job(
        {
            "datasets": inference_datasets,
            "models": [model["name"]],
            "launch_strategy": "compatible_only",
            "selected_datasets": inference_datasets,
            "submitted_datasets": inference_datasets,
            "skipped_datasets": [],
        },
        start=False,
    )
    inference = service.run_inference_job_inline(inference_job["id"])
    _require_succeeded(inference, "inference")
    _require_files(
        Path(inference["run_dir"]),
        [
            "inference_results.parquet",
            "inference_results.csv",
            "inference_benchmark.json",
            "inference_run_config.json",
            "model_catalog.json",
        ],
    )
    _require_frontend_files(frontend_data_dir)
    _require_inference_success(frontend_data_dir / "inference_benchmark.json", inference_datasets)

    model_summary = {
        "name": model["name"],
        "path": model.get("path", ""),
        "featureset": model.get("featureset", ""),
        "feature_columns": list(model.get("feature_columns", [])),
        "trained_on": list(model.get("trained_on", [])),
        "created_by_run": model.get("created_by_run", ""),
    }
    summary = {
        "status": "succeeded",
        "training_dataset": config.train_dataset,
        "datasets": inference_datasets,
        "training": _job_summary(training),
        "inference": _job_summary(inference),
        "model": model_summary,
        "frontend_data_dir": str(frontend_data_dir),
    }
    profile = service.publish_frontend_profile(
        config.profile_name,
        {
            "label": config.profile_label,
            "kind": "migration_acceptance",
            "datasets": inference_datasets,
            "training_dataset": config.train_dataset,
            "estimator": config.estimator,
            "featureset": config.featureset,
            "protocol": config.protocol,
            "seeds": list(config.seeds),
            "training_job_id": training["id"],
            "inference_job_id": inference["id"],
            "training_run_dir": training.get("run_dir", ""),
            "inference_run_dir": inference.get("run_dir", ""),
            "model": model_summary,
            "models": [model_summary],
            "launch_strategy": "compatible_only",
            "selected_datasets": inference_datasets,
            "submitted_datasets": inference_datasets,
            "skipped_datasets": [],
        },
        source_dirs=[Path(training["run_dir"]), Path(inference["run_dir"])],
    )
    summary["profile"] = {
        "name": profile["name"],
        "path": profile["path"],
        "manifest_path": profile["manifest_path"],
    }
    if config.summary_json:
        Path(config.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(config.summary_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def _require_dataset_files(data_dir: Path, names: list[str]) -> None:
    missing = [name for name in names if not (data_dir / "processed" / f"{name}_features.pkl").exists()]
    if missing:
        raise FileNotFoundError(f"missing processed datasets under {data_dir / 'processed'}: {missing}")
    for name in names:
        table = pd.read_pickle(data_dir / "processed" / f"{name}_features.pkl")
        required = {"dataset", "cell_id", "cycle_index", "soh", "voltage_mean", "voltage_std", "voltage_min", "voltage_max"}
        absent = sorted(required - set(table.columns))
        if absent:
            raise ValueError(f"{name}: missing required migration columns: {absent}")


def parse_args() -> RealMigrationAcceptanceConfig:
    parser = argparse.ArgumentParser("run a small real-data migration acceptance")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--work-dir", default="work/migration_acceptance")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--frontend-data-dir", default="frontend/app/data")
    parser.add_argument("--train-dataset", default="nasa")
    parser.add_argument("--inference-datasets", nargs="*", default=["nasa", "calce_cs2"])
    parser.add_argument("--estimator", default="ridge")
    parser.add_argument("--featureset", default="voltage")
    parser.add_argument("--protocol", default="loco")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--model-name", default="")
    parser.add_argument("--profile-name", default="migration-smoke")
    parser.add_argument("--profile-label", default="真实数据小迁移验收")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()
    return RealMigrationAcceptanceConfig(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        runs_dir=args.runs_dir,
        frontend_data_dir=args.frontend_data_dir,
        train_dataset=args.train_dataset,
        inference_datasets=tuple(args.inference_datasets),
        estimator=args.estimator,
        featureset=args.featureset,
        protocol=args.protocol,
        seeds=tuple(args.seeds),
        model_name=args.model_name,
        profile_name=args.profile_name,
        profile_label=args.profile_label,
        summary_json=args.summary_json,
    )


def main() -> None:
    summary = run_real_migration_acceptance(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
