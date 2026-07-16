from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sohbench.evaluation_service import EvaluationService
from sohbench.frontend_publish import publish_frontend_artifacts


ESTIMATOR_TEMPLATE = '''
import numpy as np
from sohbench.contracts import feature_columns


class Estimator:
    name = {name!r}
    feature_mode = "unified"
    backend = "trusted-local"

    def fit(self, X, y, meta=None):
        self.cols_ = feature_columns(X)
        self.offset_ = float(np.mean(y))
        self.center_ = {{col: float(X[col].mean()) for col in self.cols_}}
        return self

    def predict(self, X, meta=None):
        if not self.cols_:
            return np.full(len(X), self.offset_)
        score = np.zeros(len(X), dtype=float)
        for col in self.cols_:
            score += X[col].to_numpy(dtype=float) - self.center_.get(col, 0.0)
        score = score / max(len(self.cols_), 1)
        return np.clip(self.offset_ - 0.08 * score, 0.0, 1.3)
'''


FEATURESET_TEMPLATE = '''
import numpy as np

FEATURESET_NAME = {name!r}
FEATURE_COLUMNS = ["smoke_log_cycle", "smoke_voltage_margin"]


def transform(table, meta=None):
    out = table.copy()
    cycle = out["cycle_index"].to_numpy(dtype=float)
    voltage_mean = out["voltage_mean"].to_numpy(dtype=float)
    out["smoke_log_cycle"] = np.log1p(np.maximum(cycle, 0.0))
    out["smoke_voltage_margin"] = 4.25 - voltage_mean
    return out
'''


@dataclass
class SmokeAcceptanceConfig:
    data_dir: str | Path = Path(r"D:\battery_SOH\data")
    work_dir: str | Path = Path("work/trusted_plugins")
    runs_dir: str | Path = Path("runs")
    frontend_data_dir: str | Path = Path("frontend/app/data")
    dataset_name: str = "smoke_eval_demo"
    estimator_name: str = "smoke_linear_decay"
    featureset_name: str = "smoke_cycle_shape"
    model_name: str = ""
    inference_datasets: tuple[str, ...] | None = None
    protocol: str = "loco"
    seeds: tuple[int, ...] = (0,)
    cells: int = 3
    cycles: int = 14
    summary_json: str | Path | None = None
    profile_name: str = "smoke"
    profile_label: str = "Smoke 闭环验收"


def run_smoke_acceptance(config: SmokeAcceptanceConfig) -> dict:
    data_dir = Path(config.data_dir)
    work_dir = Path(config.work_dir)
    runs_dir = Path(config.runs_dir)
    frontend_data_dir = Path(config.frontend_data_dir)
    selected_datasets = list(config.inference_datasets or (config.dataset_name,))

    service = EvaluationService(
        data_dir=data_dir,
        work_dir=work_dir,
        runs_dir=runs_dir,
        frontend_data_dir=frontend_data_dir,
    )
    _upload_smoke_resources(service, config)

    training_job = service.create_job(
        {
            "datasets": [config.dataset_name],
            "estimators": [config.estimator_name],
            "featuresets": [config.featureset_name],
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
            "datasets": selected_datasets,
            "models": [model["name"]],
            "launch_strategy": "compatible_only",
            "selected_datasets": selected_datasets,
            "submitted_datasets": selected_datasets,
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
    publish_frontend_artifacts(
        [Path(training["run_dir"]), Path(inference["run_dir"])], frontend_data_dir
    )
    _require_frontend_files(frontend_data_dir)
    _require_inference_success(frontend_data_dir / "inference_benchmark.json", selected_datasets)

    summary = {
        "status": "succeeded",
        "dataset": config.dataset_name,
        "training": _job_summary(training),
        "inference": _job_summary(inference),
        "model": {
            "name": model["name"],
            "path": model.get("path", ""),
            "featureset": model.get("featureset", ""),
            "feature_columns": list(model.get("feature_columns", [])),
            "trained_on": list(model.get("trained_on", [])),
            "created_by_run": model.get("created_by_run", ""),
        },
        "frontend_data_dir": str(frontend_data_dir),
    }
    profile = service.publish_frontend_profile(
        config.profile_name,
        {
            "label": config.profile_label,
            "kind": "smoke_acceptance",
            "datasets": selected_datasets,
            "training_dataset": config.dataset_name,
            "estimator": config.estimator_name,
            "featureset": config.featureset_name,
            "protocol": config.protocol,
            "seeds": list(config.seeds),
            "training_job_id": training["id"],
            "inference_job_id": inference["id"],
            "training_run_dir": training.get("run_dir", ""),
            "inference_run_dir": inference.get("run_dir", ""),
            "model": summary["model"],
            "models": [summary["model"]],
            "launch_strategy": "compatible_only",
            "selected_datasets": selected_datasets,
            "submitted_datasets": selected_datasets,
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


def _upload_smoke_resources(service: EvaluationService, config: SmokeAcceptanceConfig) -> None:
    service.store.upload_dataset(
        config.dataset_name,
        f"{config.dataset_name}.csv",
        _smoke_dataset_csv(config),
        replace=True,
    )
    service.store.upload_estimator_plugin(
        config.estimator_name,
        f"{config.estimator_name}.py",
        ESTIMATOR_TEMPLATE.format(name=config.estimator_name).encode("utf-8"),
        replace=True,
    )
    service.store.upload_featureset_plugin(
        config.featureset_name,
        f"{config.featureset_name}.py",
        FEATURESET_TEMPLATE.format(name=config.featureset_name).encode("utf-8"),
        replace=True,
    )


def _smoke_dataset_csv(config: SmokeAcceptanceConfig) -> bytes:
    rows = []
    for cell_index in range(int(config.cells)):
        cell_id = f"smoke_cell_{cell_index + 1:02d}"
        cell_offset = cell_index * 0.0015
        for cycle in range(int(config.cycles)):
            fade = 0.006 * cycle + cell_offset
            soh = max(0.72, 1.0 - fade)
            voltage_mean = 3.62 - 0.008 * cycle - cell_offset
            voltage_max = 4.10 - 0.004 * cycle
            rows.append(
                {
                    "dataset": config.dataset_name,
                    "cell_id": cell_id,
                    "cycle_index": cycle,
                    "soh": soh,
                    "voltage_mean": voltage_mean,
                    "voltage_std": 0.012 + 0.0003 * cycle,
                    "voltage_min": 3.30 - 0.005 * cycle,
                    "voltage_max": voltage_max,
                    "resistance_proxy": 0.012 + 0.0004 * cycle + cell_offset,
                    "smoke_log_cycle": float(np.log1p(cycle)),
                    "smoke_voltage_margin": 4.25 - voltage_mean,
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _select_model(service: EvaluationService, training: dict, requested_name: str = "") -> dict:
    catalog = service.model_store.catalog()
    by_name = {item["name"]: item for item in catalog}
    if requested_name:
        if requested_name not in by_name:
            raise RuntimeError(f"requested model not found after training: {requested_name}")
        return by_name[requested_name]

    saved = [str(name) for name in training.get("saved_models", [])]
    for name in saved:
        if name in by_name:
            return by_name[name]
    for item in catalog:
        if item.get("created_by_run") == training.get("id"):
            return item
    raise RuntimeError(f"training job produced no selectable model: {training.get('id')}")


def _require_succeeded(job: dict, label: str) -> None:
    if job.get("status") != "succeeded":
        detail = job.get("error") or job.get("traceback") or "unknown error"
        raise RuntimeError(f"{label} job failed: {detail}")
    warnings = job.get("model_warnings") or []
    if warnings:
        raise RuntimeError(f"{label} job produced model warnings: {warnings}")


def _require_files(directory: Path, filenames: Iterable[str]) -> None:
    missing = [name for name in filenames if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"missing expected files under {directory}: {missing}")


def _require_frontend_files(frontend_data_dir: Path) -> None:
    _require_files(
        frontend_data_dir,
        [
            "leaderboard.json",
            "run_config.json",
            "model_catalog.json",
            "inference_benchmark.json",
            "inference_run_config.json",
        ],
    )


def _require_inference_success(path: Path, expected_datasets: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    seen = sorted({str(row.get("dataset")) for row in records})
    if seen != sorted(expected_datasets):
        raise RuntimeError(f"inference records cover {seen}, expected {sorted(expected_datasets)}")
    failed = [row for row in records if row.get("status") != "succeeded"]
    if failed:
        raise RuntimeError(f"inference records failed: {failed}")
    for row in records:
        for key in ("mae", "rmse", "predict_ms", "ms_per_1000_samples", "samples_per_second"):
            value = row.get(key)
            if value is None or not np.isfinite(float(value)):
                raise RuntimeError(f"inference record has invalid {key}: {row}")
        if int(row.get("n_samples") or 0) <= 0:
            raise RuntimeError(f"inference record has no samples: {row}")


def _job_summary(job: dict) -> dict:
    return {
        "id": job.get("id", ""),
        "status": job.get("status", ""),
        "created_at": job.get("created_at", ""),
        "started_at": job.get("started_at", ""),
        "finished_at": job.get("finished_at", ""),
        "run_dir": job.get("run_dir", ""),
        "raw_run_dir": job.get("raw_run_dir", ""),
        "saved_models": list(job.get("saved_models", [])),
    }


def parse_args() -> SmokeAcceptanceConfig:
    parser = argparse.ArgumentParser("run a one-click SOH evaluation smoke acceptance")
    parser.add_argument("--data-dir", default=r"D:\battery_SOH\data")
    parser.add_argument("--work-dir", default="work/trusted_plugins")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--frontend-data-dir", default="frontend/app/data")
    parser.add_argument("--dataset-name", default="smoke_eval_demo")
    parser.add_argument("--estimator-name", default="smoke_linear_decay")
    parser.add_argument("--featureset-name", default="smoke_cycle_shape")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--inference-datasets", nargs="*", default=None)
    parser.add_argument("--protocol", default="loco")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--cells", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=14)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--profile-name", default="smoke")
    parser.add_argument("--profile-label", default="Smoke 闭环验收")
    args = parser.parse_args()
    return SmokeAcceptanceConfig(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        runs_dir=args.runs_dir,
        frontend_data_dir=args.frontend_data_dir,
        dataset_name=args.dataset_name,
        estimator_name=args.estimator_name,
        featureset_name=args.featureset_name,
        model_name=args.model_name,
        inference_datasets=tuple(args.inference_datasets) if args.inference_datasets else None,
        protocol=args.protocol,
        seeds=tuple(args.seeds),
        cells=args.cells,
        cycles=args.cycles,
        summary_json=args.summary_json,
        profile_name=args.profile_name,
        profile_label=args.profile_label,
    )


def main() -> None:
    summary = run_smoke_acceptance(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
