from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import registry as R
from .contracts import feature_columns
from .data import DatasetLoader
from .model_store import ModelStore
from .protocols import LeaveOneCellOut, LeaveOneChemistryOut, LeaveOneDatasetOut
from .report import write_artifacts
from .runner import BenchmarkRunner
from .tasks import SOHEstimationTask
from .trusted_plugins import TrustedPluginStore


PROTOCOLS = {
    "loco": LeaveOneCellOut,
    "lodo": LeaveOneDatasetOut,
    "lcho": LeaveOneChemistryOut,
}


def ensure_builtin_resources() -> None:
    import sohbench.estimators.baselines as baselines
    import sohbench.estimators.byo as byo
    import sohbench.estimators.neural as neural
    import sohbench.estimators.physics as physics
    import sohbench.featuresets as featuresets

    for cls in (
        baselines.RidgeEstimator,
        baselines.GBDTEstimator,
        baselines.SVREstimator,
        baselines.GPREstimator,
        physics.PhysicsResidualEstimator,
        neural.MLPEstimator,
        byo.ByoSelectKBestRidge,
    ):
        R._ESTIMATORS.setdefault(cls.name, cls)

    R._FEATURESETS.setdefault("full", featuresets._full_fs)
    for fs in (
        featuresets.FeatureSet(
            name="voltage",
            candidates=["voltage_mean", "voltage_std", "voltage_min", "voltage_max"],
        ),
        featuresets.FeatureSet(
            name="ic_dva",
            candidates=["ic_peak_ah_per_v", "ic_peak_voltage_v"],
        ),
        featuresets.FeatureSet(
            name="partial_window",
            candidates=["q_win_ah", "t_win_s"],
        ),
        featuresets.FeatureSet(
            name="capacity_time",
            candidates=["charge_capacity_max", "time_s_max", "cycle_index_feature", "cycle_fraction"],
        ),
    ):
        R._FEATURESETS.setdefault(fs.name, fs)

    try:
        import torch  # noqa: F401
    except ImportError:
        return
    import sohbench.estimators.deep  # noqa: F401
    import sohbench.estimators.pinn  # noqa: F401


class EvaluationService:
    def __init__(
        self,
        *,
        data_dir: str | Path,
        work_dir: str | Path = Path("work/trusted_plugins"),
        runs_dir: str | Path = Path("runs"),
        frontend_data_dir: str | Path = Path("frontend/app/data"),
    ):
        ensure_builtin_resources()
        self.data_dir = Path(data_dir)
        self.work_dir = Path(work_dir)
        self.runs_dir = Path(runs_dir)
        self.frontend_data_dir = Path(frontend_data_dir)
        self.jobs_dir = self.work_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.store = TrustedPluginStore(self.work_dir, self.data_dir)
        self.model_store = ModelStore(self.work_dir / "models")

    def catalog(self) -> dict:
        catalog = self.store.catalog()
        catalog["jobs"] = self.list_jobs()
        catalog["models"] = self.model_store.catalog()
        return catalog

    def create_job(self, payload: dict, *, start: bool = True) -> dict:
        request = self._normalize_request(payload)
        job_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": _now(),
            "request": request,
        }
        self._save_job(job)
        if start:
            thread = threading.Thread(target=self.run_job_inline, args=(job_id,), daemon=True)
            thread.start()
        return job

    def run_job_inline(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        job.update({"status": "running", "started_at": _now()})
        self._save_job(job)
        try:
            request = job["request"]
            loader = DatasetLoader(self.data_dir)
            datasets = loader.resolve_datasets(request["datasets"])
            table, meta, qc = loader.load(datasets)
            table = self.store.apply_feature_transforms(table, meta, request["featuresets"])
            estimators = [R.get_estimator(name) for name in request["estimators"]]
            protocols = {name: PROTOCOLS[name]() for name in request["protocols"]}
            saved_models = []
            model_warnings = []

            def save_fitted_model(**kwargs) -> None:
                record = kwargs["record"]
                fitted = kwargs["estimator"]
                train_table = kwargs["train_table"]
                fs_name = str(record.get("featureset", "full"))
                model_name = "_".join(
                    [
                        _model_name_token(record["estimator"]),
                        _model_name_token(fs_name),
                        _model_name_token(record["protocol"]),
                        _model_name_token(record["held_out_key"]),
                        f"seed{int(record['seed'])}",
                    ]
                )
                manifest = {
                    "model_name": model_name,
                    "model_kind": "pickle",
                    "estimator": str(record["estimator"]),
                    "featureset": fs_name,
                    "feature_columns": feature_columns(train_table),
                    "input_columns": list(map(str, train_table.columns)),
                    "target": "soh",
                    "trained_on": sorted(map(str, train_table["dataset"].dropna().unique())),
                    "held_out_key": str(record["held_out_key"]),
                    "protocol": str(record["protocol"]),
                    "seed": int(record["seed"]),
                    "created_by_run": job_id,
                    "predict_contract": "predict(X, meta=None)",
                }
                try:
                    saved_models.append(
                        self.model_store.save_training_model(
                            model_name,
                            fitted,
                            manifest,
                            replace=True,
                        )
                    )
                except Exception as exc:
                    model_warnings.append(f"{model_name}: {exc}")

            results = BenchmarkRunner(seeds=request["seeds"]).run(
                SOHEstimationTask(),
                table=table,
                meta=meta,
                protocols=protocols,
                estimators=estimators,
                featuresets=request["featuresets"],
                on_model_fitted=save_fitted_model,
            )
            results["dataset"] = ",".join(datasets)
            raw_run_dir = Path(
                write_artifacts(
                    results,
                    qc_report=qc,
                    task=SOHEstimationTask(),
                    config={
                        **request,
                        "datasets": datasets,
                        "trusted_local_plugins": True,
                        "saved_models": [item["name"] for item in saved_models],
                        "model_warnings": model_warnings,
                    },
                    out_dir=self.runs_dir,
                )
            )
            (raw_run_dir / "model_catalog.json").write_text(
                json.dumps(
                    {"models": self.model_store.catalog()},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            published_run_dir = self._publish_frontend_artifacts(raw_run_dir)
            job.update(
                {
                    "status": "succeeded",
                    "finished_at": _now(),
                    "raw_run_dir": str(raw_run_dir),
                    "run_dir": str(published_run_dir),
                    "saved_models": [item["name"] for item in saved_models],
                    "model_warnings": model_warnings,
                }
            )
        except Exception as exc:
            job.update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=12),
                }
            )
        self._save_job(job)
        return job

    def get_job(self, job_id: str) -> dict:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_jobs(self) -> list[dict]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return jobs

    def _normalize_request(self, payload: dict) -> dict:
        datasets = _string_list(payload.get("datasets") or ["all"])
        estimators = _string_list(payload.get("estimators") or ["ridge"])
        featuresets = _string_list(payload.get("featuresets") or ["full"])
        protocols = _string_list(payload.get("protocols") or ["loco"])
        seeds = [int(seed) for seed in (payload.get("seeds") or [0])]
        unknown_protocols = [name for name in protocols if name not in PROTOCOLS]
        if unknown_protocols:
            raise ValueError(f"unknown protocols: {unknown_protocols}")
        for name in estimators:
            R.get_estimator(name)
        for name in featuresets:
            R.get_featureset(name)
        return {
            "datasets": datasets,
            "estimators": estimators,
            "featuresets": featuresets,
            "protocols": protocols,
            "seeds": seeds,
        }

    def _publish_frontend_artifacts(self, raw_run_dir: Path) -> Path:
        _sync_frontend_files(raw_run_dir, self.frontend_data_dir)
        return raw_run_dir

    def _save_job(self, job: dict) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        (self.jobs_dir / f"{job['id']}.json").write_text(
            json.dumps(job, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _model_name_token(value) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return token or "model"


def _sync_frontend_files(run_dir: Path, frontend_data_dir: Path) -> None:
    import shutil

    frontend_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("leaderboard.json", "dataset_leaderboard.json", "qc_report.json", "run_config.json"):
        shutil.copy2(run_dir / filename, frontend_data_dir / filename)
    for filename in ("model_catalog.json", "inference_benchmark.json"):
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, frontend_data_dir / filename)
