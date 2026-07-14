from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

import pandas as pd

from . import registry as R
from .battery_soh_plugins import register_battery_soh_plugins
from .contracts import feature_columns
from .data import DatasetLoader
from .battery_dashboard_import import publish_battery_dashboard_profile
from .frontend_publish import publish_frontend_profile, sync_frontend_artifacts
from .inference import run_inference_benchmark
from .model_store import ModelStore
from .protocols import LeaveOneCellOut, LeaveOneChemistryOut, LeaveOneDatasetOut
from .report import write_artifacts, write_inference_artifacts
from .runner import BenchmarkRunner
from .tasks import SOHEstimationTask
from .trusted_plugins import TrustedPluginStore
from .fabric import FeatureState, IdentityFeaturePipeline, scoped_meta_for_frame


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
        self.inference_jobs_dir = self.work_dir / "inference_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.inference_jobs_dir.mkdir(parents=True, exist_ok=True)
        self.store = TrustedPluginStore(self.work_dir, self.data_dir)
        self.model_store = ModelStore(self.work_dir / "models")

    def catalog(
        self,
        *,
        model_limit: int | None = None,
        job_limit: int | None = None,
        compact_jobs: bool = False,
    ) -> dict:
        catalog = self.store.catalog()
        jobs = self.list_jobs()
        inference_jobs = self.list_inference_jobs()
        models = self.model_store.catalog()
        catalog["job_count"] = len(jobs)
        catalog["inference_job_count"] = len(inference_jobs)
        catalog["model_count"] = len(models)
        catalog["jobs_limited"] = job_limit is not None and len(jobs) > job_limit
        catalog["inference_jobs_limited"] = job_limit is not None and len(inference_jobs) > job_limit
        catalog["models_limited"] = model_limit is not None and len(models) > model_limit
        if job_limit is not None:
            jobs = jobs[: max(0, int(job_limit))]
            inference_jobs = inference_jobs[: max(0, int(job_limit))]
        if compact_jobs:
            jobs = [_compact_job(job) for job in jobs]
            inference_jobs = [_compact_job(job) for job in inference_jobs]
        if model_limit is not None:
            models = models[: max(0, int(model_limit))]
        catalog["jobs"] = jobs
        catalog["inference_jobs"] = inference_jobs
        catalog["models"] = models
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

    def create_inference_job(self, payload: dict, *, start: bool = True) -> dict:
        request = self._normalize_inference_request(payload)
        job_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "kind": "inference",
            "status": "queued",
            "created_at": _now(),
            "request": request,
        }
        self._save_inference_job(job)
        if start:
            thread = threading.Thread(target=self.run_inference_job_inline, args=(job_id,), daemon=True)
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
            estimators = [R.get_estimator(name) for name in request["estimators"]]
            protocols = {name: PROTOCOLS[name]() for name in request["protocols"]}
            feature_pipelines = {
                name: pipeline
                for name in request["featuresets"]
                if (pipeline := self.store.feature_pipeline(name)) is not None
            }
            saved_models = []
            model_warnings = []

            def save_fitted_model(**kwargs) -> None:
                record = kwargs["record"]
                fitted = kwargs["estimator"]
                train_table = kwargs["train_table"]
                feature_state = kwargs["feature_state"]
                feature_pipeline_name = kwargs["pipeline_name"]
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
                    "input_columns": [str(column) for column in train_table.columns if column != "soh"],
                    "target": "soh",
                    "trained_on": sorted(map(str, train_table["dataset"].dropna().unique())),
                    "held_out_key": str(record["held_out_key"]),
                    "protocol": str(record["protocol"]),
                    "seed": int(record["seed"]),
                    "created_by_run": job_id,
                    "predict_contract": "predict(X, meta=None)",
                    "feature_state": feature_state.to_dict(),
                    "feature_pipeline": feature_pipeline_name,
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
                feature_pipelines=feature_pipelines,
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

    def run_inference_job_inline(self, job_id: str) -> dict:
        job = self.get_inference_job(job_id)
        job.update({"status": "running", "started_at": _now()})
        self._save_inference_job(job)
        try:
            request = job["request"]
            loader = DatasetLoader(self.data_dir)
            datasets = loader.resolve_datasets(request["datasets"])
            table, meta, _qc = loader.load(datasets)
            inferred_featuresets = self._featuresets_for_models(request["models"])
            records = []
            for model_name in request["models"]:
                prepared = self._prepare_inference_table(model_name, table, meta)
                records.extend(run_inference_benchmark(self.model_store, prepared, meta, [model_name]))
            raw_run_dir = Path(
                write_inference_artifacts(
                    records,
                    {
                        **request,
                        "datasets": datasets,
                        "job_id": job_id,
                        "trusted_local_pickle_models": True,
                        "applied_featureset_transforms": inferred_featuresets,
                    },
                    self.runs_dir,
                )
            )
            config_path = raw_run_dir / "inference_run_config.json"
            if config_path.exists():
                run_config = json.loads(config_path.read_text(encoding="utf-8"))
                run_config.update({"job_id": job_id, "run_dir": str(raw_run_dir), "raw_run_dir": str(raw_run_dir)})
                config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
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
        self._save_inference_job(job)
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
        return sorted(jobs, key=_job_sort_key, reverse=True)

    def get_inference_job(self, job_id: str) -> dict:
        path = self.inference_jobs_dir / f"{job_id}.json"
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_inference_jobs(self) -> list[dict]:
        jobs = []
        for path in sorted(self.inference_jobs_dir.glob("*.json"), reverse=True):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return sorted(jobs, key=_job_sort_key, reverse=True)

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

    def _normalize_inference_request(self, payload: dict) -> dict:
        datasets = _string_list(payload.get("datasets") or ["all"])
        models = _string_list(payload.get("models") or [])
        if not models:
            raise ValueError("at least one model is required")
        known = {item["name"] for item in self.model_store.catalog()}
        missing = [name for name in models if name not in known]
        if missing:
            raise ValueError(f"unknown models: {missing}")
        request = {
            "datasets": datasets,
            "models": models,
            "launch_strategy": str(payload.get("launch_strategy") or "unspecified"),
            "selected_datasets": _string_list(payload.get("selected_datasets") or datasets),
            "submitted_datasets": _string_list(payload.get("submitted_datasets") or datasets),
            "skipped_datasets": _string_list(payload.get("skipped_datasets") or []),
        }
        return request

    def _featuresets_for_models(self, model_names: list[str]) -> list[str]:
        selected = set(model_names)
        featuresets: list[str] = []
        for item in self.model_store.catalog():
            if item.get("name") not in selected:
                continue
            name = str(item.get("featureset") or "").strip()
            if name and name not in featuresets:
                featuresets.append(name)
        return featuresets

    def _prepare_inference_table(self, model_name: str, table, meta):
        loaded = self.model_store.load_model(model_name)
        pipeline_name = loaded.manifest.get("feature_pipeline")
        state_document = loaded.manifest.get("feature_state")
        if pipeline_name:
            pipeline = self.store.feature_pipeline(str(pipeline_name))
            if pipeline is None:
                raise ValueError(f"feature pipeline {pipeline_name!r} required by model {model_name!r} is unavailable")
            if state_document is None:
                raise ValueError(f"feature state required by model {model_name!r} is missing")
            state = FeatureState.from_dict(state_document)
        else:
            pipeline = IdentityFeaturePipeline()
            state = FeatureState.from_dict(state_document) if state_document is not None else pipeline.fit(table.drop(columns=["soh"], errors="ignore"), [], pd.DataFrame())
        source = table.drop(columns=["soh"], errors="ignore")
        transformed = pipeline.transform(source, state, scoped_meta_for_frame(meta, source))
        if not isinstance(transformed, type(table)) and not hasattr(transformed, "columns"):
            raise ValueError(f"feature pipeline for model {model_name!r} did not return a DataFrame")
        if len(transformed) != len(table) or not transformed.index.equals(table.index):
            raise ValueError(f"feature pipeline for model {model_name!r} changed inference rows")
        if "soh" in table.columns:
            transformed = transformed.copy(deep=True)
            transformed["soh"] = table["soh"].copy(deep=True)
        return transformed

    def _publish_frontend_artifacts(self, raw_run_dir: Path) -> Path:
        sync_frontend_artifacts(raw_run_dir, self.frontend_data_dir)
        return raw_run_dir

    def publish_frontend_profile(
        self,
        profile_name: str,
        manifest: dict,
        *,
        source_dirs: list[str | Path] | tuple[str | Path, ...] = (),
    ) -> dict:
        return publish_frontend_profile(
            self.frontend_data_dir,
            profile_name,
            manifest,
            source_dirs=source_dirs,
        )

    def import_battery_dashboard_profile(
        self,
        *,
        source_dir: str | Path = r"D:\battery_SOH\dashboard\data",
        profile_name: str = "battery-soh-diagnostics",
        include_curves: bool = False,
    ) -> dict:
        return publish_battery_dashboard_profile(
            self.frontend_data_dir,
            Path(source_dir),
            include_curves=include_curves,
            profile_name=profile_name,
        )

    def battery_curve_dataset(
        self,
        *,
        profile_name: str = "battery-soh-diagnostics",
        dataset: str,
        limit: int | None = None,
    ) -> dict:
        profile = _safe_profile_name(profile_name)
        dataset_name = str(dataset or "").strip()
        if not dataset_name:
            raise ValueError("dataset is required")
        manifest_path = self.frontend_data_dir / "profiles" / profile / "manifest.json"
        if not manifest_path.exists():
            raise KeyError(f"profile not found: {profile}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.get("curve_artifacts") if isinstance(manifest.get("curve_artifacts"), list) else []
        artifact = next((item for item in artifacts if str(item.get("dataset") or "") == dataset_name), None)
        if artifact is None:
            raise KeyError(f"curve dataset not found: {dataset_name}")
        source = Path(str(artifact.get("source") or ""))
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"curve source not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("curve source must be a JSON object")
        curve_items = list(payload.items())
        max_items = None if limit is None else max(0, int(limit))
        selected_items = curve_items if max_items is None else curve_items[:max_items]
        return {
            "ok": True,
            "profile": profile,
            "dataset": dataset_name,
            "source": str(source),
            "bytes": source.stat().st_size,
            "total_curves": len(curve_items),
            "returned_curves": len(selected_items),
            "truncated": len(selected_items) < len(curve_items),
            "curves": {key: value for key, value in selected_items},
        }

    def import_battery_soh_plugins(
        self,
        *,
        battery_soh_root: str | Path = r"D:\battery_SOH",
        replace: bool = False,
    ) -> dict:
        return register_battery_soh_plugins(
            self.store,
            battery_root=battery_soh_root,
            replace=replace,
        )

    def sync_native_dashboard(
        self,
        *,
        source: str | Path | None = None,
        target: str | Path | None = None,
        include_curves: bool = True,
        include_detection: bool = False,
    ) -> dict:
        from scripts.sync_battery_dashboard_native import (
            DEFAULT_SOURCE,
            DEFAULT_TARGET,
            sync_native_dashboard,
        )

        return sync_native_dashboard(
            Path(source) if source else DEFAULT_SOURCE,
            Path(target) if target else DEFAULT_TARGET,
            include_curves=include_curves,
            include_detection=include_detection,
        )

    def package_portable_frontend(
        self,
        *,
        app_dir: str | Path | None = None,
        output: str | Path | None = None,
    ) -> dict:
        from scripts.package_portable_frontend import (
            DEFAULT_APP_DIR,
            DEFAULT_OUTPUT,
            package_portable_frontend,
        )

        return package_portable_frontend(
            Path(app_dir) if app_dir else DEFAULT_APP_DIR,
            Path(output) if output else DEFAULT_OUTPUT,
        )

    def package_standalone_frontend(
        self,
        *,
        app_dir: str | Path | None = None,
        output: str | Path | None = None,
    ) -> dict:
        from scripts.build_standalone_frontend import APP_DIR, DEFAULT_OUTPUT, build_standalone_html

        source_app = Path(app_dir) if app_dir else APP_DIR
        output_path = Path(output) if output else DEFAULT_OUTPUT
        html, artifact_count = build_standalone_html(source_app)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8", newline="\n")
        embedded_payload = _standalone_data_payload(html)
        return {
            "standalone": True,
            "app_dir": str(source_app),
            "output": str(output_path),
            "artifact_count": artifact_count,
            "total_bytes": output_path.stat().st_size,
            "excluded_detection_assets": not _payload_contains_detection_asset(embedded_payload),
        }

    def package_status(
        self,
        *,
        native_dashboard: str | Path | None = None,
        portable_output: str | Path | None = None,
        standalone_output: str | Path | None = None,
    ) -> dict:
        from scripts.build_standalone_frontend import DEFAULT_OUTPUT as DEFAULT_STANDALONE_OUTPUT
        from scripts.package_portable_frontend import DEFAULT_OUTPUT as DEFAULT_PORTABLE_OUTPUT
        from scripts.sync_battery_dashboard_native import DEFAULT_TARGET as DEFAULT_NATIVE_DASHBOARD

        native = _package_native_dashboard_info(Path(native_dashboard) if native_dashboard else DEFAULT_NATIVE_DASHBOARD)
        portable = _package_portable_info(Path(portable_output) if portable_output else DEFAULT_PORTABLE_OUTPUT)
        standalone = _package_standalone_info(Path(standalone_output) if standalone_output else DEFAULT_STANDALONE_OUTPUT)

        checks = [
            _package_check(
                "native_dashboard",
                "SOH ????",
                native["present"] and native["scenario_count"] > 0,
                f"{native['scenario_count']} ??? / {native['file_count']} ???",
            ),
            _package_check(
                "native_curves",
                "SOH ????",
                native["include_soh_curves"],
                f"{native['curve_file_count']} ?????",
            ),
            _package_check(
                "portable_package",
                "???",
                portable["present"] and portable["file_count"] > 0 and portable["has_manifest"],
                f"{portable['file_count']} ??? / {_format_bytes(portable['total_bytes'])}",
            ),
            _package_check(
                "standalone_html",
                "? HTML",
                standalone["present"] and standalone["standalone"] and standalone["artifact_count"] > 0,
                f"{standalone['artifact_count']} ??? JSON / {_format_bytes(standalone['total_bytes'])}",
            ),
            _package_check(
                "detection_excluded",
                "????????",
                not native["detection_assets"]
                and not portable["detection_assets"]
                and not standalone["contains_detection_asset"],
                "??? ess_detection ??",
            ),
        ]
        passed = sum(1 for check in checks if check["passed"])
        return {
            "generated_at": _now(),
            "summary": {
                "ready": passed == len(checks),
                "passed": passed,
                "failed": len(checks) - passed,
                "total": len(checks),
            },
            "checks": checks,
            "outputs": {
                "native_dashboard": native,
                "portable": portable,
                "standalone": standalone,
            },
        }

    def _save_job(self, job: dict) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        (self.jobs_dir / f"{job['id']}.json").write_text(
            json.dumps(job, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_inference_job(self, job: dict) -> None:
        self.inference_jobs_dir.mkdir(parents=True, exist_ok=True)
        (self.inference_jobs_dir / f"{job['id']}.json").write_text(
            json.dumps(job, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _compact_job(job: dict) -> dict:
    keep = {
        "id",
        "kind",
        "status",
        "created_at",
        "started_at",
        "finished_at",
        "run_dir",
        "raw_run_dir",
        "request",
        "error",
    }
    return {key: job[key] for key in keep if key in job}


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _safe_profile_name(value: str) -> str:
    name = str(value or "battery-soh-diagnostics").strip()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", name):
        raise ValueError(f"invalid profile name: {name}")
    return name


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _package_check(check_id: str, label: str, passed: bool, detail: str) -> dict:
    return {
        "id": check_id,
        "label": label,
        "passed": bool(passed),
        "detail": detail,
        "severity": "required",
    }


def _read_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _directory_file_stats(path: Path) -> tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return file_count, total_bytes


def _format_bytes(value: int | float) -> str:
    size = float(value or 0)
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{int(size)} B"


def _package_native_dashboard_info(path: Path) -> dict:
    path = Path(path)
    data_dir = path / "data"
    manifest = _read_json_if_exists(path / "SYNC_MANIFEST.json")
    source_manifest = _read_json_if_exists(data_dir / "manifest.json")
    scenarios = source_manifest.get("scenarios") or []
    curve_files = [
        item
        for item in data_dir.glob("*_curves.json")
        if item.is_file() and not item.name.startswith("ess_detection")
    ]
    detection_assets = [item.relative_to(path).as_posix() for item in path.rglob("*ess_detection*.json") if item.is_file()]
    file_count, total_bytes = _directory_file_stats(path)
    scenario_count = int(manifest.get("scenario_count") or len(scenarios))
    return {
        "path": str(path),
        "present": (path / "index.html").is_file() and data_dir.is_dir(),
        "has_manifest": bool(manifest),
        "scenario_count": scenario_count,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "include_soh_curves": bool(manifest.get("include_soh_curves") or curve_files),
        "curve_file_count": len(curve_files),
        "excluded_detection_assets": not detection_assets,
        "detection_assets": detection_assets,
    }


def _package_portable_info(path: Path) -> dict:
    path = Path(path)
    manifest = _read_json_if_exists(path / "MANIFEST.json")
    file_count, total_bytes = _directory_file_stats(path)
    detection_assets = [item.relative_to(path).as_posix() for item in path.rglob("*ess_detection*.json") if item.is_file()]
    return {
        "path": str(path),
        "present": path.is_dir(),
        "has_manifest": bool(manifest),
        "file_count": int(manifest.get("file_count") or file_count),
        "total_bytes": int(manifest.get("total_bytes") or total_bytes),
        "include_soh_curves": bool(manifest.get("include_soh_curves")),
        "excluded_detection_assets": bool(manifest.get("excluded_detection_assets")) and not detection_assets,
        "detection_assets": detection_assets,
    }


def _package_standalone_info(path: Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    payload = _standalone_data_payload(text)
    total_bytes = path.stat().st_size if path.is_file() else 0
    return {
        "path": str(path),
        "present": path.is_file(),
        "standalone": "window.__SOH_STANDALONE__ = true" in text,
        "artifact_count": len(payload),
        "total_bytes": total_bytes,
        "contains_detection_asset": _payload_contains_detection_asset(payload),
        "has_package_operations": "package-operations-card" in text,
        "has_package_report": "package-report-card" in text,
    }


def _standalone_data_payload(text: str) -> dict:
    if not text:
        return {}
    match = re.search(
        r"const\s+SOH_STANDALONE_DATA\s*=\s*(\{.*?\});\s*window\.__SOH_STANDALONE__",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_contains_detection_asset(value) -> bool:
    if isinstance(value, dict):
        return any(
            "ess_detection" in str(key) or _payload_contains_detection_asset(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_payload_contains_detection_asset(item) for item in value)
    return isinstance(value, str) and "ess_detection" in value


def _job_sort_key(job: dict) -> tuple[str, str]:
    timestamp = job.get("finished_at") or job.get("started_at") or job.get("created_at") or ""
    return str(timestamp), str(job.get("id") or "")


def _model_name_token(value) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return token or "model"


def _sync_frontend_files(run_dir: Path, frontend_data_dir: Path) -> None:
    sync_frontend_artifacts(run_dir, frontend_data_dir)
