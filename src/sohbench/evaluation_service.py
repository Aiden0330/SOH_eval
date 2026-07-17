from __future__ import annotations

import inspect
import json
import os
import re
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

import pandas as pd

from . import registry as R
from .battery_soh_plugins import register_battery_soh_plugins
from .contracts import feature_columns
from .data import DatasetLoader, normalize_cycle_index
from .battery_dashboard_import import publish_battery_dashboard_profile
from .frontend_publish import (
    publish_frontend_artifacts,
    publish_frontend_profile,
    resolve_published_directory,
    sync_frontend_artifacts,
)
from .inference import run_inference_benchmark
from .model_store import (
    LoadedModel,
    ModelReadFailure,
    ModelReadSnapshot,
    ModelStore,
)
from .protocols import LeaveOneCellOut, LeaveOneChemistryOut, LeaveOneDatasetOut
from .report import write_artifacts, write_inference_artifacts
from .runner import BenchmarkRunner
from .tasks import SOHEstimationTask
from .trusted_plugins import TrustedPluginStore
from .fabric import (
    CandidateSpec,
    DatasetSnapshot,
    FeatureState,
    IdentityFeaturePipeline,
    RunOutcome,
    RunSpec,
    SpecRef,
    frame_digest,
    new_id,
    scoped_meta_for_frame,
)
from .fabric.evidence import verify_evidence_pack, write_evidence_pack
from .fabric.ids import sha256_file, sha256_json


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
        run_id = new_id("RUN")
        job = {
            "id": job_id,
            "status": "queued",
            "run_id": run_id,
            "execution_state": "QUEUED",
            "outcome": None,
            "verdict": "not_reviewed",
            "evidence_path": None,
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
        run_id = new_id("RUN")
        job = {
            "id": job_id,
            "kind": "inference",
            "status": "queued",
            "run_id": run_id,
            "execution_state": "QUEUED",
            "outcome": None,
            "verdict": "not_reviewed",
            "evidence_path": None,
            "created_at": _now(),
            "request": request,
        }
        self._save_inference_job(job)
        if start:
            thread = threading.Thread(target=self.run_inference_job_inline, args=(job_id,), daemon=True)
            thread.start()
        return job

    def run_job_inline(self, job_id: str) -> dict:
        with self.store.execution_lease():
            return self._run_job_inline_leased(job_id)

    def _run_job_inline_leased(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if not job.get("run_id"):
            job["run_id"] = new_id("RUN")
        job.update({
            "status": "running",
            "execution_state": "RUNNING",
            "outcome": None,
            "verdict": "not_reviewed",
            "evidence_path": None,
            "started_at": _now(),
        })
        self._save_job(job)
        raw_run_dir = None
        try:
            request = job["request"]
            run_id = job["run_id"]
            loader = DatasetLoader(self.data_dir)
            datasets = loader.resolve_datasets(request["datasets"])
            table, meta, qc = loader.load(datasets)
            sample_ids = _intrinsic_sample_ids(table)
            run_spec = self._build_run_spec(
                run_type="training", request=request, datasets=datasets,
                table=table, qc=qc,
            )
            estimators = [R.get_estimator(name) for name in request["estimators"]]
            protocols = {name: PROTOCOLS[name]() for name in request["protocols"]}
            feature_pipelines = {
                name: pipeline
                for name in request["featuresets"]
                if (pipeline := self.store.feature_pipeline(name)) is not None
            }
            saved_models = []
            model_warnings = []
            prediction_rows = []

            def save_fitted_model(**kwargs) -> None:
                record = kwargs["record"]
                fitted = kwargs["estimator"]
                train_table = kwargs["train_table"]
                feature_state = kwargs["feature_state"]
                feature_pipeline_name = kwargs["pipeline_name"]
                fs_name = str(record.get("featureset", "full"))
                model_name = "_".join(
                    [
                        _model_name_token(run_id),
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
                    "created_by_run": run_id,
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
                        )
                    )
                except Exception as exc:
                    model_warnings.append(f"{model_name}: {exc}")

            def collect_predictions(**kwargs) -> None:
                record = kwargs["record"]
                sample_frame = kwargs["sample_frame"]
                y_pred = kwargs["y_pred"]
                interval_low = kwargs["interval_low"]
                interval_high = kwargs["interval_high"]
                for position, (_, sample) in enumerate(sample_frame.iterrows()):
                    dataset = str(sample["dataset"])
                    cell_id = str(sample["cell_id"])
                    cycle_index = int(sample["cycle_index"])
                    row = {
                        "run_id": run_id,
                        "candidate_id": str(record["estimator"]),
                        "featureset": str(record["featureset"]),
                        "protocol": str(record["protocol"]),
                        "fold": int(record["fold"]),
                        "seed": int(record["seed"]),
                        "dataset": dataset,
                        "cell_id": cell_id,
                        "cycle_index": cycle_index,
                        "sample_id": sample_ids[(dataset, cell_id, cycle_index)],
                        "y_pred": float(y_pred[position]),
                        "interval_low": float(interval_low[position]),
                        "interval_high": float(interval_high[position]),
                    }
                    if "soh" in sample:
                        row["y_true"] = float(sample["soh"])
                    prediction_rows.append(row)

            results = BenchmarkRunner(seeds=request["seeds"]).run(
                SOHEstimationTask(),
                table=table,
                meta=meta,
                protocols=protocols,
                estimators=estimators,
                featuresets=request["featuresets"],
                feature_pipelines=feature_pipelines,
                on_model_fitted=save_fitted_model,
                on_predictions=collect_predictions,
            )
            results["dataset"] = ",".join(datasets)
            results["candidate_id"] = results["estimator"].astype(str)
            predictions = pd.DataFrame.from_records(prediction_rows) if prediction_rows else None
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
                    run_id=run_id,
                    evidence_context={"run_spec": run_spec},
                    predictions=predictions,
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
            self._rebuild_evidence_pack(raw_run_dir)
            published_run_dir = self._publish_frontend_artifacts(raw_run_dir)
            outcome = RunOutcome.from_results(results)
            job.update(
                {
                    "run_id": run_id,
                    "execution_state": "REVIEW" if outcome.outcome != "failed" else "FAILED",
                    "outcome": outcome.outcome,
                    "verdict": "not_reviewed" if outcome.outcome != "failed" else "fail",
                    "status": "succeeded" if outcome.outcome == "complete" else "partial" if outcome.outcome == "partial" else "failed",
                    "evidence_path": str(raw_run_dir / "evidence-pack.json"),
                    "finished_at": _now(),
                    "raw_run_dir": str(raw_run_dir),
                    "run_dir": str(published_run_dir),
                    "saved_models": [item["name"] for item in saved_models],
                    "model_warnings": model_warnings,
                }
            )
        except Exception as exc:
            evidence_path = _verified_evidence_path(raw_run_dir)
            job.update(
                {
                    "status": "failed",
                    "execution_state": "FAILED",
                    "outcome": "failed",
                    "verdict": "fail",
                    "evidence_path": evidence_path,
                    "finished_at": _now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=12),
                }
            )
            if evidence_path is not None:
                job.update({"raw_run_dir": str(raw_run_dir), "run_dir": str(raw_run_dir)})
        self._save_job(job)
        return job

    def run_inference_job_inline(self, job_id: str) -> dict:
        with self.store.execution_lease():
            return self._run_inference_job_inline_leased(job_id)

    def _run_inference_job_inline_leased(self, job_id: str) -> dict:
        job = self.get_inference_job(job_id)
        if not job.get("run_id"):
            job["run_id"] = new_id("RUN")
        job.update({
            "status": "running",
            "execution_state": "RUNNING",
            "outcome": None,
            "verdict": "not_reviewed",
            "evidence_path": None,
            "started_at": _now(),
        })
        self._save_inference_job(job)
        raw_run_dir = None
        try:
            request = job["request"]
            run_id = job["run_id"]
            loader = DatasetLoader(self.data_dir)
            datasets = loader.resolve_datasets(
                request["datasets"], require_labels=False,
            )
            table, meta, _qc = loader.load(
                datasets, require_labels=False, qc=False,
            )
            with self.model_store.partitioned_read_lease(
                request["models"]
            ) as (model_snapshots, model_failures):
                run_spec = self._build_run_spec(
                    run_type="inference", request=request, datasets=datasets,
                    table=table, qc={}, model_snapshots=model_snapshots,
                    model_failures=model_failures,
                )
                inferred_featuresets = self._featuresets_for_models(
                    request["models"], model_snapshots=model_snapshots,
                )
                records = []
                for model_name in request["models"]:
                    pair_datasets = self._pair_datasets_for_model(
                        request, model_name, datasets,
                    )
                    if not pair_datasets:
                        continue
                    model_table = table[
                        table["dataset"].astype(str).isin(pair_datasets)
                    ].copy()
                    if model_name in model_failures:
                        records.extend(
                            self._inference_preparation_failures(
                                model_name,
                                pair_datasets,
                                model_table,
                                model_failures[model_name].error,
                            )
                        )
                        continue
                    try:
                        loaded = model_snapshots[model_name].loaded
                        prepared = self._prepare_inference_table(
                            model_name, model_table, meta, loaded_model=loaded,
                        )
                        records.extend(
                            run_inference_benchmark(
                                self.model_store,
                                prepared,
                                meta,
                                [model_name],
                                loaded_models={model_name: loaded},
                            )
                        )
                    except Exception as exc:
                        records.extend(
                            self._inference_preparation_failures(
                                model_name, pair_datasets, model_table, exc,
                            )
                        )
                model_catalog = self.model_store.catalog()
                raw_run_dir = Path(
                    write_inference_artifacts(
                        records,
                        {
                            **request,
                            "datasets": datasets,
                            "job_id": job_id,
                            "run_id": run_id,
                            "run_dir": str(self.runs_dir / run_id),
                            "raw_run_dir": str(self.runs_dir / run_id),
                            "trusted_local_pickle_models": True,
                            "applied_featureset_transforms": inferred_featuresets,
                        },
                        self.runs_dir,
                        run_id=run_id,
                        evidence_context={"run_spec": run_spec},
                    )
                )
                (raw_run_dir / "model_catalog.json").write_text(
                    json.dumps(
                        {"models": model_catalog},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                self._rebuild_evidence_pack(raw_run_dir)
            training_run_dirs = self._training_run_dirs_for_models(
                model_snapshots, model_catalog=model_catalog
            )
            published_run_dir = self._publish_frontend_composite(
                raw_run_dir, training_run_dirs
            )
            outcome = self._inference_outcome(records)
            job.update(
                {
                    "run_id": run_id,
                    "execution_state": "REVIEW" if outcome.outcome != "failed" else "FAILED",
                    "outcome": outcome.outcome,
                    "verdict": "not_reviewed" if outcome.outcome != "failed" else "fail",
                    "status": "succeeded" if outcome.outcome == "complete" else "partial" if outcome.outcome == "partial" else "failed",
                    "evidence_path": str(raw_run_dir / "evidence-pack.json"),
                    "finished_at": _now(),
                    "raw_run_dir": str(raw_run_dir),
                    "run_dir": str(published_run_dir),
                }
            )
        except Exception as exc:
            evidence_path = _verified_evidence_path(raw_run_dir)
            job.update(
                {
                    "status": "failed",
                    "execution_state": "FAILED",
                    "outcome": "failed",
                    "verdict": "fail",
                    "evidence_path": evidence_path,
                    "finished_at": _now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=12),
                }
            )
            if evidence_path is not None:
                job.update({"raw_run_dir": str(raw_run_dir), "run_dir": str(raw_run_dir)})
        self._save_inference_job(job)
        return job

    def get_job(self, job_id: str) -> dict:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            raise KeyError(job_id)
        return _read_job_json(path)

    def list_jobs(self) -> list[dict]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            try:
                jobs.append(_read_job_json(path))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=_job_sort_key, reverse=True)

    def get_inference_job(self, job_id: str) -> dict:
        path = self.inference_jobs_dir / f"{job_id}.json"
        if not path.exists():
            raise KeyError(job_id)
        return _read_job_json(path)

    def list_inference_jobs(self) -> list[dict]:
        jobs = []
        for path in sorted(self.inference_jobs_dir.glob("*.json"), reverse=True):
            try:
                jobs.append(_read_job_json(path))
            except (OSError, json.JSONDecodeError):
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
        if "pairs" in payload:
            raw_pairs = payload.get("pairs")
            if not isinstance(raw_pairs, list) or not raw_pairs:
                raise ValueError("pairs must be a non-empty list")
            selected_models = set(models)
            selected_datasets = set(datasets)
            pairs: list[dict[str, str]] = []
            seen_pairs: set[tuple[str, str]] = set()
            for item in raw_pairs:
                if not isinstance(item, dict):
                    raise ValueError("each inference pair must be an object")
                model = str(item.get("model") or "").strip()
                dataset = str(item.get("dataset") or "").strip()
                if not model or not dataset:
                    raise ValueError("each inference pair requires model and dataset")
                if model not in selected_models:
                    raise ValueError(f"pair references unselected model: {model}")
                if "all" not in selected_datasets and dataset not in selected_datasets:
                    raise ValueError(f"pair references unselected dataset: {dataset}")
                identity = (model, dataset)
                if identity in seen_pairs:
                    continue
                seen_pairs.add(identity)
                pairs.append({"model": model, "dataset": dataset})
            request["pairs"] = pairs
        return request

    def _pair_datasets_for_model(
        self,
        request: dict,
        model_name: str,
        resolved_datasets: list[str],
    ) -> list[str]:
        if "pairs" not in request:
            return list(resolved_datasets)
        resolved = set(map(str, resolved_datasets))
        selected = [
            str(item["dataset"])
            for item in request["pairs"]
            if item.get("model") == model_name
        ]
        unknown = [dataset for dataset in selected if dataset not in resolved]
        if unknown:
            raise ValueError(
                f"inference pairs reference unresolved datasets: {unknown}"
            )
        return selected

    def _build_run_spec(
        self,
        *,
        run_type: str,
        request: dict,
        datasets: list[str],
        table: pd.DataFrame,
        qc,
        model_snapshots: dict[str, ModelReadSnapshot] | None = None,
        model_failures: dict[str, ModelReadFailure] | None = None,
    ) -> RunSpec:
        task = SOHEstimationTask()
        raw_protocols = request.get("protocols") or ["inference"]
        if isinstance(raw_protocols, str):
            raw_protocols = [raw_protocols]
        canonical_protocols = sorted(set(map(str, raw_protocols)))
        canonical_request = {**request, "protocols": canonical_protocols}
        label_digest = (
            frame_digest(table[[task.target]])
            if task.target in table.columns
            else sha256_json({"target": task.target, "state": "absent"})
        )
        snapshot = DatasetSnapshot.from_frame(
            table,
            datasets,
            label_digest,
            qc.__dict__ if hasattr(qc, "__dict__") else qc,
        )

        def ref(kind: str, spec_id: str, document: dict) -> SpecRef:
            return SpecRef(kind=kind, spec_id=spec_id, version="1.0.0", digest=sha256_json(document))

        candidates = request["estimators"] if run_type == "training" else request["models"]
        return RunSpec(
            run_type=run_type,
            task=ref("task", task.name, {"name": task.name, "target": task.target}),
            dataset=ref("dataset_snapshot", snapshot.snapshot_id, {
                "snapshot_id": snapshot.snapshot_id,
                "digest": snapshot.digest,
                "datasets": list(snapshot.datasets),
                "rows": snapshot.rows,
                "assets": snapshot.assets,
                "label_digest": snapshot.label_digest,
                "qc": dict(snapshot.qc),
            }),
            protocol=ref("protocol", ",".join(canonical_protocols), {
                "protocols": canonical_protocols,
            }),
            metric=ref("metric", "soh_metrics", {
                "point": list(task.point_metrics),
                "interval": list(task.interval_metrics),
                "trajectory": list(task.trajectory_metrics),
            }),
            policy=ref("policy", "trusted_local", {
                "trusted_local_plugins": run_type == "training",
                "trusted_local_pickle_models": run_type == "inference",
            }),
            candidates=tuple(
                self._candidate_spec(
                    run_type=run_type,
                    candidate_name=str(candidate),
                    request=request,
                    model_snapshot=(model_snapshots or {}).get(str(candidate)),
                    model_failure=(model_failures or {}).get(str(candidate)),
                )
                for candidate in candidates
            ),
            seeds=tuple(request.get("seeds", [0])),
            idempotency_key=sha256_json({
                "run_type": run_type,
                "request": canonical_request,
                "dataset": snapshot.digest,
            }),
        )

    def _candidate_spec(
        self,
        *,
        run_type: str,
        candidate_name: str,
        request: dict,
        model_snapshot: ModelReadSnapshot | None = None,
        model_failure: ModelReadFailure | None = None,
    ) -> CandidateSpec | SpecRef:
        if run_type == "training":
            estimator = R.get_estimator(candidate_name)
            artifacts = [
                _file_spec_ref(
                    "estimator_plugin",
                    candidate_name,
                    _source_path(estimator),
                )
            ]
            artifacts.extend(
                self._plugin_dependency_refs("estimators", candidate_name)
            )
            for featureset_name in request.get("featuresets", []):
                artifacts.append(self._feature_content_ref(featureset_name))
                artifacts.extend(
                    self._plugin_dependency_refs(
                        "featuresets", featureset_name
                    )
                )
        else:
            if model_failure is not None:
                return SpecRef(
                    kind="candidate_unavailable",
                    spec_id=candidate_name,
                    version="content-identity-v1",
                    digest=sha256_json(model_failure.content_identity),
                )
            known_models = {
                item["name"] for item in self.model_store.catalog()
            }
            if candidate_name not in known_models:
                return SpecRef(
                    kind="candidate",
                    spec_id=candidate_name,
                    version="1.0.0",
                    digest=sha256_json(
                        {
                            "name": candidate_name,
                            "run_type": run_type,
                            "content": "unavailable",
                        }
                    ),
                )
            digests = (
                model_snapshot.digests
                if model_snapshot is not None
                else self.model_store.content_digests(candidate_name)
            )
            artifacts = [
                SpecRef(
                    kind="model_pickle",
                    spec_id=f"{candidate_name}/model.pkl",
                    version="sha256",
                    digest=digests["model_pickle"],
                ),
                SpecRef(
                    kind="model_manifest",
                    spec_id=f"{candidate_name}/model_manifest.json",
                    version="sha256",
                    digest=digests["model_manifest"],
                ),
            ]
            manifest = (
                model_snapshot.loaded.manifest
                if model_snapshot is not None
                else self.model_store.load_model(candidate_name).manifest
            )
            featureset_name = str(
                manifest.get("feature_pipeline")
                or manifest.get("featureset")
                or ""
            ).strip()
            estimator_name = str(manifest.get("estimator") or "").strip()
            if estimator_name:
                try:
                    estimator = R.get_estimator(estimator_name)
                    estimator_source = _source_path(estimator)
                except (KeyError, OSError, ValueError):
                    pass
                else:
                    artifacts.append(
                        _file_spec_ref(
                            "estimator_plugin",
                            estimator_name,
                            estimator_source,
                        )
                    )
                    artifacts.extend(
                        self._plugin_dependency_refs(
                            "estimators", estimator_name
                        )
                    )
            if featureset_name:
                artifacts.append(
                    self._feature_content_ref(
                        featureset_name,
                        manifest=manifest,
                    )
                )
                artifacts.extend(
                    self._plugin_dependency_refs(
                        "featuresets", featureset_name
                    )
                )
        return CandidateSpec(
            spec_id=candidate_name,
            artifacts=tuple(artifacts),
        )

    def _plugin_dependency_refs(
        self, section: str, plugin_name: str
    ) -> list[SpecRef]:
        return [
            SpecRef(
                kind=f"executable_dependency_{item['kind']}",
                spec_id=f"{section}/{plugin_name}/{item['name']}",
                version=(
                    "sha256-tree-v1" if item["kind"] == "tree" else "sha256"
                ),
                digest=str(item["digest"]),
            )
            for item in self.store.declared_plugin_dependencies(
                section, plugin_name
            )
        ]

    def _feature_content_ref(
        self, featureset_name: str, *, manifest: dict | None = None
    ) -> SpecRef:
        name = str(featureset_name)
        pipeline = self.store.feature_pipeline(name)
        plugin_path = getattr(pipeline, "_path", None)
        if plugin_path is not None:
            return _file_spec_ref("feature_plugin", name, Path(plugin_path))
        try:
            featureset = R.get_featureset(name)
        except KeyError:
            return SpecRef(
                kind="feature_definition",
                spec_id=name,
                version="sha256",
                digest=sha256_json(
                    {
                        "featureset": name,
                        "feature_columns": list(
                            (manifest or {}).get("feature_columns", [])
                        ),
                    }
                ),
            )
        return _file_spec_ref(
            "feature_definition", name, _source_path(type(featureset))
        )

    def _inference_outcome(self, records: list[dict]) -> RunOutcome:
        rows = [
            {
                "estimator": record.get("model", "unknown"),
                "featureset": record.get("featureset", "full"),
                "protocol": "inference",
                "held_out_key": record.get("dataset", "all"),
                "fold": index,
                "seed": 0,
                "unit_status": record.get("status", "failed"),
            }
            for index, record in enumerate(records)
        ]
        return RunOutcome.from_results(pd.DataFrame.from_records(rows))

    def _inference_preparation_failures(
        self,
        model_name: str,
        datasets: list[str],
        table: pd.DataFrame,
        exc: Exception,
    ) -> list[dict]:
        catalog_item = next(
            (
                item for item in self.model_store.catalog()
                if item.get("name") == model_name
            ),
            {},
        )
        return [
            {
                "model": model_name,
                "dataset": dataset,
                "featureset": catalog_item.get("featureset"),
                "estimator": catalog_item.get("estimator"),
                "n_samples": int(table["dataset"].eq(dataset).sum()),
                "load_ms": 0.0,
                "mae": None,
                "rmse": None,
                "mean_bias_error": None,
                "monotonicity_violation_rate": None,
                "value_range_violation": None,
                "predict_ms": None,
                "ms_per_1000_samples": None,
                "samples_per_second": None,
                "status": "failed",
                "error": str(exc),
            }
            for dataset in sorted(map(str, datasets))
        ]

    def _rebuild_evidence_pack(self, run_dir: Path) -> dict:
        pack = verify_evidence_pack(run_dir / "evidence-pack.json")
        manifest = {
            key: value
            for key, value in pack.items()
            if key not in {"schema_version", "artifacts", "ok"}
        }
        artifact_paths = [run_dir / item["path"] for item in pack["artifacts"]]
        catalog = run_dir / "model_catalog.json"
        if catalog.exists():
            artifact_paths.append(catalog)
        write_evidence_pack(run_dir, manifest, artifact_paths)
        return verify_evidence_pack(run_dir / "evidence-pack.json")

    def _featuresets_for_models(
        self,
        model_names: list[str],
        *,
        model_snapshots: dict[str, ModelReadSnapshot] | None = None,
    ) -> list[str]:
        selected = set(model_names)
        featuresets: list[str] = []
        if model_snapshots is not None:
            items = [
                {
                    "name": name,
                    "featureset": snapshot.loaded.manifest.get("featureset"),
                }
                for name, snapshot in model_snapshots.items()
            ]
        else:
            items = self.model_store.catalog()
        for item in items:
            if item.get("name") not in selected:
                continue
            name = str(item.get("featureset") or "").strip()
            if name and name not in featuresets:
                featuresets.append(name)
        return featuresets

    def _prepare_inference_table(
        self,
        model_name: str,
        table,
        meta,
        *,
        loaded_model: LoadedModel | None = None,
    ):
        loaded = loaded_model or self.model_store.load_model(model_name)
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

    def _publish_frontend_composite(
        self, inference_run_dir: Path, training_run_dirs: list[Path]
    ) -> Path:
        publish_frontend_artifacts(
            [*training_run_dirs, inference_run_dir], self.frontend_data_dir
        )
        return inference_run_dir

    def _training_run_dirs_for_models(
        self,
        model_snapshots: dict[str, ModelReadSnapshot],
        *,
        model_catalog: list[dict] | None = None,
    ) -> list[Path]:
        catalog_by_name = {
            str(item.get("name")): item
            for item in (
                model_catalog
                if model_catalog is not None
                else self.model_store.catalog()
            )
        }
        jobs_by_run = {
            str(job.get("run_id")): job for job in self.list_jobs()
        }
        run_dirs = []
        seen = set()
        for model_name, snapshot in model_snapshots.items():
            run_id = str(
                snapshot.loaded.manifest.get("created_by_run") or ""
            ).strip()
            catalog_item = catalog_by_name.get(str(model_name), {})
            run_dir = self.runs_dir / run_id
            job = jobs_by_run.get(run_id, {})
            if (
                not run_id
                or run_id in seen
                or Path(run_id).name != run_id
                or catalog_item.get("source") != "training"
                or str(catalog_item.get("created_by_run") or "") != run_id
                or snapshot.loaded.manifest.get("model_name") != model_name
                or catalog_item.get("model_sha256")
                != snapshot.digests.get("model_pickle")
                or catalog_item.get("manifest_sha256")
                != snapshot.digests.get("model_manifest")
                or job.get("status") != "succeeded"
                or str(model_name) not in set(map(str, job.get("saved_models", [])))
                or not run_dir.is_dir()
                or not self._training_run_matches_catalog(
                    run_dir, run_id, str(model_name), catalog_item, job
                )
            ):
                continue
            seen.add(run_id)
            run_dirs.append(run_dir)
        return run_dirs

    def _training_run_matches_catalog(
        self,
        run_dir: Path,
        run_id: str,
        model_name: str,
        catalog_item: dict,
        job: dict,
    ) -> bool:
        try:
            expected_run_dir = run_dir.resolve()
            if expected_run_dir.parent != self.runs_dir.resolve():
                return False
            if (
                Path(str(job.get("raw_run_dir") or "")).resolve()
                != expected_run_dir
            ):
                return False
            evidence = verify_evidence_pack(run_dir / "evidence-pack.json")
            if evidence.get("run_id") != run_id:
                return False
            run_config = json.loads(
                (run_dir / "run_config.json").read_text(encoding="utf-8")
            )
            if model_name not in set(
                map(str, run_config.get("saved_models", []))
            ):
                return False
            run_catalog = json.loads(
                (run_dir / "model_catalog.json").read_text(encoding="utf-8")
            )
            run_item = next(
                item
                for item in run_catalog.get("models", [])
                if item.get("name") == model_name
            )
        except (OSError, ValueError, KeyError, StopIteration):
            return False
        authenticated_fields = (
            "name",
            "source",
            "created_by_run",
            "model_sha256",
            "manifest_sha256",
        )
        return all(
            run_item.get(field) == catalog_item.get(field)
            for field in authenticated_fields
        )

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
        frontend_root = resolve_published_directory(self.frontend_data_dir)
        manifest_path = frontend_root / "profiles" / profile / "manifest.json"
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
        from scripts.build_standalone_frontend import (
            APP_DIR,
            DEFAULT_OUTPUT,
            build_standalone_html,
            publish_standalone_html,
        )

        source_app = Path(app_dir) if app_dir else APP_DIR
        output_path = Path(output) if output else DEFAULT_OUTPUT
        html, artifact_count = build_standalone_html(source_app)
        publish_standalone_html(output_path, html)
        embedded_payload = _standalone_data_payload(html)
        return {
            "standalone": True,
            "app_dir": str(source_app),
            "output": str(output_path),
            "artifact_count": artifact_count,
            "total_bytes": output_path.stat().st_size,
            "native_dashboard_included": _standalone_native_dashboard_included(html),
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
                "SOH 完整看板",
                native["present"] and native["scenario_count"] > 0,
                f"{native['scenario_count']} 个场景 / {native['file_count']} 个文件",
            ),
            _package_check(
                "native_curves",
                "SOH 曲线数据",
                native["include_soh_curves"],
                f"{native['curve_file_count']} 个曲线文件",
            ),
            _package_check(
                "portable_package",
                "便携目录包",
                portable["present"] and portable["file_count"] > 0 and portable["has_manifest"],
                f"{portable['file_count']} 个文件 / {_format_bytes(portable['total_bytes'])}",
            ),
            _package_check(
                "standalone_html",
                "单文件 HTML",
                standalone["present"] and standalone["standalone"] and standalone["artifact_count"] > 0,
                f"{standalone['artifact_count']} 个内嵌 JSON / {_format_bytes(standalone['total_bytes'])}",
            ),
            _package_check(
                "detection_excluded",
                "检测数据已排除",
                not native["detection_assets"]
                and not portable["detection_assets"]
                and not standalone["contains_detection_asset"],
                "未包含 ess_detection 资产",
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
        _atomic_json_write(self.jobs_dir / f"{job['id']}.json", job)

    def _save_inference_job(self, job: dict) -> None:
        self.inference_jobs_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(self.inference_jobs_dir / f"{job['id']}.json", job)


def _intrinsic_sample_ids(table: pd.DataFrame) -> dict[tuple[str, str, int], str]:
    identity_columns = ["dataset", "cell_id", "cycle_index"]
    missing = [column for column in identity_columns if column not in table.columns]
    if missing:
        raise ValueError(
            f"prediction identity requires columns: {', '.join(missing)}"
        )
    identity_table = normalize_cycle_index(table, context="prediction identity")
    sample_ids = {}
    for dataset, cell_id, cycle_index in identity_table[identity_columns].itertuples(
        index=False, name=None,
    ):
        key = (str(dataset), str(cell_id), int(cycle_index))
        if key in sample_ids:
            raise ValueError(
                "duplicate intrinsic sample identity: "
                f"dataset={key[0]!r}, cell_id={key[1]!r}, cycle_index={key[2]}"
            )
        sample_ids[key] = "SAMPLE-" + sha256_json({
            "dataset": key[0],
            "cell_id": key[1],
            "cycle_index": key[2],
        })
    return sample_ids


def _verified_evidence_path(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    evidence_path = Path(run_dir) / "evidence-pack.json"
    if not evidence_path.is_file():
        return None
    try:
        verify_evidence_pack(evidence_path)
    except (OSError, ValueError):
        return None
    return str(evidence_path)


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
        "run_id",
        "execution_state",
        "outcome",
        "verdict",
        "evidence_path",
    }
    return {key: job[key] for key in keep if key in job}


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _atomic_json_write(path: Path, document: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        deadline = time.monotonic() + 5.0
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_job_json(path: Path) -> dict:
    deadline = time.monotonic() + 5.0
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


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
    path = resolve_published_directory(path)
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
    path = resolve_published_directory(path)
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
        "native_dashboard_included": _standalone_native_dashboard_included(text),
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


def _standalone_native_dashboard_included(text: str) -> bool:
    marker = "window.__SOH_STANDALONE_NATIVE_DASHBOARD__ = "
    return marker in text and f"{marker}null;" not in text


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


def _source_path(value) -> Path:
    try:
        source = inspect.getsourcefile(value)
    except TypeError:
        source = None
    if source is None:
        module = inspect.getmodule(value)
        source = getattr(module, "__file__", None)
    if source is None:
        raise ValueError(f"source bytes are unavailable for {value!r}")
    path = Path(source).resolve()
    if not path.is_file():
        raise ValueError(f"source file is unavailable: {path}")
    return path


def _file_spec_ref(kind: str, spec_id: str, path: Path) -> SpecRef:
    return SpecRef(
        kind=kind,
        spec_id=str(spec_id),
        version="sha256",
        digest=sha256_file(Path(path)),
    )


def _sync_frontend_files(run_dir: Path, frontend_data_dir: Path) -> None:
    sync_frontend_artifacts(run_dir, frontend_data_dir)
