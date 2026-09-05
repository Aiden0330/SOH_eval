"""sohkernel.api — the kernel surface any tool calls (Spec 01 §9).

``load`` → ``preflight`` → ``transform`` → ``score`` → ``verdicts`` → ``write_artifacts``.

Guarantees: pure with respect to jobs and cases, deterministic given seeds and data,
and never imports the workbench (``sohbench``) or the platform (``sohagent``).  The
admission logic here is what the workbench's ``EvaluationService.training_preflight``
delegates to; the workbench only adds its plugin-aware transform and provenance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import registry as R
from .acceptance import evaluate_verdicts
from .contracts import QCReport, Verdict, feature_columns
from .data import ALL_DATASETS, REQUIRED_COLUMNS, DatasetLoader
from .datasheets import chemistry_verified, load_datasheets
from .feature_safety import FeatureSafetyError, apply_transform_without_label, audit_feature_frame
from .metadata import supervision_kind
from .method_contracts import incompatible_pairing_reason, method_contract
from .protocols import PROTOCOLS
from .report import write_artifacts as _write_artifacts
from .runner import BenchmarkRunner
from .tasks import SOHEstimationTask

__all__ = [
    "DATA_DIR_ENV",
    "PROTOCOLS",
    "Preflight",
    "effective_training_featuresets",
    "load",
    "normalize_request",
    "preflight",
    "resolve_data_dir",
    "score",
    "supervision_errors",
    "training_dataset_profiles",
    "training_feature_profile",
    "training_protocol_profile",
    "transform",
    "verdicts",
    "write_artifacts",
]

DATA_DIR_ENV = "SOHBENCH_DATA_DIR"
TransformFn = Callable[[pd.DataFrame, pd.DataFrame | None, Sequence[str]], pd.DataFrame]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """The supervised data root: the argument, else ``SOHBENCH_DATA_DIR``."""
    if data_dir is not None:
        return Path(data_dir).expanduser()
    configured = os.environ.get(DATA_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    raise ValueError(f"no data root: pass data_dir or set {DATA_DIR_ENV}")


def load(
    datasets: Iterable[str] | str = ("all",),
    *,
    data_dir: str | Path | None = None,
    qc: bool = True,
    **qc_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, QCReport]:
    """Load ``processed/<name>_features.pkl`` tables → ``(table, meta, qc_report)``.

    ``"all"`` discovers every dataset satisfying the ``dataset / cell_id / cycle_index / soh``
    contract.  QC (SOH bounds, availability mask, BMS-label rejection) runs unless ``qc=False``.
    """
    loader = DatasetLoader(resolve_data_dir(data_dir))
    names = loader.resolve_datasets([datasets] if isinstance(datasets, str) else list(datasets))
    return loader.load(names, qc=qc, **qc_kwargs)


# ---------------------------------------------------------------------------
# request normalization
# ---------------------------------------------------------------------------

def _string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _require_known_datasets(datasets: list[str], data_dir: Path) -> None:
    """Unknown names must fail at normalization (a clean ValueError), never as a
    FileNotFoundError deep inside the pipeline that leaks the data path."""
    missing = [
        name for name in datasets
        if str(name).lower() not in ALL_DATASETS
        and not (data_dir / "processed" / f"{name}_features.pkl").is_file()
    ]
    if missing:
        raise ValueError(f"unknown datasets: {missing}")


def normalize_request(payload: Mapping, *, data_dir: str | Path | None = None) -> dict:
    """Validate a ``datasets × estimators × featuresets × protocols × seeds`` request.

    Defaults: ``all`` datasets, ``ridge``, ``full``, ``loco``, seed ``0``.  Unknown datasets
    (when a data root is known), unknown protocols → ``ValueError``; unknown estimators or
    featuresets → ``KeyError`` from the registry.
    """
    datasets = _string_list(payload.get("datasets") or ["all"])
    if data_dir is not None:
        _require_known_datasets(datasets, Path(data_dir))
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


# ---------------------------------------------------------------------------
# transform (label-isolated)
# ---------------------------------------------------------------------------

def transform(
    table: pd.DataFrame,
    meta: pd.DataFrame | None,
    featuresets: Iterable[str],
) -> pd.DataFrame:
    """Apply the registered featuresets' transforms behind the label-isolation boundary.

    A transform never receives the label column (``feature_safety.apply_transform_without_label``);
    its output is audited before it is accepted, it may not change the row count, and every
    candidate column it could not derive is materialised as NaN so per-model compatibility
    can block exactly the affected models.  Featuresets without a transform are no-ops.
    """
    out = table
    for name in featuresets or []:
        registered = R._FEATURESETS.get(str(name))
        fn = getattr(registered, "transform", None)
        if fn is None:
            continue
        candidates = list(getattr(registered, "candidates", []) or [])
        before_rows = len(out)
        transformed, _audit = apply_transform_without_label(
            fn, out, meta, name=str(name), candidates=candidates
        )
        if len(transformed) != before_rows:
            raise FeatureSafetyError(f"feature transform {name!r} changed row count")
        missing_required = REQUIRED_COLUMNS - set(transformed.columns)
        if missing_required:
            raise FeatureSafetyError(
                f"feature transform {name!r} dropped required columns: {sorted(missing_required)}"
            )
        for column in candidates:
            if column not in transformed.columns:
                transformed[column] = float("nan")
        out = transformed
    return out


# ---------------------------------------------------------------------------
# preflight (admission oracle)
# ---------------------------------------------------------------------------

def effective_training_featuresets(estimator_names: list[str], requested: list[str]) -> list[str]:
    """Include a joint method's preferred package when no selected feature is compatible."""
    effective = list(dict.fromkeys(map(str, requested)))
    for estimator_name in estimator_names:
        contract = method_contract(R.get_estimator(estimator_name))
        for feature_name in contract.feature_iterations(requested):
            if feature_name != "(byo)" and feature_name not in effective:
                effective.append(feature_name)
    return effective


def training_dataset_profiles(table: pd.DataFrame, datasets: list[str]) -> list[dict]:
    """Per dataset: rows, units, usable feature columns and the supervision kind (Spec 12 §4);
    for field sets also the labeled rows and the reference measurements per unit."""
    profiles = []
    for name in datasets:
        frame = table[table["dataset"].astype(str) == str(name)]
        usable = [
            column for column in feature_columns(frame)
            if column in frame.columns and bool(frame[column].notna().any())
        ]
        kind = supervision_kind(name)
        labeled = pd.to_numeric(frame["soh"], errors="coerce").notna() if "soh" in frame.columns else pd.Series(False, index=frame.index)
        profile = {
            "dataset": name,
            "rows": int(len(frame)),
            "cells": int(frame["cell_id"].astype(str).nunique()),
            "usable_feature_columns": usable,
            "supervision": kind,
            "labeled_rows": int(labeled.sum()),
        }
        if kind != "soh_supervised" and "label_time_s" in frame.columns:
            times = pd.to_numeric(frame["label_time_s"], errors="coerce")
            per_unit = frame.loc[times.notna()].groupby(frame.loc[times.notna(), "cell_id"].astype(str))["label_time_s"].nunique()
            profile["tests_per_unit"] = {str(k): int(v) for k, v in per_unit.items()}
        profiles.append(profile)
    return profiles


def supervision_errors(datasets: list[str]) -> list[str]:
    """Why a training request over these datasets is refused (Spec 12 §4): an ``unlabeled_field``
    dataset has no labels, so it can neither train nor be claimed — inference, audit and drift only."""
    return [
        f"dataset {name!r} is unlabeled_field: no labels, so it cannot train or be claimed "
        "(Spec 12 §4; roles inference, audit, drift)"
        for name in datasets if supervision_kind(name) == "unlabeled_field"
    ]


def training_feature_profile(
    table: pd.DataFrame,
    datasets: list[str],
    name: str,
    *,
    transform_error: str = "",
) -> dict:
    feature_set = R.get_featureset(name)
    selected = list(feature_set.columns(table))
    expected = list(getattr(feature_set, "candidates", []) or selected)
    dataset_coverage = []
    errors: list[str] = []
    warnings: list[str] = []
    if transform_error:
        errors.append(transform_error)
    if not selected:
        errors.append(f"feature set {name!r} selects zero usable columns")

    label_safety = audit_feature_frame(table, selected)
    errors.extend(item["reason"] for item in label_safety["violations"])
    warnings.extend(item["reason"] for item in label_safety.get("warnings", []))

    for dataset in datasets:
        frame = table[table["dataset"].astype(str) == str(dataset)]
        usable = [
            column for column in selected
            if column in frame.columns and bool(frame[column].notna().any())
        ]
        missing = [column for column in expected if column not in usable]
        status = "ready"
        if not usable:
            status = "blocked"
            errors.append(f"{dataset} has no usable columns for feature set {name}")
        elif missing:
            status = "warning"
            warnings.append(f"{dataset} is missing {len(missing)} expected feature columns")
        dataset_coverage.append(
            {
                "dataset": dataset,
                "status": status,
                "usable_columns": usable,
                "missing_columns": missing,
            }
        )
    return {
        "name": name,
        "selected_columns": selected,
        "expected_columns": expected,
        "dataset_coverage": dataset_coverage,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "label_safety": label_safety,
    }


def training_protocol_profile(table: pd.DataFrame, meta: pd.DataFrame | None, name: str,
                              *, datasheets: Mapping[str, Mapping] | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict = {}
    dataset_count = int(table["dataset"].astype(str).nunique())
    cell_count = int(
        table[["dataset", "cell_id"]].astype(str).drop_duplicates().shape[0]
    )
    if name == "loco":
        details = {"cells": cell_count}
        if cell_count < 2:
            errors.append("LOCO requires at least two cells")
    elif name == "lodo":
        details = {"datasets": dataset_count}
        if dataset_count < 2:
            errors.append("LODO requires at least two datasets")
    elif name == "lcho":
        chemistry = meta.get("chemistry") if hasattr(meta, "get") else None
        values = [] if chemistry is None else [str(value) for value in chemistry.dropna().unique()]
        unknown = [value for value in values if not value or value.lower() == "unknown"]
        known = sorted({value for value in values if value and value.lower() != "unknown"})
        details = {"chemistries": known, "unknown_metadata": unknown}
        if unknown:
            errors.append("LCHO requires known chemistry metadata for every selected cell")
        if len(known) < 2:
            errors.append("LCHO requires at least two known chemistries")
        # Spec 01 §1 / Spec 06: a dataset whose datasheet still carries VERIFY on its chemistry
        # cannot support a chemistry-level claim — the fold runs, the claim is flagged.
        if datasheets:
            unverified = sorted(
                name_ for name_ in map(str, table["dataset"].unique())
                if name_ in datasheets and not chemistry_verified(datasheets[name_])
            )
            details["unverified_chemistry"] = unverified
            if unverified:
                warnings.append(
                    f"chemistry not yet verified for {unverified} (datasheet VERIFY); this LCHO fold cannot support a chemistry claim"
                )
    elif name == "forward":
        protocol = PROTOCOLS["forward"]()
        plan = protocol.plan(table)
        eligible = [key for key, (n, n_train) in plan.items() if n_train < n]
        details = {
            "cells": cell_count,
            "eligible_cells": len(eligible),
            "train_frac": protocol.train_frac,
            "min_train_cycles": protocol.min_train_cycles,
        }
        if not eligible:
            errors.append(
                f"forward requires at least one cell with more than {protocol.min_train_cycles} cycles"
            )
        elif len(eligible) < cell_count:
            warnings.append(
                f"{cell_count - len(eligible)} of {cell_count} cells are too short to hold out a future segment"
            )
    else:
        errors.append(f"unknown protocol: {name}")
    if len(table) < 4:
        warnings.append("selected scope contains fewer than four QC-passed rows")
    return {
        "name": name,
        "status": "blocked" if errors else ("warning" if warnings else "ready"),
        "details": details,
        "errors": errors,
        "warnings": warnings,
    }


@dataclass
class Preflight:
    """Admission verdict for an exact ``data × feature × estimator × protocol`` matrix.

    ``combinations`` carry the per-cell status (``ready | warning | blocked | not_applicable``)
    with their ``reasons`` and ``warnings``; ``reasons``/``warnings`` here are the flattened,
    de-duplicated union.  ``table``/``meta`` are the QC-filtered, transformed inputs ready
    for :func:`score`.
    """

    ready: bool
    request: dict
    datasets: list[dict]
    featuresets: list[dict]
    protocols: list[dict]
    combinations: list[dict]
    summary: dict
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    table: pd.DataFrame | None = field(default=None, repr=False)
    meta: pd.DataFrame | None = field(default=None, repr=False)
    qc_report: QCReport | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "request": self.request,
            "datasets": self.datasets,
            "featuresets": self.featuresets,
            "protocols": self.protocols,
            "combinations": self.combinations,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def preflight(
    request: Mapping,
    *,
    data_dir: str | Path | None = None,
    table: pd.DataFrame | None = None,
    meta: pd.DataFrame | None = None,
    qc_report: QCReport | None = None,
    transform: TransformFn | None = None,
) -> Preflight:
    """Admission oracle: label safety, window coverage, feature coverage, protocol feasibility.

    ``request`` is a normalized request (:func:`normalize_request`).  Pass ``table``/``meta``
    to preflight already-loaded data (then ``request["datasets"]`` must be the resolved names);
    otherwise the datasets are resolved and loaded from ``data_dir``.  ``transform`` defaults to
    the kernel's label-isolated :func:`transform`; a caller may inject a plugin-aware one with
    the same signature.  A transform that fails (label-safety audit included) blocks every
    combination using that featureset with the error as the reason — excluded candidates never
    execute.
    """
    request = dict(request)
    apply = transform if transform is not None else globals()["transform"]
    if table is None:
        loader = DatasetLoader(resolve_data_dir(data_dir))
        datasets = loader.resolve_datasets(request.get("datasets") or ["all"])
        request["datasets"] = datasets
        table, meta, qc_report = loader.load(datasets)
    else:
        datasets = list(request.get("datasets") or [])
        if not datasets:
            datasets = [str(x) for x in pd.unique(table["dataset"].astype(str))]
            request["datasets"] = datasets
    request.setdefault("estimators", ["ridge"])
    request.setdefault("featuresets", ["full"])
    request.setdefault("protocols", ["loco"])
    request.setdefault("seeds", [0])

    effective = effective_training_featuresets(request["estimators"], request["featuresets"])
    transform_errors: dict[str, str] = {}
    for feature_name in effective:
        try:
            table = apply(table, meta, [feature_name])
        except Exception as exc:  # the reason is reported on every affected combination
            transform_errors[feature_name] = str(exc)

    dataset_profiles = training_dataset_profiles(table, datasets)
    feature_profiles = {
        name: training_feature_profile(
            table, datasets, name, transform_error=transform_errors.get(name, "")
        )
        for name in effective
    }
    try:
        datasheets = load_datasheets(data_dir)
    except Exception:  # a malformed datasheet must not break admission; the claim flow re-checks
        datasheets = {}
    protocol_profiles = {
        name: training_protocol_profile(table, meta, name, datasheets=datasheets) for name in request["protocols"]
    }
    refused = supervision_errors(datasets)

    seed_count = len(request["seeds"])
    combinations: list[dict] = []
    for estimator_name in request["estimators"]:
        estimator = R.get_estimator(estimator_name)
        contract = method_contract(estimator)
        feature_names = ["(byo)"] if contract.feature_mode == "byo" else list(dict.fromkeys([
            *request["featuresets"],
            *contract.feature_iterations(request["featuresets"]),
        ]))
        for feature_name in feature_names:
            feature = None if feature_name == "(byo)" else feature_profiles[feature_name]
            for protocol_name in request["protocols"]:
                protocol = protocol_profiles[protocol_name]
                incompatible = bool(
                    feature is not None and not contract.accepts_featureset(feature_name)
                )
                if incompatible:
                    combinations.append(
                        {
                            "estimator": estimator_name,
                            "featureset": feature_name,
                            "protocol": protocol_name,
                            **contract.catalog_metadata(),
                            "selected_columns": [],
                            "status": "not_applicable",
                            "reasons": [],
                            "warnings": [incompatible_pairing_reason(contract, feature_name)],
                            "seed_count": seed_count,
                            "run_instances": 0,
                        }
                    )
                    continue
                reasons = list(refused) + list(protocol["errors"])
                warnings = list(protocol["warnings"])
                selected_columns: list[str] = []
                if feature is not None:
                    selected_columns = list(feature["selected_columns"])
                    reasons.extend(feature["errors"])
                    warnings.extend(feature["warnings"])
                if contract.feature_mode not in {"unified", "byo"}:
                    reasons.append(f"unsupported estimator feature_mode: {contract.feature_mode}")
                status = "blocked" if reasons else ("warning" if warnings else "ready")
                combinations.append(
                    {
                        "estimator": estimator_name,
                        "featureset": feature_name,
                        "protocol": protocol_name,
                        **contract.catalog_metadata(),
                        "selected_columns": selected_columns,
                        "status": status,
                        "reasons": reasons,
                        "warnings": warnings,
                        "seed_count": seed_count,
                        "run_instances": seed_count,
                    }
                )

    blocked = [item for item in combinations if item["status"] == "blocked"]
    warned = [item for item in combinations if item["status"] == "warning"]
    not_applicable = [item for item in combinations if item["status"] == "not_applicable"]
    runnable = [item for item in combinations if item["status"] in {"ready", "warning"}]
    summary = {
        "rows": int(len(table)),
        "datasets": len(datasets),
        "base_combinations": len(combinations),
        "applicable_combinations": len(runnable) + len(blocked),
        "run_instances": sum(item["run_instances"] for item in combinations),
        "runnable_combinations": len(runnable),
        "warning_combinations": len(warned),
        "blocked_combinations": len(blocked),
        "not_applicable_combinations": len(not_applicable),
        "qc_pass_rate": float(qc_report.pass_rate) if qc_report is not None else 1.0,
    }
    return Preflight(
        ready=bool(runnable) and not blocked,
        request=request,
        datasets=dataset_profiles,
        featuresets=list(feature_profiles.values()),
        protocols=list(protocol_profiles.values()),
        combinations=combinations,
        summary=summary,
        reasons=list(dict.fromkeys(
            f"{item['estimator']} × {item['featureset']} × {item['protocol']}: {reason}"
            for item in blocked for reason in item["reasons"]
        )),
        warnings=list(dict.fromkeys(
            f"{item['estimator']} × {item['featureset']} × {item['protocol']}: {warning}"
            for item in combinations if item["status"] != "blocked" for warning in item["warnings"]
        )),
        table=table,
        meta=meta,
        qc_report=qc_report,
    )


# ---------------------------------------------------------------------------
# score / verdicts / artifacts
# ---------------------------------------------------------------------------

def _resolve_task(task):
    if task is None:
        return SOHEstimationTask()
    if isinstance(task, str):
        return R.get_task(task)()
    return task


def score(
    task,
    table: pd.DataFrame,
    meta: pd.DataFrame,
    protocols: Iterable[str] | Mapping[str, object],
    estimators: Iterable,
    featuresets: Iterable[str] | None = None,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    *,
    on_model_fitted: Callable | None = None,
    feature_pipelines: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Run the ``estimators × protocols × featuresets × seeds`` matrix (``BenchmarkRunner``).

    Protocols and estimators may be given by registry name or as objects.  Returns the
    long-format results frame (one row per metric per fold per seed).
    """
    if isinstance(protocols, Mapping):
        protos = dict(protocols)
    else:
        protos = {}
        for item in protocols:
            if isinstance(item, str):
                protos[item] = PROTOCOLS[item]()
            else:
                protos[getattr(item, "name", type(item).__name__)] = item
    ests = [R.get_estimator(item) if isinstance(item, str) else item for item in estimators]
    return BenchmarkRunner(seeds=list(seeds)).run(
        _resolve_task(task),
        table=table,
        meta=meta,
        protocols=protos,
        estimators=ests,
        featuresets=list(featuresets) if featuresets is not None else None,
        on_model_fitted=on_model_fitted,
        feature_pipelines=feature_pipelines,
    )


def verdicts(metrics: Mapping[str, float], task=None) -> list[Verdict]:
    """Acceptance verdicts (generalization, calibration, physical consistency)."""
    return evaluate_verdicts(dict(metrics), _resolve_task(task))


def write_artifacts(results: pd.DataFrame, qc_report, task, config: Mapping, out_dir) -> str:
    """Write ``results.parquet/csv``, ``leaderboard.json``, ``dataset_leaderboard.json``,
    ``qc_report.json`` and ``run_config.json`` into a fresh run directory; returns its path."""
    return _write_artifacts(results, qc_report=qc_report, task=_resolve_task(task),
                            config=dict(config), out_dir=out_dir)
