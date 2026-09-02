from __future__ import annotations

import importlib.util
import json
import threading
import uuid
import re
import sys
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np
import pandas as pd

from . import registry as R
from .contracts import feature_columns
from .data import DatasetLoader, REQUIRED_COLUMNS
from .featuresets import FeatureSet
from .feature_safety import (
    FeatureSafetyError,
    apply_transform_without_label,
    audit_feature_frame,
    passed_label_safety,
)
from .metadata import DATASET_CHEMISTRY, DATASET_C_RATED_AH
from .method_contracts import method_contract


SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


class TrustedPluginError(ValueError):
    """Raised for invalid trusted local uploads."""


@dataclass
class FeaturePlugin:
    name: str
    candidates: list[str]
    transform: Callable | None
    path: Path
    label_safety: dict


class TrustedPluginStore:
    """Validated local resource store for trusted sohbench plugins.

    This class deliberately does not sandbox uploaded Python. It validates the
    public contract and smoke-tests behavior before adding classes/featuresets
    to the in-process registry.
    """

    def __init__(
        self,
        base_dir: str | Path,
        data_dir: str | Path,
        *,
        load_plugins: bool = True,
    ):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.estimator_dir = self.base_dir / "estimators"
        self.featureset_dir = self.base_dir / "featuresets"
        self.catalog_path = self.base_dir / "catalog.json"
        self._feature_plugins: dict[str, FeaturePlugin] = {}

        self.estimator_dir.mkdir(parents=True, exist_ok=True)
        self.featureset_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        self._catalog_lock = threading.RLock()
        self._catalog = self._read_catalog()
        if load_plugins:
            self.load_trusted_plugins()

    def upload_dataset(
        self,
        name: str,
        filename: str,
        payload: bytes,
        *,
        replace: bool = False,
    ) -> dict:
        name = _safe_resource_name(name)
        table = _read_table(filename, payload)
        _validate_dataset_table(table)
        table = table.copy()
        table["dataset"] = name

        out_path = self.data_dir / "processed" / f"{name}_features.pkl"
        if out_path.exists() and not replace:
            raise TrustedPluginError(f"dataset already exists: {name}")
        table.to_pickle(out_path)

        # Ingestion-time label-source scan: columns that reconstruct the
        # label are recorded on the dataset itself, once, so every downstream
        # consumer (catalog, preflight, training gate) can see them.  The
        # dataset is not rejected — carrying its label source is legitimate;
        # using it as a feature is what the training/inference gates block.
        feature_columns = [
            col for col in table.columns
            if col not in ("dataset", "cell_id", "cycle_index", "soh")
        ]
        ingest_audit = audit_feature_frame(table, feature_columns)
        label_source_columns = sorted(
            {item["column"] for item in ingest_audit["violations"]}
        )

        resource = {
            "name": name,
            "kind": "dataset",
            "path": str(out_path),
            "rows": int(len(table)),
            "columns": list(map(str, table.columns)),
            "label_source_columns": label_source_columns,
            "source_filename": filename,
        }
        self._upsert_catalog("datasets", resource)
        return resource

    def upload_estimator_plugin(
        self,
        name: str,
        filename: str,
        payload: bytes,
        *,
        replace: bool = False,
    ) -> dict:
        _safe_resource_name(name)
        path = self._write_plugin_file(self.estimator_dir, filename, payload, replace=replace)
        try:
            cls = self._load_estimator_file(
                path, replace=replace, reserved=self._reserved_names("estimators", R._ESTIMATORS)
            )
        except Exception:
            # A rejected upload must not leave executable Python behind.
            path.unlink(missing_ok=True)
            raise
        resource = {
            "name": cls.name,
            "kind": "estimator",
            "path": str(path),
            **_estimator_method_metadata(cls),
            "source_filename": filename,
        }
        self._upsert_catalog("estimators", resource)
        return resource

    def upload_featureset_plugin(
        self,
        name: str,
        filename: str,
        payload: bytes,
        *,
        replace: bool = False,
    ) -> dict:
        _safe_resource_name(name)
        path = self._write_plugin_file(self.featureset_dir, filename, payload, replace=replace)
        try:
            plugin = self._load_featureset_file(
                path,
                fallback_name=name,
                replace=replace,
                enforce_label_safety=True,
                reserved=self._reserved_names("featuresets", R._FEATURESETS),
            )
        except Exception:
            # A rejected upload must not leave executable Python behind.
            path.unlink(missing_ok=True)
            raise
        resource = {
            "name": plugin.name,
            "kind": "featureset",
            "path": str(path),
            "columns": plugin.candidates,
            "has_transform": plugin.transform is not None,
            "label_safety": plugin.label_safety,
            "source_filename": filename,
        }
        self._upsert_catalog("featuresets", resource)
        return resource

    def apply_feature_transforms(
        self,
        table: pd.DataFrame,
        meta: pd.DataFrame | None,
        featuresets: list[str] | tuple[str, ...],
    ) -> pd.DataFrame:
        out = table
        for name in featuresets or []:
            plugin = self._feature_plugins.get(str(name))
            registered = R._FEATURESETS.get(str(name))
            transform = (
                plugin.transform
                if plugin is not None and plugin.transform is not None
                else getattr(registered, "transform", None)
            )
            if transform is None:
                continue
            candidates = list(
                plugin.candidates
                if plugin is not None
                else getattr(registered, "candidates", [])
            )
            before_rows = len(out)
            try:
                transformed, audit = apply_transform_without_label(
                    transform,
                    out,
                    meta,
                    name=str(name),
                    candidates=candidates,
                )
            except FeatureSafetyError as exc:
                raise TrustedPluginError(str(exc)) from exc
            if len(transformed) != before_rows:
                raise TrustedPluginError(f"feature transform {name!r} changed row count")
            _validate_dataset_table(transformed)
            missing = [col for col in candidates if col not in transformed.columns]
            for col in missing:
                # Candidate columns the transform could not derive on this
                # dataset (absent source signals) stay materialized as NaN so
                # per-model compatibility can block exactly the affected
                # models, instead of one passthrough column failing the whole
                # job before any model runs.
                transformed[col] = float("nan")
            if plugin is not None:
                plugin.label_safety = audit
            out = transformed
        return out

    def catalog(self) -> dict:
        manifest = self._data_manifest()
        manifest_by_name = {
            str(item.get("name") or ""): item
            for item in manifest.get("datasets", [])
            if item.get("name")
        }
        datasets = []
        for name in DatasetLoader(self.data_dir).discover_datasets():
            path = self.data_dir / "processed" / f"{name}_features.pkl"
            rows = 0
            columns: list[str] = []
            cycle_count = 0
            battery_count = 0
            chemistry = str(DATASET_CHEMISTRY.get(name) or "unknown")
            max_capacity_ah = None
            capacity_source = "unknown"
            try:
                frame = pd.read_pickle(path)
                rows = int(len(frame))
                columns = list(map(str, frame.columns))
                if {"cell_id", "cycle_index"}.issubset(frame.columns):
                    cycle_count = int(
                        frame[["cell_id", "cycle_index"]]
                        .dropna()
                        .drop_duplicates()
                        .shape[0]
                    )
                elif "cycle_index" in frame.columns:
                    cycle_count = int(frame["cycle_index"].dropna().nunique())
                if "cell_id" in frame.columns:
                    battery_count = int(frame["cell_id"].dropna().astype(str).nunique())
                if "chemistry" in frame.columns:
                    values = frame["chemistry"].dropna().astype(str).str.strip()
                    values = values[(values != "") & (values.str.lower() != "unknown")]
                    if not values.empty:
                        chemistry = str(values.mode().iloc[0])
                if "capacity_ah" in frame.columns:
                    values = pd.to_numeric(frame["capacity_ah"], errors="coerce")
                    values = values[np.isfinite(values) & (values > 0)]
                    if not values.empty:
                        max_capacity_ah = float(values.max())
                        capacity_source = "measured_capacity"
                if max_capacity_ah is None and "c_rated_ah" in frame.columns:
                    values = pd.to_numeric(frame["c_rated_ah"], errors="coerce")
                    values = values[np.isfinite(values) & (values > 0)]
                    if not values.empty:
                        max_capacity_ah = float(values.max())
                        capacity_source = "rated_column"
            except Exception:
                pass
            if max_capacity_ah is None:
                rated = DATASET_C_RATED_AH.get(name)
                if rated is not None and np.isfinite(rated) and float(rated) > 0:
                    max_capacity_ah = float(rated)
                    capacity_source = "rated_metadata"
            catalog_item = self._catalog_item("datasets", name)
            manifest_item = manifest_by_name.get(name, {})
            datasets.append({
                "name": name,
                "kind": "dataset",
                "rows": rows,
                "record_count": rows,
                "cycle_count": cycle_count,
                "battery_count": battery_count,
                "chemistry": chemistry,
                "max_capacity_ah": max_capacity_ah,
                "capacity_source": capacity_source,
                "held_out_ready": battery_count > 1,
                "status": str(manifest_item.get("status") or "ready"),
                "roles": list(manifest_item.get("roles") or []),
                "raw_local": bool(manifest_item.get("raw_local", False)),
                "curve_available": bool(manifest_item.get("curve_path")),
                "source": str(manifest_item.get("source") or ""),
                "license": str(manifest_item.get("license") or "unknown"),
                "processed_sha256": str(
                    (manifest_item.get("processed") or {}).get("sha256") or ""
                ),
                "columns": columns,
                "uploaded": catalog_item is not None,
            })

        estimators = [
            {
                "name": name,
                "kind": "estimator",
                **_estimator_method_metadata(cls),
                "uploaded": self._catalog_item("estimators", name) is not None,
            }
            for name, cls in sorted(R._ESTIMATORS.items())
        ]
        featuresets = [
            {
                "name": name,
                "kind": "featureset",
                "columns": list(getattr(fs, "candidates", [])),
                "has_transform": bool(
                    (name in self._feature_plugins and self._feature_plugins[name].transform is not None)
                    or getattr(fs, "transform", None) is not None
                ),
                "family": str(getattr(fs, "family", "data_driven")),
                "mechanism_level": str(getattr(fs, "mechanism_level", "none")),
                "data_granularity": str(getattr(fs, "data_granularity", "cycle_table")),
                "description": str(getattr(fs, "description", "")),
                "required_signals": list(getattr(fs, "required_signals", []) or []),
                "label_safety": dict(
                    self._feature_plugins[name].label_safety
                    if name in self._feature_plugins
                    else passed_label_safety(checked_columns=getattr(fs, "candidates", []))
                ),
                "uploaded": self._catalog_item("featuresets", name) is not None,
            }
            for name, fs in sorted(R._FEATURESETS.items())
        ]
        return {
            "datasets": datasets,
            "estimators": estimators,
            "featuresets": featuresets,
            "data_manifest": {
                "schema_version": manifest.get("schema_version", ""),
                "generated_at": manifest.get("generated_at", ""),
                "migration_strategy": manifest.get("migration_strategy", ""),
                "summary": manifest.get("summary", {}),
                "verified": bool(
                    (manifest.get("verification") or {}).get(
                        "all_inventory_checks_passed", False
                    )
                ),
            },
        }

    def _data_manifest(self) -> dict:
        path = self.data_dir / "manifest.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def load_trusted_plugins(self) -> None:
        for item in list(self._catalog.get("estimators", [])):
            path = Path(item.get("path", ""))
            if path.exists() and item.get("name") not in R._ESTIMATORS:
                self._load_estimator_file(path, replace=True)
        for item in list(self._catalog.get("featuresets", [])):
            path = Path(item.get("path", ""))
            name = str(item.get("name") or path.stem)
            if path.exists() and name not in R._FEATURESETS:
                self._load_featureset_file(
                    path,
                    fallback_name=name,
                    replace=True,
                    enforce_label_safety=False,
                )

    def _write_plugin_file(self, directory: Path, filename: str, payload: bytes, *, replace: bool) -> Path:
        path = directory / Path(filename).name
        if path.suffix.lower() != ".py":
            raise TrustedPluginError("trusted plugin uploads must be .py files")
        if path.exists() and not replace:
            raise TrustedPluginError(f"plugin file already exists: {path.name}")
        path.write_bytes(payload)
        return path

    def _reserved_names(self, section: str, registry: dict) -> frozenset[str]:
        """Registered names that are NOT plugins of this store, i.e. built-ins.

        A plugin may replace an earlier plugin of the same name, but never a
        built-in: the override only lived in the uploading process (start-up
        skips catalog entries whose name is already registered), so jobs
        silently switched back to the built-in after the next restart."""
        plugins = {str(item.get("name") or "") for item in self._catalog.get(section, [])}
        return frozenset(name for name in registry if name not in plugins)

    def _load_estimator_file(
        self, path: Path, *, replace: bool, reserved: frozenset[str] = frozenset()
    ) -> type:
        module = _load_module(path, "estimator")
        cls = _extract_estimator_class(module)
        if not hasattr(cls, "name") or not str(cls.name):
            raise TrustedPluginError("estimator plugin must define a non-empty name")
        if cls.name in reserved:
            raise TrustedPluginError(
                f"estimator name {cls.name!r} is reserved by a built-in method; choose another name"
            )
        if not hasattr(cls, "feature_mode"):
            cls.feature_mode = "unified"
        if not hasattr(cls, "backend"):
            cls.backend = "external"
        if cls.name in R._ESTIMATORS and not replace:
            raise TrustedPluginError(f"estimator already registered: {cls.name}")
        _smoke_estimator(cls)
        R._ESTIMATORS[cls.name] = cls
        return cls

    def _load_featureset_file(
        self,
        path: Path,
        *,
        fallback_name: str,
        replace: bool,
        enforce_label_safety: bool = False,
        reserved: frozenset[str] = frozenset(),
    ) -> FeaturePlugin:
        module = _load_module(path, "featureset")
        plugin = _extract_feature_plugin(module, path, fallback_name)
        if plugin.name in reserved:
            raise TrustedPluginError(
                f"featureset name {plugin.name!r} is reserved by a built-in featureset; choose another name"
            )
        if plugin.name in R._FEATURESETS and not replace:
            raise TrustedPluginError(f"featureset already registered: {plugin.name}")
        _smoke_featureset_plugin(plugin, enforce_label_safety=enforce_label_safety)
        R._FEATURESETS[plugin.name] = FeatureSet(plugin.name, plugin.candidates)
        self._feature_plugins[plugin.name] = plugin
        return plugin

    def _read_catalog(self) -> dict:
        if not self.catalog_path.exists():
            return {"datasets": [], "estimators": [], "featuresets": []}
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def _write_catalog(self) -> None:
        with self._catalog_lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.catalog_path.with_name(
                f".{self.catalog_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                temporary.write_text(
                    json.dumps(self._catalog, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                temporary.replace(self.catalog_path)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def _upsert_catalog(self, section: str, resource: dict) -> None:
        with self._catalog_lock:
            items = [item for item in self._catalog.get(section, []) if item.get("name") != resource["name"]]
            items.append(resource)
            self._catalog[section] = sorted(items, key=lambda item: item["name"])
            self._write_catalog()

    def _catalog_item(self, section: str, name: str) -> dict | None:
        for item in self._catalog.get(section, []):
            if item.get("name") == name:
                return item
        return None


def _estimator_method_metadata(cls: type) -> dict:
    """Expose method semantics without changing the estimator fit contract."""
    return method_contract(cls).catalog_metadata()


def _safe_resource_name(name: str) -> str:
    value = str(name or "").strip()
    if not SAFE_NAME.match(value):
        raise TrustedPluginError(f"invalid resource name: {name!r}")
    return value


def _read_table(filename: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    bio = BytesIO(payload)
    if suffix == ".csv":
        return pd.read_csv(bio)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(bio)
    if suffix == ".parquet":
        return pd.read_parquet(bio)
    raise TrustedPluginError("dataset upload must be .csv, .pkl, or .parquet")


def _validate_dataset_table(table: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise TrustedPluginError(f"missing required columns: {sorted(missing)}")
    if table.empty:
        raise TrustedPluginError("dataset table is empty")


def _load_module(path: Path, kind: str) -> ModuleType:
    module_name = f"sohbench_trusted_{kind}_{path.stem}_{_stable_module_token(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise TrustedPluginError(f"cannot load plugin file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise TrustedPluginError(f"plugin import failed: {exc}") from exc
    return module


def _stable_module_token(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/").lower()
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def _extract_estimator_class(module: ModuleType) -> type:
    plugin = getattr(module, "PLUGIN", None)
    if isinstance(plugin, dict) and "estimator" in plugin:
        candidate = plugin["estimator"]
    elif hasattr(module, "get_estimator"):
        candidate = module.get_estimator()
    elif hasattr(module, "Estimator"):
        candidate = module.Estimator
    else:
        raise TrustedPluginError("estimator plugin must expose Estimator, get_estimator(), or PLUGIN['estimator']")

    if not isinstance(candidate, type):
        raise TrustedPluginError("estimator plugin must provide an estimator class, not an instance")
    for method in ("fit", "predict"):
        if not callable(getattr(candidate, method, None)):
            raise TrustedPluginError(f"estimator class must define {method}()")
    return candidate


def _extract_feature_plugin(module: ModuleType, path: Path, fallback_name: str) -> FeaturePlugin:
    plugin = getattr(module, "PLUGIN", None)
    provided = getattr(module, "FEATURESET", None)

    if isinstance(plugin, dict):
        name = plugin.get("name") or plugin.get("featureset_name") or fallback_name
        candidates = plugin.get("columns") or plugin.get("candidates") or []
    elif provided is not None and hasattr(provided, "name"):
        name = provided.name
        candidates = list(getattr(provided, "candidates", []))
    else:
        name = getattr(module, "FEATURESET_NAME", fallback_name)
        candidates = getattr(module, "FEATURE_COLUMNS", getattr(module, "CANDIDATES", []))

    name = _safe_resource_name(str(name))
    candidates = [str(col) for col in candidates]
    transform = getattr(module, "transform", None)
    if transform is not None and not callable(transform):
        raise TrustedPluginError("feature transform must be callable")
    if not candidates and transform is None:
        raise TrustedPluginError("featureset plugin must define FEATURE_COLUMNS or transform()")
    return FeaturePlugin(
        name=name,
        candidates=candidates,
        transform=transform,
        path=path,
        label_safety=passed_label_safety(checked_columns=candidates),
    )


def _smoke_table() -> pd.DataFrame:
    cycles = np.arange(12)
    return pd.DataFrame(
        {
            "dataset": ["smoke"] * len(cycles),
            "cell_id": ["c1"] * len(cycles),
            "cycle_index": cycles,
            "soh": 1.0 - 0.01 * cycles,
            "voltage_mean": 3.5 - 0.02 * cycles,
            "resistance_proxy": 0.02 + 0.001 * cycles,
        }
    )


def _smoke_estimator(cls: type) -> None:
    table = _smoke_table()
    est = cls()
    train = table.iloc[:8]
    test = table.iloc[8:]
    fitted = est.fit(train, train["soh"].to_numpy(), pd.DataFrame())
    if fitted is not None:
        est = fitted
    pred = np.asarray(est.predict(test, pd.DataFrame()), dtype=float)
    if pred.shape[0] != len(test):
        raise TrustedPluginError("estimator predict() must return one value per row")
    if not np.isfinite(pred).all():
        raise TrustedPluginError("estimator predict() returned non-finite values")


def _smoke_featureset_plugin(
    plugin: FeaturePlugin,
    *,
    enforce_label_safety: bool,
) -> None:
    table = _smoke_table()
    if plugin.transform:
        try:
            transformed, audit = apply_transform_without_label(
                plugin.transform,
                table,
                pd.DataFrame(),
                name=plugin.name,
                candidates=plugin.candidates,
            )
            plugin.label_safety = audit
        except FeatureSafetyError as exc:
            plugin.label_safety = {
                "contract": "sohbench.label-safety.v1",
                "target": "soh",
                "status": "blocked",
                "checked_columns": list(plugin.candidates),
                "violations": [{"kind": "target_access", "reason": str(exc)}],
            }
            if enforce_label_safety:
                raise TrustedPluginError(str(exc)) from exc
            # Keep legacy unsafe plugins discoverable for migration/audit, but
            # never execute them through apply_feature_transforms().
            transformed = plugin.transform(table.copy(), pd.DataFrame())
    else:
        transformed = table
        plugin.label_safety = passed_label_safety(checked_columns=plugin.candidates)
    if not isinstance(transformed, pd.DataFrame):
        raise TrustedPluginError("feature transform did not return a DataFrame")
    if len(transformed) != len(table):
        raise TrustedPluginError("feature transform changed row count")
    if plugin.transform and not plugin.candidates:
        plugin.candidates = [col for col in feature_columns(transformed) if col not in feature_columns(table)]
    missing = [col for col in plugin.candidates if col not in feature_columns(transformed)]
    if missing:
        raise TrustedPluginError(f"featureset columns are missing after transform: {missing}")
