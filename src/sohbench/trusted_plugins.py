from __future__ import annotations

import json
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
from .contracts import feature_columns, validate_finite_vector
from .data import (
    DatasetLoader,
    INFERENCE_REQUIRED_COLUMNS,
    REQUIRED_COLUMNS,
    normalize_cycle_index,
)
from .featuresets import FeatureSet
from .fabric import LegacyStatelessPipeline, assert_prefix_invariant


SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


class TrustedPluginError(ValueError):
    """Raised for invalid trusted local uploads."""


@dataclass
class FeaturePlugin:
    name: str
    candidates: list[str]
    transform: Callable | None
    path: Path


class TrustedPluginStore:
    """Validated local resource store for trusted sohbench plugins.

    This class deliberately does not sandbox uploaded Python. It validates the
    public contract and smoke-tests behavior before adding classes/featuresets
    to the in-process registry.
    """

    def __init__(self, base_dir: str | Path, data_dir: str | Path):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.estimator_dir = self.base_dir / "estimators"
        self.featureset_dir = self.base_dir / "featuresets"
        self.catalog_path = self.base_dir / "catalog.json"
        self._feature_plugins: dict[str, FeaturePlugin] = {}

        self.estimator_dir.mkdir(parents=True, exist_ok=True)
        self.featureset_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        self._catalog = self._read_catalog()
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
        _validate_dataset_table(table, require_labels=False)
        try:
            table = normalize_cycle_index(table, context=name)
        except ValueError as exc:
            raise TrustedPluginError(str(exc)) from exc
        table = table.copy()
        table["dataset"] = name

        out_path = self.data_dir / "processed" / f"{name}_features.pkl"
        if out_path.exists() and not replace:
            raise TrustedPluginError(f"dataset already exists: {name}")
        table.to_pickle(out_path)

        resource = {
            "name": name,
            "kind": "dataset",
            "path": str(out_path),
            "rows": int(len(table)),
            "columns": list(map(str, table.columns)),
            "has_labels": "soh" in table.columns,
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
        cls = self._load_estimator_file(path, replace=replace)
        resource = {
            "name": cls.name,
            "kind": "estimator",
            "path": str(path),
            "feature_mode": getattr(cls, "feature_mode", "unified"),
            "backend": getattr(cls, "backend", "external"),
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
        plugin = self._load_featureset_file(path, fallback_name=name, replace=replace)
        resource = {
            "name": plugin.name,
            "kind": "featureset",
            "path": str(path),
            "columns": plugin.candidates,
            "has_transform": plugin.transform is not None,
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
        out = table.copy(deep=True)
        for name in featuresets or []:
            pipeline = self.feature_pipeline(str(name))
            if pipeline is None:
                continue
            target = out["soh"].copy(deep=True) if "soh" in out.columns else None
            state = pipeline.fit(out.drop(columns=["soh"], errors="ignore"), np.empty(len(out)), meta)
            transformed = pipeline.transform(out.drop(columns=["soh"], errors="ignore"), state, meta)
            if target is not None:
                transformed = transformed.copy(deep=True)
                transformed["soh"] = target
            _validate_dataset_table(transformed, require_labels=target is not None)
            out = transformed
        return out

    def feature_pipeline(self, name: str):
        plugin = self._feature_plugins.get(str(name))
        if plugin is None or plugin.transform is None:
            return None
        return LegacyStatelessPipeline(plugin.name, plugin.candidates, plugin.transform, plugin.path)

    def catalog(self) -> dict:
        datasets = []
        for name in DatasetLoader(self.data_dir).discover_datasets(require_labels=False):
            path = self.data_dir / "processed" / f"{name}_features.pkl"
            rows = 0
            columns: list[str] = []
            try:
                frame = pd.read_pickle(path)
                rows = int(len(frame))
                columns = list(map(str, frame.columns))
            except Exception:
                pass
            catalog_item = self._catalog_item("datasets", name)
            datasets.append({
                "name": name,
                "kind": "dataset",
                "rows": rows,
                "columns": columns,
                "has_labels": "soh" in columns,
                "uploaded": catalog_item is not None,
            })

        estimators = [
            {
                "name": name,
                "kind": "estimator",
                "feature_mode": getattr(cls, "feature_mode", "unified"),
                "backend": getattr(cls, "backend", "external"),
                "uploaded": self._catalog_item("estimators", name) is not None,
            }
            for name, cls in sorted(R._ESTIMATORS.items())
        ]
        featuresets = [
            {
                "name": name,
                "kind": "featureset",
                "columns": list(getattr(fs, "candidates", [])),
                "has_transform": name in self._feature_plugins and self._feature_plugins[name].transform is not None,
                "uploaded": self._catalog_item("featuresets", name) is not None,
            }
            for name, fs in sorted(R._FEATURESETS.items())
        ]
        return {"datasets": datasets, "estimators": estimators, "featuresets": featuresets}

    def load_trusted_plugins(self) -> None:
        for item in list(self._catalog.get("estimators", [])):
            path = Path(item.get("path", ""))
            if path.exists() and item.get("name") not in R._ESTIMATORS:
                self._load_estimator_file(path, replace=True)
        for item in list(self._catalog.get("featuresets", [])):
            path = Path(item.get("path", ""))
            name = str(item.get("name") or path.stem)
            if path.exists() and name not in R._FEATURESETS:
                self._load_featureset_file(path, fallback_name=name, replace=True)

    def _write_plugin_file(self, directory: Path, filename: str, payload: bytes, *, replace: bool) -> Path:
        path = directory / Path(filename).name
        if path.suffix.lower() != ".py":
            raise TrustedPluginError("trusted plugin uploads must be .py files")
        if path.exists() and not replace:
            raise TrustedPluginError(f"plugin file already exists: {path.name}")
        path.write_bytes(payload)
        return path

    def _load_estimator_file(self, path: Path, *, replace: bool) -> type:
        module = _load_module(path, "estimator")
        cls = _extract_estimator_class(module)
        if not hasattr(cls, "name") or not str(cls.name):
            raise TrustedPluginError("estimator plugin must define a non-empty name")
        if not hasattr(cls, "feature_mode"):
            cls.feature_mode = "unified"
        if not hasattr(cls, "backend"):
            cls.backend = "external"
        if cls.name in R._ESTIMATORS and not replace:
            raise TrustedPluginError(f"estimator already registered: {cls.name}")
        _smoke_estimator(cls)
        R._ESTIMATORS[cls.name] = cls
        return cls

    def _load_featureset_file(self, path: Path, *, fallback_name: str, replace: bool) -> FeaturePlugin:
        module = _load_module(path, "featureset")
        plugin = _extract_feature_plugin(module, path, fallback_name)
        if plugin.name in R._FEATURESETS and not replace:
            raise TrustedPluginError(f"featureset already registered: {plugin.name}")
        _smoke_featureset_plugin(plugin)
        pipeline = LegacyStatelessPipeline(plugin.name, plugin.candidates, plugin.transform, plugin.path) if plugin.transform else None
        if pipeline is not None:
            try:
                assert_prefix_invariant(pipeline, _smoke_table(), pd.DataFrame())
            except ValueError as exc:
                raise TrustedPluginError(str(exc)) from exc
        R._FEATURESETS[plugin.name] = FeatureSet(plugin.name, plugin.candidates)
        self._feature_plugins[plugin.name] = plugin
        return plugin

    def _read_catalog(self) -> dict:
        if not self.catalog_path.exists():
            return {"datasets": [], "estimators": [], "featuresets": []}
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def _write_catalog(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path.write_text(
            json.dumps(self._catalog, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _upsert_catalog(self, section: str, resource: dict) -> None:
        items = [item for item in self._catalog.get(section, []) if item.get("name") != resource["name"]]
        items.append(resource)
        self._catalog[section] = sorted(items, key=lambda item: item["name"])
        self._write_catalog()

    def _catalog_item(self, section: str, name: str) -> dict | None:
        for item in self._catalog.get(section, []):
            if item.get("name") == name:
                return item
        return None


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


def _validate_dataset_table(table: pd.DataFrame, *, require_labels: bool = True) -> None:
    required = REQUIRED_COLUMNS if require_labels else INFERENCE_REQUIRED_COLUMNS
    missing = required - set(table.columns)
    if missing:
        raise TrustedPluginError(f"missing required columns: {sorted(missing)}")
    if table.empty:
        raise TrustedPluginError("dataset table is empty")


def _load_module(path: Path, kind: str) -> ModuleType:
    module_name = f"sohbench_trusted_{kind}_{path.stem}_{_stable_module_token(path)}"
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise TrustedPluginError(f"cannot load plugin file: {path}") from exc
    module = ModuleType(module_name)
    module.__file__ = str(path)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        code = compile(source_bytes, str(path), "exec")
        exec(code, module.__dict__)
    except Exception as exc:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
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
    return FeaturePlugin(name=name, candidates=candidates, transform=transform, path=path)


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
    try:
        validate_finite_vector(
            est.predict(test, pd.DataFrame()),
            name="predictions",
            n_samples=len(test),
        )
        predict_interval = getattr(est, "predict_interval", None)
        if callable(predict_interval):
            low, high = predict_interval(test, pd.DataFrame())
            validate_finite_vector(
                low, name="interval_low", n_samples=len(test)
            )
            validate_finite_vector(
                high, name="interval_high", n_samples=len(test)
            )
    except (TypeError, ValueError) as exc:
        raise TrustedPluginError(str(exc)) from exc


def _smoke_featureset_plugin(plugin: FeaturePlugin) -> None:
    table = _smoke_table()
    if plugin.transform:
        pipeline = LegacyStatelessPipeline(plugin.name, plugin.candidates, plugin.transform, plugin.path)
        try:
            state = pipeline.fit(table.drop(columns=["soh"]), table["soh"].to_numpy(), pd.DataFrame())
            transformed = pipeline.transform(table.drop(columns=["soh"]), state, pd.DataFrame())
        except ValueError as exc:
            raise TrustedPluginError(str(exc)) from exc
    else:
        transformed = table
    if not isinstance(transformed, pd.DataFrame):
        raise TrustedPluginError("feature transform did not return a DataFrame")
    if len(transformed) != len(table):
        raise TrustedPluginError("feature transform changed row count")
    if plugin.transform and not plugin.candidates:
        plugin.candidates = [col for col in feature_columns(transformed) if col not in feature_columns(table.drop(columns=["soh"]))]
    missing = [col for col in plugin.candidates if col not in feature_columns(transformed)]
    if missing:
        raise TrustedPluginError(f"featureset columns are missing after transform: {missing}")
