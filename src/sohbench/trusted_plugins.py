from __future__ import annotations

import json
import re
import sys
import hashlib
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from . import registry as R
from .contracts import (
    feature_columns,
    validate_finite_vector,
    validate_interval_bounds,
)
from .data import (
    DatasetLoader,
    INFERENCE_REQUIRED_COLUMNS,
    REQUIRED_COLUMNS,
    normalize_cycle_index,
)
from .featuresets import FeatureSet
from .fabric import LegacyStatelessPipeline, assert_prefix_invariant


SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
_MISSING = object()


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
        self.dependency_dir = self.base_dir / "dependencies"
        self.catalog_path = self.base_dir / "catalog.json"
        self._lock = threading.RLock()
        self._feature_plugins: dict[str, FeaturePlugin] = {}

        self.estimator_dir.mkdir(parents=True, exist_ok=True)
        self.featureset_dir.mkdir(parents=True, exist_ok=True)
        self.dependency_dir.mkdir(parents=True, exist_ok=True)
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
        resource = {
            "name": name,
            "kind": "dataset",
            "path": str(out_path),
            "rows": int(len(table)),
            "columns": list(map(str, table.columns)),
            "has_labels": "soh" in table.columns,
            "source_filename": filename,
        }
        with self._lock:
            if out_path.exists() and not replace:
                raise TrustedPluginError(f"dataset already exists: {name}")
            staged = self._stage_dataset_pickle(out_path, table)
            try:
                next_catalog = self._catalog_with_upsert(
                    "datasets", resource
                )
                catalog_staging = self._stage_catalog(next_catalog)
                self._activate_plugin_transaction(
                    out_path,
                    staged,
                    catalog_staging,
                    next_catalog,
                    lambda: None,
                )
                return resource
            finally:
                staged.unlink(missing_ok=True)

    def upload_estimator_plugin(
        self,
        name: str,
        filename: str,
        payload: bytes,
        *,
        replace: bool = False,
        dependencies: dict[str, str | Path] | None = None,
    ) -> dict:
        with self._lock:
            _safe_resource_name(name)
            dependency_records = _declared_dependency_records(dependencies)
            path = self._plugin_destination(
                self.estimator_dir, filename, replace=replace
            )
            staged = self._stage_plugin_payload(path, payload)
            module_name = _trusted_module_name(path, "estimator")
            previous_module = sys.modules.get(module_name, _MISSING)
            committed = False
            try:
                cls = self._validate_estimator_file(
                    staged, logical_path=path
                )
                self._validate_replacement_identity(
                    "estimators", path, str(cls.name), replace
                )
                if cls.name in R._ESTIMATORS and not replace:
                    raise TrustedPluginError(
                        f"estimator already registered: {cls.name}"
                    )
                resource = {
                    "name": cls.name,
                    "kind": "estimator",
                    "path": str(path),
                    "feature_mode": getattr(cls, "feature_mode", "unified"),
                    "backend": getattr(cls, "backend", "external"),
                    "source_filename": filename,
                    "dependencies": dependency_records,
                }
                next_catalog = self._catalog_with_upsert(
                    "estimators", resource
                )
                catalog_staging = self._stage_catalog(next_catalog)
                self._activate_plugin_transaction(
                    path,
                    staged,
                    catalog_staging,
                    next_catalog,
                    lambda: R._ESTIMATORS.__setitem__(cls.name, cls),
                )
                committed = True
                return resource
            finally:
                staged.unlink(missing_ok=True)
                if not committed:
                    _restore_module(module_name, previous_module)

    def upload_featureset_plugin(
        self,
        name: str,
        filename: str,
        payload: bytes,
        *,
        replace: bool = False,
        dependencies: dict[str, str | Path] | None = None,
    ) -> dict:
        with self._lock:
            _safe_resource_name(name)
            dependency_records = _declared_dependency_records(dependencies)
            path = self._plugin_destination(
                self.featureset_dir, filename, replace=replace
            )
            staged = self._stage_plugin_payload(path, payload)
            module_name = _trusted_module_name(path, "featureset")
            previous_module = sys.modules.get(module_name, _MISSING)
            committed = False
            try:
                plugin = self._validate_featureset_file(
                    staged, fallback_name=name, logical_path=path
                )
                self._validate_replacement_identity(
                    "featuresets", path, plugin.name, replace
                )
                if plugin.name in R._FEATURESETS and not replace:
                    raise TrustedPluginError(
                        f"featureset already registered: {plugin.name}"
                    )
                resource = {
                    "name": plugin.name,
                    "kind": "featureset",
                    "path": str(path),
                    "columns": plugin.candidates,
                    "has_transform": plugin.transform is not None,
                    "source_filename": filename,
                    "dependencies": dependency_records,
                }
                next_catalog = self._catalog_with_upsert(
                    "featuresets", resource
                )
                catalog_staging = self._stage_catalog(next_catalog)

                def activate_feature() -> None:
                    R._FEATURESETS[plugin.name] = FeatureSet(
                        plugin.name, plugin.candidates
                    )
                    self._feature_plugins[plugin.name] = plugin

                self._activate_plugin_transaction(
                    path,
                    staged,
                    catalog_staging,
                    next_catalog,
                    activate_feature,
                )
                committed = True
                return resource
            finally:
                staged.unlink(missing_ok=True)
                if not committed:
                    _restore_module(module_name, previous_module)

    def snapshot_dependency_tree(
        self, owner: str, source: str | Path
    ) -> Path:
        owner_name = _safe_resource_name(owner)
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise TrustedPluginError(
                f"declared dependency tree is unavailable: {source_path}"
            )
        digest = _dependency_digest(source_path, "tree")
        owner_root = self.dependency_dir / owner_name
        generation = owner_root / digest
        destination = generation / source_path.name

        with self._lock:
            if destination.is_dir():
                if _dependency_digest(destination, "tree") != digest:
                    raise TrustedPluginError(
                        f"managed dependency snapshot is corrupted: {destination}"
                    )
                return generation.resolve()

            owner_root.mkdir(parents=True, exist_ok=True)
            staging = owner_root / f".{digest}.tmp-{uuid.uuid4().hex}"
            try:
                staging.mkdir()
                shutil.copytree(
                    source_path,
                    staging / source_path.name,
                    ignore=_ignore_generated_dependency_files,
                )
                staged_dependency = staging / source_path.name
                if _dependency_digest(staged_dependency, "tree") != digest:
                    raise TrustedPluginError(
                        "managed dependency snapshot digest changed while copying"
                    )
                _fsync_tree(staging)
                try:
                    os.replace(staging, generation)
                except FileExistsError:
                    if not destination.is_dir():
                        raise
                _fsync_directory(owner_root)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            return generation.resolve()

    def declared_plugin_dependencies(
        self, section: str, name: str
    ) -> list[dict]:
        with self._lock:
            item = self._catalog_item(section, str(name))
            records = list((item or {}).get("dependencies") or [])
            verified = []
            for record in sorted(records, key=lambda value: str(value.get("name"))):
                path = Path(str(record.get("path") or ""))
                kind = str(record.get("kind") or "")
                digest = _dependency_digest(path, kind)
                if digest != record.get("digest"):
                    raise TrustedPluginError(
                        f"declared dependency changed after registration: {path}"
                    )
                verified.append(dict(record))
            return verified

    @contextmanager
    def execution_lease(self) -> Iterator[None]:
        with self._lock:
            yield

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

    def _plugin_destination(
        self, directory: Path, filename: str, *, replace: bool
    ) -> Path:
        path = directory / Path(filename).name
        if path.suffix.lower() != ".py":
            raise TrustedPluginError("trusted plugin uploads must be .py files")
        if path.exists() and not replace:
            raise TrustedPluginError(f"plugin file already exists: {path.name}")
        return path

    def _stage_plugin_payload(self, path: Path, payload: bytes) -> Path:
        staged = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
        try:
            with staged.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if staged.read_bytes() != payload:
                raise TrustedPluginError("staged plugin bytes failed validation")
            return staged
        except BaseException:
            staged.unlink(missing_ok=True)
            raise

    def _stage_dataset_pickle(
        self, path: Path, table: pd.DataFrame
    ) -> Path:
        staged = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
        try:
            table.to_pickle(staged)
            with staged.open("r+b") as stream:
                os.fsync(stream.fileno())
            try:
                persisted = pd.read_pickle(staged)
                pd.testing.assert_frame_equal(persisted, table)
            except Exception as exc:
                raise TrustedPluginError(
                    "staged dataset pickle failed validation"
                ) from exc
            return staged
        except BaseException:
            staged.unlink(missing_ok=True)
            raise

    def _validate_replacement_identity(
        self, section: str, path: Path, candidate_name: str, replace: bool
    ) -> None:
        existing = next(
            (
                item
                for item in self._catalog.get(section, [])
                if Path(str(item.get("path") or "")) == path
            ),
            None,
        )
        if (
            replace
            and existing is not None
            and str(existing.get("name")) != candidate_name
        ):
            raise TrustedPluginError(
                "replacement plugin must keep its registered name"
            )

    def _catalog_with_upsert(self, section: str, resource: dict) -> dict:
        document = json.loads(json.dumps(self._catalog))
        items = [
            item
            for item in document.get(section, [])
            if item.get("name") != resource["name"]
        ]
        items.append(resource)
        document[section] = sorted(items, key=lambda item: item["name"])
        return document

    def _stage_catalog(self, document: dict) -> Path:
        staged = self.base_dir / f".{self.catalog_path.name}.tmp-{uuid.uuid4().hex}"
        try:
            staged.write_text(
                json.dumps(document, indent=2, ensure_ascii=False),
                encoding="utf-8",
                newline="\n",
            )
            with staged.open("r+b") as stream:
                os.fsync(stream.fileno())
            if json.loads(staged.read_text(encoding="utf-8")) != document:
                raise TrustedPluginError("staged plugin catalog failed validation")
            return staged
        except BaseException:
            staged.unlink(missing_ok=True)
            raise

    def _activate_plugin_transaction(
        self,
        path: Path,
        staged: Path,
        catalog_staging: Path,
        next_catalog: dict,
        activate_registry: Callable[[], None],
    ) -> None:
        previous_payload = path.read_bytes() if path.is_file() else None
        previous_catalog_payload = (
            self.catalog_path.read_bytes() if self.catalog_path.is_file() else None
        )
        previous_catalog = json.loads(json.dumps(self._catalog))
        estimator_registry = dict(R._ESTIMATORS)
        featureset_registry = dict(R._FEATURESETS)
        feature_plugins = dict(self._feature_plugins)
        rollback = path.parent / f".{path.name}.rollback-{uuid.uuid4().hex}"
        catalog_rollback = self.base_dir / (
            f".{self.catalog_path.name}.rollback-{uuid.uuid4().hex}"
        )
        committed = False
        restored = False
        try:
            if previous_payload is not None:
                with rollback.open("wb") as stream:
                    stream.write(previous_payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            if previous_catalog_payload is not None:
                with catalog_rollback.open("wb") as stream:
                    stream.write(previous_catalog_payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            os.replace(staged, path)
            try:
                os.replace(catalog_staging, self.catalog_path)
            except BaseException:
                if previous_payload is None:
                    path.unlink(missing_ok=True)
                else:
                    os.replace(rollback, path)
                restored = True
                raise
            try:
                activate_registry()
            except BaseException:
                R._ESTIMATORS.clear()
                R._ESTIMATORS.update(estimator_registry)
                R._FEATURESETS.clear()
                R._FEATURESETS.update(featureset_registry)
                self._feature_plugins.clear()
                self._feature_plugins.update(feature_plugins)
                self._catalog = previous_catalog
                if previous_catalog_payload is None:
                    self.catalog_path.unlink(missing_ok=True)
                else:
                    os.replace(catalog_rollback, self.catalog_path)
                if previous_payload is None:
                    path.unlink(missing_ok=True)
                else:
                    os.replace(rollback, path)
                restored = True
                raise
            self._catalog = next_catalog
            _fsync_directory(path.parent)
            _fsync_directory(self.base_dir)
            committed = True
        finally:
            if committed or restored:
                rollback.unlink(missing_ok=True)
                catalog_rollback.unlink(missing_ok=True)
            catalog_staging.unlink(missing_ok=True)

    def _load_estimator_file(self, path: Path, *, replace: bool) -> type:
        cls = self._validate_estimator_file(path)
        if cls.name in R._ESTIMATORS and not replace:
            raise TrustedPluginError(f"estimator already registered: {cls.name}")
        R._ESTIMATORS[cls.name] = cls
        return cls

    def _validate_estimator_file(
        self, path: Path, *, logical_path: Path | None = None
    ) -> type:
        module = _load_module(path, "estimator", logical_path=logical_path)
        cls = _extract_estimator_class(module)
        if not hasattr(cls, "name") or not str(cls.name):
            raise TrustedPluginError("estimator plugin must define a non-empty name")
        if not hasattr(cls, "feature_mode"):
            cls.feature_mode = "unified"
        if not hasattr(cls, "backend"):
            cls.backend = "external"
        _smoke_estimator(cls)
        return cls

    def _load_featureset_file(self, path: Path, *, fallback_name: str, replace: bool) -> FeaturePlugin:
        plugin = self._validate_featureset_file(path, fallback_name=fallback_name)
        if plugin.name in R._FEATURESETS and not replace:
            raise TrustedPluginError(f"featureset already registered: {plugin.name}")
        R._FEATURESETS[plugin.name] = FeatureSet(plugin.name, plugin.candidates)
        self._feature_plugins[plugin.name] = plugin
        return plugin

    def _validate_featureset_file(
        self,
        path: Path,
        *,
        fallback_name: str,
        logical_path: Path | None = None,
    ) -> FeaturePlugin:
        module = _load_module(path, "featureset", logical_path=logical_path)
        plugin = _extract_feature_plugin(module, path, fallback_name)
        _smoke_featureset_plugin(plugin)
        pipeline = LegacyStatelessPipeline(plugin.name, plugin.candidates, plugin.transform, plugin.path) if plugin.transform else None
        if pipeline is not None:
            try:
                assert_prefix_invariant(pipeline, _smoke_table(), pd.DataFrame())
            except ValueError as exc:
                raise TrustedPluginError(str(exc)) from exc
        return FeaturePlugin(
            name=plugin.name,
            candidates=plugin.candidates,
            transform=plugin.transform,
            path=Path(logical_path) if logical_path is not None else path,
        )

    def _read_catalog(self) -> dict:
        if not self.catalog_path.exists():
            return {"datasets": [], "estimators": [], "featuresets": []}
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def _upsert_catalog(self, section: str, resource: dict) -> None:
        document = self._catalog_with_upsert(section, resource)
        staged = self._stage_catalog(document)
        try:
            os.replace(staged, self.catalog_path)
            _fsync_directory(self.base_dir)
        finally:
            staged.unlink(missing_ok=True)
        self._catalog = document

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


def _declared_dependency_records(
    dependencies: dict[str, str | Path] | None,
) -> list[dict]:
    if dependencies is None:
        return []
    if not isinstance(dependencies, dict):
        raise TrustedPluginError("plugin dependencies must be a name-to-path mapping")
    records = []
    for name, raw_path in sorted(dependencies.items(), key=lambda item: str(item[0])):
        dependency_name = _safe_resource_name(str(name))
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            kind = "file"
        elif path.is_dir():
            kind = "tree"
        else:
            raise TrustedPluginError(
                f"declared plugin dependency is unavailable: {path}"
            )
        records.append(
            {
                "name": dependency_name,
                "path": str(path),
                "kind": kind,
                "digest": _dependency_digest(path, kind),
            }
        )
    return records


def _dependency_digest(path: Path, kind: str) -> str:
    path = Path(path)
    if kind == "file":
        if not path.is_file() or path.is_symlink():
            raise TrustedPluginError(f"declared dependency file is unavailable: {path}")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if kind != "tree":
        raise TrustedPluginError(f"unsupported dependency kind: {kind!r}")
    if not path.is_dir() or path.is_symlink():
        raise TrustedPluginError(f"declared dependency tree is unavailable: {path}")
    hasher = hashlib.sha256()
    hasher.update(b"sohbench-dependency-tree-v1\0")
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        relative_path = item.relative_to(path)
        if "__pycache__" in relative_path.parts or item.suffix.lower() == ".pyc":
            continue
        if item.is_symlink():
            raise TrustedPluginError(
                f"declared dependency trees must not contain symlinks: {item}"
            )
        if not item.is_file():
            continue
        relative = relative_path.as_posix().encode("utf-8")
        payload_digest = hashlib.sha256(item.read_bytes()).digest()
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(payload_digest)
    return hasher.hexdigest()


def _ignore_generated_dependency_files(
    _directory: str, names: list[str]
) -> set[str]:
    return {
        name
        for name in names
        if name == "__pycache__" or Path(name).suffix.lower() == ".pyc"
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_file():
            with item.open("r+b") as stream:
                os.fsync(stream.fileno())
    for directory in sorted(
        (item for item in path.rglob("*") if item.is_dir()), reverse=True
    ):
        _fsync_directory(directory)
    _fsync_directory(path)


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


def _trusted_module_name(path: Path, kind: str) -> str:
    return f"sohbench_trusted_{kind}_{path.stem}_{_stable_module_token(path)}"


def _restore_module(module_name: str, previous_module: ModuleType | object) -> None:
    if previous_module is _MISSING:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous_module


def _load_module(
    path: Path, kind: str, *, logical_path: Path | None = None
) -> ModuleType:
    identity_path = Path(logical_path) if logical_path is not None else path
    module_name = _trusted_module_name(identity_path, kind)
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise TrustedPluginError(f"cannot load plugin file: {path}") from exc
    module = ModuleType(module_name)
    module.__file__ = str(identity_path)
    previous_module = sys.modules.get(module_name, _MISSING)
    sys.modules[module_name] = module
    try:
        code = compile(source_bytes, str(identity_path), "exec")
        exec(code, module.__dict__)
    except Exception as exc:
        _restore_module(module_name, previous_module)
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
            low = validate_finite_vector(
                low, name="interval_low", n_samples=len(test)
            )
            high = validate_finite_vector(
                high, name="interval_high", n_samples=len(test)
            )
            validate_interval_bounds(low, high)
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
