from __future__ import annotations

import copy
import inspect
import json
import os
import pickle
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ID_COLUMNS, validate_finite_vector
from .fabric.ids import sha256_file


SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


class ModelStoreError(ValueError):
    """Raised for invalid trusted local model packages."""


@dataclass
class LoadedModel:
    name: str
    model: Any
    manifest: dict
    path: Path
    load_ms: float = 0.0

    def feature_columns(self) -> list[str]:
        return [str(col) for col in self.manifest.get("feature_columns", [])]

    def input_columns(self, table: pd.DataFrame) -> list[str]:
        configured = self.manifest.get("input_columns")
        if configured:
            return [str(col) for col in configured]
        return [col for col in [*ID_COLUMNS, *self.feature_columns()] if col in table.columns]

    def compatibility(self, table: pd.DataFrame) -> dict:
        missing = [col for col in self.feature_columns() if col not in table.columns]
        return {"ok": not missing, "missing_feature_columns": missing}

    def predict(self, table: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        self._ensure_compatible(table)
        X = table[self.input_columns(table)].copy()
        predict = getattr(self.model, "predict", None)
        if not callable(predict):
            raise ModelStoreError(f"model {self.name!r} does not expose predict()")
        values = _call_predict(predict, X, meta)
        try:
            return validate_finite_vector(
                values, name="predictions", n_samples=len(table)
            )
        except ValueError as exc:
            if "finite" in str(exc):
                raise ModelStoreError(
                    f"model {self.name!r} returned non-finite predictions"
                ) from exc
            raise ModelStoreError(f"model {self.name!r} {exc}") from exc

    def predict_interval(
        self,
        table: pd.DataFrame,
        meta: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        fn = getattr(self.model, "predict_interval", None)
        if not callable(fn):
            return None
        self._ensure_compatible(table)
        X = table[self.input_columns(table)].copy()
        try:
            lo, hi = _call_predict(fn, X, meta)
            return (
                validate_finite_vector(
                    lo, name="interval_low", n_samples=len(table)
                ),
                validate_finite_vector(
                    hi, name="interval_high", n_samples=len(table)
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ModelStoreError(f"model {self.name!r} {exc}") from exc

    def _ensure_compatible(self, table: pd.DataFrame) -> None:
        missing = self.compatibility(table)["missing_feature_columns"]
        if missing:
            raise ModelStoreError(f"missing feature columns for {self.name}: {missing}")


class ModelStore:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.catalog_path = self.base_dir / "model_catalog.json"
        self._lock = threading.RLock()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._catalog = self._read_catalog()

    def save_training_model(
        self,
        name: str,
        estimator: object,
        manifest: dict,
        replace: bool = False,
    ) -> dict:
        return self._write_package(name, estimator, manifest, source="training", replace=replace)

    def upload_model(
        self,
        name: str,
        model_filename: str,
        model_payload: bytes,
        manifest_filename: str,
        manifest_payload: bytes,
        replace: bool = False,
    ) -> dict:
        if Path(model_filename).suffix.lower() not in {".pkl", ".pickle"}:
            raise ModelStoreError("model upload must be .pkl or .pickle")
        if Path(manifest_filename).name != "model_manifest.json":
            raise ModelStoreError("manifest upload must be model_manifest.json")
        try:
            model = pickle.loads(model_payload)
        except Exception as exc:
            raise ModelStoreError(f"model pickle could not be loaded: {exc}") from exc
        try:
            manifest = json.loads(manifest_payload.decode("utf-8"))
        except Exception as exc:
            raise ModelStoreError(f"model manifest could not be parsed: {exc}") from exc
        return self._write_package(name, model, manifest, source="upload", replace=replace)

    def load_model(self, name: str) -> LoadedModel:
        with self._lock:
            item = dict(self._find(name))
            start = time.perf_counter()
            with open(item["model_path"], "rb") as fh:
                model = pickle.load(fh)
            load_ms = (time.perf_counter() - start) * 1000.0
            manifest = json.loads(
                Path(item["manifest_path"]).read_text(encoding="utf-8")
            )
            return LoadedModel(
                name=item["name"],
                model=model,
                manifest=manifest,
                path=Path(item["path"]),
                load_ms=load_ms,
            )

    def catalog(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._catalog.get("models", []))

    def content_digests(self, name: str) -> dict[str, str]:
        with self._lock:
            item = self._find(name)
            values = {
                "model_pickle": sha256_file(Path(item["model_path"])),
                "model_manifest": sha256_file(Path(item["manifest_path"])),
            }
            for field, kind in (
                ("model_sha256", "model_pickle"),
                ("manifest_sha256", "model_manifest"),
            ):
                expected = item.get(field)
                if expected is not None and expected != values[kind]:
                    raise ModelStoreError(
                        f"model {name!r} {kind} digest mismatch"
                    )
            return values

    def _write_package(
        self,
        name: str,
        model: object,
        manifest: dict,
        source: str,
        replace: bool,
    ) -> dict:
        name = _safe_name(name)
        manifest = _validate_manifest(name, manifest)
        if not callable(getattr(model, "predict", None)):
            raise ModelStoreError("model must expose predict()")
        try:
            model_payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise ModelStoreError(f"model could not be serialized: {exc}") from exc
        manifest_payload = json.dumps(
            manifest, indent=2, ensure_ascii=False
        ).encode("utf-8")

        token = uuid.uuid4().hex
        model_dir = self.base_dir / name
        staged_dir = self.base_dir / f".{name}.tmp-{token}"
        backup_dir = self.base_dir / f".{name}.backup-{token}"
        staged_dir.mkdir()
        staged_model_path = staged_dir / "model.pkl"
        staged_manifest_path = staged_dir / "model_manifest.json"
        try:
            _write_fsynced(staged_model_path, model_payload)
            _write_fsynced(staged_manifest_path, manifest_payload)
        except BaseException:
            shutil.rmtree(staged_dir, ignore_errors=True)
            raise
        try:
            staged_model = pickle.loads(staged_model_path.read_bytes())
            staged_manifest = json.loads(
                staged_manifest_path.read_text(encoding="utf-8")
            )
            loaded = LoadedModel(
                name=name,
                model=staged_model,
                manifest=staged_manifest,
                path=staged_dir,
            )
            smoke = _smoke_table(list(manifest.get("feature_columns", [])))
            loaded.predict(smoke, pd.DataFrame())
            loaded.predict_interval(smoke, pd.DataFrame())
        except ModelStoreError:
            shutil.rmtree(staged_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(staged_dir, ignore_errors=True)
            raise ModelStoreError(
                f"model package validation failed: {exc}"
            ) from exc

        model_path = model_dir / "model.pkl"
        manifest_path = model_dir / "model_manifest.json"
        resource = {
            "name": name,
            "kind": "model",
            "source": source,
            "path": str(model_dir),
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "estimator": manifest.get("estimator"),
            "featureset": manifest.get("featureset"),
            "feature_columns": list(manifest.get("feature_columns", [])),
            "trained_on": list(manifest.get("trained_on", [])),
            "created_by_run": manifest.get("created_by_run"),
            "model_sha256": sha256_file(staged_model_path),
            "manifest_sha256": sha256_file(staged_manifest_path),
        }
        catalog_temporary = None
        published = False
        try:
            with self._lock:
                existing = any(
                    item.get("name") == name
                    for item in self._catalog.get("models", [])
                )
                if (existing or model_dir.exists()) and not replace:
                    raise ModelStoreError(f"model already exists: {name}")
                items = [
                    item
                    for item in self._catalog.get("models", [])
                    if item.get("name") != name
                ]
                items.append(resource)
                candidate_catalog = {
                    "models": sorted(items, key=lambda item: item["name"])
                }
                catalog_temporary = self._prepare_catalog(candidate_catalog)
                moved_existing = False
                moved_staged = False
                try:
                    if model_dir.exists():
                        model_dir.replace(backup_dir)
                        moved_existing = True
                    staged_dir.replace(model_dir)
                    moved_staged = True
                    catalog_temporary.replace(self.catalog_path)
                    self._catalog = candidate_catalog
                    published = True
                except BaseException:
                    if moved_staged and model_dir.exists():
                        shutil.rmtree(model_dir, ignore_errors=True)
                    if moved_existing and backup_dir.exists():
                        backup_dir.replace(model_dir)
                    raise
        finally:
            if catalog_temporary is not None:
                catalog_temporary.unlink(missing_ok=True)
            if staged_dir.exists():
                shutil.rmtree(staged_dir, ignore_errors=True)
            if published and backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
        return copy.deepcopy(resource)

    def _read_catalog(self) -> dict:
        if not self.catalog_path.exists():
            return {"models": []}
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def _prepare_catalog(self, document: dict) -> Path:
        temporary = self.base_dir / f".model_catalog.{uuid.uuid4().hex}.tmp"
        _write_fsynced(
            temporary,
            json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8"),
        )
        return temporary

    def _find(self, name: str) -> dict:
        for item in self._catalog.get("models", []):
            if item.get("name") == name:
                return item
        raise ModelStoreError(f"unknown model: {name}")


def _safe_name(name: str) -> str:
    value = str(name or "").strip()
    if not SAFE_NAME.match(value):
        raise ModelStoreError(f"invalid model name: {name!r}")
    return value


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_manifest(name: str, manifest: dict) -> dict:
    required = ["model_name", "model_kind", "feature_columns", "target", "predict_contract"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ModelStoreError(f"model manifest missing keys: {missing}")
    if manifest["model_kind"] != "pickle":
        raise ModelStoreError("model manifest model_kind must be pickle")
    if not manifest.get("feature_columns"):
        raise ModelStoreError("model manifest must define feature_columns")
    manifest = dict(manifest)
    manifest["model_name"] = name
    return manifest


def _call_predict(fn, X: pd.DataFrame, meta: pd.DataFrame | None):
    params = inspect.signature(fn).parameters
    if len(params) >= 2:
        return fn(X, meta)
    return fn(X)


def _smoke_table(columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "dataset": ["smoke", "smoke", "smoke"],
            "cell_id": ["c1", "c1", "c1"],
            "cycle_index": [0, 1, 2],
            "soh": [1.0, 0.99, 0.98],
        }
    )
    for col in columns:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame
