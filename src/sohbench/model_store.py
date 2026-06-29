from __future__ import annotations

import inspect
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ID_COLUMNS


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
        pred = np.asarray(values, dtype=float)
        if pred.shape[0] != len(table):
            raise ModelStoreError(f"model {self.name!r} returned {pred.shape[0]} predictions for {len(table)} rows")
        if not np.isfinite(pred).all():
            raise ModelStoreError(f"model {self.name!r} returned non-finite predictions")
        return pred

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
        lo, hi = _call_predict(fn, X, meta)
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def _ensure_compatible(self, table: pd.DataFrame) -> None:
        missing = self.compatibility(table)["missing_feature_columns"]
        if missing:
            raise ModelStoreError(f"missing feature columns for {self.name}: {missing}")


class ModelStore:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.catalog_path = self.base_dir / "model_catalog.json"
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
        item = self._find(name)
        start = time.perf_counter()
        with open(item["model_path"], "rb") as fh:
            model = pickle.load(fh)
        load_ms = (time.perf_counter() - start) * 1000.0
        manifest = json.loads(Path(item["manifest_path"]).read_text(encoding="utf-8"))
        return LoadedModel(
            name=item["name"],
            model=model,
            manifest=manifest,
            path=Path(item["path"]),
            load_ms=load_ms,
        )

    def catalog(self) -> list[dict]:
        return list(self._catalog.get("models", []))

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
        model_dir = self.base_dir / name
        if model_dir.exists() and not replace:
            raise ModelStoreError(f"model already exists: {name}")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        manifest_path = model_dir / "model_manifest.json"
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
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
        }
        self._upsert(resource)
        self.load_model(name).predict(_smoke_table(resource["feature_columns"]), pd.DataFrame())
        return resource

    def _read_catalog(self) -> dict:
        if not self.catalog_path.exists():
            return {"models": []}
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def _write_catalog(self) -> None:
        self.catalog_path.write_text(
            json.dumps(self._catalog, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _upsert(self, resource: dict) -> None:
        items = [
            item
            for item in self._catalog.get("models", [])
            if item.get("name") != resource["name"]
        ]
        items.append(resource)
        self._catalog["models"] = sorted(items, key=lambda item: item["name"])
        self._write_catalog()

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
