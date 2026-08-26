from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ID_COLUMNS
from .feature_safety import LABEL_SAFETY_CONTRACT, audit_feature_frame, suspicious_feature_names


SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
MODEL_PACKAGE_CONTRACT_VERSION = "1.0.0"


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
        target = str(self.manifest.get("target") or "soh")
        configured = self.manifest.get("input_columns")
        if configured:
            return [
                str(col) for col in configured
                if str(col) != target and str(col) in table.columns
            ]
        return [
            col for col in [*ID_COLUMNS, *self.feature_columns()]
            if col != target and col in table.columns
        ]

    def compatibility(self, table: pd.DataFrame) -> dict:
        feature_columns = self.feature_columns()
        missing = [col for col in feature_columns if col not in table.columns]
        # A column materialized as all-NaN (a transform whose source signals
        # are absent from this dataset) is missing in every way that matters
        # for prediction; report it so preflight can filter the combination.
        all_nan = [
            col for col in feature_columns
            if col in table.columns and bool(table[col].isna().all())
        ]
        return {
            "ok": not missing and not all_nan,
            "missing_feature_columns": missing,
            "all_nan_feature_columns": all_nan,
        }

    def predict(self, table: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        self._ensure_compatible(table)
        self._ensure_label_safe(table)
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
        self._ensure_label_safe(table)
        X = table[self.input_columns(table)].copy()
        lo, hi = _call_predict(fn, X, meta)
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def _ensure_compatible(self, table: pd.DataFrame) -> None:
        report = self.compatibility(table)
        if report["missing_feature_columns"]:
            raise ModelStoreError(
                f"missing feature columns for {self.name}: {report['missing_feature_columns']}"
            )
        if report["all_nan_feature_columns"]:
            raise ModelStoreError(
                f"feature columns are all-NaN on this dataset for {self.name} "
                f"(source signals absent): {report['all_nan_feature_columns']}"
            )

    def _ensure_label_safe(self, table: pd.DataFrame) -> None:
        target = str(self.manifest.get("target") or "soh")
        audit = audit_feature_frame(table, self.feature_columns(), target=target)
        if audit["status"] == "passed":
            return
        reasons = "; ".join(item["reason"] for item in audit["violations"])
        raise ModelStoreError(
            f"model {self.name!r} inference blocked by {LABEL_SAFETY_CONTRACT}: {reasons}"
        )


class ModelStore:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.catalog_path = self.base_dir / "model_catalog.json"
        self.invalidations_path = self.base_dir / "artifact_invalidations.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._catalog = self._read_catalog()
        self._invalidations = self._read_invalidations()

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
        self._ensure_artifact_valid(_model_artifact_id(item))
        return self._load_item(item)

    def _load_item(self, item: dict) -> LoadedModel:
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

    def load_artifact(self, artifact_id: str) -> LoadedModel:
        """Load an immutable model package and verify its content digest."""
        self._ensure_artifact_valid(artifact_id)
        item = self._find_artifact(artifact_id)
        return self._load_item(item)

    def artifact_id(self, name: str) -> str:
        return _model_artifact_id(self._find(name))

    def catalog(self) -> list[dict]:
        output = []
        changed = False
        for stored in self._catalog.get("models", []):
            item = dict(stored)
            if not item.get("artifact_id"):
                try:
                    item["artifact_id"] = _model_artifact_id(item)
                    stored["artifact_id"] = item["artifact_id"]
                    changed = True
                except OSError:
                    item["artifact_id"] = ""
            invalidation = self.invalidation(str(item.get("artifact_id") or ""))
            validation = dict(item.get("validation") or {})
            if invalidation:
                validation.update(
                    {
                        "label_safety": "blocked",
                        "usable": False,
                        "invalidation": invalidation,
                    }
                )
                item["status"] = "invalidated"
                item["usable"] = False
            else:
                validation.setdefault("label_safety", "passed")
                validation.setdefault("usable", True)
                item.setdefault("status", "ready")
                item["usable"] = bool(validation.get("usable", True))
            item["validation"] = validation
            output.append(item)
        if changed:
            self._write_catalog()
        return output

    def invalidate_artifact(
        self,
        artifact_id: str,
        *,
        reason: str,
        details: dict | None = None,
        source: str = "label-safety-migration",
    ) -> dict:
        expected = str(artifact_id or "").strip().lower()
        if not re.fullmatch(r"model@sha256:[0-9a-f]{64}", expected):
            raise ModelStoreError("invalid immutable model artifact reference")
        # Verify that the immutable bytes are known before recording policy.
        self._find_artifact(expected)
        record = {
            "artifact_id": expected,
            "status": "invalidated",
            "reason": str(reason or "model artifact invalidated"),
            "details": dict(details or {}),
            "source": str(source or "label-safety-migration"),
            "invalidated_at": _now_iso(),
            "contract": LABEL_SAFETY_CONTRACT,
        }
        self._invalidations[expected] = record
        self._write_invalidations()
        return dict(record)

    def invalidation(self, artifact_id: str) -> dict | None:
        record = self._invalidations.get(str(artifact_id or "").strip().lower())
        return dict(record) if isinstance(record, dict) else None

    def is_artifact_invalidated(self, artifact_id: str) -> bool:
        return self.invalidation(artifact_id) is not None

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

        # Validate the executable contract before changing an existing package.
        # A failed smoke test must never replace a previously usable artifact.
        _smoke_model(name, model, manifest)
        try:
            model_payload = pickle.dumps(model)
        except Exception as exc:
            raise ModelStoreError(f"model could not be serialized: {exc}") from exc
        manifest_payload = json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")

        existing = self._find_optional(name)
        if existing is not None and replace:
            self._archive_artifact(existing)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        manifest_path = model_dir / "model_manifest.json"
        model_path.write_bytes(model_payload)
        manifest_path.write_bytes(manifest_payload)
        resource = {
            "name": name,
            "kind": "model",
            "source": source,
            "path": str(model_dir),
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "estimator": manifest.get("estimator"),
            "featureset": manifest.get("featureset"),
            "training_featuresets": [
                str(item) for item in manifest.get("training_featuresets", []) if str(item)
            ],
            "feature_columns": list(manifest.get("feature_columns", [])),
            "trained_on": list(manifest.get("trained_on", [])),
            "created_by_run": manifest.get("created_by_run"),
            "protocol": manifest.get("protocol"),
            "held_out_key": manifest.get("held_out_key"),
            "seed": manifest.get("seed"),
            "target": manifest.get("target"),
            "predict_contract": manifest.get("predict_contract"),
            "contract_version": manifest.get("contract_version"),
            "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "model_sha256": hashlib.sha256(model_payload).hexdigest(),
            "validation": {
                "manifest": "passed",
                "predict_smoke_test": "passed",
                "immutable_history": True,
            },
        }
        resource["artifact_id"] = _model_artifact_id(resource)
        self._upsert(resource)
        return resource

    def _read_catalog(self) -> dict:
        if not self.catalog_path.exists():
            return {"models": []}
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        for item in catalog.get("models", []):
            if isinstance(item, dict):
                _relocate_stale_paths(item, self.base_dir)
        return catalog

    def _read_invalidations(self) -> dict[str, dict]:
        if not self.invalidations_path.exists():
            return {}
        try:
            payload = json.loads(self.invalidations_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        items = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
        return {
            str(key).lower(): dict(value)
            for key, value in items.items()
            if isinstance(value, dict)
        }

    def _write_invalidations(self) -> None:
        self.invalidations_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "contract": LABEL_SAFETY_CONTRACT,
                    "artifacts": self._invalidations,
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

    def _ensure_artifact_valid(self, artifact_id: str) -> None:
        invalidation = self.invalidation(artifact_id)
        if invalidation:
            raise ModelStoreError(
                f"model artifact invalidated by label-safety policy: {invalidation.get('reason', '')}"
            )

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

    def _find_optional(self, name: str) -> dict | None:
        for item in self._catalog.get("models", []):
            if item.get("name") == name:
                return item
        return None

    def _find_artifact(self, artifact_id: str) -> dict:
        expected = str(artifact_id or "").strip().lower()
        if not re.fullmatch(r"model@sha256:[0-9a-f]{64}", expected):
            raise ModelStoreError("invalid immutable model artifact reference")
        for item in self._catalog.get("models", []):
            stored = str(item.get("artifact_id") or "").lower()
            if stored and stored != expected:
                continue
            try:
                current = _model_artifact_id(item)
            except OSError:
                continue
            if current == expected:
                return item
        archived = self._archived_artifact(expected)
        if archived is not None:
            return archived
        raise ModelStoreError(f"unknown or changed model artifact: {artifact_id}")

    def _archive_artifact(self, item: dict) -> None:
        """Preserve the current bytes before a logical model name is replaced."""
        artifact_id = _model_artifact_id(item)
        digest = artifact_id.split(":", 1)[1]
        archive_dir = self.base_dir / "_artifacts" / digest
        archive_model = archive_dir / "model.pkl"
        archive_manifest = archive_dir / "model_manifest.json"
        if archive_model.exists() and archive_manifest.exists():
            archived = {
                **item,
                "path": str(archive_dir),
                "model_path": str(archive_model),
                "manifest_path": str(archive_manifest),
            }
            if _model_artifact_id(archived) != artifact_id:
                raise ModelStoreError(f"immutable archive digest mismatch: {artifact_id}")
            return
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(item["model_path"]), archive_model)
        shutil.copy2(Path(item["manifest_path"]), archive_manifest)
        archived = {
            **item,
            "path": str(archive_dir),
            "model_path": str(archive_model),
            "manifest_path": str(archive_manifest),
        }
        if _model_artifact_id(archived) != artifact_id:
            raise ModelStoreError(f"immutable archive digest mismatch: {artifact_id}")

    def _archived_artifact(self, artifact_id: str) -> dict | None:
        digest = artifact_id.split(":", 1)[1]
        archive_dir = self.base_dir / "_artifacts" / digest
        model_path = archive_dir / "model.pkl"
        manifest_path = archive_dir / "model_manifest.json"
        if not model_path.exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        item = {
            "name": str(manifest.get("model_name") or f"artifact-{digest[:12]}"),
            "path": str(archive_dir),
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
            "artifact_id": artifact_id,
        }
        try:
            return item if _model_artifact_id(item) == artifact_id else None
        except OSError:
            return None


def _stored_path_parts(value: object) -> tuple[str, ...]:
    """Split a catalog path that may come from another OS.

    Windows-era catalogs store ``D:\\SOH_eval\\...`` absolute paths and
    ``work\\trusted_plugins\\...`` backslash-relative paths; on POSIX those
    are single opaque filenames, so parse them with the Windows rules when
    they contain a backslash or a drive prefix.
    """
    text = str(value or "")
    if not text:
        return ()
    if "\\" in text or re.match(r"^[A-Za-z]:", text):
        return PureWindowsPath(text).parts
    return PurePosixPath(text).parts


def _relocate_stale_paths(item: dict, base_dir: Path) -> None:
    """Repair catalog entries migrated from another machine (B7 debt).

    When a stored path does not resolve on this machine but the same package
    exists under the local store, point the entry at the local files.  The
    path suffix after the store's ``models`` directory is preserved so both
    top-level packages and ``_artifacts`` archive files relocate correctly.
    Entries whose files exist nowhere are left untouched (they stay
    unloadable, exactly as before).
    """

    def local_candidate(parts: tuple[str, ...], *, is_dir: bool) -> Path | None:
        if not parts:
            return None
        anchor = base_dir.name  # "models"
        if anchor in parts[:-1]:
            index = len(parts) - 1 - parts[:-1][::-1].index(anchor)
            suffix = parts[index:]
        else:
            keep = 1 if is_dir else 2
            suffix = parts[-keep:]
        return base_dir.joinpath(*suffix) if suffix else None

    for key in ("model_path", "manifest_path", "path"):
        stored = str(item.get(key) or "")
        if not stored or Path(stored).exists():
            continue
        candidate = local_candidate(_stored_path_parts(stored), is_dir=key == "path")
        if candidate is not None and candidate.exists():
            item[key] = str(candidate)


def _model_artifact_id(item: dict) -> str:
    digest = hashlib.sha256()
    for label, key in (("manifest", "manifest_path"), ("model", "model_path")):
        path = Path(str(item.get(key) or ""))
        payload = path.read_bytes()
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"model@sha256:{digest.hexdigest()}"


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
    columns = [str(col).strip() for col in manifest.get("feature_columns", []) if str(col).strip()]
    if not columns:
        raise ModelStoreError("model manifest must define feature_columns")
    if len(columns) != len(set(columns)):
        raise ModelStoreError("model manifest feature_columns must be unique")
    target = str(manifest.get("target") or "").strip()
    if target in columns:
        raise ModelStoreError("model manifest target must not appear in feature_columns")
    suspicious = suspicious_feature_names(columns, target=target)
    if suspicious:
        raise ModelStoreError(
            f"model manifest contains label-derived or label-aliased features: {suspicious}"
        )
    predict_contract = str(manifest.get("predict_contract") or "").strip()
    if not predict_contract.startswith("predict("):
        raise ModelStoreError("model manifest predict_contract must describe predict(...)")
    manifest = dict(manifest)
    manifest["model_name"] = name
    manifest["feature_columns"] = columns
    manifest["target"] = target
    manifest["predict_contract"] = predict_contract
    manifest.setdefault("contract_version", MODEL_PACKAGE_CONTRACT_VERSION)
    manifest.setdefault(
        "feature_schema",
        {
            "columns": [{"name": column, "dtype": "numeric"} for column in columns],
            "target": target,
        },
    )
    return manifest


def _smoke_model(name: str, model: object, manifest: dict) -> None:
    loaded = LoadedModel(
        name=name,
        model=model,
        manifest=manifest,
        path=Path("."),
    )
    try:
        loaded.predict(_smoke_table(loaded.feature_columns()), pd.DataFrame())
    except Exception as exc:
        raise ModelStoreError(f"model predict smoke test failed: {exc}") from exc


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
