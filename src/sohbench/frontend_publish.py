from __future__ import annotations

import os
import json
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence


FRONTEND_ARTIFACTS = (
    "leaderboard.json",
    "dataset_leaderboard.json",
    "qc_report.json",
    "run_config.json",
    "model_catalog.json",
    "inference_benchmark.json",
    "inference_run_config.json",
    "recommendation_summary.json",
)

PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


def sync_frontend_artifacts(run_dir: str | Path, frontend_data_dir: str | Path) -> list[str]:
    source_dir = Path(run_dir)
    target_dir = Path(frontend_data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for filename in FRONTEND_ARTIFACTS:
        source = source_dir / filename
        if source.exists():
            _copy_file_atomic(source, target_dir / filename)
            copied.append(filename)
    return copied


def publish_frontend_profile(
    frontend_data_dir: str | Path,
    profile_name: str,
    manifest: dict,
    *,
    source_dirs: Sequence[str | Path] = (),
    artifact_files: Sequence[str] = FRONTEND_ARTIFACTS,
) -> dict:
    name = _safe_profile_name(profile_name)
    root = Path(frontend_data_dir)
    profile_root = root / "profiles"
    profile_root.mkdir(parents=True, exist_ok=True)
    with _profile_publish_lock(profile_root):
        profile_dir = profile_root / name
        profile_dir.mkdir(parents=True, exist_ok=True)

        copied: list[str] = []
        source_artifacts: dict[str, str] = {}
        sources = [Path(item) for item in source_dirs] or [root]
        for filename in artifact_files:
            source = _find_artifact(filename, sources)
            if source is None:
                continue
            _copy_file_atomic(source, profile_dir / filename)
            copied.append(filename)
            source_artifacts[filename] = str(source)

        manifest_source_artifacts = manifest.get("source_artifacts") or {}
        if not isinstance(manifest_source_artifacts, dict):
            manifest_source_artifacts = {}
        reported_source_artifacts = {
            **source_artifacts,
            **{str(key): str(value) for key, value in manifest_source_artifacts.items()},
        }

        payload = {
            "schema_version": 1,
            "profile": name,
            "label": manifest.get("label") or name,
            "kind": manifest.get("kind") or "evaluation_profile",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifact_files": copied,
            "source_artifacts": reported_source_artifacts,
            **manifest,
        }
        payload["schema_version"] = 1
        payload["profile"] = name
        payload["artifact_files"] = copied
        payload["source_artifacts"] = reported_source_artifacts
        manifest_path = profile_dir / "manifest.json"
        _write_json_atomic(manifest_path, payload)
        _upsert_profile_index_unlocked(profile_root, payload)
    return {
        "name": name,
        "path": str(profile_dir),
        "manifest_path": str(manifest_path),
        "manifest": payload,
    }


def _find_artifact(filename: str, source_dirs: Sequence[Path]) -> Path | None:
    for directory in source_dirs:
        path = directory / filename
        if path.exists():
            return path
    return None


def _upsert_profile_index(profile_root: Path, manifest: dict) -> None:
    profile_root.mkdir(parents=True, exist_ok=True)
    with _profile_publish_lock(profile_root):
        _upsert_profile_index_unlocked(profile_root, manifest)


def _upsert_profile_index_unlocked(profile_root: Path, manifest: dict) -> None:
    index_path = profile_root / "index.json"
    index = _read_index(index_path)
    existing = []
    for item in index.get("profiles") or []:
        if not isinstance(item, dict):
            continue
        try:
            entry = _profile_index_entry(item)
        except ValueError:
            continue
        if entry["name"] != manifest["profile"]:
            existing.append(entry)
    existing.append(_profile_index_entry(manifest, name=manifest["profile"]))
    index["schema_version"] = 1
    index["profiles"] = sorted(existing, key=lambda item: item.get("name", ""))
    _write_json_atomic(index_path, index)


def load_frontend_profile_index(frontend_data_dir: str | Path) -> dict:
    """Return the current profile catalog, including manifests added after API startup."""
    profile_root = Path(frontend_data_dir) / "profiles"
    index = _read_index(profile_root / "index.json")
    merged: dict[str, dict] = {}
    for item in index.get("profiles") or []:
        if not isinstance(item, dict):
            continue
        try:
            entry = _profile_index_entry(item)
        except ValueError:
            continue
        merged[entry["name"]] = entry

    if profile_root.exists():
        for manifest_path in sorted(profile_root.glob("*/manifest.json")):
            name = manifest_path.parent.name
            try:
                _safe_profile_name(name)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or manifest.get("profile") != name:
                    continue
                merged[name] = _profile_index_entry(manifest, name=name)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return {
        "schema_version": 1,
        "profiles": sorted(merged.values(), key=lambda item: item["name"]),
    }


def _profile_index_entry(payload: dict, *, name: str | None = None) -> dict:
    profile_name = _safe_profile_name(name or payload.get("name") or payload.get("profile") or "")
    raw_datasets = payload.get("datasets") or []
    if isinstance(raw_datasets, str):
        raw_datasets = [raw_datasets]
    datasets = []
    for value in raw_datasets if isinstance(raw_datasets, (list, tuple)) else []:
        text = str(value or "").strip()
        if text and text not in datasets:
            datasets.append(text)
    return {
        "name": profile_name,
        "label": str(payload.get("label") or profile_name),
        "kind": str(payload.get("kind") or "evaluation_profile"),
        "created_at": str(payload.get("created_at") or ""),
        "datasets": datasets,
        "training_job_id": str(payload.get("training_job_id") or ""),
        "inference_job_id": str(payload.get("inference_job_id") or ""),
    }


def _read_index(index_path: Path) -> dict:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles", []), list):
        payload = {}
    return {"schema_version": 1, "profiles": list(payload.get("profiles") or [])}


def _copy_file_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        _replace_with_retry(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.025 * (attempt + 1))


@contextmanager
def _profile_publish_lock(profile_root: Path, *, timeout_seconds: float = 10.0):
    lock_path = profile_root / ".publish.lock"
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()} time={time.time()}\n".encode("ascii"))
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 120
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for profile publish lock: {lock_path}")
            time.sleep(0.025)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _safe_profile_name(name: str) -> str:
    value = str(name or "").strip()
    if not PROFILE_NAME_RE.match(value):
        raise ValueError(f"invalid profile name: {name!r}")
    return value
