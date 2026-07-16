from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Sequence


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
_PUBLICATION_LOCK = threading.RLock()


def _remove_owned_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


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
        if not item.is_file():
            continue
        with item.open("r+b") as stream:
            os.fsync(stream.fileno())
    for directory in sorted((item for item in path.rglob("*") if item.is_dir()), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(path)


def _publish_directory(target: Path, build: Callable[[Path], None]) -> None:
    target = Path(target)
    parent = target.parent
    token = uuid.uuid4().hex
    staging = parent / f".{target.name}.tmp-{token}"
    backup = parent / f".{target.name}.bak-{token}"

    with _PUBLICATION_LOCK:
        parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not target.is_dir():
            raise NotADirectoryError(f"publication target is not a directory: {target}")
        try:
            build(staging)
            if not staging.is_dir():
                raise RuntimeError(f"publication did not build a directory: {staging}")
            _fsync_tree(staging)
            _fsync_directory(parent)

            had_target = target.exists()
            if had_target:
                os.replace(target, backup)
            try:
                os.replace(staging, target)
            except BaseException:
                if had_target and backup.exists() and not target.exists():
                    os.replace(backup, target)
                raise
            _fsync_directory(parent)
            _remove_owned_directory(backup)
        finally:
            _remove_owned_directory(staging)
            if backup.exists() and target.exists():
                _remove_owned_directory(backup)


def sync_frontend_artifacts(run_dir: str | Path, frontend_data_dir: str | Path) -> list[str]:
    source_dir = Path(run_dir)
    target_dir = Path(frontend_data_dir)
    copied = [filename for filename in FRONTEND_ARTIFACTS if (source_dir / filename).is_file()]

    def build(staging: Path) -> None:
        if target_dir.is_dir():
            shutil.copytree(target_dir, staging)
        else:
            staging.mkdir(parents=True)
        for filename in FRONTEND_ARTIFACTS:
            (staging / filename).unlink(missing_ok=True)
        for filename in copied:
            shutil.copy2(source_dir / filename, staging / filename)

    _publish_directory(target_dir, build)
    return copied


def publish_frontend_artifacts(
    source_dirs: Sequence[str | Path], frontend_data_dir: str | Path
) -> list[str]:
    target_dir = Path(frontend_data_dir)
    sources = [Path(item) for item in source_dirs]
    selected_sources = {
        filename: source
        for filename in FRONTEND_ARTIFACTS
        if (source := _find_artifact(filename, sources)) is not None
    }
    copied = list(selected_sources)

    def build(staging: Path) -> None:
        if target_dir.is_dir():
            shutil.copytree(target_dir, staging)
        else:
            staging.mkdir(parents=True)
        for filename in FRONTEND_ARTIFACTS:
            (staging / filename).unlink(missing_ok=True)
        for filename, source in selected_sources.items():
            shutil.copy2(source, staging / filename)

    _publish_directory(target_dir, build)
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
    sources = [Path(item) for item in source_dirs] or [root]
    selected_sources = {
        filename: source
        for filename in artifact_files
        if (source := _find_artifact(filename, sources)) is not None
    }
    copied = list(selected_sources)
    source_artifacts = {filename: str(source) for filename, source in selected_sources.items()}

    manifest_source_artifacts = manifest.get("source_artifacts") or {}
    if not isinstance(manifest_source_artifacts, dict):
        manifest_source_artifacts = {}
    reported_source_artifacts = {
        **source_artifacts,
        **{str(key): str(value) for key, value in manifest_source_artifacts.items()},
    }

    result: dict = {}

    def build(staging: Path) -> None:
        if root.is_dir():
            shutil.copytree(root, staging)
        else:
            staging.mkdir(parents=True)
        profile_root = staging / "profiles"
        profile_dir = profile_root / name
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
        profile_dir.mkdir(parents=True)
        for filename, source in selected_sources.items():
            shutil.copy2(source, profile_dir / filename)

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
        (profile_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _upsert_profile_index(profile_root, payload)
        result["manifest"] = payload

    _publish_directory(root, build)
    profile_dir = root / "profiles" / name
    manifest_path = profile_dir / "manifest.json"
    return {
        "name": name,
        "path": str(profile_dir),
        "manifest_path": str(manifest_path),
        "manifest": result["manifest"],
    }


def _find_artifact(filename: str, source_dirs: Sequence[Path]) -> Path | None:
    for directory in source_dirs:
        path = directory / filename
        if path.exists():
            return path
    return None


def _upsert_profile_index(profile_root: Path, manifest: dict) -> None:
    index_path = profile_root / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"profiles": []}
    except json.JSONDecodeError:
        index = {"profiles": []}
    existing = [item for item in index.get("profiles", []) if item.get("name") != manifest["profile"]]
    existing.append(
        {
            "name": manifest["profile"],
            "label": manifest.get("label") or manifest["profile"],
            "kind": manifest.get("kind") or "evaluation_profile",
            "created_at": manifest.get("created_at") or "",
            "datasets": list(manifest.get("datasets") or []),
            "training_job_id": manifest.get("training_job_id") or "",
            "inference_job_id": manifest.get("inference_job_id") or "",
        }
    )
    index["profiles"] = sorted(existing, key=lambda item: item.get("name", ""))
    profile_root.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_profile_name(name: str) -> str:
    value = str(name or "").strip()
    if not PROFILE_NAME_RE.match(value):
        raise ValueError(f"invalid profile name: {name!r}")
    return value
