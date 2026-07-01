from __future__ import annotations

import json
import re
import shutil
import time
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
            shutil.copy2(source, target_dir / filename)
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
    profile_dir = profile_root / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    source_artifacts: dict[str, str] = {}
    sources = [Path(item) for item in source_dirs] or [root]
    for filename in artifact_files:
        source = _find_artifact(filename, sources)
        if source is None:
            continue
        shutil.copy2(source, profile_dir / filename)
        copied.append(filename)
        source_artifacts[filename] = str(source)

    payload = {
        "schema_version": 1,
        "profile": name,
        "label": manifest.get("label") or name,
        "kind": manifest.get("kind") or "evaluation_profile",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_files": copied,
        "source_artifacts": source_artifacts,
        **manifest,
    }
    payload["schema_version"] = 1
    payload["profile"] = name
    payload["artifact_files"] = copied
    payload["source_artifacts"] = source_artifacts
    manifest_path = profile_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _upsert_profile_index(profile_root, payload)
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
