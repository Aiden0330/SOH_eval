from __future__ import annotations

import errno
import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

if os.name == "nt":
    import msvcrt
else:
    import fcntl


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
GENERATION_NAME_RE = re.compile(r"^[0-9a-f]{32}$")
PUBLICATION_PATH_PART_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
GENERATIONS_DIRNAME = ".generations"
CURRENT_POINTER_NAME = "CURRENT.json"
_PUBLICATION_CONTROL_NAMES = frozenset(
    name.casefold()
    for name in (GENERATIONS_DIRNAME, CURRENT_POINTER_NAME)
)
_PUBLICATION_LOCK = threading.RLock()
_ATOMIC_REPLACE_TIMEOUT_SECONDS = 5.0
_POINTER_READ_ATTEMPTS = 5
_POINTER_READ_DELAY_SECONDS = 0.01


def _remove_owned_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _is_publication_control_name(name: str) -> bool:
    return str(name).casefold() in _PUBLICATION_CONTROL_NAMES


def resolve_safe_publication_asset(
    root: str | Path,
    relative_path: str,
    *,
    label: str = "publication asset",
) -> Path:
    value = str(relative_path or "")
    pure_path = PurePosixPath(value)
    parts = pure_path.parts
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or pure_path.is_absolute()
        or pure_path.as_posix() != value
        or not parts
        or any(
            part in ("", ".", "..")
            or not PUBLICATION_PATH_PART_RE.fullmatch(part)
            or part.casefold() in {".generations", "current", "current.json"}
            for part in parts
        )
    ):
        raise ValueError(f"invalid {label}: {relative_path!r}")

    canonical_root = Path(root).resolve()
    candidate = canonical_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes publication directory: {relative_path!r}"
        ) from exc
    return candidate


def _validated_compatibility_sources(source_root: Path) -> list[Path]:
    sources = sorted(
        source_root.iterdir(), key=lambda item: item.name
    )
    collisions = [
        source.name
        for source in sources
        if _is_publication_control_name(source.name)
    ]
    if collisions:
        raise ValueError(
            "generation contains a reserved publication control: "
            + ", ".join(collisions)
        )
    return sources


def _ignore_publication_controls_at_root(source_root: Path):
    canonical_root = source_root.resolve()

    def ignore(directory: str, names: list[str]) -> list[str]:
        if Path(directory).resolve() != canonical_root:
            return []
        return [
            name
            for name in names
            if _is_publication_control_name(name)
        ]

    return ignore


def _lock_publication_file(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        while True:
            try:
                msvcrt.locking(
                    stream.fileno(), msvcrt.LK_NBLCK, 1
                )
                return
            except OSError as exc:
                if exc.errno not in (
                    errno.EACCES,
                    errno.EAGAIN,
                    errno.EDEADLK,
                ):
                    raise
                time.sleep(0.01)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)


def _unlock_publication_file(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _interprocess_publication_lock(target: Path):
    lock_path = target.parent / f".{target.name}.publish.lock"
    with lock_path.open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        _lock_publication_file(stream)
        try:
            yield
        finally:
            _unlock_publication_file(stream)


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


def resolve_published_directory(target: str | Path) -> Path:
    root = Path(target)
    pointer = root / CURRENT_POINTER_NAME
    document = _read_pointer_document(pointer)
    if document is None:
        return root
    generation = str(document.get("generation") or "")
    if (
        document.get("schema_version") != 1
        or not GENERATION_NAME_RE.fullmatch(generation)
    ):
        raise ValueError(f"invalid publication pointer: {pointer}")
    active = root / GENERATIONS_DIRNAME / generation
    if not active.is_dir():
        raise FileNotFoundError(
            f"active publication generation is unavailable: {active}"
        )
    return active


def _read_pointer_document(pointer: Path) -> dict | None:
    for attempt in range(_POINTER_READ_ATTEMPTS):
        try:
            payload = pointer.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except PermissionError:
            if attempt + 1 >= _POINTER_READ_ATTEMPTS:
                raise
            time.sleep(_POINTER_READ_DELAY_SECONDS * (2**attempt))
            continue
        return json.loads(payload)
    raise RuntimeError("unreachable pointer read retry state")


def _atomic_replace(source: Path, destination: Path) -> None:
    deadline = time.monotonic() + _ATOMIC_REPLACE_TIMEOUT_SECONDS
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def publish_text_file_atomic(
    target: str | Path,
    text: str,
    *,
    validate: Callable[[Path], None] | None = None,
) -> None:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if validate is not None:
            validate(temporary)
        _atomic_replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_atomic_json(path: Path, document: dict) -> None:
    payload = json.dumps(document, indent=2, ensure_ascii=False)

    def validate(temporary: Path) -> None:
        if json.loads(temporary.read_text(encoding="utf-8")) != document:
            raise ValueError(f"atomic JSON validation failed: {path}")

    publish_text_file_atomic(path, payload, validate=validate)


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.compat-{uuid.uuid4().hex}"
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _launcher_html(generation: str) -> bytes:
    destination = f"./{GENERATIONS_DIRNAME}/{generation}/index.html"
    script_destination = json.dumps(destination)
    return (
        "<!doctype html><meta charset=\"utf-8\">"
        "<title>SOH</title><script>"
        f"location.replace({script_destination}+location.search+location.hash);"
        "</script>"
    ).encode("utf-8")


def _sync_compatibility_view(
    target: Path, generation: Path, generation_name: str
) -> None:
    snapshot = target.parent / (
        f".{target.name}.compat-snapshot-{uuid.uuid4().hex}"
    )
    snapshot_ready = False
    completed = False
    restored = False
    try:
        snapshot.mkdir()
        _copy_compatibility_content(target, snapshot)
        snapshot_ready = True
        _apply_compatibility_view(target, generation, generation_name)
        completed = True
    except BaseException:
        if snapshot_ready:
            _restore_compatibility_content(target, snapshot)
            restored = True
        raise
    finally:
        if completed or restored or not snapshot_ready:
            _remove_owned_directory(snapshot)


def _copy_compatibility_content(
    source_root: Path, target_root: Path
) -> None:
    for source in sorted(source_root.iterdir(), key=lambda item: item.name):
        if _is_publication_control_name(source.name):
            continue
        destination = target_root / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _restore_compatibility_content(
    target: Path, snapshot: Path
) -> None:
    for existing in list(target.iterdir()):
        if _is_publication_control_name(existing.name):
            continue
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink(missing_ok=True)
    _copy_compatibility_content(snapshot, target)


def _apply_compatibility_view(
    target: Path,
    generation: Path,
    generation_name: str,
) -> None:
    sources = _validated_compatibility_sources(generation)
    desired = {item.name for item in sources}
    for existing in list(target.iterdir()):
        if (
            _is_publication_control_name(existing.name)
            or existing.name in desired
        ):
            continue
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink(missing_ok=True)

    for source in sources:
        destination = target / source.name
        if source.is_file():
            payload = (
                _launcher_html(generation_name)
                if source.name == "index.html"
                else source.read_bytes()
            )
            _write_atomic_bytes(destination, payload)
            continue

        staging = target / f".{source.name}.compat-{uuid.uuid4().hex}"
        backup = target / f".{source.name}.backup-{uuid.uuid4().hex}"
        backup_created = False
        activated = False
        restored = False
        try:
            shutil.copytree(source, staging)
            if destination.exists():
                os.replace(destination, backup)
                backup_created = True
            try:
                os.replace(staging, destination)
                activated = True
            except BaseException:
                if backup_created:
                    os.replace(backup, destination)
                    restored = True
                raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            if backup.exists() and (activated or restored):
                shutil.rmtree(backup)


def _publish_directory(target: Path, build: Callable[[Path], None]) -> None:
    target = Path(target).expanduser().resolve()
    parent = target.parent
    token = uuid.uuid4().hex
    staging = parent / f".{target.name}.tmp-{token}"

    with _PUBLICATION_LOCK:
        parent.mkdir(parents=True, exist_ok=True)
        with _interprocess_publication_lock(target):
            if target.exists() and not target.is_dir():
                raise NotADirectoryError(
                    f"publication target is not a directory: {target}"
                )
            try:
                build(staging)
                if not staging.is_dir():
                    raise RuntimeError(
                        f"publication did not build a directory: {staging}"
                    )
                _validated_compatibility_sources(staging)
                _fsync_tree(staging)
                _fsync_directory(parent)

                target.mkdir(exist_ok=True)
                generations = target / GENERATIONS_DIRNAME
                generations.mkdir(exist_ok=True)
                generation = generations / token
                os.replace(staging, generation)
                _fsync_directory(generations)
                try:
                    _write_atomic_json(
                        target / CURRENT_POINTER_NAME,
                        {
                            "schema_version": 1,
                            "generation": token,
                            "path": f"{GENERATIONS_DIRNAME}/{token}",
                        },
                    )
                except BaseException:
                    if _generation_is_definitely_unreferenced(
                        target, generation
                    ):
                        _remove_owned_directory(generation)
                    raise
                try:
                    _sync_compatibility_view(
                        target, generation, token
                    )
                except OSError:
                    pass
            finally:
                _remove_owned_directory(staging)


def _generation_is_definitely_unreferenced(
    target: Path, generation: Path
) -> bool:
    try:
        return resolve_published_directory(target) != generation
    except BaseException:
        return False


def _copy_current_publication(target: Path, staging: Path) -> None:
    current = resolve_published_directory(target)
    if not current.is_dir():
        staging.mkdir(parents=True)
        return
    if current == target:
        shutil.copytree(
            current,
            staging,
            ignore=_ignore_publication_controls_at_root(current),
        )
        return
    shutil.copytree(current, staging)


def sync_frontend_artifacts(run_dir: str | Path, frontend_data_dir: str | Path) -> list[str]:
    source_dir = Path(run_dir)
    target_dir = Path(frontend_data_dir)
    copied = [filename for filename in FRONTEND_ARTIFACTS if (source_dir / filename).is_file()]

    def build(staging: Path) -> None:
        _copy_current_publication(target_dir, staging)
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
    sources = [resolve_published_directory(item) for item in source_dirs]
    selected_sources = {
        filename: source
        for filename in FRONTEND_ARTIFACTS
        if (source := _find_artifact(filename, sources)) is not None
    }
    copied = list(selected_sources)

    def build(staging: Path) -> None:
        _copy_current_publication(target_dir, staging)
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
    current_root = resolve_published_directory(root)
    sources = [resolve_published_directory(item) for item in source_dirs] or [
        current_root
    ]
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
        _copy_current_publication(root, staging)
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
    profile_dir = resolve_published_directory(root) / "profiles" / name
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
