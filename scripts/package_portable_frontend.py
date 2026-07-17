from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sohbench.frontend_publish import (
    _publish_directory,
    publish_text_file_atomic,
    resolve_published_directory,
    resolve_safe_publication_asset,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP_DIR = ROOT / "frontend" / "app"
DEFAULT_OUTPUT = ROOT / "dist" / "soh-eval-portable"


APP_ASSET_NAMES = {
    "index.html",
    "app.js",
    "dashboard-core.mjs",
    "styles.css",
}


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _is_detection_asset(path: Path) -> bool:
    return "ess_detection" in path.name


def _is_detection_entry(value) -> bool:
    if isinstance(value, str):
        return "ess_detection" in value
    if not isinstance(value, dict):
        return False
    fields = [
        value.get("dataset"),
        value.get("filename"),
        value.get("source"),
        value.get("id"),
    ]
    return any("ess_detection" in str(field) for field in fields if field is not None)


def _strip_detection_text(value: str) -> str:
    if "ess_detection" not in value:
        return value
    parts = value.split(";")
    if len(parts) > 1:
        return ";".join(part for part in parts if "ess_detection" not in part)
    return ""


def _strip_detection_entries(value):
    if isinstance(value, list):
        return [
            _strip_detection_entries(item)
            for item in value
            if not _is_detection_entry(item)
        ]
    if isinstance(value, dict):
        return {
            key: _strip_detection_entries(item)
            for key, item in value.items()
            if "ess_detection" not in str(key)
        }
    if isinstance(value, str):
        return _strip_detection_text(value)
    return value


def _copy_data_tree_without_detection(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        if _is_detection_asset(path):
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            payload = _strip_detection_entries(json.loads(path.read_text(encoding="utf-8")))
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        else:
            shutil.copy2(path, destination)


def _directory_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return file_count, total_bytes


def _remove_detection_assets(path: Path) -> None:
    for item in path.rglob("*ess_detection*.json"):
        if item.is_file():
            item.unlink()


def _rewrite_native_sync_manifest_without_detection(path: Path) -> None:
    sync_manifest_path = path / "SYNC_MANIFEST.json"
    if not sync_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing native dashboard sync manifest: {sync_manifest_path}"
        )
    sync_manifest = json.loads(
        sync_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(sync_manifest, dict):
        raise ValueError("native dashboard sync manifest must be an object")
    copied_files = sync_manifest.get("copied_files")
    if not isinstance(copied_files, list):
        raise ValueError("native dashboard sync manifest has no copied_files")
    rewritten = {
        **sync_manifest,
        "copied_files": [
            filename
            for filename in copied_files
            if "ess_detection" not in str(filename)
        ],
        "excluded_detection_assets": True,
    }
    payload = json.dumps(rewritten, ensure_ascii=False, indent=2)

    def validate(temporary: Path) -> None:
        if json.loads(temporary.read_text(encoding="utf-8")) != rewritten:
            raise ValueError("native dashboard sync manifest rewrite failed")

    publish_text_file_atomic(
        sync_manifest_path, payload, validate=validate
    )


def _validate_native_dashboard(path: Path) -> None:
    index = path / "index.html"
    data = path / "data"
    manifest_path = data / "manifest.json"
    sync_manifest_path = path / "SYNC_MANIFEST.json"
    if not index.is_file():
        raise FileNotFoundError(f"Missing native dashboard index.html: {index}")
    index.read_text(encoding="utf-8")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing native dashboard manifest: {manifest_path}")
    if not sync_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing native dashboard sync manifest: {sync_manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenarios = manifest.get("scenarios") if isinstance(manifest, dict) else None
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("native dashboard manifest has no scenarios")
    required = [
        resolve_safe_publication_asset(
            data, "overview.json", label="native dashboard asset path"
        )
    ]
    for item in scenarios:
        if not isinstance(item, dict):
            raise ValueError("native dashboard scenario must be an object")
        name = item.get("mode") or item.get("id")
        if not name:
            raise ValueError("native dashboard scenario has no mode or id")
        required.append(
            resolve_safe_publication_asset(
                data,
                f"{name}.json",
                label="native dashboard scenario path",
            )
        )
    for asset in required:
        if not asset.is_file():
            raise FileNotFoundError(f"Missing native dashboard asset: {asset}")
        json.loads(asset.read_text(encoding="utf-8"))

    sync_manifest = json.loads(sync_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(sync_manifest, dict):
        raise ValueError("native dashboard sync manifest must be an object")
    scenario_count = sync_manifest.get("scenario_count")
    if (
        isinstance(scenario_count, bool)
        or not isinstance(scenario_count, int)
        or scenario_count <= 0
    ):
        raise ValueError(
            "native dashboard sync manifest has invalid scenario_count"
        )
    copied_files = sync_manifest.get("copied_files")
    if not isinstance(copied_files, list) or not copied_files:
        raise ValueError(
            "native dashboard sync manifest has no copied_files"
        )
    for filename in copied_files:
        asset = resolve_safe_publication_asset(
            data,
            str(filename),
            label="synced native dashboard asset path",
        )
        if asset.suffix.casefold() != ".json":
            raise ValueError(
                f"Synced native dashboard asset is not JSON: {asset}"
            )
        if not asset.is_file():
            raise FileNotFoundError(f"Missing synced native dashboard asset: {asset}")
        json.loads(asset.read_text(encoding="utf-8"))


def _native_dashboard_is_available(path: Path) -> bool:
    if not path.is_dir():
        return False
    generated_entries = [item for item in path.iterdir() if item.name.lower() != "readme.md"]
    if not generated_entries:
        return False
    _validate_native_dashboard(path)
    return True


def package_portable_frontend(app_dir: Path = DEFAULT_APP_DIR, output: Path = DEFAULT_OUTPUT) -> dict:
    app_dir = Path(app_dir)
    output = Path(output)
    native_dashboard = resolve_published_directory(
        app_dir / "battery-dashboard"
    )

    if not (app_dir / "index.html").is_file():
        raise FileNotFoundError(f"Missing frontend index.html: {app_dir / 'index.html'}")
    native_dashboard_included = _native_dashboard_is_available(native_dashboard)
    result: dict = {}

    def build(staging: Path) -> None:
        staging.mkdir(parents=True)
        copied_assets: list[str] = []
        for name in sorted(APP_ASSET_NAMES):
            src = app_dir / name
            if src.is_file():
                _copy_file(src, staging / name)
                copied_assets.append(name)

        data_source = app_dir / "data"
        if data_source.is_dir():
            _copy_data_tree_without_detection(
                resolve_published_directory(data_source), staging / "data"
            )
            copied_assets.append("data/")

        excluded_detection_assets = True
        soh_curve_count = 0
        if native_dashboard_included:
            staged_dashboard = staging / "battery-dashboard"
            _copy_tree(native_dashboard, staged_dashboard)
            _remove_detection_assets(staged_dashboard / "data")
            _rewrite_native_sync_manifest_without_detection(
                staged_dashboard
            )
            copied_assets.append("battery-dashboard/")
            _validate_native_dashboard(staged_dashboard)

            native_data = staged_dashboard / "data"
            detection_assets = [
                native_data / "ess_detection.json",
                native_data / "ess_detection_curves.json",
            ]
            excluded_detection_assets = not any(path.exists() for path in detection_assets)
            soh_curve_count = len(
                [path for path in native_data.glob("*_curves.json") if not path.name.startswith("ess_detection")]
            )
        file_count, total_bytes = _directory_stats(staging)
        manifest = {
            "portable": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "app_dir": str(app_dir),
            "output": str(output),
            "copied_assets": copied_assets,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "include_soh_curves": soh_curve_count > 0,
            "excluded_detection_assets": excluded_detection_assets,
            "battery_dashboard": {
                "included": native_dashboard_included,
                "sync_required": not native_dashboard_included,
                "include_soh_curves": soh_curve_count > 0,
                "soh_curve_file_count": soh_curve_count,
                "excluded_detection_assets": excluded_detection_assets,
            },
        }
        (staging / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        json.loads((staging / "MANIFEST.json").read_text(encoding="utf-8"))
        result["manifest"] = manifest

    _publish_directory(output, build)
    return result["manifest"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable directory bundle for the SOH frontend.")
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR, help="Source frontend/app directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output portable directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = package_portable_frontend(args.app_dir, args.output)
    print(
        f"Wrote {args.output} "
        f"({manifest['file_count']} files, {manifest['total_bytes'] / (1024 * 1024):.2f} MiB)"
    )


if __name__ == "__main__":
    main()
