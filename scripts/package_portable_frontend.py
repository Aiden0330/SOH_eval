from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


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


def package_portable_frontend(app_dir: Path = DEFAULT_APP_DIR, output: Path = DEFAULT_OUTPUT) -> dict:
    app_dir = Path(app_dir)
    output = Path(output)
    native_dashboard = app_dir / "battery-dashboard"

    if not (app_dir / "index.html").is_file():
        raise FileNotFoundError(f"Missing frontend index.html: {app_dir / 'index.html'}")
    if not native_dashboard.is_dir():
        raise FileNotFoundError(f"Missing native battery dashboard: {native_dashboard}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copied_assets: list[str] = []
    for name in sorted(APP_ASSET_NAMES):
        src = app_dir / name
        if src.is_file():
            _copy_file(src, output / name)
            copied_assets.append(name)

    data_source = app_dir / "data"
    if data_source.is_dir():
        _copy_data_tree_without_detection(data_source, output / "data")
        copied_assets.append("data/")

    _copy_tree(native_dashboard, output / "battery-dashboard")
    _remove_detection_assets(output / "battery-dashboard" / "data")
    copied_assets.append("battery-dashboard/")

    native_data = output / "battery-dashboard" / "data"
    detection_assets = [
        native_data / "ess_detection.json",
        native_data / "ess_detection_curves.json",
    ]
    excluded_detection_assets = not any(path.exists() for path in detection_assets)
    soh_curve_count = len([path for path in native_data.glob("*_curves.json") if not path.name.startswith("ess_detection")])

    file_count, total_bytes = _directory_stats(output)
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
            "included": True,
            "include_soh_curves": soh_curve_count > 0,
            "soh_curve_file_count": soh_curve_count,
            "excluded_detection_assets": excluded_detection_assets,
        },
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


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
