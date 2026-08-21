from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sohbench.metadata import DATASET_CHEMISTRY, DATASET_C_RATED_AH, DATASET_LABEL_SOURCE  # noqa: E402


RAW_BINDINGS = {
    "nasa": ["raw/nasa", "raw/nasa_full", "raw/nasa_full_zips", "raw/nasa_B0005_sample.mat", "raw/nasa_battery_dataset.zip"],
    "calce_cs2": ["raw/calce"],
    "hust": ["raw/hust"],
    "oxford": ["raw/oxford"],
    "snl_lfp": ["raw/snl_batterylife"],
    "uconn": ["raw/uconn_lfp"],
}

LICENSES = {
    "nasa": {"license": "public", "source": "NASA PCoE Li-ion Battery Aging Datasets"},
    "calce_cs2": {"license": "academic-use", "source": "CALCE Battery Data"},
    "hust": {"license": "CC-BY-4.0", "source": "Ma et al. 2022 / Mendeley Data"},
    "oxford": {"license": "ODC-ODbL", "source": "Oxford Battery Degradation Dataset 1"},
    "snl_lfp": {"license": "CC-BY-4.0", "source": "BatteryLife / BatteryArchive / Preger 2020"},
    "uconn": {"license": "CC-BY-4.0", "source": "UConn-ISU-ILCC / Nowacki et al. 2025"},
    "rwth_isea_public": {"license": "CC-BY-4.0", "source": "RWTH ISEA public battery data"},
}

CURVE_TOKENS = {"calce_cs2": "calce", "rwth_isea_public": "rwth"}

MIGRATED_RELATIVE_PATHS = [
    "processed",
    "checksums",
    "registry",
    "raw/nasa",
    "raw/nasa_full",
    "raw/nasa_full_zips",
    "raw/calce",
    "raw/hust",
    "raw/oxford",
    "raw/snl_batterylife",
    "raw/uconn_lfp",
    "raw/nasa_B0005_sample.mat",
    "raw/nasa_battery_dataset.zip",
]

EXTERNAL_RESOURCES = [
    ("rwth_hss", "raw/rwth_hss", "field_no_supervised_soh", "RWTH 户储现场原始数据；当前不进入监督 SOH 训练"),
    ("m5bat", "raw/m5bat", "field_no_supervised_soh", "M5BAT 电站数据；当前仅用于工况画像"),
    ("real_ess", "raw/real_ESS", "field_no_supervised_soh", "ESS 现场数据；尚未接入监督标签"),
    ("isu_ilcc", "raw/isu_ilcc", "registered_not_connected", "已登记但尚未完成 adapter 与质检"),
    ("calce_a123", "raw/calce_a123", "registered_not_connected", "CALCE A123 尚未形成项目 processed 数据集"),
    ("gbrain_evals", "raw/gbrain-evals", "unrelated", "历史工程依赖，不属于电池数据资产"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file())


def inventory(path: Path) -> dict:
    files = files_under(path)
    return {"present": path.exists(), "files": len(files), "bytes": sum(item.stat().st_size for item in files)}


def verify_copy(source_root: Path, data_root: Path) -> list[dict]:
    results = []
    for relative in MIGRATED_RELATIVE_PATHS:
        source = source_root / Path(relative)
        target = data_root / Path(relative)
        source_files = files_under(source)
        target_files = files_under(target)
        source_map = {str(path.relative_to(source)) if source.is_dir() else path.name: path.stat().st_size for path in source_files}
        target_map = {str(path.relative_to(target)) if target.is_dir() else path.name: path.stat().st_size for path in target_files}
        results.append({
            "path": relative.replace("\\", "/"),
            "source_files": len(source_files),
            "target_files": len(target_files),
            "bytes": sum(source_map.values()),
            "inventory_match": source_map == target_map,
        })
    return results


def dataset_entry(path: Path, data_root: Path) -> dict:
    name = path.name.removesuffix("_features.pkl")
    frame = pd.read_pickle(path)
    required = {"dataset", "cell_id", "cycle_index", "soh"}
    missing = sorted(required - set(frame.columns))
    cycle_count = int(frame[["cell_id", "cycle_index"]].dropna().drop_duplicates().shape[0]) if not missing else 0
    battery_count = int(frame["cell_id"].dropna().astype(str).nunique()) if "cell_id" in frame else 0
    capacity = None
    capacity_source = "unknown"
    if "capacity_ah" in frame:
        values = pd.to_numeric(frame["capacity_ah"], errors="coerce")
        values = values[np.isfinite(values) & (values > 0)]
        if not values.empty:
            capacity = float(values.max())
            capacity_source = "measured_capacity"
    if capacity is None:
        rated = DATASET_C_RATED_AH.get(name)
        if rated is not None and np.isfinite(rated) and float(rated) > 0:
            capacity = float(rated)
            capacity_source = "rated_metadata"
    token = CURVE_TOKENS.get(name, name)
    curve_path = data_root / "curves" / f"{token}_curves.json"
    raw_paths = RAW_BINDINGS.get(name, [])
    license_info = LICENSES.get(name, {"license": "internal-demo" if "demo" in name else "unknown", "source": "generated or pending registration"})
    return {
        "name": name,
        "status": "ready" if not missing else "blocked",
        "supervision": "soh_supervised" if not missing else "unknown",
        "roles": ["training", "same_dataset_holdout", "inference", "audit"] if not missing else [],
        "processed": {
            "path": str(path.relative_to(data_root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "required_columns_missing": missing,
            "columns": list(map(str, frame.columns)),
        },
        "record_count": int(len(frame)),
        "cycle_count": cycle_count,
        "battery_count": battery_count,
        "chemistry": str(DATASET_CHEMISTRY.get(name) or "unknown"),
        "max_capacity_ah": capacity,
        "capacity_source": capacity_source,
        "label_source": str(DATASET_LABEL_SOURCE.get(name) or "unknown"),
        "held_out_ready": battery_count > 1,
        "raw_paths": raw_paths,
        "raw_local": bool(raw_paths) and all((data_root / Path(item)).exists() for item in raw_paths),
        "curve_path": str(curve_path.relative_to(data_root)).replace("\\", "/") if curve_path.is_file() else "",
        "curve_sha256": sha256(curve_path) if curve_path.is_file() else "",
        **license_info,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\battery_SOH\data"))
    args = parser.parse_args(argv)
    data_root = args.data_root.resolve()
    source_root = args.source_root.resolve()
    datasets = [dataset_entry(path, data_root) for path in sorted((data_root / "processed").glob("*_features.pkl"))]
    verification = verify_copy(source_root, data_root)
    external = []
    for name, relative, status, reason in EXTERNAL_RESOURCES:
        details = inventory(source_root / Path(relative))
        external.append({"name": name, "source_path": str(source_root / Path(relative)), "status": status, "reason": reason, **details})
    all_files = [item for item in data_root.rglob("*") if item.is_file() and item.name != "manifest.json"]
    manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "migration_strategy": "B-curated-supervised",
        "source_root": str(source_root),
        "data_root": str(data_root),
        "summary": {
            "local_files": len(all_files),
            "local_bytes": sum(item.stat().st_size for item in all_files),
            "processed_datasets": len(datasets),
            "records": sum(item["record_count"] for item in datasets),
            "cycles": sum(item["cycle_count"] for item in datasets),
            "batteries": sum(item["battery_count"] for item in datasets),
        },
        "verification": {
            "inventory_checks": verification,
            "all_inventory_checks_passed": all(item["inventory_match"] for item in verification),
            "runtime_artifacts_sha256_recorded": True,
        },
        "datasets": datasets,
        "external_resources": external,
        "exclusions": [
            "source root duplicate archives",
            "SOHbenchmark code archive",
            "gbrain-evals historical engineering dependencies",
            "RWTH HSS and other field data without supervised SOH labels",
        ],
    }
    output = data_root / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "summary": manifest["summary"], "verified": manifest["verification"]["all_inventory_checks_passed"]}, ensure_ascii=False))
    return 0 if manifest["verification"]["all_inventory_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
