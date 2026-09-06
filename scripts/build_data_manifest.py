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
from sohkernel.metadata import SUPERVISION_ROLES, supervision_kind  # noqa: E402
from sohbench.paths import battery_soh_root  # noqa: E402


RAW_BINDINGS = {
    "nasa": ["raw/nasa", "raw/nasa_full", "raw/nasa_full_zips", "raw/nasa_B0005_sample.mat", "raw/nasa_battery_dataset.zip"],
    "calce_cs2": ["raw/calce"],
    "hust": ["raw/hust"],
    "oxford": ["raw/oxford"],
    "snl_lfp": ["raw/snl_batterylife"],
    "snl_nmc": ["raw/snl_batterylife"],
    "snl_nca": ["raw/snl_batterylife"],
    "uconn": ["raw/uconn_lfp"],
    "tongji_nca": ["raw/tongji"],
    "tongji_ncm": ["raw/tongji"],
    "tongji_blend": ["raw/tongji"],
    "forklift_lfp180": ["raw/forklift_lfp180"],
    "matr": ["raw/matr"],
    "home_storage_lfp": ["raw/home_storage_lfp"],
    "turku_rc_nca": ["raw/turku_rc"],
    "turku_rc_nmc": ["raw/turku_rc"],
    "turku_rc_lfp": ["raw/turku_rc"],
    "field_lfp_28": ["raw/field_lfp_28"],
}

LICENSES = {
    "nasa": {"license": "public", "source": "NASA PCoE Li-ion Battery Aging Datasets"},
    "calce_cs2": {"license": "academic-use", "source": "CALCE Battery Data"},
    "hust": {"license": "CC-BY-4.0", "source": "Ma et al. 2022 / Mendeley Data"},
    "oxford": {"license": "ODC-ODbL", "source": "Oxford Battery Degradation Dataset 1"},
    "snl_lfp": {"license": "CC-BY-4.0", "source": "BatteryLife / BatteryArchive / Preger 2020"},
    "snl_nmc": {"license": "VERIFY (BatteryArchive study page states no license)", "source": "BatteryLife / BatteryArchive / Preger 2020 (NMC subset)"},
    "snl_nca": {"license": "VERIFY (BatteryArchive study page states no license)", "source": "BatteryLife / BatteryArchive / Preger 2020 (NCA subset)"},
    "uconn": {"license": "CC-BY-4.0", "source": "UConn-ISU-ILCC / Nowacki et al. 2025"},
    "rwth_isea_public": {"license": "CC-BY-4.0", "source": "RWTH ISEA public battery data"},
    "tongji_nca": {"license": "CC-BY-4.0", "source": "Zhu et al. 2022 / Zenodo 6405084 (Dataset_1 NCA)"},
    "tongji_ncm": {"license": "CC-BY-4.0", "source": "Zhu et al. 2022 / Zenodo 6405084 (Dataset_2 NCM)"},
    "tongji_blend": {"license": "CC-BY-4.0", "source": "Zhu et al. 2022 / Zenodo 6405084 (Dataset_3 NCM+NCA)"},
    "forklift_lfp180": {"license": "CC-BY-4.0", "source": "Vilsen & Stroe 2024 / Mendeley Data yz4pttm73n v2"},
    "matr": {"license": "cite-the-paper (data.matr.io terms; VERIFY)", "source": "Severson et al. 2019 + Attia et al. 2020 / data.matr.io (4 batches)"},
    "home_storage_lfp": {"license": "CC-BY-4.0", "source": "Figgener et al. 2024 / Zenodo 12091223 (LFP systems 14-21)"},
    "turku_rc_nca": {"license": "CC-BY-NC-3.0 (FairData record; the article is CC BY 4.0)", "source": "Heydarzadeh et al. 2025 / FairData 10.23729/b07462e8-9e31-428c-bdd5-deb1890c608b (NCA cells)"},
    "turku_rc_nmc": {"license": "CC-BY-NC-3.0 (FairData record; the article is CC BY 4.0)", "source": "Heydarzadeh et al. 2025 / FairData 10.23729/b07462e8-9e31-428c-bdd5-deb1890c608b (NMC cells)"},
    "turku_rc_lfp": {"license": "CC-BY-NC-3.0 (FairData record; the article is CC BY 4.0)", "source": "Heydarzadeh et al. 2025 / FairData 10.23729/b07462e8-9e31-428c-bdd5-deb1890c608b (LFP cells)"},
    "field_lfp_28": {"license": "CC-BY-NC-4.0", "source": "Schaeffer et al. 2024 / Zenodo 13715694 (28 LFP systems, unlabeled)"},
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
    kind = supervision_kind(name)
    labeled = int(pd.to_numeric(frame["soh"], errors="coerce").notna().sum()) if "soh" in frame else 0
    return {
        "name": name,
        "status": "ready" if not missing else "blocked",
        "supervision": kind if not missing else "unknown",   # Spec 12 §4 (K13, 2026-09-05)
        "roles": list(SUPERVISION_ROLES[kind]) if not missing else [],
        "labeled_record_count": labeled,
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
        "held_out_ready": battery_count > 1 and kind != "unlabeled_field",
        "raw_paths": raw_paths,
        "raw_local": bool(raw_paths) and all((data_root / Path(item)).exists() for item in raw_paths),
        "curve_path": str(curve_path.relative_to(data_root)).replace("\\", "/") if curve_path.is_file() else "",
        "curve_sha256": sha256(curve_path) if curve_path.is_file() else "",
        **license_info,
    }


def update_entries(data_root: Path, names: list[str], partial_windows: dict[str, list[float]]) -> dict:
    """Recompute the named dataset entries inside the existing manifest (no source-root verification).

    Used when a processed table is (re)built on a machine without the migration source root
    (D1, 2026-09-04): the entry fields are the same ``dataset_entry`` fields the full build writes;
    ``partial_window_v`` is carried over from the previous entry unless given explicitly, and the
    summary counters are re-summed.
    """
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {item["name"]: item for item in manifest.get("datasets", [])}
    for name in names:
        path = data_root / "processed" / f"{name}_features.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = dataset_entry(path, data_root)
        previous = entries.get(name, {})
        window = partial_windows.get(name) or (previous.get("processed") or {}).get("partial_window_v")
        if window:
            entry["processed"]["partial_window_v"] = [float(window[0]), float(window[1])]
        entries[name] = entry
    manifest["datasets"] = [entries[name] for name in sorted(entries)]
    manifest["summary"].update({
        "processed_datasets": len(manifest["datasets"]),
        "records": sum(item["record_count"] for item in manifest["datasets"]),
        "cycles": sum(item["cycle_count"] for item in manifest["datasets"]),
        "batteries": sum(item["battery_count"] for item in manifest["datasets"]),
    })
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--source-root", type=Path, default=battery_soh_root() / "data")
    parser.add_argument("--update-only", nargs="+", metavar="DATASET", default=None,
                        help="recompute only these dataset entries in the existing manifest (no source-root verification)")
    parser.add_argument("--partial-window", nargs=3, action="append", metavar=("DATASET", "LO", "HI"), default=[],
                        help="partial_window_v to record for a dataset (with --update-only)")
    args = parser.parse_args(argv)
    data_root = args.data_root.resolve()
    if args.update_only:
        windows = {name: [float(lo), float(hi)] for name, lo, hi in args.partial_window}
        manifest = update_entries(data_root, args.update_only, windows)
        updated = {item["name"]: {k: item[k] for k in ("record_count", "cycle_count", "battery_count")} | {"sha256": item["processed"]["sha256"]}
                   for item in manifest["datasets"] if item["name"] in args.update_only}
        print(json.dumps({"manifest": str(data_root / "manifest.json"), "updated": updated, "summary": manifest["summary"]}, ensure_ascii=False))
        return 0
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
