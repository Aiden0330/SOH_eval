from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

from .frontend_publish import publish_frontend_profile


ARTIFACT_PREFIX = "battery_dashboard_"
ALGORITHM_DESIGN_ARTIFACT = "battery_algorithm_design.json"
MODEL_COMPARISONS_ARTIFACT = "battery_model_comparisons.json"
FEATURE_WINDOWS_ARTIFACT = "battery_feature_windows.json"
CURVE_PREVIEWS_ARTIFACT = "battery_curve_previews.json"
MAX_PREVIEW_CURVES_PER_DATASET = 3
MAX_PREVIEW_POINTS_PER_CURVE = 80
KNOWN_SOH_FEATURES = ("q_win_ah", "t_win_s", "ic_peak_ah_per_v", "ic_peak_voltage_v")
FORBIDDEN_FEATURE_HINTS = ("DCIR", "RPT 脉冲", "cycle_fraction")
EXCLUDED_CAPABILITIES = ("ess_detection",)


def stage_battery_dashboard_artifacts(
    source_dir: Path,
    target_dir: Path,
    include_curves: bool = False,
) -> list[str]:
    source = Path(source_dir)
    target = Path(target_dir)
    if not source.exists():
        raise FileNotFoundError(f"battery dashboard source directory not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"battery dashboard source path is not a directory: {source}")

    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(source.glob("*.json"), key=lambda item: item.name):
        if _is_excluded_dashboard_asset(path):
            continue
        if path.name.endswith("_curves.json") and not include_curves:
            continue
        filename = _dashboard_artifact_name(path.name)
        shutil.copy2(path, target / filename)
        copied.append(filename)
    return sorted(copied)


def publish_battery_dashboard_profile(
    frontend_data_dir: Path,
    source_dir: Path,
    include_curves: bool = False,
    profile_name: str = "battery-soh-diagnostics",
) -> dict:
    with tempfile.TemporaryDirectory(prefix="battery-dashboard-") as tmp:
        staged_dir = Path(tmp)
        artifacts = stage_battery_dashboard_artifacts(source_dir, staged_dir, include_curves=include_curves)
        optional_artifacts, optional_sources = stage_battery_optional_assets(Path(source_dir), staged_dir)
        preview_artifacts, preview_sources = stage_battery_curve_previews(Path(source_dir), staged_dir)
        optional_artifacts = sorted([*optional_artifacts, *preview_artifacts])
        optional_sources = {**optional_sources, **preview_sources}
        curve_artifacts = _dashboard_curve_artifacts(Path(source_dir), include_curves=include_curves)
        manifest = {
            "source": "battery_SOH_dashboard",
            "kind": "battery_soh_diagnostics",
            "scope": {
                "soh_evaluation": True,
                "ess_detection": False,
            },
            "excluded_capabilities": list(EXCLUDED_CAPABILITIES),
            "include_curves": bool(include_curves),
            "diagnostic_artifacts": artifacts,
            "optional_artifacts": optional_artifacts,
            "curve_artifacts": curve_artifacts,
            "source_artifacts": {
                **_dashboard_source_artifacts(Path(source_dir), include_curves=include_curves),
                **optional_sources,
            },
        }
        result = publish_frontend_profile(
            frontend_data_dir,
            profile_name,
            manifest,
            source_dirs=[staged_dir],
            artifact_files=[*artifacts, *optional_artifacts],
        )
        _remove_excluded_profile_assets(Path(result["path"]))
        return result


def stage_battery_optional_assets(source_dir: Path, target_dir: Path) -> tuple[list[str], dict[str, str]]:
    root = _battery_root_from_dashboard_source(Path(source_dir))
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    sources: dict[str, str] = {}

    algorithm_source = root / "docs" / "design" / "soh-algorithm-design.md"
    algorithm_payload = _algorithm_design_payload(algorithm_source)
    if algorithm_payload:
        _write_json(target / ALGORITHM_DESIGN_ARTIFACT, algorithm_payload)
        copied.append(ALGORITHM_DESIGN_ARTIFACT)
        sources[ALGORITHM_DESIGN_ARTIFACT] = str(algorithm_source)

    feature_source = root / "experiments" / "reports" / "phase4_voltage_feature_windows.json"
    feature_payload = _feature_windows_payload(feature_source)
    if feature_payload:
        _write_json(target / FEATURE_WINDOWS_ARTIFACT, feature_payload)
        copied.append(FEATURE_WINDOWS_ARTIFACT)
        sources[FEATURE_WINDOWS_ARTIFACT] = str(feature_source)

    comparison_payload, comparison_sources = _model_comparisons_payload(root / "experiments" / "reports")
    if comparison_payload:
        _write_json(target / MODEL_COMPARISONS_ARTIFACT, comparison_payload)
        copied.append(MODEL_COMPARISONS_ARTIFACT)
        sources[MODEL_COMPARISONS_ARTIFACT] = ";".join(str(item) for item in comparison_sources)

    return sorted(copied), sources


def stage_battery_curve_previews(source_dir: Path, target_dir: Path) -> tuple[list[str], dict[str, str]]:
    payload = _curve_previews_payload(Path(source_dir))
    if not payload:
        return [], {}
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / CURVE_PREVIEWS_ARTIFACT, payload)
    sources = [
        dataset["source_file"]
        for dataset in payload.get("datasets", [])
        if dataset.get("source_file")
    ]
    return [CURVE_PREVIEWS_ARTIFACT], {CURVE_PREVIEWS_ARTIFACT: ";".join(sources)}


def _dashboard_artifact_name(filename: str) -> str:
    return filename if filename.startswith(ARTIFACT_PREFIX) else f"{ARTIFACT_PREFIX}{filename}"


def _is_excluded_dashboard_asset(path: Path) -> bool:
    return Path(path).name.lower().startswith("ess_detection")


def _remove_excluded_profile_assets(profile_dir: Path) -> None:
    for path in Path(profile_dir).glob(f"{ARTIFACT_PREFIX}ess_detection*.json"):
        if path.is_file():
            path.unlink()


def _dashboard_source_artifacts(source_dir: Path, include_curves: bool = False) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(Path(source_dir).glob("*.json"), key=lambda item: item.name):
        if _is_excluded_dashboard_asset(path):
            continue
        if path.name.endswith("_curves.json") and not include_curves:
            continue
        mapping[_dashboard_artifact_name(path.name)] = str(path)
    return mapping


def _dashboard_curve_artifacts(source_dir: Path, include_curves: bool = False) -> list[dict]:
    artifacts = []
    for path in sorted(Path(source_dir).glob("*_curves.json"), key=lambda item: item.name):
        if _is_excluded_dashboard_asset(path):
            continue
        filename = _dashboard_artifact_name(path.name)
        dataset = path.name.removesuffix("_curves.json")
        artifacts.append(
            {
                "dataset": dataset,
                "filename": filename,
                "source": str(path),
                "bytes": path.stat().st_size,
                "copied": bool(include_curves),
            }
        )
    return artifacts


def _battery_root_from_dashboard_source(source_dir: Path) -> Path:
    source = Path(source_dir)
    if source.name == "data" and source.parent.name == "dashboard":
        return source.parent.parent
    return source.parent


def _algorithm_design_payload(source: Path) -> dict | None:
    if not source.exists():
        return None
    text = source.read_text(encoding="utf-8", errors="ignore")
    title = next((line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#")), "SOH 算法设计说明书")
    features = [name for name in KNOWN_SOH_FEATURES if name in text]
    forbidden = [name for name in FORBIDDEN_FEATURE_HINTS if name in text]
    return {
        "schema_version": 1,
        "title": title,
        "task": "SOH 点估计 + 95% CI",
        "model_family": "TabularSOHEnsemble" if "TabularSOHEnsemble" in text else "",
        "model_structure": ["ridge", "hist_gradient_boosting", "random_forest"],
        "feature_policy": {
            "allowed": ["容量侧充电窗特征"],
            "forbidden": forbidden,
        },
        "features": features,
        "source_file": str(source),
    }


def _model_comparisons_payload(reports_dir: Path) -> tuple[dict | None, list[Path]]:
    specs = [
        ("phase2_tabular_reproduction_metrics.json", "phase2_tabular_reproduction", "XJTU 固定结构复现"),
        ("phase2_tabular_adaptation_metrics.json", "phase2_tabular_adaptation", "跨数据集接口适配"),
        ("phase3_xjtu_r2_comparison.json", "phase3_xjtu_r2_comparison", "XJTU R2 对比"),
    ]
    comparisons = []
    sources = []
    for filename, item_id, title in specs:
        source = Path(reports_dir) / filename
        payload = _read_json(source)
        if payload is None:
            continue
        sources.append(source)
        comparisons.append(
            {
                "id": item_id,
                "title": title,
                "scope": payload.get("validation_scope") or payload.get("scope") or payload.get("goal") or "",
                "goal": payload.get("goal") or "",
                "metrics": payload.get("metrics") or {
                    key: payload[key]
                    for key in ("mean_r2", "mean_mae_percent")
                    if key in payload
                },
                "fixed_structure": payload.get("fixed_structure") or [],
                "domains": payload.get("adaptation_domains") or [],
                "rows": payload.get("metrics_by_cell") or [],
                "source_file": str(source),
            }
        )
    if not comparisons:
        return None, []
    return {"schema_version": 1, "comparisons": comparisons}, sources


def _feature_windows_payload(source: Path) -> dict | None:
    payload = _read_json(source)
    if payload is None:
        return None
    plots = payload.get("plots") if isinstance(payload.get("plots"), dict) else {}
    windows = []
    for dataset, item in sorted(plots.items()):
        windows.append(
            {
                "dataset": str(dataset),
                "feature_voltage_window": item.get("feature_voltage_window") or [],
                "rows": item.get("rows"),
                "curves": item.get("curves"),
                "source_plot": item.get("path") or "",
            }
        )
    return {"schema_version": 1, "windows": windows}


def _curve_previews_payload(source_dir: Path) -> dict | None:
    datasets = []
    for path in sorted(Path(source_dir).glob("*_curves.json"), key=lambda item: item.name):
        if _is_excluded_dashboard_asset(path):
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        preview = _dataset_curve_preview(path.name.removesuffix("_curves.json"), path, payload)
        if preview:
            datasets.append(preview)
    if not datasets:
        return None
    return {
        "schema_version": 1,
        "max_curves_per_dataset": MAX_PREVIEW_CURVES_PER_DATASET,
        "max_points_per_curve": MAX_PREVIEW_POINTS_PER_CURVE,
        "datasets": datasets,
    }


def _dataset_curve_preview(dataset: str, source: Path, payload: dict) -> dict | None:
    return _charge_curve_preview(dataset, source, payload)


def _charge_curve_preview(dataset: str, source: Path, payload: dict) -> dict | None:
    preview_curves = []
    for curve_id, curve in payload.items():
        if not isinstance(curve, dict):
            continue
        x_values = _numeric_list(curve.get("t_s"))
        series = []
        for key in ("voltage_v", "current_a", "temperature_c"):
            points = _sample_xy_points(x_values, _numeric_list(curve.get(key)), MAX_PREVIEW_POINTS_PER_CURVE)
            if not points:
                continue
            series.append(_curve_series(key, points))
        if not series:
            continue
        preview_curves.append(
            {
                "id": str(curve_id),
                "point_count": max(len(x_values), *(len(item["points"]) for item in series)),
                "sampled_point_count": max(len(item["points"]) for item in series),
                "x_key": "t_s",
                "series": series,
            }
        )
        if len(preview_curves) >= MAX_PREVIEW_CURVES_PER_DATASET:
            break
    if not preview_curves:
        return None
    return {
        "dataset": dataset,
        "kind": "charge_curve",
        "source_file": str(source),
        "total_curves": len(payload),
        "preview_curves": preview_curves,
    }


def _curve_series(key: str, points: list[list[float]], label: str | None = None) -> dict:
    return {
        "key": key,
        "label": label or _curve_series_label(key),
        "min": min(point[1] for point in points),
        "max": max(point[1] for point in points),
        "points": points,
    }


def _numeric_list(values) -> list[float]:
    if not isinstance(values, list):
        return []
    numbers = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def _sample_xy_points(x_values: list[float], y_values: list[float], max_points: int) -> list[list[float]]:
    if not y_values:
        return []
    if not x_values:
        x_values = [float(index) for index in range(len(y_values))]
    count = min(len(x_values), len(y_values))
    if count <= 0:
        return []
    if count <= max_points:
        indexes = range(count)
    else:
        indexes = sorted({round(index * (count - 1) / (max_points - 1)) for index in range(max_points)})
    return [[round(x_values[index], 6), round(y_values[index], 6)] for index in indexes]


def _curve_series_label(key: str) -> str:
    return {
        "voltage_v": "Voltage",
        "current_a": "Current",
        "temperature_c": "Temperature",
    }.get(key, key)


def _read_json(source: Path) -> dict | None:
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
