from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE_DASHBOARD = ROOT / "frontend" / "app" / "battery-dashboard"
NATIVE_DATA = NATIVE_DASHBOARD / "data"


def test_native_battery_dashboard_assets_are_vendored() -> None:
    assert (NATIVE_DASHBOARD / "index.html").is_file()
    assert NATIVE_DATA.is_dir()


def test_native_battery_dashboard_manifest_scenarios_have_data_and_curves() -> None:
    manifest_path = NATIVE_DATA / "manifest.json"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenarios = manifest.get("scenarios") or []
    assert scenarios
    scenario_names = {scenario.get("mode") or scenario.get("id") for scenario in scenarios}
    assert "ess_detection" not in scenario_names

    for scenario in scenarios:
        name = scenario.get("mode") or scenario.get("id")
        assert name
        assert (NATIVE_DATA / f"{name}.json").is_file()
        assert (NATIVE_DATA / f"{name}_curves.json").is_file()

    curve_datasets = {path.name.removesuffix("_curves.json") for path in NATIVE_DATA.glob("*_curves.json")}
    assert curve_datasets == scenario_names


def test_native_battery_dashboard_detection_assets_are_excluded_by_default() -> None:
    assert not (NATIVE_DATA / "ess_detection.json").exists()
    assert not (NATIVE_DATA / "ess_detection_curves.json").exists()
