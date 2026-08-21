from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_source_dashboard(root: Path) -> Path:
    dashboard = root / "dashboard"
    data = dashboard / "data"
    data.mkdir(parents=True)
    (dashboard / "index.html").write_text(
        """
<!doctype html>
<html><head><style>.modetabs{display:block}</style></head>
<body>
<span class="modetabs"><button data-mode="soh">SOH</button><button data-mode="detection">Detection</button></span>
<canvas id="soh"></canvas>
<script>
"use strict";
const $ = id => document.getElementById(id);
(async () => {
  state.manifest = await fetchJSON("data/manifest.json");
  await setMode("soh");
})();
</script>
</body></html>
""".strip(),
        encoding="utf-8",
    )
    _write_json(
        data / "manifest.json",
        {
            "scenarios": [
                {"mode": "xjtu", "title": "XJTU"},
                {"mode": "calce", "title": "CALCE"},
            ]
        },
    )
    _write_json(data / "overview.json", {"scenarios": []})
    _write_json(data / "xjtu.json", {"points": []})
    _write_json(data / "xjtu_curves.json", {"cell-1|1": {"t_s": [0], "voltage_v": [4.1]}})
    _write_json(data / "calce.json", {"points": []})
    _write_json(data / "calce_curves.json", {"cell-2|1": {"t_s": [0], "voltage_v": [4.0]}})
    _write_json(data / "ess_detection.json", {"fleet": []})
    _write_json(data / "ess_detection_curves.json", {"module": {}})
    return dashboard


def test_sync_native_dashboard_excludes_detection_and_keeps_soh_curves(tmp_path: Path) -> None:
    source = _write_source_dashboard(tmp_path / "source")
    target = tmp_path / "frontend" / "app" / "battery-dashboard"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sync_battery_dashboard_native.py"),
            "--source",
            str(source),
            "--target",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (target / "index.html").is_file()
    assert (target / "data" / "manifest.json").is_file()
    assert (target / "data" / "overview.json").is_file()
    assert (target / "data" / "xjtu.json").is_file()
    assert (target / "data" / "xjtu_curves.json").is_file()
    assert (target / "data" / "calce.json").is_file()
    assert (target / "data" / "calce_curves.json").is_file()
    assert not (target / "data" / "ess_detection.json").exists()
    assert not (target / "data" / "ess_detection_curves.json").exists()

    html = (target / "index.html").read_text(encoding="utf-8")
    assert "URLSearchParams(location.search)" in html
    assert "setMode(initialMode)" in html
    assert "body.embedded .modetabs" in html
    assert 'params.get("dataset")' in html
    assert 'id="dataset"' in html
    assert 'id="algorithm"' in html
    assert 'id="featureset"' in html
    assert "function applyLinkedContext" in html
    assert "const text = [state.linkedContext.model, state.linkedContext.featureset]" in html
    assert "state.linkedContext.dataset = canonicalDataset(e.target.value);\n  const next = scenarioForDataset(e.target.value);" in html
    assert "rwth_isea_public" in html


def test_package_portable_frontend_copies_soh_native_dashboard_without_detection(tmp_path: Path) -> None:
    source = _write_source_dashboard(tmp_path / "source")
    native_target = tmp_path / "app" / "battery-dashboard"
    sync = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sync_battery_dashboard_native.py"),
            "--source",
            str(source),
            "--target",
            str(native_target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert sync.returncode == 0, sync.stderr + sync.stdout
    _write_json(native_target / "data" / "ess_detection.json", {"fleet": []})
    _write_json(native_target / "data" / "ess_detection_curves.json", {"module": {}})

    app_source = tmp_path / "app"
    (app_source / "index.html").write_text("<html><body>SOH</body></html>", encoding="utf-8")
    (app_source / "app.js").write_text("console.log('app')", encoding="utf-8")
    (app_source / "styles.css").write_text("body{}", encoding="utf-8")
    _write_json(app_source / "data" / "leaderboard.json", [])
    profile = app_source / "data" / "profiles" / "battery-soh-diagnostics"
    _write_json(profile / "battery_dashboard_ess_detection.json", {"dataset": "ess_detection"})
    _write_json(
        profile / "battery_curve_previews.json",
        {
            "datasets": [
                {"dataset": "ess_detection", "preview_curves": [{"id": "ess"}]},
                {"dataset": "xjtu", "preview_curves": [{"id": "xjtu"}]},
            ]
        },
    )
    _write_json(
        profile / "manifest.json",
        {
            "artifact_files": [
                "battery_dashboard_summary.json",
                "battery_dashboard_ess_detection.json",
            ],
            "source_artifacts": {
                "battery_curve_previews.json": (
                    "D:\\battery_SOH\\dashboard\\data\\xjtu_curves.json;"
                    "D:\\battery_SOH\\dashboard\\data\\ess_detection_curves.json"
                )
            },
            "diagnostic_artifacts": [
                "battery_dashboard_summary.json",
                "battery_dashboard_ess_detection.json",
            ],
        },
    )

    output = tmp_path / "dist" / "soh-eval-portable"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_portable_frontend.py"),
            "--app-dir",
            str(app_source),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (output / "index.html").is_file()
    assert (output / "battery-dashboard" / "index.html").is_file()
    assert (output / "battery-dashboard" / "data" / "xjtu_curves.json").is_file()
    assert not (output / "battery-dashboard" / "data" / "ess_detection.json").exists()
    assert not (output / "battery-dashboard" / "data" / "ess_detection_curves.json").exists()
    assert not (output / "data" / "profiles" / "battery-soh-diagnostics" / "battery_dashboard_ess_detection.json").exists()
    preview = json.loads(
        (output / "data" / "profiles" / "battery-soh-diagnostics" / "battery_curve_previews.json").read_text(
            encoding="utf-8"
        )
    )
    assert {item["dataset"] for item in preview["datasets"]} == {"xjtu"}
    profile_manifest = json.loads(
        (output / "data" / "profiles" / "battery-soh-diagnostics" / "manifest.json").read_text(encoding="utf-8")
    )
    assert "ess_detection" not in json.dumps(profile_manifest, ensure_ascii=False)

    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["portable"] is True
    assert manifest["include_soh_curves"] is True
    assert manifest["excluded_detection_assets"] is True
    assert manifest["battery_dashboard"]["included"] is True
    assert manifest["battery_dashboard"]["include_soh_curves"] is True
    assert manifest["battery_dashboard"]["excluded_detection_assets"] is True
    assert manifest["file_count"] >= 5
    assert manifest["total_bytes"] > 0
