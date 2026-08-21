import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from sohbench.api_server import create_handler
from sohbench.battery_dashboard_import import (
    publish_battery_dashboard_profile,
    stage_battery_dashboard_artifacts,
)
from sohbench.evaluation_service import EvaluationService


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_dashboard_dir(tmp_path):
    source = tmp_path / "dashboard" / "data"
    _write_json(source / "manifest.json", {"source_manifest": True})
    _write_json(source / "overview.json", {"overview": True})
    _write_json(source / "xjtu.json", {"scenario": "xjtu"})
    _write_json(source / "ess_detection.json", {"scenario": "ess_detection"})
    _write_json(source / "xjtu_curves.json", {"curve": "xjtu"})
    _write_json(source / "ess_detection_curves.json", {"curve": "ess"})
    (source / "notes.txt").write_text("not dashboard json", encoding="utf-8")
    return source


def _write_optional_battery_assets(tmp_path):
    design = tmp_path / "docs" / "design"
    design.mkdir(parents=True, exist_ok=True)
    (design / "soh-algorithm-design.md").write_text(
        "# SOH 算法设计说明书\n\n"
        "点估计器固定为 `TabularSOHEnsemble`。\n"
        "特征包括 `q_win_ah`、`t_win_s`、`ic_peak_ah_per_v`、`ic_peak_voltage_v`。\n"
        "禁内阻类 DCIR/RPT 脉冲，禁未来信息 cycle_fraction。\n",
        encoding="utf-8",
    )
    reports = tmp_path / "experiments" / "reports"
    _write_json(
        reports / "phase2_tabular_reproduction_metrics.json",
        {
            "goal": "fixed structure XJTU reproduction",
            "validation_scope": "real XJTU handcraft_features",
            "fixed_structure": ["ridge", "hist_gradient_boosting", "random_forest"],
            "metrics": {"mae": 0.019, "rmse": 0.02, "r2": 0.83},
        },
    )
    _write_json(
        reports / "phase2_tabular_adaptation_metrics.json",
        {
            "goal": "dataset skill adaptation",
            "adaptation_domains": ["MIT", "XJTU"],
            "metrics": {
                "MIT": {"mae": 0.048, "rmse": 0.064, "r2": -1.7},
                "XJTU": {"mae": 0.002, "rmse": 0.003, "r2": 0.99},
            },
        },
    )
    _write_json(
        reports / "phase3_xjtu_r2_comparison.json",
        {
            "scope": "XJTU batch-1",
            "mean_r2": 0.858,
            "mean_mae_percent": 1.85,
            "metrics_by_cell": [{"cell_id": "batch-1:cell-1", "r2": 0.83}],
        },
    )
    _write_json(
        reports / "phase4_voltage_feature_windows.json",
        {
            "plots": {
                "xjtu": {
                    "path": "experiments\\reports\\phase4_xjtu_voltage_feature_window.png",
                    "feature_voltage_window": [3.7, 4.1],
                    "rows": 768,
                    "curves": 6,
                },
                "nasa": {
                    "path": "experiments\\reports\\phase4_nasa_voltage_feature_window.png",
                    "feature_voltage_window": [3.7, 4.2],
                    "rows": 8250,
                    "curves": 3,
                },
            }
        },
    )


def _post_json(base_url, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _serve(service):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), create_handler(service, allow_dangerous_operations=True)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_default_staging_excludes_curves_and_prefixes_json_names(tmp_path):
    source = _source_dashboard_dir(tmp_path)
    target = tmp_path / "staged"

    copied = stage_battery_dashboard_artifacts(source, target)

    assert copied == [
        "battery_dashboard_manifest.json",
        "battery_dashboard_overview.json",
        "battery_dashboard_xjtu.json",
    ]
    assert sorted(path.name for path in target.iterdir()) == copied
    assert json.loads((target / "battery_dashboard_manifest.json").read_text(encoding="utf-8")) == {
        "source_manifest": True
    }


def test_staging_can_include_curves(tmp_path):
    source = _source_dashboard_dir(tmp_path)
    target = tmp_path / "staged"

    copied = stage_battery_dashboard_artifacts(source, target, include_curves=True)

    assert copied == [
        "battery_dashboard_manifest.json",
        "battery_dashboard_overview.json",
        "battery_dashboard_xjtu.json",
        "battery_dashboard_xjtu_curves.json",
    ]
    assert (target / "battery_dashboard_xjtu_curves.json").exists()
    assert not (target / "battery_dashboard_ess_detection.json").exists()
    assert not (target / "battery_dashboard_ess_detection_curves.json").exists()


def test_publish_creates_profile_manifest_without_source_manifest_collision(tmp_path):
    source = _source_dashboard_dir(tmp_path)
    frontend_data = tmp_path / "frontend_data"
    stale_detection = frontend_data / "profiles" / "battery-soh-diagnostics" / "battery_dashboard_ess_detection.json"
    _write_json(stale_detection, {"stale": True})

    result = publish_battery_dashboard_profile(frontend_data, source, include_curves=False)

    profile_dir = frontend_data / "profiles" / "battery-soh-diagnostics"
    manifest_path = profile_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["name"] == "battery-soh-diagnostics"
    assert manifest["source"] == "battery_SOH_dashboard"
    assert manifest["kind"] == "battery_soh_diagnostics"
    assert manifest["scope"] == {"soh_evaluation": True, "ess_detection": False}
    assert manifest["excluded_capabilities"] == ["ess_detection"]
    assert manifest["include_curves"] is False
    assert manifest["diagnostic_artifacts"] == [
        "battery_dashboard_manifest.json",
        "battery_dashboard_overview.json",
        "battery_dashboard_xjtu.json",
    ]
    assert manifest["artifact_files"] == manifest["diagnostic_artifacts"]
    assert manifest["source_artifacts"]["battery_dashboard_manifest.json"] == str(source / "manifest.json")
    assert all("ess_detection" not in key for key in manifest["source_artifacts"])
    assert "source_manifest" not in manifest
    assert (profile_dir / "battery_dashboard_manifest.json").exists()
    assert not stale_detection.exists()
    assert (frontend_data / "profiles" / "index.json").exists()


def test_publish_generates_optional_algorithm_comparison_feature_and_curve_index(tmp_path):
    source = _source_dashboard_dir(tmp_path)
    _write_optional_battery_assets(tmp_path)
    frontend_data = tmp_path / "frontend_data"

    result = publish_battery_dashboard_profile(frontend_data, source, include_curves=False)

    profile_dir = frontend_data / "profiles" / "battery-soh-diagnostics"
    manifest = result["manifest"]
    assert manifest["optional_artifacts"] == [
        "battery_algorithm_design.json",
        "battery_feature_windows.json",
        "battery_model_comparisons.json",
    ]
    assert "battery_algorithm_design.json" in manifest["artifact_files"]
    assert "battery_dashboard_xjtu_curves.json" not in manifest["artifact_files"]
    assert {
        "dataset": "xjtu",
        "filename": "battery_dashboard_xjtu_curves.json",
        "source": str(source / "xjtu_curves.json"),
        "bytes": (source / "xjtu_curves.json").stat().st_size,
        "copied": False,
    } in manifest["curve_artifacts"]

    algorithm = json.loads((profile_dir / "battery_algorithm_design.json").read_text(encoding="utf-8"))
    assert algorithm["schema_version"] == 1
    assert algorithm["model_family"] == "TabularSOHEnsemble"
    assert "q_win_ah" in algorithm["features"]
    assert "DCIR" in algorithm["feature_policy"]["forbidden"]

    comparisons = json.loads((profile_dir / "battery_model_comparisons.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in comparisons["comparisons"]] == [
        "phase2_tabular_reproduction",
        "phase2_tabular_adaptation",
        "phase3_xjtu_r2_comparison",
    ]
    assert comparisons["comparisons"][0]["metrics"]["mae"] == 0.019

    feature_windows = json.loads((profile_dir / "battery_feature_windows.json").read_text(encoding="utf-8"))
    assert feature_windows["windows"][0]["dataset"] == "nasa"
    assert feature_windows["windows"][1]["dataset"] == "xjtu"
    assert feature_windows["windows"][1]["feature_voltage_window"] == [3.7, 4.1]


def test_publish_generates_curve_preview_artifact(tmp_path):
    source = _source_dashboard_dir(tmp_path)
    _write_json(
        source / "xjtu_curves.json",
        {
            "cell-1|1": {
                "t_s": list(range(120)),
                "voltage_v": [3.0 + index * 0.01 for index in range(120)],
                "current_a": [1.0 for _ in range(120)],
                "temperature_c": [25.0 + index * 0.05 for index in range(120)],
            },
            "cell-1|2": {
                "t_s": list(range(20)),
                "voltage_v": [3.2 + index * 0.02 for index in range(20)],
                "current_a": [0.8 for _ in range(20)],
            },
        },
    )
    frontend_data = tmp_path / "frontend_data"

    result = publish_battery_dashboard_profile(frontend_data, source, include_curves=False)

    profile_dir = frontend_data / "profiles" / "battery-soh-diagnostics"
    manifest = result["manifest"]
    assert "battery_curve_previews.json" in manifest["optional_artifacts"]
    assert "battery_curve_previews.json" in manifest["artifact_files"]
    assert "battery_dashboard_xjtu_curves.json" not in manifest["artifact_files"]

    preview = json.loads((profile_dir / "battery_curve_previews.json").read_text(encoding="utf-8"))
    assert preview["schema_version"] == 1
    assert preview["max_curves_per_dataset"] == 3
    assert preview["max_points_per_curve"] == 80
    xjtu = next(item for item in preview["datasets"] if item["dataset"] == "xjtu")
    assert xjtu["kind"] == "charge_curve"
    assert xjtu["total_curves"] == 2
    assert xjtu["preview_curves"][0]["id"] == "cell-1|1"
    assert xjtu["preview_curves"][0]["sampled_point_count"] <= 80
    assert xjtu["preview_curves"][0]["series"][0]["key"] == "voltage_v"
    assert len(xjtu["preview_curves"][0]["series"][0]["points"]) <= 80
    assert all(item["dataset"] != "ess_detection" for item in preview["datasets"])


def test_api_import_battery_dashboard_publishes_profile(tmp_path):
    source = _source_dashboard_dir(tmp_path)
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        response = _post_json(
            base_url,
            "/api/import/battery-dashboard",
            {"source_dir": str(source), "include_curves": True},
        )
    finally:
        server.shutdown()

    assert response["ok"] is True
    manifest = response["profile"]["manifest"]
    assert manifest["profile"] == "battery-soh-diagnostics"
    assert manifest["include_curves"] is True
    assert "battery_dashboard_xjtu_curves.json" in manifest["diagnostic_artifacts"]
    assert all("ess_detection" not in item for item in manifest["diagnostic_artifacts"])
    assert all(item["dataset"] != "ess_detection" for item in manifest["curve_artifacts"])
    assert manifest["source_artifacts"]["battery_dashboard_xjtu_curves.json"] == str(source / "xjtu_curves.json")
    assert (tmp_path / "frontend_data" / "profiles" / "battery-soh-diagnostics" / "manifest.json").exists()
