from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.build_standalone_frontend import (
    APP_DIR,
    build_standalone_html,
    strip_module_exports,
)
from scripts.package_portable_frontend import package_portable_frontend
from sohbench.evaluation_service import EvaluationService


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_synced_native_dashboard(app_dir: Path) -> Path:
    native = app_dir / "battery-dashboard"
    data = native / "data"
    data.mkdir(parents=True, exist_ok=True)
    (native / "index.html").write_text(
        """<!doctype html><html><head><meta charset="utf-8"></head><body>
<div id="native-fixture">SOH 完整看板</div>
<script>fetch("data/xjtu.json").then((response) => response.json()).then((value) => {
  document.querySelector("#native-fixture").dataset.rows = String(value.rows.length);
});</script>
</body></html>""",
        encoding="utf-8",
    )
    _write_json(data / "manifest.json", {"scenarios": [{"mode": "xjtu"}]})
    _write_json(data / "overview.json", {"scenarios": ["xjtu"]})
    _write_json(data / "xjtu.json", {"rows": [{"cycle": 1, "soh": 0.99}]})
    _write_json(data / "xjtu_curves.json", {"cell-1": {"curve": [1, 2, 3]}})
    _write_json(
        native / "SYNC_MANIFEST.json",
        {
            "scenario_count": 1,
            "include_soh_curves": True,
            "excluded_detection_assets": True,
            "copied_files": [
                "manifest.json",
                "overview.json",
                "xjtu.json",
                "xjtu_curves.json",
            ],
        },
    )
    return native


def _headless_browser() -> Path | None:
    configured = os.environ.get("SOHBENCH_BROWSER")
    candidates = [
        configured,
        shutil.which("msedge"),
        shutil.which("chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    return next(
        (Path(candidate) for candidate in candidates if candidate and Path(candidate).is_file()),
        None,
    )


def test_real_app_portable_package_is_deterministic_without_native_dashboard(tmp_path):
    portable_output = tmp_path / "portable"

    manifest = package_portable_frontend(APP_DIR, portable_output)

    assert manifest["battery_dashboard"] == {
        "included": False,
        "sync_required": True,
        "include_soh_curves": False,
        "soh_curve_file_count": 0,
        "excluded_detection_assets": True,
    }
    assert not (portable_output / "battery-dashboard").exists()
    assert (portable_output / "index.html").is_file()


def test_real_app_standalone_is_deterministic_without_native_dashboard():
    standalone, artifact_count = build_standalone_html(APP_DIR)

    assert artifact_count > 0
    assert "window.__SOH_STANDALONE_NATIVE_DASHBOARD__ = null;" in standalone
    assert "尚未同步原生看板" in standalone


def test_standalone_module_transform_removes_sync_and_async_exports():
    source = (
        "export async function loadData() {}\n"
        "export function render() {}\n"
        "export const status = 'ready';\n"
    )

    transformed = strip_module_exports(source)

    assert "export " not in transformed
    assert "async function loadData() {}" in transformed
    assert "function render() {}" in transformed
    assert "const status = 'ready';" in transformed


def test_real_app_portable_package_includes_complete_synced_native_dashboard(tmp_path):
    app_dir = tmp_path / "frontend" / "app"
    shutil.copytree(APP_DIR, app_dir)
    native = _write_synced_native_dashboard(app_dir)
    portable_output = tmp_path / "portable"

    manifest = package_portable_frontend(app_dir, portable_output)

    assert manifest["battery_dashboard"]["included"] is True
    assert manifest["battery_dashboard"]["sync_required"] is False
    assert (portable_output / "battery-dashboard" / "index.html").read_bytes() == (native / "index.html").read_bytes()
    assert (portable_output / "battery-dashboard" / "data" / "xjtu_curves.json").read_bytes() == (
        native / "data" / "xjtu_curves.json"
    ).read_bytes()


def test_real_app_standalone_includes_complete_synced_native_dashboard(tmp_path):
    app_dir = tmp_path / "frontend" / "app"
    shutil.copytree(APP_DIR, app_dir)
    _write_synced_native_dashboard(app_dir)

    standalone, _ = build_standalone_html(app_dir)

    assert "window.__SOH_STANDALONE_NATIVE_DASHBOARD__ = null;" not in standalone
    assert "SOH 完整看板" in standalone
    assert "native-fixture" in standalone
    assert '\\"curve\\":[1,2,3]' in standalone


def test_standalone_partial_write_failure_preserves_last_known_good(
    tmp_path, monkeypatch
):
    output = tmp_path / "soh-eval-standalone.html"
    old_bytes = b"last-known-good-standalone"
    output.write_bytes(old_bytes)
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    real_write_text = Path.write_text

    def partial_write(path, text, *args, **kwargs):
        is_owned_temp = (
            path.parent == output.parent
            and path.name.startswith(f".{output.name}.tmp-")
        )
        if path == output or is_owned_temp:
            with path.open("wb") as stream:
                stream.write(str(text).encode("utf-8")[:31])
            raise OSError("injected standalone partial write")
        return real_write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", partial_write)

    with pytest.raises(OSError, match="injected standalone partial write"):
        service.package_standalone_frontend(app_dir=APP_DIR, output=output)

    assert output.read_bytes() == old_bytes
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


def test_standalone_bundle_executes_and_renders_embedded_data(tmp_path):
    browser = _headless_browser()
    if browser is None:
        pytest.skip("no headless Edge or Chrome executable is available")

    html, artifact_count = build_standalone_html(APP_DIR)
    output = tmp_path / "soh-eval-standalone.html"
    output.write_text(html, encoding="utf-8")
    result = subprocess.run(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--allow-file-access-from-files",
            "--virtual-time-budget=5000",
            f"--user-data-dir={tmp_path / 'browser-profile'}",
            "--dump-dom",
            output.resolve().as_uri(),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )

    assert artifact_count > 0
    assert result.returncode == 0, result.stderr
    assert re.search(
        r'id="training-workflow"[^>]*>\s*<div class="panel-title"',
        result.stdout,
    )
    assert re.search(
        r'id="dataset-summary"[^>]*>\s*<div class="scope-metric"',
        result.stdout,
    )
    assert re.search(
        r'id="leaderboard"[^>]*>.*?<table class="lb"',
        result.stdout,
        flags=re.DOTALL,
    )
    assert re.search(
        r'id="foot"[^>]*>sohbench 实测运行\s*/',
        result.stdout,
    )
