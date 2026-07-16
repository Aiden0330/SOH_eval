from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.build_standalone_frontend import APP_DIR, build_standalone_html


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
