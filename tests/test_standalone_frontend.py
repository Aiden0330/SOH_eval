from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _standalone_payload(html: str) -> dict:
    match = re.search(r"const SOH_STANDALONE_DATA = (\{.*?\});\nwindow\.__SOH_STANDALONE__", html, re.DOTALL)
    assert match, "standalone data payload not found"
    return json.loads(match.group(1))


def test_build_standalone_frontend_inlines_app_assets(tmp_path: Path) -> None:
    output = tmp_path / "soh-eval-standalone.html"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_standalone_frontend.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    html = output.read_text(encoding="utf-8")

    assert "<script type=\"module\"" not in html
    assert "src=\"app.js" not in html
    assert "href=\"styles.css" not in html
    assert "window.__SOH_STANDALONE__ = true" in html
    assert "window.__SOH_STANDALONE_DATA__" in html
    assert "./data/dataset_leaderboard.json" in html
    assert "./data/recommendation_summary.json" in html
    assert "./data/profiles/index.json" in html
    assert "./data/profiles/smoke/model_catalog.json" in html
    assert "function datasetOptions" in html
    assert "离线静态包不包含后端接口" in html
    assert "D:\\battery_SOH\\data" not in html
    payload = _standalone_payload(html)
    assert all("ess_detection" not in key for key in payload)
    assert "ess_detection" not in json.dumps(payload, ensure_ascii=False)
    preview = payload.get("./data/profiles/battery-soh-diagnostics/battery_curve_previews.json")
    if preview:
        datasets = {item.get("dataset") for item in preview.get("datasets", [])}
        assert "ess_detection" not in datasets
        assert datasets - {None}
    assert "离线内嵌数据" in html
