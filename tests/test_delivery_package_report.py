from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_package_status_fixture(root: Path) -> tuple[Path, Path, Path]:
    native = root / "frontend" / "app" / "battery-dashboard"
    native_data = native / "data"
    native_data.mkdir(parents=True)
    (native / "index.html").write_text("<html><body>SOH dashboard</body></html>", encoding="utf-8")
    (native / "SYNC_MANIFEST.json").write_text(
        json.dumps(
            {
                "scenario_count": 1,
                "include_soh_curves": True,
                "excluded_detection_assets": True,
                "copied_files": ["manifest.json", "overview.json", "xjtu.json", "xjtu_curves.json"],
            }
        ),
        encoding="utf-8",
    )
    (native_data / "manifest.json").write_text(json.dumps({"scenarios": [{"mode": "xjtu"}]}), encoding="utf-8")
    (native_data / "overview.json").write_text(json.dumps({"scenarios": []}), encoding="utf-8")
    (native_data / "xjtu.json").write_text(json.dumps({"points": []}), encoding="utf-8")
    (native_data / "xjtu_curves.json").write_text(json.dumps({"cell": {}}), encoding="utf-8")

    portable = root / "dist" / "soh-eval-portable"
    portable.mkdir(parents=True)
    (portable / "index.html").write_text("<html><body>portable</body></html>", encoding="utf-8")
    (portable / "MANIFEST.json").write_text(
        json.dumps(
            {
                "portable": True,
                "file_count": 2,
                "total_bytes": 1024,
                "include_soh_curves": True,
                "excluded_detection_assets": True,
            }
        ),
        encoding="utf-8",
    )

    standalone = root / "dist" / "soh-eval-standalone.html"
    standalone.write_text(
        """
<!-- Standalone SOH evaluation bundle -->
<html><body>
<section id="package-operations-card"></section>
<section id="package-report-card"></section>
<script>
const SOH_STANDALONE_DATA = {"./data/leaderboard.json":[],"./data/profiles/index.json":{"profiles":[]}};
window.__SOH_STANDALONE__ = true;
</script>
</body></html>
""".strip(),
        encoding="utf-8",
    )
    return native, portable, standalone


def test_build_delivery_package_report_writes_manifest_and_usage_md(tmp_path: Path) -> None:
    native, portable, standalone = _write_package_status_fixture(tmp_path)
    json_output = tmp_path / "dist" / "soh-eval-delivery-report.json"
    md_output = tmp_path / "dist" / "soh-eval-delivery-report.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_delivery_package_report.py"),
            "--native-dashboard",
            str(native),
            "--portable-output",
            str(portable),
            "--standalone-output",
            str(standalone),
            "--json-output",
            str(json_output),
            "--md-output",
            str(md_output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = md_output.read_text(encoding="utf-8")

    assert payload["kind"] == "soh_eval_delivery_report"
    assert payload["report"]["summary"]["ready"] is True
    assert payload["artifacts"]["portable"]["path"] == str(portable)
    assert payload["artifacts"]["standalone"]["path"] == str(standalone)
    assert payload["usage"]["standalone"][0].endswith("soh-eval-standalone.html")
    assert "验收通过：5/5" in markdown
    assert "迁移使用步骤" in markdown
    assert "soh-eval-portable" in markdown
    assert "soh-eval-standalone.html" in markdown
    assert f"cd {portable}" in markdown
    assert "python -m http.server" in markdown
