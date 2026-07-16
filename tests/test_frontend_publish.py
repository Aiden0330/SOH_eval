from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import scripts.package_portable_frontend as portable_module
from scripts.package_portable_frontend import package_portable_frontend
from scripts.sync_battery_dashboard_native import sync_native_dashboard
import sohbench.frontend_publish as frontend_publish_module
from sohbench.frontend_publish import publish_frontend_profile


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _snapshot(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _write_native_dashboard(root: Path, *, include_curves: bool = True) -> Path:
    data = root / "data"
    data.mkdir(parents=True)
    (root / "index.html").write_text(
        """
<!doctype html><html><head><style>footer { color:var(--dim); font-size:12px; margin-top:14px; }</style></head>
<body><span class="modetabs"><button data-mode="soh">SOH</button></span>
<script>const $ = id => document.getElementById(id); (async () => { await setMode("soh"); })();</script>
</body></html>
""".strip(),
        encoding="utf-8",
    )
    _write_json(data / "manifest.json", {"scenarios": [{"mode": "xjtu", "title": "XJTU"}]})
    _write_json(data / "overview.json", {"scenarios": []})
    _write_json(data / "xjtu.json", {"points": []})
    if include_curves:
        _write_json(data / "xjtu_curves.json", {"cell-1|1": {"t_s": [0], "voltage_v": [4.1]}})
    return root


def _write_frontend_app(root: Path) -> Path:
    root.mkdir(parents=True)
    for name, content in {
        "index.html": "<!doctype html><html><body>SOH</body></html>",
        "app.js": "console.log('SOH');",
        "dashboard-core.mjs": "export const ready = true;",
        "styles.css": "body { font-family: sans-serif; }",
    }.items():
        (root / name).write_text(content, encoding="utf-8")
    _write_json(root / "data" / "leaderboard.json", [{"model": "demo"}])
    _write_native_dashboard(root / "battery-dashboard")
    return root


def test_native_sync_missing_manifest_asset_preserves_last_known_good(tmp_path):
    source = _write_native_dashboard(tmp_path / "source", include_curves=False)
    target = tmp_path / "app" / "battery-dashboard"
    _write_json(target / "data" / "last-known-good.json", {"version": "old"})
    (target / "index.html").write_text("old dashboard", encoding="utf-8")
    before = _snapshot(target)

    with pytest.raises(FileNotFoundError, match="xjtu_curves.json"):
        sync_native_dashboard(source, target, include_curves=True)

    assert _snapshot(target) == before
    assert not list(target.parent.glob(f".{target.name}.tmp-*"))


def test_portable_copy_failure_preserves_last_known_good(tmp_path, monkeypatch):
    app_dir = _write_frontend_app(tmp_path / "frontend" / "app")
    output = tmp_path / "dist" / "soh-eval-portable"
    _write_json(output / "MANIFEST.json", {"version": "old"})
    (output / "old.txt").write_text("last-known-good", encoding="utf-8")
    before = _snapshot(output)
    real_copy2 = portable_module.shutil.copy2

    def failing_copy(source, target, *args, **kwargs):
        if Path(source).name == "styles.css":
            raise OSError("injected portable copy failure")
        return real_copy2(source, target, *args, **kwargs)

    monkeypatch.setattr(portable_module.shutil, "copy2", failing_copy)

    with pytest.raises(OSError, match="injected portable copy failure"):
        package_portable_frontend(app_dir, output)

    assert _snapshot(output) == before
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


def test_concurrent_profile_publication_never_mixes_runs(tmp_path, monkeypatch):
    frontend_data = tmp_path / "frontend" / "data"
    source_a = tmp_path / "run-a"
    source_b = tmp_path / "run-b"
    for source, run in ((source_a, "A"), (source_b, "B")):
        _write_json(source / "leaderboard.json", {"run": run, "artifact": "leaderboard"})
        _write_json(source / "qc_report.json", {"run": run, "artifact": "qc"})

    first_a_copy = threading.Event()
    real_copy2 = frontend_publish_module.shutil.copy2

    def interleaved_copy(source, target, *args, **kwargs):
        result = real_copy2(source, target, *args, **kwargs)
        source_path = Path(source)
        if source_path.parent == source_a and source_path.name == "leaderboard.json":
            first_a_copy.set()
            time.sleep(0.4)
        return result

    monkeypatch.setattr(frontend_publish_module.shutil, "copy2", interleaved_copy)
    errors: list[BaseException] = []

    def publish(source: Path, run: str) -> None:
        try:
            publish_frontend_profile(
                frontend_data,
                "shared",
                {"label": run, "run": run},
                source_dirs=[source],
                artifact_files=["leaderboard.json", "qc_report.json"],
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first = threading.Thread(target=publish, args=(source_a, "A"))
    second = threading.Thread(target=publish, args=(source_b, "B"))
    first.start()
    assert first_a_copy.wait(timeout=2)
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    profile = frontend_data / "profiles" / "shared"
    runs = {
        json.loads((profile / "leaderboard.json").read_text(encoding="utf-8"))["run"],
        json.loads((profile / "qc_report.json").read_text(encoding="utf-8"))["run"],
        json.loads((profile / "manifest.json").read_text(encoding="utf-8"))["run"],
    }
    assert len(runs) == 1
