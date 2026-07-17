from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

import scripts.package_portable_frontend as portable_module
from scripts.package_portable_frontend import package_portable_frontend
from scripts.sync_battery_dashboard_native import _scenario_names, sync_native_dashboard
import sohbench.frontend_publish as frontend_publish_module
from sohbench.frontend_publish import (
    publish_frontend_profile,
    sync_frontend_artifacts,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _snapshot(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _active_publication(path: Path) -> Path:
    resolver = getattr(
        frontend_publish_module, "resolve_published_directory", None
    )
    return resolver(path) if resolver is not None else path


def _write_native_dashboard(
    root: Path,
    *,
    include_curves: bool = True,
    include_detection: bool = False,
) -> Path:
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
    if include_detection:
        _write_json(data / "ess_detection.json", {"detections": []})
        _write_json(
            data / "ess_detection_curves.json",
            {"cell-1": {"sample": [0], "dev_mv": {"cell-1": [0.1]}}},
        )
    copied_files = ["manifest.json", "overview.json", "xjtu.json"]
    if include_curves:
        copied_files.append("xjtu_curves.json")
    if include_detection:
        copied_files.extend(
            ["ess_detection.json", "ess_detection_curves.json"]
        )
    _write_json(
        root / "SYNC_MANIFEST.json",
        {
            "source": str(root),
            "target": str(root),
            "scenario_count": 1,
            "include_soh_curves": include_curves,
            "excluded_detection_assets": not include_detection,
            "copied_files": copied_files,
        },
    )
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


def _multiprocess_publish_worker(
    target_value,
    version,
    pointer_ready,
    result_queue,
):
    target = Path(target_value)
    real_sync = frontend_publish_module._sync_compatibility_view

    if version == "A":
        def delayed_sync(target_dir, generation, generation_name):
            pointer_ready.set()
            time.sleep(2.0)
            return real_sync(target_dir, generation, generation_name)

        frontend_publish_module._sync_compatibility_view = delayed_sync

    def build(staging):
        staging.mkdir(parents=True)
        _write_json(staging / "manifest.json", {"version": version})
        _write_json(staging / "payload.json", {"version": version})

    try:
        frontend_publish_module._publish_directory(target, build)
        result_queue.put({"version": version, "status": "succeeded"})
    except BaseException as exc:
        result_queue.put(
            {
                "version": version,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


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


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("", id="empty"),
        pytest.param(".", id="dot"),
        pytest.param("..", id="dot-dot"),
        pytest.param("../escape", id="posix-traversal"),
        pytest.param(r"..\escape", id="windows-traversal"),
        pytest.param(r"C:\escape", id="windows-absolute"),
        pytest.param("C:escape", id="windows-drive-relative"),
        pytest.param("%2e%2e", id="encoded-traversal"),
        pytest.param("CuRrEnT", id="control-stem-casefold"),
        pytest.param("current.JSON", id="control-file-casefold"),
        pytest.param("CON", id="windows-device-con"),
        pytest.param("cOm1", id="windows-device-com-mixed-case"),
        pytest.param("LpT9", id="windows-device-lpt-mixed-case"),
    ],
)
def test_native_sync_rejects_unsafe_scenario_filename_stems(name):
    with pytest.raises(ValueError, match="invalid scenario name"):
        _scenario_names({"scenarios": [{"mode": name}]})


def test_native_sync_traversal_preserves_active_generation_and_sibling(tmp_path):
    source = _write_native_dashboard(tmp_path / "source")
    target = tmp_path / "app" / "battery-dashboard"
    sync_native_dashboard(source, target, include_curves=True)
    before = _snapshot(target)
    sibling = target.parent / "sibling.json"
    sibling.write_bytes(b"last-known-good-sibling")

    _write_json(
        source / "data" / "manifest.json",
        {"scenarios": [{"mode": "../../sibling", "title": "unsafe"}]},
    )
    _write_json(tmp_path / "sibling.json", {"version": "attacker"})

    with pytest.raises(ValueError, match="invalid scenario name"):
        sync_native_dashboard(source, target, include_curves=False)

    assert _snapshot(target) == before
    assert sibling.read_bytes() == b"last-known-good-sibling"
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


def test_portable_package_rewrites_detection_enabled_native_sync_manifest(
    tmp_path,
):
    app_dir = _write_frontend_app(tmp_path / "frontend" / "app")
    native_source = _write_native_dashboard(
        tmp_path / "native-source", include_detection=True
    )
    synced = sync_native_dashboard(
        native_source,
        app_dir / "battery-dashboard",
        include_curves=True,
        include_detection=True,
    )
    assert {
        "ess_detection.json",
        "ess_detection_curves.json",
    } <= set(synced["copied_files"])

    output = tmp_path / "dist" / "soh-eval-portable"
    package_portable_frontend(app_dir, output)
    active = frontend_publish_module.resolve_published_directory(output)
    packaged_dashboard = active / "battery-dashboard"
    packaged_sync = json.loads(
        (packaged_dashboard / "SYNC_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )

    assert all(
        "ess_detection" not in filename
        for filename in packaged_sync["copied_files"]
    )
    assert not list(
        (packaged_dashboard / "data").glob("*ess_detection*.json")
    )
    assert packaged_sync["excluded_detection_assets"] is True
    for key in (
        "source",
        "target",
        "scenario_count",
        "include_soh_curves",
    ):
        assert packaged_sync[key] == synced[key]
    portable_module._validate_native_dashboard(packaged_dashboard)


def test_portable_missing_native_sync_manifest_preserves_last_known_good(
    tmp_path,
):
    app_dir = _write_frontend_app(tmp_path / "frontend" / "app")
    output = tmp_path / "dist" / "soh-eval-portable"
    package_portable_frontend(app_dir, output)
    before = _snapshot(output)
    (app_dir / "battery-dashboard" / "SYNC_MANIFEST.json").unlink()

    with pytest.raises(FileNotFoundError, match="SYNC_MANIFEST.json"):
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


def test_publishers_are_serialized_across_real_processes(tmp_path):
    target = tmp_path / "published"
    context = multiprocessing.get_context("spawn")
    pointer_ready = context.Event()
    results = context.Queue()
    first = context.Process(
        target=_multiprocess_publish_worker,
        args=(str(target), "A", pointer_ready, results),
    )
    second = context.Process(
        target=_multiprocess_publish_worker,
        args=(str(target), "B", pointer_ready, results),
    )

    first.start()
    assert pointer_ready.wait(timeout=15)
    second.start()
    first.join(timeout=20)
    second.join(timeout=20)

    assert not first.is_alive() and not second.is_alive()
    assert first.exitcode == 0 and second.exitcode == 0
    messages = sorted(
        [results.get(timeout=2), results.get(timeout=2)],
        key=lambda item: item["version"],
    )
    assert messages == [
        {"version": "A", "status": "succeeded"},
        {"version": "B", "status": "succeeded"},
    ]
    pointer = json.loads(
        (target / "CURRENT.json").read_text(encoding="utf-8")
    )
    active = target / ".generations" / pointer["generation"]
    active_versions = {
        json.loads((active / name).read_text(encoding="utf-8"))["version"]
        for name in ("manifest.json", "payload.json")
    }
    compatibility_versions = {
        json.loads((target / name).read_text(encoding="utf-8"))["version"]
        for name in ("manifest.json", "payload.json")
    }

    assert len(active_versions) == 1
    assert compatibility_versions == active_versions
    lock_path = target.parent / f".{target.name}.publish.lock"
    assert lock_path.is_file()
    assert lock_path.parent == target.parent


def test_root_publication_removes_omitted_artifacts_but_preserves_profiles(
    tmp_path,
):
    frontend_data = tmp_path / "frontend" / "data"
    run_dir = tmp_path / "run-b"
    _write_json(
        frontend_data / "leaderboard.json",
        {"run": "A", "artifact": "leaderboard"},
    )
    _write_json(
        frontend_data / "recommendation_summary.json",
        {"run": "A", "artifact": "recommendation"},
    )
    _write_json(
        frontend_data / "profiles" / "accepted" / "manifest.json",
        {"profile": "accepted", "persistent": True},
    )
    _write_json(
        run_dir / "leaderboard.json",
        {"run": "B", "artifact": "leaderboard"},
    )
    profile_before = _snapshot(frontend_data / "profiles")

    copied = sync_frontend_artifacts(run_dir, frontend_data)

    assert copied == ["leaderboard.json"]
    assert json.loads(
        (frontend_data / "leaderboard.json").read_text(encoding="utf-8")
    )["run"] == "B"
    assert not (frontend_data / "recommendation_summary.json").exists()
    assert _snapshot(frontend_data / "profiles") == profile_before


def test_legacy_root_migration_preserves_nested_control_named_business_files(
    tmp_path,
):
    frontend_data = tmp_path / "frontend" / "data"
    run_dir = tmp_path / "run"
    nested_pointer = frontend_data / "assets" / "CURRENT.json"
    nested_generation = (
        frontend_data / "assets" / ".generations" / "payload.bin"
    )
    nested_pointer.parent.mkdir(parents=True)
    nested_pointer.write_bytes(b"nested-business-pointer")
    nested_generation.parent.mkdir(parents=True)
    nested_generation.write_bytes(b"nested-business-generation")
    _write_json(run_dir / "leaderboard.json", {"run": "migration"})

    sync_frontend_artifacts(run_dir, frontend_data)

    active = frontend_publish_module.resolve_published_directory(
        frontend_data
    )
    for base in (active, frontend_data):
        assert (
            base / "assets" / "CURRENT.json"
        ).read_bytes() == b"nested-business-pointer"
        assert (
            base / "assets" / ".generations" / "payload.bin"
        ).read_bytes() == b"nested-business-generation"


def test_composite_root_publication_uses_first_source_and_preserves_profiles(tmp_path):
    frontend_data = tmp_path / "frontend" / "data"
    training_dir = tmp_path / "training"
    inference_dir = tmp_path / "inference"
    _write_json(frontend_data / "recommendation_summary.json", {"run": "stale"})
    _write_json(
        frontend_data / "profiles" / "accepted" / "manifest.json",
        {"profile": "accepted", "persistent": True},
    )
    _write_json(training_dir / "leaderboard.json", {"run": "training"})
    _write_json(training_dir / "run_config.json", {"run": "training"})
    _write_json(training_dir / "model_catalog.json", {"run": "training"})
    _write_json(inference_dir / "model_catalog.json", {"run": "inference"})
    _write_json(inference_dir / "inference_benchmark.json", {"run": "inference"})
    _write_json(inference_dir / "inference_run_config.json", {"run": "inference"})
    profile_before = _snapshot(frontend_data / "profiles")

    copied = frontend_publish_module.publish_frontend_artifacts(
        [training_dir, inference_dir], frontend_data
    )

    assert copied == [
        "leaderboard.json",
        "run_config.json",
        "model_catalog.json",
        "inference_benchmark.json",
        "inference_run_config.json",
    ]
    assert json.loads((frontend_data / "leaderboard.json").read_text(encoding="utf-8"))["run"] == "training"
    assert json.loads((frontend_data / "model_catalog.json").read_text(encoding="utf-8"))["run"] == "training"
    assert json.loads((frontend_data / "inference_benchmark.json").read_text(encoding="utf-8"))["run"] == "inference"
    assert not (frontend_data / "recommendation_summary.json").exists()
    assert _snapshot(frontend_data / "profiles") == profile_before


def test_directory_publication_keeps_one_complete_generation_visible_to_reader(
    tmp_path, monkeypatch
):
    target = tmp_path / "published"

    def build_version(version):
        def build(staging):
            staging.mkdir(parents=True)
            _write_json(staging / "manifest.json", {"version": version})
            _write_json(staging / "payload.json", {"version": version})

        return build

    frontend_publish_module._publish_directory(target, build_version("old"))
    stop = threading.Event()
    observed = []
    errors = []

    def read_active_generation():
        while not stop.is_set():
            try:
                active = _active_publication(target)
                manifest = json.loads(
                    (active / "manifest.json").read_text(encoding="utf-8")
                )
                payload = json.loads(
                    (active / "payload.json").read_text(encoding="utf-8")
                )
                observed.append((manifest["version"], payload["version"]))
            except BaseException as exc:
                errors.append(exc)
                return

    real_replace = frontend_publish_module.os.replace

    def pause_after_pointer_activation(source, destination):
        result = real_replace(source, destination)
        if Path(destination) == target / "CURRENT.json":
            time.sleep(0.25)
        return result

    monkeypatch.setattr(
        frontend_publish_module.os, "replace", pause_after_pointer_activation
    )
    reader = threading.Thread(target=read_active_generation)
    reader.start()
    try:
        frontend_publish_module._publish_directory(
            target, build_version("new")
        )
    finally:
        stop.set()
        reader.join(timeout=5)

    assert not reader.is_alive()
    assert errors == []
    assert ("old", "old") in observed
    assert ("new", "new") in observed
    assert set(observed) <= {("old", "old"), ("new", "new")}


def test_directory_activation_interruption_preserves_last_known_good(
    tmp_path, monkeypatch
):
    target = tmp_path / "published"

    def build(version):
        def write(staging):
            staging.mkdir(parents=True)
            _write_json(staging / "manifest.json", {"version": version})
            _write_json(staging / "payload.json", {"version": version})

        return write

    frontend_publish_module._publish_directory(target, build("old"))
    real_replace = frontend_publish_module.os.replace

    def interrupt_activation(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name == "CURRENT.json":
            raise KeyboardInterrupt("injected pointer interruption")
        if source_path == target and ".bak-" in destination_path.name:
            real_replace(source, destination)
            raise KeyboardInterrupt("injected legacy interruption")
        return real_replace(source, destination)

    monkeypatch.setattr(
        frontend_publish_module.os, "replace", interrupt_activation
    )

    with pytest.raises(KeyboardInterrupt, match="injected"):
        frontend_publish_module._publish_directory(target, build("new"))

    active = _active_publication(target)
    assert json.loads(
        (active / "manifest.json").read_text(encoding="utf-8")
    )["version"] == "old"
    assert json.loads(
        (active / "payload.json").read_text(encoding="utf-8")
    )["version"] == "old"


def test_compatibility_view_restores_directory_when_second_replace_fails(
    tmp_path, monkeypatch
):
    target = tmp_path / "published"
    generation = tmp_path / "generation"
    _write_json(target / "assets" / "manifest.json", {"version": "old"})
    (target / "assets" / "payload.bin").write_bytes(b"last-known-good")
    _write_json(generation / "assets" / "manifest.json", {"version": "new"})
    (generation / "assets" / "payload.bin").write_bytes(b"replacement")
    before = _snapshot(target / "assets")
    real_replace = frontend_publish_module.os.replace

    def fail_second_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name.startswith(".assets.compat-")
            and destination_path == target / "assets"
        ):
            raise OSError("injected compatibility activation failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        frontend_publish_module.os, "replace", fail_second_replace
    )

    with pytest.raises(OSError, match="compatibility activation failure"):
        frontend_publish_module._sync_compatibility_view(
            target, generation, "a" * 32
        )

    assert _snapshot(target / "assets") == before
    assert not list(target.glob(".assets.compat-*"))
    assert not list(target.glob(".assets.backup-*"))


def test_compatibility_view_restores_entire_mirror_after_late_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "published"
    generation = tmp_path / "generation"
    _write_json(target / "a.json", {"version": "old"})
    (target / "b-assets" / "payload.bin").parent.mkdir(parents=True)
    (target / "b-assets" / "payload.bin").write_bytes(b"old-assets")
    (target / "obsolete.txt").write_bytes(b"keep-on-failure")
    _write_json(
        target / "CURRENT.json",
        {"schema_version": 1, "generation": "c" * 32},
    )
    (target / ".generations" / ("c" * 32)).mkdir(parents=True)
    (target / ".generations" / ("c" * 32) / "marker").write_bytes(
        b"generation-control"
    )
    _write_json(generation / "a.json", {"version": "new"})
    (generation / "b-assets" / "payload.bin").parent.mkdir(parents=True)
    (generation / "b-assets" / "payload.bin").write_bytes(b"new-assets")

    controls = {".generations", "CURRENT.json"}

    def compatibility_snapshot():
        return {
            item.relative_to(target).as_posix(): item.read_bytes()
            for item in sorted(target.rglob("*"))
            if item.is_file()
            and item.relative_to(target).parts[0] not in controls
        }

    before = compatibility_snapshot()
    control_before = {
        item.relative_to(target).as_posix(): item.read_bytes()
        for item in sorted(target.rglob("*"))
        if item.is_file()
        and item.relative_to(target).parts[0] in controls
    }
    real_replace = frontend_publish_module.os.replace

    def fail_late_directory_activation(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name.startswith(".b-assets.compat-")
            and destination_path == target / "b-assets"
        ):
            raise OSError("injected late compatibility failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        frontend_publish_module.os,
        "replace",
        fail_late_directory_activation,
    )

    with pytest.raises(OSError, match="late compatibility failure"):
        frontend_publish_module._sync_compatibility_view(
            target, generation, "d" * 32
        )

    assert compatibility_snapshot() == before
    assert {
        item.relative_to(target).as_posix(): item.read_bytes()
        for item in sorted(target.rglob("*"))
        if item.is_file()
        and item.relative_to(target).parts[0] in controls
    } == control_before


def test_recovery_snapshot_is_preserved_when_restoration_fails(
    tmp_path, monkeypatch
):
    target = tmp_path / "published"
    generation = tmp_path / "generation"
    _write_json(target / "payload.json", {"version": "old"})
    (target / "obsolete.bin").write_bytes(b"old-only")
    pointer = target / "CURRENT.json"
    pointer.write_bytes(b"pointer-control")
    marker = target / ".generations" / ("a" * 32) / "marker.bin"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"generation-control")
    _write_json(generation / "payload.json", {"version": "new"})
    old_compatibility = {
        "obsolete.bin": b"old-only",
        "payload.json": (target / "payload.json").read_bytes(),
    }
    real_apply = frontend_publish_module._apply_compatibility_view

    def fail_sync_after_mutation(target_dir, generation_dir, name):
        real_apply(target_dir, generation_dir, name)
        raise OSError("injected sync failure")

    def fail_restoration(target_dir, snapshot):
        raise OSError("injected restoration failure")

    monkeypatch.setattr(
        frontend_publish_module,
        "_apply_compatibility_view",
        fail_sync_after_mutation,
    )
    monkeypatch.setattr(
        frontend_publish_module,
        "_restore_compatibility_content",
        fail_restoration,
    )

    with pytest.raises(OSError, match="restoration failure"):
        frontend_publish_module._sync_compatibility_view(
            target, generation, "b" * 32
        )

    snapshots = list(
        target.parent.glob(f".{target.name}.compat-snapshot-*")
    )
    assert len(snapshots) == 1
    assert _snapshot(snapshots[0]) == old_compatibility
    assert pointer.read_bytes() == b"pointer-control"
    assert marker.read_bytes() == b"generation-control"


def test_casefolded_publication_controls_survive_compatibility_sync(
    tmp_path,
):
    target = tmp_path / "published"
    generation = tmp_path / "generation"
    active_marker = (
        target / ".Generations" / ("a" * 32) / "marker.bin"
    )
    active_marker.parent.mkdir(parents=True)
    active_marker.write_bytes(b"active-generation")
    pointer = target / "current.JSON"
    pointer.write_bytes(b"active-pointer")
    _write_json(generation / "payload.json", {"version": "new"})

    frontend_publish_module._sync_compatibility_view(
        target, generation, "b" * 32
    )

    assert active_marker.read_bytes() == b"active-generation"
    assert pointer.read_bytes() == b"active-pointer"
    assert json.loads(
        (target / "payload.json").read_text(encoding="utf-8")
    ) == {"version": "new"}


def test_casefolded_controls_are_never_mutated_during_failed_sync(
    tmp_path, monkeypatch
):
    target = tmp_path / "published"
    generation = tmp_path / "generation"
    active_marker = (
        target / ".GENERATIONS" / ("a" * 32) / "marker.bin"
    )
    active_marker.parent.mkdir(parents=True)
    active_marker.write_bytes(b"active-generation")
    pointer = target / "Current.Json"
    pointer.write_bytes(b"active-pointer")
    _write_json(target / "a.json", {"version": "old"})
    _write_json(generation / "a.json", {"version": "new"})
    (generation / "b-assets" / "payload.bin").parent.mkdir(parents=True)
    (generation / "b-assets" / "payload.bin").write_bytes(b"new-assets")
    control_mutations = []
    real_rmtree = frontend_publish_module.shutil.rmtree
    real_unlink = Path.unlink
    real_replace = frontend_publish_module.os.replace

    def track_rmtree(path, *args, **kwargs):
        if Path(path).name.casefold() == ".generations":
            control_mutations.append(Path(path).name)
        return real_rmtree(path, *args, **kwargs)

    def track_unlink(path, *args, **kwargs):
        if (
            path.parent == target
            and path.name.casefold() == "current.json"
        ):
            control_mutations.append(path.name)
        return real_unlink(path, *args, **kwargs)

    def fail_late_activation(source, destination):
        if (
            Path(source).name.startswith(".b-assets.compat-")
            and Path(destination) == target / "b-assets"
        ):
            raise OSError("injected casefold sync failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        frontend_publish_module.shutil, "rmtree", track_rmtree
    )
    monkeypatch.setattr(Path, "unlink", track_unlink)
    monkeypatch.setattr(
        frontend_publish_module.os, "replace", fail_late_activation
    )

    with pytest.raises(OSError, match="casefold sync failure"):
        frontend_publish_module._sync_compatibility_view(
            target, generation, "b" * 32
        )

    assert control_mutations == []
    assert active_marker.read_bytes() == b"active-generation"
    assert pointer.read_bytes() == b"active-pointer"


def test_generation_source_case_collision_with_control_is_rejected(
    tmp_path,
):
    target = tmp_path / "published"
    generation = tmp_path / "generation"
    pointer = target / "CURRENT.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_bytes(b"active-pointer")
    (generation / "Current.JSON").parent.mkdir(parents=True)
    (generation / "Current.JSON").write_bytes(b"collision")

    with pytest.raises(ValueError, match="publication control"):
        frontend_publish_module._sync_compatibility_view(
            target, generation, "b" * 32
        )

    assert pointer.read_bytes() == b"active-pointer"


def test_publication_rejects_control_collision_before_pointer_activation(
    tmp_path,
):
    target = tmp_path / "published"

    def build_old(staging):
        staging.mkdir(parents=True)
        _write_json(staging / "payload.json", {"version": "old"})

    frontend_publish_module._publish_directory(target, build_old)
    pointer_before = (target / "CURRENT.json").read_bytes()

    def build_collision(staging):
        staging.mkdir(parents=True)
        _write_json(staging / "payload.json", {"version": "new"})
        (staging / "Current.Json").write_bytes(b"collision")

    with pytest.raises(ValueError, match="publication control"):
        frontend_publish_module._publish_directory(
            target, build_collision
        )

    assert (target / "CURRENT.json").read_bytes() == pointer_before
    active = frontend_publish_module.resolve_published_directory(target)
    assert json.loads(
        (active / "payload.json").read_text(encoding="utf-8")
    ) == {"version": "old"}


def test_resolve_published_directory_retries_transient_pointer_read_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "published"

    def build(staging):
        staging.mkdir(parents=True)
        _write_json(staging / "manifest.json", {"version": "current"})

    frontend_publish_module._publish_directory(target, build)
    pointer = target / "CURRENT.json"
    real_read_text = Path.read_text
    attempts = 0

    def transient_read_error(path, *args, **kwargs):
        nonlocal attempts
        if path == pointer and attempts < 2:
            attempts += 1
            raise PermissionError("injected pointer sharing violation")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient_read_error)

    active = frontend_publish_module.resolve_published_directory(target)

    assert attempts == 2
    assert json.loads(
        (active / "manifest.json").read_text(encoding="utf-8")
    )["version"] == "current"


def test_resolve_published_directory_rejects_stable_invalid_pointer_immediately(
    tmp_path, monkeypatch
):
    target = tmp_path / "published"
    _write_json(
        target / "CURRENT.json",
        {"schema_version": 1, "generation": "not-a-generation"},
    )
    sleeps = []
    monkeypatch.setattr(
        frontend_publish_module.time,
        "sleep",
        lambda delay: sleeps.append(delay),
    )

    with pytest.raises(ValueError, match="invalid publication pointer"):
        frontend_publish_module.resolve_published_directory(target)

    assert sleeps == []


def test_repeated_directory_publication_keeps_active_reader_resolvable(
    tmp_path,
):
    target = tmp_path / "published"
    stop = threading.Event()
    errors = []

    def build_version(version):
        def build(staging):
            staging.mkdir(parents=True)
            _write_json(staging / "manifest.json", {"version": version})
            _write_json(staging / "payload.json", {"version": version})

        return build

    frontend_publish_module._publish_directory(target, build_version(0))

    def read_active_generation():
        while not stop.is_set():
            try:
                active = frontend_publish_module.resolve_published_directory(
                    target
                )
                manifest = json.loads(
                    (active / "manifest.json").read_text(encoding="utf-8")
                )
                payload = json.loads(
                    (active / "payload.json").read_text(encoding="utf-8")
                )
                assert manifest["version"] == payload["version"]
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)
                return

    reader = threading.Thread(target=read_active_generation)
    reader.start()
    try:
        for version in range(1, 6):
            frontend_publish_module._publish_directory(
                target, build_version(version)
            )
    finally:
        stop.set()
        reader.join(timeout=5)

    assert not reader.is_alive()
    assert errors == []
    active = frontend_publish_module.resolve_published_directory(target)
    assert json.loads(
        (active / "manifest.json").read_text(encoding="utf-8")
    )["version"] == 5


def test_post_replace_failure_never_deletes_referenced_generation(
    tmp_path, monkeypatch
):
    target = tmp_path / "published"

    def build_version(version):
        def build(staging):
            staging.mkdir(parents=True)
            _write_json(staging / "manifest.json", {"version": version})
            _write_json(staging / "payload.json", {"version": version})

        return build

    frontend_publish_module._publish_directory(target, build_version("old"))
    real_write_pointer = frontend_publish_module._write_atomic_json

    def install_pointer_then_fail(path, document):
        real_write_pointer(path, document)
        raise OSError("injected post-replace durability failure")

    monkeypatch.setattr(
        frontend_publish_module,
        "_write_atomic_json",
        install_pointer_then_fail,
    )

    with pytest.raises(OSError, match="post-replace durability failure"):
        frontend_publish_module._publish_directory(
            target, build_version("new")
        )

    active = frontend_publish_module.resolve_published_directory(target)
    manifest = json.loads(
        (active / "manifest.json").read_text(encoding="utf-8")
    )
    payload = json.loads(
        (active / "payload.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == payload["version"]
    assert manifest["version"] in {"old", "new"}
