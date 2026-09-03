"""Task 0 of the kernel extraction: every filesystem root comes from the
environment or a repo-relative default — never from a machine-specific literal."""
from pathlib import Path

from sohbench import paths


def test_data_root_defaults_to_repo_data(monkeypatch):
    monkeypatch.delenv("SOHBENCH_DATA_DIR", raising=False)
    assert paths.data_root() == paths.PROJECT_ROOT / "data"
    assert paths.default_data_dir() == paths.data_root()


def test_data_root_honours_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SOHBENCH_DATA_DIR", str(tmp_path / "elsewhere"))
    assert paths.data_root() == tmp_path / "elsewhere"
    monkeypatch.delenv("SOHBENCH_CURVE_DIR", raising=False)
    assert paths.default_curve_dir() == tmp_path / "elsewhere" / "curves"


def test_battery_soh_root_defaults_to_sibling_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("BATTERY_SOH_ROOT", raising=False)
    assert paths.battery_soh_root() == paths.PROJECT_ROOT.parent / "battery_SOH"
    monkeypatch.setenv("BATTERY_SOH_ROOT", str(tmp_path / "battery_SOH"))
    assert paths.battery_soh_root() == tmp_path / "battery_SOH"


def test_no_machine_specific_defaults_remain():
    """No absolute machine path may be a default anywhere in the package."""
    src = Path(paths.__file__).resolve().parents[1]
    offenders = []
    for path in sorted(src.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if 'r"D:\\' in line or "r'D:\\" in line or '"D:\\\\' in line:
                offenders.append(f"{path.relative_to(src)}:{lineno}")
    assert not offenders, f"machine-specific default paths remain: {offenders}"


def test_battery_soh_root_request_resolves_to_server_default(monkeypatch, tmp_path):
    from sohbench.battery_soh_plugins import resolve_battery_soh_root

    monkeypatch.setattr(paths, "DEFAULT_BATTERY_SOH_ROOT", tmp_path / "default_root")
    monkeypatch.setattr(
        "sohbench.battery_soh_plugins.DEFAULT_BATTERY_SOH_ROOT", tmp_path / "default_root"
    )
    assert resolve_battery_soh_root("") == tmp_path / "default_root"
    assert resolve_battery_soh_root(None) == tmp_path / "default_root"
    assert resolve_battery_soh_root("  ") == tmp_path / "default_root"
    assert resolve_battery_soh_root(str(tmp_path / "given")) == tmp_path / "given"
