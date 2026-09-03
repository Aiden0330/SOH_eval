"""Filesystem roots for the workbench, resolved from the environment.

No default path points at a specific machine any more (the Windows-era
``D:\\`` literals are gone, see ADR-001 / extraction plan Task 0):

* ``SOHBENCH_DATA_DIR``  — supervised data root (default ``<repo>/data``)
* ``SOHBENCH_CURVE_DIR`` — raw curve assets (default ``<data root>/curves``)
* ``BATTERY_SOH_ROOT``   — the sibling ``battery_SOH`` project that provides the
  optional TabularSOHEnsemble plugin and the native dashboard
  (default ``<repo parent>/battery_SOH``)
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str) -> Path | None:
    configured = os.environ.get(name, "").strip()
    return Path(configured).expanduser() if configured else None


def data_root() -> Path:
    """Supervised data root: ``SOHBENCH_DATA_DIR`` when set, else ``<repo>/data``."""
    return _env_path("SOHBENCH_DATA_DIR") or PROJECT_ROOT / "data"


def default_data_dir() -> Path:
    """Backwards-compatible alias of :func:`data_root`."""
    return data_root()


def default_curve_dir() -> Path:
    return _env_path("SOHBENCH_CURVE_DIR") or data_root() / "curves"


def battery_soh_root() -> Path:
    """Root of the optional ``battery_SOH`` checkout (plugins, native dashboard)."""
    return _env_path("BATTERY_SOH_ROOT") or PROJECT_ROOT.parent / "battery_SOH"


DEFAULT_DATA_DIR = data_root()
DEFAULT_CURVE_DIR = default_curve_dir()
DEFAULT_BATTERY_SOH_ROOT = battery_soh_root()
