from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Return the project-owned data root unless explicitly overridden."""
    configured = os.environ.get("SOHBENCH_DATA_DIR", "").strip()
    return Path(configured).expanduser() if configured else PROJECT_ROOT / "data"


def default_curve_dir() -> Path:
    configured = os.environ.get("SOHBENCH_CURVE_DIR", "").strip()
    return Path(configured).expanduser() if configured else default_data_dir() / "curves"


DEFAULT_DATA_DIR = default_data_dir()
DEFAULT_CURVE_DIR = default_curve_dir()
