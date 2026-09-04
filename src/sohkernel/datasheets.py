"""Dataset datasheets (Spec 06): the human-signed record behind every chemistry or label claim.

Datasheets are YAML files under ``<data root>/datasheets/<name>.yaml``.  Any field the agent could
not verify carries the literal ``VERIFY``; a dataset with ``VERIFY`` on its chemistry cannot support
a chemistry-level claim (Spec 01 §1), which the preflight surfaces on ``lcho`` folds.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

VERIFY = "VERIFY"


def default_data_dir() -> Path:
    configured = os.environ.get("SOHBENCH_DATA_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parents[2] / "data"


def datasheets_dir(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir).expanduser() / "datasheets" if data_dir is not None else default_data_dir() / "datasheets"


def load_datasheet(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or "name" not in payload:
        raise ValueError(f"datasheet {path} must be a mapping with a 'name'")
    return payload


def load_datasheets(data_dir: str | Path | None = None) -> dict[str, dict]:
    directory = datasheets_dir(data_dir)
    if not directory.is_dir():
        return {}
    sheets = {}
    for path in sorted(directory.glob("*.yaml")):
        sheet = load_datasheet(path)
        sheets[str(sheet["name"])] = sheet
    return sheets


def unverified_fields(sheet: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Dotted paths of every field still marked ``VERIFY``."""
    found: list[str] = []
    for key, value in sheet.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            found.extend(unverified_fields(value, path + "."))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    found.extend(unverified_fields(item, f"{path}[{index}]."))
                elif isinstance(item, str) and VERIFY in item:
                    found.append(f"{path}[{index}]")
        elif isinstance(value, str) and VERIFY in value:
            found.append(path)
    return found


def chemistry_verified(sheet: Mapping[str, Any]) -> bool:
    chemistry = sheet.get("chemistry") or {}
    if not isinstance(chemistry, Mapping):
        return False
    return not any(field.startswith("chemistry.") for field in unverified_fields({"chemistry": dict(chemistry)}))


def is_signed(sheet: Mapping[str, Any]) -> bool:
    signed = sheet.get("signed") or {}
    return isinstance(signed, Mapping) and bool(signed.get("by")) and VERIFY not in str(signed.get("by")) and bool(signed.get("on"))
