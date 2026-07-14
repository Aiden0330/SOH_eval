from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON contains a non-finite float")
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        _finite(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_id(prefix: str) -> str:
    clean = str(prefix).strip().upper()
    if not clean or not clean.replace("_", "").isalnum():
        raise ValueError(f"invalid id prefix: {prefix!r}")
    return f"{clean}-{uuid.uuid4().hex}"
