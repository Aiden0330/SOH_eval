from pathlib import Path

from sohbench.fabric.ids import canonical_json, new_id, sha256_file, sha256_json


def test_sha256_json_is_key_order_independent():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_canonical_json_rejects_nan():
    import pytest

    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"value": float("nan")})


def test_sha256_file_and_new_id(tmp_path: Path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"evidence")
    assert len(sha256_file(path)) == 64
    assert new_id("RUN").startswith("RUN-")
