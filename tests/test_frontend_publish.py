import json
from concurrent.futures import ThreadPoolExecutor

from sohbench.frontend_publish import (
    load_frontend_profile_index,
    publish_frontend_profile,
)


def test_concurrent_profile_publish_is_atomic_and_keeps_every_profile(tmp_path):
    frontend_data = tmp_path / "frontend_data"
    source = tmp_path / "source"
    source.mkdir()
    (source / "leaderboard.json").write_text(
        json.dumps([{"estimator": "ridge", "featureset": "full"}]),
        encoding="utf-8",
    )

    def publish(name):
        return publish_frontend_profile(
            frontend_data,
            name,
            {
                "label": f"Profile {name}",
                "datasets": ["nasa"],
                "training_job_id": f"train-{name}",
            },
            source_dirs=[source],
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(publish, ["alpha", "beta", "gamma", "delta"]))

    index = load_frontend_profile_index(frontend_data)
    assert [item["name"] for item in index["profiles"]] == [
        "alpha", "beta", "delta", "gamma"
    ]
    for name in ("alpha", "beta", "gamma", "delta"):
        assert json.loads(
            (frontend_data / "profiles" / name / "leaderboard.json").read_text(encoding="utf-8")
        )[0]["estimator"] == "ridge"
    assert not (frontend_data / "profiles" / ".publish.lock").exists()
    assert list((frontend_data / "profiles").rglob("*.tmp")) == []


def test_profile_catalog_recovers_new_manifests_when_static_index_is_stale(tmp_path):
    frontend_data = tmp_path / "frontend_data"
    publish_frontend_profile(frontend_data, "fresh", {"datasets": ["nasa"]})
    (frontend_data / "profiles" / "index.json").write_text(
        '{"profiles": [{"name": "legacy", "datasets": ["calce_cs2"]}]}',
        encoding="utf-8",
    )

    index = load_frontend_profile_index(frontend_data)

    assert [item["name"] for item in index["profiles"]] == ["fresh", "legacy"]
    assert next(item for item in index["profiles"] if item["name"] == "fresh")["datasets"] == ["nasa"]
