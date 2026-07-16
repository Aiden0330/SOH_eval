import json
from pathlib import Path

import pandas as pd

from scripts.build_frontend_artifacts import BuildFrontendArtifactsConfig, build_frontend_artifacts


def _write_run(path: Path, rows: list[dict], config: dict, qc: dict | None = None) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path / "results.parquet")
    (path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "qc_report.json").write_text(json.dumps(qc or {"per_dataset": {}}), encoding="utf-8")


def _metric_rows(
    estimator: str,
    featureset: str,
    held_out: str,
    mae: float,
    seed: int = 0,
    candidate_id: str | None = None,
) -> list[dict]:
    rows = []
    for metric, value in [
        ("mae", mae),
        ("picp", 0.91),
        ("monotonicity_violation_rate", 0.0),
        ("value_range_violation", 0.0),
    ]:
        row = {
                "estimator": estimator,
                "feature_mode": "unified",
                "backend": "sklearn",
                "protocol": "lodo",
                "dataset": "all",
                "held_out_key": held_out,
                "fold": 0,
                "seed": seed,
                "metric": metric,
                "value": value,
                "featureset": featureset,
                "track": "unified",
                "ci_source": "weighted_conformal",
        }
        if candidate_id is not None:
            row["candidate_id"] = candidate_id
        rows.append(row)
    return rows


def test_build_frontend_artifacts_merges_runs_and_syncs_frontend(tmp_path):
    base_run = tmp_path / "runs" / "base"
    addon_run = tmp_path / "runs" / "addon"
    archive_dir = tmp_path / "archive"
    frontend_dir = tmp_path / "frontend-data"

    _write_run(
        base_run,
        _metric_rows("ridge", "full", "nasa", 0.20)
        + _metric_rows("mlp", "partial_window", "nasa", 0.30),
        {
            "phase": "base",
            "datasets": ["nasa"],
            "estimators": ["ridge", "mlp"],
            "featuresets": ["full", "partial_window"],
            "protocols": ["lodo"],
            "seeds": [0],
        },
        {"per_dataset": {"nasa": {"rows": 10}}},
    )
    addon_rows = []
    for seed, mae in [(0, 0.11), (1, 0.12), (2, 0.13)]:
        addon_rows += _metric_rows("mlp", "partial_window", "nasa", mae, seed=seed)
    addon_rows += _metric_rows("new_algo", "ic_dva", "future_ds", 0.08)
    _write_run(
        addon_run,
        addon_rows,
        {
            "phase": "addon",
            "datasets": ["nasa", "future_ds"],
            "estimators": ["mlp", "new_algo"],
            "featuresets": ["partial_window", "ic_dva"],
            "protocols": ["lodo"],
            "seeds": [0, 1, 2],
        },
        {"per_dataset": {"nasa": {"rows": 10}, "future_ds": {"rows": 5}}},
    )
    pd.DataFrame(
        [
            {
                "estimator": "mlp",
                "featureset": "partial_window",
                "mae_mean": 0.12,
                "mae_std": 0.01,
                "mae_min": 0.11,
                "mae_max": 0.13,
            }
        ]
    ).to_csv(addon_run / "seed_stability_summary.csv", index=False)

    artifact_dir = build_frontend_artifacts(
        BuildFrontendArtifactsConfig(
            source_runs=[base_run, addon_run],
            archive_dir=archive_dir,
            frontend_data_dir=frontend_dir,
            phase="combined_for_frontend",
            replace_cells=[("mlp", "partial_window")],
        )
    )

    assert artifact_dir.exists()
    for filename in ["leaderboard.json", "dataset_leaderboard.json", "qc_report.json", "run_config.json"]:
        assert (artifact_dir / filename).exists()
        assert (frontend_dir / filename).exists()

    leaderboard = json.loads((frontend_dir / "leaderboard.json").read_text(encoding="utf-8"))
    dataset_leaderboard = json.loads((frontend_dir / "dataset_leaderboard.json").read_text(encoding="utf-8"))
    config = json.loads((frontend_dir / "run_config.json").read_text(encoding="utf-8"))

    assert {entry["estimator"] for entry in leaderboard} == {"ridge", "mlp", "new_algo"}
    mlp = next(entry for entry in leaderboard if entry["estimator"] == "mlp")
    assert mlp["metrics"]["mae"]["mean"] == 0.12
    assert {entry["held_out_key"] for entry in dataset_leaderboard} == {"nasa", "future_ds"}
    assert config["phase"] == "combined_for_frontend"
    assert config["datasets"] == ["future_ds", "nasa"]
    assert config["estimators"] == ["mlp", "new_algo", "ridge"]
    assert config["seeds"] == [0]
    assert config["source_seeds"] == [0, 1, 2]
    assert config["stability_replaced_cells"] == [{"estimator": "mlp", "featureset": "partial_window"}]
    assert config["stability_summary"][0]["mae_mean"] == 0.12


def test_build_frontend_artifacts_collapses_loco_held_out_keys_to_dataset(tmp_path):
    run = tmp_path / "runs" / "rwth"
    archive_dir = tmp_path / "archive"
    frontend_dir = tmp_path / "frontend-data"

    _write_run(
        run,
        _metric_rows("trusted_huber_decay", "rwth_cycle_shape", "rwth_isea_public/rwth_cell_001", 0.02)
        + _metric_rows("trusted_huber_decay", "rwth_cycle_shape", "rwth_isea_public/rwth_cell_002", 0.04),
        {
            "phase": "trusted",
            "datasets": ["rwth_isea_public"],
            "estimators": ["trusted_huber_decay"],
            "featuresets": ["rwth_cycle_shape"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        {"per_dataset": {"rwth_isea_public": {"rows": 100}}},
    )

    build_frontend_artifacts(
        BuildFrontendArtifactsConfig(
            source_runs=[run],
            archive_dir=archive_dir,
            frontend_data_dir=frontend_dir,
        )
    )

    config = json.loads((frontend_dir / "run_config.json").read_text(encoding="utf-8"))

    assert config["datasets"] == ["rwth_isea_public"]


def test_build_frontend_artifacts_copies_inference_and_model_files(tmp_path):
    run = tmp_path / "runs" / "with_inference"
    archive_dir = tmp_path / "archive"
    frontend_dir = tmp_path / "frontend-data"

    _write_run(
        run,
        _metric_rows("ridge", "full", "nasa", 0.20),
        {
            "phase": "base",
            "datasets": ["nasa"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["lodo"],
            "seeds": [0],
        },
        {"per_dataset": {"nasa": {"rows": 10}}},
    )
    (run / "inference_benchmark.json").write_text(
        json.dumps({"records": [{"model": "ridge_full", "dataset": "nasa"}]}),
        encoding="utf-8",
    )
    (run / "model_catalog.json").write_text(
        json.dumps({"models": [{"name": "ridge_full"}]}),
        encoding="utf-8",
    )

    artifact_dir = build_frontend_artifacts(
        BuildFrontendArtifactsConfig(
            source_runs=[run],
            archive_dir=archive_dir,
            frontend_data_dir=frontend_dir,
        )
    )

    assert (artifact_dir / "inference_benchmark.json").exists()
    assert (artifact_dir / "model_catalog.json").exists()
    assert (frontend_dir / "inference_benchmark.json").exists()
    assert (frontend_dir / "model_catalog.json").exists()


def test_build_frontend_artifacts_preserves_distinct_candidate_ids_during_deduplication(tmp_path):
    run = tmp_path / "runs" / "candidates"
    archive_dir = tmp_path / "archive"
    frontend_dir = tmp_path / "frontend-data"

    _write_run(
        run,
        _metric_rows("ridge", "full", "nasa", 0.10, candidate_id="candidate-a")
        + _metric_rows("ridge", "full", "nasa", 0.20, candidate_id="candidate-b"),
        {
            "phase": "candidates",
            "datasets": ["nasa"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["lodo"],
            "seeds": [0],
        },
    )

    build_frontend_artifacts(
        BuildFrontendArtifactsConfig(
            source_runs=[run],
            archive_dir=archive_dir,
            frontend_data_dir=frontend_dir,
        )
    )

    leaderboard = json.loads((frontend_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert {(entry["candidate_id"], entry["metrics"]["mae"]["mean"]) for entry in leaderboard} == {
        ("candidate-a", 0.10),
        ("candidate-b", 0.20),
    }


def test_build_frontend_artifacts_normalizes_mixed_legacy_and_candidate_runs(tmp_path):
    legacy_run = tmp_path / "runs" / "legacy"
    candidate_run = tmp_path / "runs" / "candidates"
    config = {
        "phase": "mixed",
        "datasets": ["nasa"],
        "estimators": ["ridge"],
        "featuresets": ["full"],
        "protocols": ["lodo"],
        "seeds": [0],
    }
    _write_run(
        legacy_run,
        _metric_rows("ridge", "full", "nasa", 0.30),
        config,
    )
    _write_run(
        candidate_run,
        _metric_rows("ridge", "full", "nasa", 0.10, candidate_id="candidate-a")
        + _metric_rows("ridge", "full", "nasa", 0.20, candidate_id="candidate-b"),
        config,
    )

    for label, source_runs in [
        ("legacy-first", [legacy_run, candidate_run]),
        ("candidate-first", [candidate_run, legacy_run]),
    ]:
        frontend_dir = tmp_path / f"frontend-{label}"
        build_frontend_artifacts(
            BuildFrontendArtifactsConfig(
                source_runs=source_runs,
                archive_dir=tmp_path / f"archive-{label}",
                frontend_data_dir=frontend_dir,
            )
        )

        leaderboard = json.loads((frontend_dir / "leaderboard.json").read_text(encoding="utf-8"))
        assert {(entry["candidate_id"], entry["metrics"]["mae"]["mean"]) for entry in leaderboard} == {
            ("ridge", 0.30),
            ("candidate-a", 0.10),
            ("candidate-b", 0.20),
        }


def test_build_frontend_artifacts_keeps_all_legacy_output_candidate_free(tmp_path):
    run = tmp_path / "runs" / "legacy"
    frontend_dir = tmp_path / "frontend"
    _write_run(
        run,
        _metric_rows("ridge", "full", "nasa", 0.30),
        {
            "phase": "legacy",
            "datasets": ["nasa"],
            "estimators": ["ridge"],
            "featuresets": ["full"],
            "protocols": ["lodo"],
            "seeds": [0],
        },
    )

    build_frontend_artifacts(
        BuildFrontendArtifactsConfig(
            source_runs=[run],
            archive_dir=tmp_path / "archive",
            frontend_data_dir=frontend_dir,
        )
    )

    leaderboard = json.loads((frontend_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert all("candidate_id" not in entry for entry in leaderboard)
