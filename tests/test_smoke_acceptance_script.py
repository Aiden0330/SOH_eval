import json
from pathlib import Path

import pytest

from scripts.run_smoke_acceptance import SmokeAcceptanceConfig, run_smoke_acceptance


def test_smoke_acceptance_runs_training_inference_and_syncs_frontend(tmp_path):
    config = SmokeAcceptanceConfig(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
        dataset_name="smoke_eval_demo",
    )

    summary = run_smoke_acceptance(config)

    assert summary["status"] == "succeeded"
    assert summary["training"]["status"] == "succeeded"
    assert summary["inference"]["status"] == "succeeded"
    assert summary["model"]["name"] in summary["training"]["saved_models"]

    training_dir = Path(summary["training"]["run_dir"])
    inference_dir = Path(summary["inference"]["run_dir"])
    assert (training_dir / "leaderboard.json").exists()
    assert (training_dir / "model_catalog.json").exists()
    assert (inference_dir / "inference_benchmark.json").exists()
    assert (inference_dir / "inference_run_config.json").exists()

    frontend_dir = tmp_path / "frontend_data"
    assert (frontend_dir / "leaderboard.json").exists()
    assert (frontend_dir / "model_catalog.json").exists()
    assert (frontend_dir / "inference_benchmark.json").exists()
    assert (frontend_dir / "inference_run_config.json").exists()
    assert (frontend_dir / "profiles" / "smoke" / "manifest.json").exists()
    assert (frontend_dir / "profiles" / "smoke" / "inference_benchmark.json").exists()
    assert (frontend_dir / "profiles" / "index.json").exists()

    benchmark = json.loads((frontend_dir / "inference_benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["summary"]["n_succeeded"] == 1
    assert benchmark["records"][0]["dataset"] == "smoke_eval_demo"
    assert benchmark["records"][0]["status"] == "succeeded"
    assert benchmark["records"][0]["n_samples"] > 0

    inference_config = json.loads((frontend_dir / "inference_run_config.json").read_text(encoding="utf-8"))
    assert inference_config["job_id"] == summary["inference"]["id"]
    assert inference_config["launch_strategy"] == "compatible_only"
    assert inference_config["selected_datasets"] == ["smoke_eval_demo"]
    assert inference_config["submitted_datasets"] == ["smoke_eval_demo"]
    assert inference_config["skipped_datasets"] == []
    profile_manifest = json.loads(
        (frontend_dir / "profiles" / "smoke" / "manifest.json").read_text(encoding="utf-8")
    )
    assert profile_manifest["profile"] == "smoke"
    assert profile_manifest["kind"] == "smoke_acceptance"
    assert profile_manifest["training_job_id"] == summary["training"]["id"]
    assert profile_manifest["inference_job_id"] == summary["inference"]["id"]
    assert summary["profile"]["name"] == "smoke"


def test_smoke_acceptance_fails_when_requested_model_is_missing(tmp_path):
    config = SmokeAcceptanceConfig(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
        dataset_name="smoke_eval_demo",
        model_name="missing_model",
    )

    with pytest.raises(RuntimeError, match="missing_model"):
        run_smoke_acceptance(config)
