import ast
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sohbench.frontend_publish as frontend_publish_module

from scripts.run_smoke_acceptance import (
    FEATURESET_TEMPLATE,
    SmokeAcceptanceConfig,
    _smoke_dataset_csv,
    run_smoke_acceptance,
)


def test_smoke_featureset_template_is_target_free_and_prefix_invariant():
    source = FEATURESET_TEMPLATE.format(name="smoke_cycle_shape")
    tree = ast.parse(source)
    forbidden_subscripts = []
    forbidden_methods = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "soh":
            forbidden_subscripts.append(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"groupby", "max", "min"}:
            forbidden_methods.append(node.func.attr)

    assert "smoke_log_cycle" in source
    assert "smoke_voltage_margin" in source
    assert 'out["voltage_mean"]' in source
    assert not forbidden_subscripts
    assert not forbidden_methods


def test_smoke_dataset_matches_the_causal_generated_feature_contract():
    config = SmokeAcceptanceConfig(dataset_name="smoke_eval_demo", cells=1, cycles=3)
    frame = pd.read_csv(BytesIO(_smoke_dataset_csv(config)))

    assert {"smoke_log_cycle", "smoke_voltage_margin"} <= set(frame.columns)
    assert {"smoke_cycle_norm", "smoke_fade_from_full"}.isdisjoint(frame.columns)
    np.testing.assert_allclose(frame["smoke_log_cycle"], np.log1p(frame["cycle_index"]))
    np.testing.assert_allclose(frame["smoke_voltage_margin"], 4.25 - frame["voltage_mean"])


def test_smoke_acceptance_runs_training_inference_and_syncs_frontend(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        frontend_publish_module,
        "_sync_compatibility_view",
        lambda _target, _generation, _generation_name: None,
    )
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

    frontend_dir = frontend_publish_module.resolve_published_directory(
        tmp_path / "frontend_data"
    )
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
