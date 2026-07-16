import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import sohbench.frontend_publish as frontend_publish_module

from scripts.run_real_migration_acceptance import (
    RealMigrationAcceptanceConfig,
    run_real_migration_acceptance,
)


def _write_dataset(data_dir: Path, name: str, *, cells: int = 3, cycles: int = 12) -> None:
    rows = []
    for cell in range(cells):
        for cycle in range(cycles):
            norm = cycle / max(cycles - 1, 1)
            rows.append(
                {
                    "dataset": name,
                    "cell_id": f"{name}_cell_{cell + 1}",
                    "cycle_index": cycle,
                    "soh": 1.0 - 0.025 * cycle - 0.001 * cell,
                    "voltage_mean": 3.65 - 0.05 * norm,
                    "voltage_std": 0.01 + 0.002 * norm,
                    "voltage_min": 3.32 - 0.03 * norm,
                    "voltage_max": 4.12 - 0.02 * norm,
                }
            )
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(processed / f"{name}_features.pkl")


def test_real_migration_acceptance_runs_train_to_transfer_profile(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        frontend_publish_module,
        "_sync_compatibility_view",
        lambda _target, _generation, _generation_name: None,
    )
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, "nasa")
    _write_dataset(data_dir, "calce_cs2")
    config = RealMigrationAcceptanceConfig(
        data_dir=data_dir,
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
        train_dataset="nasa",
        inference_datasets=("nasa", "calce_cs2"),
        profile_name="migration-smoke",
    )

    summary = run_real_migration_acceptance(config)

    assert summary["status"] == "succeeded"
    assert summary["training"]["status"] == "succeeded"
    assert summary["inference"]["status"] == "succeeded"
    assert summary["profile"]["name"] == "migration-smoke"

    frontend_dir = frontend_publish_module.resolve_published_directory(
        tmp_path / "frontend_data"
    )
    profile_dir = frontend_dir / "profiles" / "migration-smoke"
    assert (profile_dir / "manifest.json").exists()
    assert (profile_dir / "inference_benchmark.json").exists()
    manifest = json.loads((profile_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "migration_acceptance"
    assert manifest["datasets"] == ["nasa", "calce_cs2"]
    assert manifest["training_dataset"] == "nasa"

    benchmark = json.loads((profile_dir / "inference_benchmark.json").read_text(encoding="utf-8"))
    assert sorted({row["dataset"] for row in benchmark["records"]}) == ["calce_cs2", "nasa"]
    assert all(row["status"] == "succeeded" for row in benchmark["records"])


def test_real_migration_acceptance_cli_help_runs_from_repo_root():
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [sys.executable, "scripts/run_real_migration_acceptance.py", "--help"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--train-dataset" in result.stdout
