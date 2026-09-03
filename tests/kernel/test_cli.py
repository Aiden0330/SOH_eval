"""The minimal kernel CLI: load, preflight, score (Task 11)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohkernel.cli import main


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for cell in range(3):
        for k in range(40):
            soh = 1.0 - 0.2 * k / 39 + rng.normal(0, 0.003)
            rows.append({"dataset": "synthetic", "cell_id": f"c{cell}", "cycle_index": k, "soh": soh,
                         "voltage_mean": 3.6 + 0.2 * soh + rng.normal(0, 0.02),
                         "voltage_std": 0.05 + rng.normal(0, 0.005),
                         "voltage_min": 3.0 + 0.1 * soh + rng.normal(0, 0.01),
                         "voltage_max": 4.1 + 0.05 * soh + rng.normal(0, 0.01)})
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(rows).to_pickle(processed / "synthetic_features.pkl")
    return tmp_path


def test_load_prints_the_qc_report(data_dir, capsys):
    assert main(["load", "--data-dir", str(data_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["datasets"] == ["synthetic"] and payload["rows"] == 120 and payload["cells"] == 3
    assert payload["qc"]["pass_rate"] == 1.0


def test_preflight_prints_the_admission_verdict(data_dir, capsys):
    assert main(["preflight", "--data-dir", str(data_dir), "--featuresets", "voltage", "--seeds", "0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True and payload["combinations"][0]["status"] == "ready"
    assert payload["request"]["datasets"] == ["synthetic"]


def test_score_writes_run_artifacts(data_dir, tmp_path, capsys):
    out_dir = tmp_path / "runs"
    assert main(["score", "--data-dir", str(data_dir), "--featuresets", "voltage", "--seeds", "0", "1",
                 "--out-dir", str(out_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    assert run_dir.parent == out_dir and (run_dir / "leaderboard.json").is_file()
    assert payload["units"] == 6  # 3 cells x 2 seeds
    leaderboard = json.loads((run_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert leaderboard[0]["estimator"] == "ridge" and leaderboard[0]["units"]["outcome"] == "complete"


def test_score_refuses_a_blocked_matrix(data_dir, capsys):
    # the 'partial_window' featureset selects nothing on this table -> blocked, exit code 2
    assert main(["score", "--data-dir", str(data_dir), "--featuresets", "partial_window", "--seeds", "0"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False and payload["reasons"]
