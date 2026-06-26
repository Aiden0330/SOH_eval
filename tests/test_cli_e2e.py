import json
import pandas as pd
from pathlib import Path
from sohbench.cli import run_benchmark


def _fixture(tmp_path, include_voltage_cols=False):
    proc = tmp_path / "processed"; proc.mkdir()
    rows = []
    for cell in ["c1", "c2", "c3"]:
        for k in range(15):
            row = ["hust", cell, k, 1.0 - 0.01 * k, float(k), float(k * k)]
            if include_voltage_cols:
                row += [float(k) * 0.1, float(k) * 0.2,
                        float(k) * 0.05, float(k) * 0.3]
            rows.append(row)
    cols = ["dataset", "cell_id", "cycle_index", "soh", "f1", "f2"]
    if include_voltage_cols:
        cols += ["voltage_mean", "voltage_std", "voltage_min", "voltage_max"]
    pd.DataFrame(rows, columns=cols).to_pickle(proc / "hust_features.pkl")
    return tmp_path


def test_end_to_end(tmp_path):
    root = _fixture(tmp_path)
    out = run_benchmark(data_dir=root, datasets=["hust"], estimators=["ridge"],
                        protocols=["loco"], seeds=[0, 1, 2, 3, 4],
                        out_dir=tmp_path / "runs")
    assert (Path(out) / "leaderboard.json").exists()
    assert (Path(out) / "results.parquet").exists()


def test_end_to_end_featuresets(tmp_path):
    """run_benchmark with multiple featuresets+estimators → leaderboard has both featuresets."""
    root = _fixture(tmp_path, include_voltage_cols=True)
    out = run_benchmark(
        data_dir=root,
        datasets=["hust"],
        estimators=["ridge", "phys_resid"],
        protocols=["loco"],
        seeds=[0, 1, 2],
        featuresets=["full", "voltage"],
        out_dir=tmp_path / "runs",
    )
    lb = json.loads((Path(out) / "leaderboard.json").read_text())
    fs_in_lb = {e["featureset"] for e in lb}
    assert fs_in_lb == {"full", "voltage"}, \
        f"expected both featuresets in leaderboard, got {fs_in_lb}"
