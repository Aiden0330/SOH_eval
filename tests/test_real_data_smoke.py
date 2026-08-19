import os, pytest
from pathlib import Path
from sohbench.cli import run_benchmark

DATA = Path(__file__).resolve().parents[1] / "data"

@pytest.mark.skipif(not (DATA/"processed").exists(), reason="real data dir absent")
def test_four_datasets_run(tmp_path):
    out = run_benchmark(data_dir=DATA, datasets=["nasa","calce_cs2","hust","oxford"],
                        estimators=["ridge"], protocols=["loco","lodo","lcho"],
                        seeds=[0,1,2,3,4], out_dir=tmp_path)
    assert (Path(out)/"leaderboard.json").exists()
