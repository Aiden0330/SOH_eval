import pytest
from pathlib import Path
from sohbench.cli import run_benchmark
from sohbench.paths import data_root

DATA = data_root()

@pytest.mark.skipif(not (DATA/"processed").exists(), reason="real data dir absent (set SOHBENCH_DATA_DIR)")
def test_four_datasets_run(tmp_path):
    out = run_benchmark(data_dir=DATA, datasets=["nasa","calce_cs2","hust","oxford"],
                        estimators=["ridge"], protocols=["loco","lodo","lcho"],
                        seeds=[0,1,2,3,4], out_dir=tmp_path)
    assert (Path(out)/"leaderboard.json").exists()
