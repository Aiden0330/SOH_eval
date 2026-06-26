import pandas as pd
from pathlib import Path
from sohbench.cli import run_benchmark

def _fixture(tmp_path):
    proc = tmp_path/"processed"; proc.mkdir()
    rows=[]
    for cell in ["c1","c2","c3"]:
        for k in range(15):
            rows.append(("hust",cell,k,1.0-0.01*k,float(k),float(k*k)))
    pd.DataFrame(rows, columns=["dataset","cell_id","cycle_index","soh","f1","f2"]
                ).to_pickle(proc/"hust_features.pkl")
    return tmp_path

def test_end_to_end(tmp_path):
    root=_fixture(tmp_path)
    out = run_benchmark(data_dir=root, datasets=["hust"], estimators=["ridge"],
                        protocols=["loco"], seeds=[0,1,2,3,4], out_dir=tmp_path/"runs")
    assert (Path(out)/"leaderboard.json").exists()
    assert (Path(out)/"results.parquet").exists()
