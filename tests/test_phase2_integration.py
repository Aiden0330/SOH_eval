from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohbench.cli import run_benchmark


def test_dl_extra_declares_torch():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    deps = data["project"]["optional-dependencies"]
    assert "dl" in deps
    assert "torch" in deps["dl"]


def test_cli_import_registers_torch_estimators_when_torch_is_available():
    pytest.importorskip("torch")
    import sohbench.cli  # noqa: F401
    from sohbench.registry import get_estimator

    for name in ("lstm", "cnn1d", "transformer", "pinn"):
        assert get_estimator(name).name == name


def test_cli_run_records_pytorch_backend_for_pinn(tmp_path):
    pytest.importorskip("torch")

    proc = tmp_path / "processed"
    proc.mkdir()
    rows = []
    for cell in ("c1", "c2"):
        for cycle in range(8):
            frac = cycle / 7
            rows.append({
                "dataset": "hust",
                "cell_id": cell,
                "cycle_index": cycle,
                "soh": 1.0 - 0.04 * frac,
                "cycle_fraction": frac,
                "f1": frac,
                "f2": float(np.sin(frac)),
            })
    pd.DataFrame(rows).to_pickle(proc / "hust_features.pkl")

    out = run_benchmark(
        data_dir=tmp_path,
        datasets=["hust"],
        estimators=["pinn"],
        protocols=["loco"],
        seeds=[0],
        featuresets=["full"],
        out_dir=tmp_path / "runs",
    )

    results = pd.read_parquet(Path(out) / "results.parquet")
    assert set(results["backend"]) == {"pytorch"}
