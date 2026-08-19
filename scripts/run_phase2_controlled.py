from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sohbench.data import DatasetLoader
from sohbench.estimators.deep import CNN1DEstimator, LSTMEstimator, TransformerEstimator
from sohbench.estimators.pinn import PINNEstimator
from sohbench.protocols import LeaveOneDatasetOut
from sohbench.report import write_artifacts
from sohbench.runner import BenchmarkRunner
from sohbench.tasks import SOHEstimationTask
from sohbench.paths import DEFAULT_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tractable Phase-2 PyTorch SOH benchmark pass."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--featuresets", nargs="+", default=["partial_window"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-max", type=int, default=6000)
    parser.add_argument("--seq-epochs", type=int, default=8)
    parser.add_argument("--pinn-epochs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = DatasetLoader(args.data_dir)
    datasets = loader.resolve_datasets(args.datasets)
    print("datasets", datasets, flush=True)
    table, meta, qc = loader.load(datasets)
    print("table", table.shape, flush=True)

    task = SOHEstimationTask()
    runner = BenchmarkRunner(seeds=args.seeds)
    protocols = {"lodo": LeaveOneDatasetOut()}
    estimators = [
        LSTMEstimator(W=8, hidden=24, n_max=args.n_max, epochs=args.seq_epochs, lr=2e-3, seed=0),
        CNN1DEstimator(W=8, channels=24, n_max=args.n_max, epochs=args.seq_epochs, lr=2e-3, seed=0),
        TransformerEstimator(
            W=8, d_model=32, nhead=4, n_max=args.n_max, epochs=args.seq_epochs, lr=2e-3, seed=0
        ),
        PINNEstimator(hidden=32, epochs=args.pinn_epochs, lr=2e-3, seed=0, n_max=args.n_max),
    ]

    frames = []
    for estimator in estimators:
        print("running", estimator.name, flush=True)
        frame = runner.run(
            task,
            table=table,
            meta=meta,
            protocols=protocols,
            estimators=[estimator],
            featuresets=args.featuresets,
        )
        frames.append(frame)
        ok_mae = frame.loc[frame["metric"].eq("mae"), "value"].notna().sum()
        print("finished", estimator.name, "mae_count", int(ok_mae), flush=True)

    results = pd.concat(frames, ignore_index=True)
    results["dataset"] = ",".join(datasets)
    out = write_artifacts(
        results,
        qc_report=qc,
        task=task,
        config={
            "phase": "phase2_controlled",
            "datasets": datasets,
            "estimators": [estimator.name for estimator in estimators],
            "protocols": ["lodo"],
            "seeds": args.seeds,
            "featuresets": args.featuresets,
            "n_max": args.n_max,
            "seq_epochs": args.seq_epochs,
            "pinn_epochs": args.pinn_epochs,
        },
        out_dir=Path(args.out_dir),
    )
    print("out", out, flush=True)


if __name__ == "__main__":
    main()
