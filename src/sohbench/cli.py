from __future__ import annotations
import argparse
from .data import DatasetLoader
from .registry import get_estimator
from .protocols import LeaveOneCellOut, LeaveOneDatasetOut, LeaveOneChemistryOut
from .tasks import SOHEstimationTask
from .runner import BenchmarkRunner
from .report import write_artifacts
import sohbench.estimators.baselines  # noqa: F401  (register baselines)

_PROTOCOLS = {"loco": LeaveOneCellOut, "lodo": LeaveOneDatasetOut, "lcho": LeaveOneChemistryOut}

def run_benchmark(data_dir, datasets, estimators, protocols, seeds, out_dir):
    table, meta, qc = DatasetLoader(data_dir).load(datasets)
    ests = [get_estimator(n) for n in estimators]
    protos = {p: _PROTOCOLS[p]() for p in protocols}
    rs = BenchmarkRunner(seeds=seeds).run(SOHEstimationTask(), table=table, meta=meta,
                                          protocols=protos, estimators=ests)
    rs["feature_mode"] = "unified"; rs["backend"] = "sklearn"; rs["dataset"] = ",".join(datasets)
    return write_artifacts(rs, qc_report=qc, task=SOHEstimationTask(),
                           config={"datasets":datasets,"estimators":estimators,
                                   "protocols":protocols,"seeds":seeds}, out_dir=out_dir)

def main(argv=None):
    p = argparse.ArgumentParser("sohbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--datasets", nargs="+", required=True)
    r.add_argument("--estimators", nargs="+", default=["ridge"])
    r.add_argument("--protocols", nargs="+", default=["loco"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    r.add_argument("--out-dir", default="runs")
    a = p.parse_args(argv)
    out = run_benchmark(a.data_dir, a.datasets, a.estimators, a.protocols, a.seeds, a.out_dir)
    print(out)

if __name__ == "__main__":
    main()
