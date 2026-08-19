from __future__ import annotations
import argparse
from .data import DatasetLoader
from .registry import get_estimator
from .protocols import LeaveOneCellOut, LeaveOneDatasetOut, LeaveOneChemistryOut
from .tasks import SOHEstimationTask
from .runner import BenchmarkRunner
from .report import write_artifacts
import sohbench.estimators.baselines  # noqa: F401  (register ridge/gbdt/svr/gpr)
import sohbench.estimators.mechanism  # noqa: F401  (register extended baselines/gray-box method)
import sohbench.estimators.physics   # noqa: F401  (register phys_resid)
import sohbench.estimators.neural    # noqa: F401  (register mlp)
import sohbench.estimators.byo       # noqa: F401  (register byo_skb_ridge)
import sohbench.featuresets          # noqa: F401  (register full/voltage/ic_dva/partial_window/capacity_time)


def _register_optional_torch_estimators():
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    import sohbench.estimators.deep  # noqa: F401  (register lstm/cnn1d/transformer)
    import sohbench.estimators.pinn  # noqa: F401  (register pinn)


_register_optional_torch_estimators()

_PROTOCOLS = {"loco": LeaveOneCellOut, "lodo": LeaveOneDatasetOut, "lcho": LeaveOneChemistryOut}


def run_benchmark(data_dir, datasets, estimators, protocols, seeds, out_dir,
                  featuresets=None):
    if featuresets is None:
        featuresets = ["full"]
    loader = DatasetLoader(data_dir)
    datasets = loader.resolve_datasets(datasets)
    table, meta, qc = loader.load(datasets)
    ests = [get_estimator(n) for n in estimators]
    protos = {p: _PROTOCOLS[p]() for p in protocols}
    rs = BenchmarkRunner(seeds=seeds).run(SOHEstimationTask(), table=table, meta=meta,
                                          protocols=protos, estimators=ests,
                                          featuresets=featuresets)
    # NOTE: do NOT force-set rs["track"], rs["feature_mode"], or rs["backend"] globally.
    # The runner sets these per-estimator; report.py reads them.
    # We add a dataset column for config traceability.
    rs["dataset"] = ",".join(datasets)
    return write_artifacts(rs, qc_report=qc, task=SOHEstimationTask(),
                           config={"datasets": datasets, "estimators": estimators,
                                   "protocols": protocols, "seeds": seeds,
                                   "featuresets": featuresets}, out_dir=out_dir)


def main(argv=None):
    p = argparse.ArgumentParser("sohbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data-dir", required=True)
    r.add_argument("--datasets", nargs="+", default=["all"])
    r.add_argument("--estimators", nargs="+", default=["ridge"])
    r.add_argument("--protocols", nargs="+", default=["loco"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    r.add_argument("--featuresets", nargs="+", default=["full"])
    r.add_argument("--out-dir", default="runs")
    a = p.parse_args(argv)
    out = run_benchmark(a.data_dir, a.datasets, a.estimators, a.protocols,
                        a.seeds, a.out_dir, featuresets=a.featuresets)
    print(out)


if __name__ == "__main__":
    main()
