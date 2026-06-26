from __future__ import annotations
import numpy as np, pandas as pd
from . import metrics as M
from .conformal import ConformalWrapper
from .features import unified_features
from .featuresets import project
from .registry import get_featureset
import sohbench.featuresets  # noqa: F401 — ensure the 5 named sets are registered

class BenchmarkRunner:
    def __init__(self, seeds=(0,1,2,3,4)):
        self.seeds = list(seeds)

    def run(self, task, table, meta, protocols, estimators,
            featuresets=None) -> pd.DataFrame:
        """Run the benchmark matrix over estimators × protocols × featuresets × seeds.

        Parameters
        ----------
        featuresets:
            List of featureset names to iterate over, e.g. ``["full", "voltage"]``.
            Defaults to ``["full"]`` when *None* (backward-compatible).
        """
        if featuresets is None:
            featuresets = ["full"]

        records = []
        for fs_name in featuresets:
            projected = project(table, fs_name)
            for est_factory in estimators:
                est_name = getattr(est_factory, "name", est_factory.__name__)
                for pname, proto in protocols.items():
                    for fi, fold in enumerate(proto.generate(projected, meta)):
                        tr = projected.iloc[fold.train_idx]
                        te = projected.iloc[fold.test_idx]
                        for seed in self.seeds:
                            np.random.seed(seed)
                            rec_base = dict(estimator=est_name, protocol=pname,
                                            held_out_key=fold.held_out_key, fold=fi,
                                            seed=seed, ci_source="conformal",
                                            featureset=fs_name)
                            try:
                                est = est_factory() if isinstance(est_factory, type) else est_factory
                                native_ci = hasattr(est, "predict_interval")
                                if not native_ci:
                                    est = ConformalWrapper(est, alpha=1 - task.acceptance["calibration"]["nominal"])
                                rec_base["ci_source"] = "native" if native_ci else "conformal"
                                ytr = tr[task.target].to_numpy()
                                est.fit(tr, ytr, meta)
                                yte = te[task.target].to_numpy()
                                yp = est.predict(te, meta)
                                row = self._point_metrics(task, yte, yp)
                                lo, hi = est.predict_interval(te, meta)
                                row["picp"] = M.picp(yte, lo, hi)
                                row.update(self._trajectory_metrics(task, te, yp))
                            except Exception as e:  # isolate failed fold
                                row = {m: float("nan") for m in
                                       (*task.point_metrics, *task.interval_metrics, *task.trajectory_metrics)}
                                rec_base["error"] = str(e)
                            for metric, value in row.items():
                                records.append({**rec_base, "metric": metric, "value": value})
        return pd.DataFrame.from_records(records)

    def _point_metrics(self, task, yt, yp):
        out = {}
        for m in task.point_metrics:
            out[m] = getattr(M, m)(yt, yp)
        return out

    def _trajectory_metrics(self, task, te, yp):
        order = np.argsort(te["cycle_index"].to_numpy())
        return {"monotonicity_violation_rate": M.monotonicity_violation_rate(yp[order]),
                "value_range_violation": M.value_range_violation(yp)}
