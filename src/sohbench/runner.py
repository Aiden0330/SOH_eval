from __future__ import annotations
import numpy as np, pandas as pd
from . import metrics as M
from .conformal import ConformalWrapper
from .features import unified_features

class BenchmarkRunner:
    def __init__(self, seeds=(0,1,2,3,4)):
        self.seeds = list(seeds)

    def run(self, task, table, meta, protocols, estimators) -> pd.DataFrame:
        records = []
        for est_factory in estimators:
            est_name = getattr(est_factory, "name", est_factory.__name__)
            for pname, proto in protocols.items():
                for fi, fold in enumerate(proto.generate(table, meta)):
                    tr, te = table.iloc[fold.train_idx], table.iloc[fold.test_idx]
                    for seed in self.seeds:
                        np.random.seed(seed)
                        rec_base = dict(estimator=est_name, protocol=pname,
                                        held_out_key=fold.held_out_key, fold=fi, seed=seed)
                        try:
                            est = est_factory() if isinstance(est_factory, type) else est_factory
                            if not hasattr(est, "predict_interval"):
                                est = ConformalWrapper(est, alpha=1 - task.acceptance["calibration"]["nominal"])
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
