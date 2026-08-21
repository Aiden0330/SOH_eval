from __future__ import annotations
import numpy as np, pandas as pd
from . import metrics as M
from .conformal import ConformalWrapper, WeightedConformalWrapper
from .contracts import feature_columns
from .feature_safety import is_label_source_feature
from .features import unified_features
from .featuresets import project
from .method_contracts import method_contract
from .registry import get_featureset
import sohbench.featuresets  # noqa: F401 — ensure the 5 named sets are registered


class BenchmarkRunner:
    def __init__(self, seeds=(0, 1, 2, 3, 4)):
        self.seeds = list(seeds)

    def run(self, task, table, meta, protocols, estimators,
            featuresets=None, on_model_fitted=None) -> pd.DataFrame:
        """Run the benchmark matrix over estimators × protocols × featuresets × seeds.

        Parameters
        ----------
        featuresets:
            List of featureset names to iterate over for *unified* estimators,
            e.g. ``["full", "voltage"]``.  Defaults to ``["full"]`` when *None*
            (backward-compatible).  BYO estimators ignore this parameter and
            always receive the full (unprojected) table.
        """
        if featuresets is None:
            featuresets = ["full"]

        records = []

        for est_factory in estimators:
            contract = method_contract(est_factory)
            est_name = contract.name
            is_byo = contract.feature_mode == "byo"

            if is_byo:
                # BYO estimators: single pseudo-featureset, no projection.
                # Label-source columns (the quantities SOH is computed from)
                # are banned as model inputs on every track, so they are
                # removed before any BYO code can observe them.
                banned = [
                    c for c in feature_columns(table) if is_label_source_feature(c)
                ]
                fs_iterations = [("(byo)", table.drop(columns=banned), "byo")]
            else:
                # Ordinary estimators remain a free Cartesian product. Joint
                # or gray-box methods may declare an explicit feature-profile
                # contract; incompatible cells are omitted consistently with
                # the server-side preflight.
                selected_featuresets = contract.feature_iterations(featuresets)
                fs_iterations = [
                    (fs_name, project(table, fs_name), "unified")
                    for fs_name in selected_featuresets
                ]

            for fs_name, data_table, track in fs_iterations:
                for pname, proto in protocols.items():
                    cross = getattr(proto, "cross_domain", False)
                    for fi, fold in enumerate(proto.generate(data_table, meta)):
                        tr = data_table.iloc[fold.train_idx]
                        te = data_table.iloc[fold.test_idx]
                        for seed in self.seeds:
                            np.random.seed(seed)
                            rec_base = dict(
                                estimator=est_name, protocol=pname,
                                held_out_key=fold.held_out_key, fold=fi,
                                seed=seed, ci_source="conformal",
                                featureset=fs_name, track=track,
                                feature_mode=contract.feature_mode,
                                backend=contract.backend,
                                method_family=contract.method_family,
                                pairing_mode=contract.pairing_mode,
                                method_category=contract.method_category,
                                combination_policy=contract.combination_policy,
                                execution_contract=contract.execution_contract,
                            )
                            try:
                                est = est_factory() if isinstance(est_factory, type) else est_factory
                                native_ci = hasattr(est, "predict_interval")
                                if not native_ci:
                                    alpha = 1 - task.acceptance["calibration"]["nominal"]
                                    if cross:
                                        est = WeightedConformalWrapper(est, alpha=alpha)
                                        rec_base["ci_source"] = "weighted_conformal"
                                    else:
                                        est = ConformalWrapper(est, alpha=alpha)
                                        rec_base["ci_source"] = "conformal"
                                else:
                                    rec_base["ci_source"] = "native"
                                ytr = tr[task.target].to_numpy()
                                est.fit(tr, ytr, meta)
                                if on_model_fitted is not None:
                                    on_model_fitted(
                                        estimator=est,
                                        record=dict(rec_base),
                                        train_table=tr.copy(),
                                        test_table=te.copy(),
                                        meta=meta,
                                    )
                                yte = te[task.target].to_numpy()
                                yp = est.predict(te, meta)
                                row = self._point_metrics(task, yte, yp)
                                lo, hi = est.predict_interval(te, meta)
                                row["picp"] = M.picp(yte, lo, hi)
                                row.update(self._trajectory_metrics(task, te, yp))
                            except Exception as e:  # isolate failed fold
                                row = {m: float("nan") for m in
                                       (*task.point_metrics, *task.interval_metrics,
                                        *task.trajectory_metrics)}
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
