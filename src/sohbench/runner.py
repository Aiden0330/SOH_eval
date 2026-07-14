from __future__ import annotations
import copy
import inspect
import numpy as np, pandas as pd
from . import metrics as M
from .conformal import ConformalWrapper, WeightedConformalWrapper
from .features import unified_features
from .featuresets import project
from .registry import get_featureset
from .fabric.views import make_fold_views
import sohbench.featuresets  # noqa: F401 — ensure the 5 named sets are registered


def _instantiate_estimator(factory, seed):
    if isinstance(factory, type):
        params = inspect.signature(factory).parameters
        return factory(random_state=seed) if "random_state" in params else factory()
    estimator = copy.deepcopy(factory)
    setter = getattr(estimator, "set_seed", None)
    if callable(setter):
        setter(seed)
    elif hasattr(estimator, "random_state"):
        estimator.random_state = seed
    return estimator


class BenchmarkRunner:
    def __init__(self, seeds=(0, 1, 2, 3, 4)):
        self.seeds = list(seeds)

    def run(self, task, table, meta, protocols, estimators,
            featuresets=None, on_model_fitted=None, on_predictions=None) -> pd.DataFrame:
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
            est_name = getattr(est_factory, "name", est_factory.__class__.__name__)
            est_feature_mode = getattr(est_factory, "feature_mode", "unified")
            est_backend = getattr(est_factory, "backend", "sklearn")
            is_byo = est_feature_mode == "byo"

            if is_byo:
                # BYO estimators: single pseudo-featureset, no projection
                fs_iterations = [("(byo)", table, "byo")]
            else:
                # Unified estimators: project table to each requested featureset
                fs_iterations = [
                    (fs_name, project(table, fs_name), "unified")
                    for fs_name in featuresets
                ]

            for fs_name, data_table, track in fs_iterations:
                for pname, proto in protocols.items():
                    cross = getattr(proto, "cross_domain", False)
                    for fi, fold in enumerate(proto.generate(data_table, meta)):
                        train_control = test_control = sample_frame = None
                        try:
                            views = make_fold_views(data_table, fold, task.target)
                            if on_model_fitted is not None:
                                train_control = data_table.iloc[fold.train_idx].copy()
                                test_control = data_table.iloc[fold.test_idx].copy()
                            if on_predictions is not None:
                                sample_frame = table.iloc[fold.test_idx].copy()
                            view_error = None
                        except Exception as error:
                            views = None
                            view_error = error
                        for seed in self.seeds:
                            np.random.seed(seed)
                            rec_base = dict(
                                estimator=est_name, protocol=pname,
                                held_out_key=fold.held_out_key, fold=fi,
                                seed=seed, ci_source="conformal",
                                featureset=fs_name, track=track,
                                feature_mode=est_feature_mode,
                                backend=est_backend,
                            )
                            try:
                                if view_error is not None:
                                    raise view_error
                                est = _instantiate_estimator(est_factory, seed)
                                native_ci = hasattr(est, "predict_interval")
                                if not native_ci:
                                    alpha = 1 - task.acceptance["calibration"]["nominal"]
                                    if cross:
                                        est = WeightedConformalWrapper(est, alpha=alpha,
                                                                        random_state=seed)
                                        rec_base["ci_source"] = "weighted_conformal"
                                    else:
                                        est = ConformalWrapper(est, alpha=alpha)
                                        rec_base["ci_source"] = "conformal"
                                else:
                                    rec_base["ci_source"] = "native"
                                est.fit(views.train_x, views.train_y, meta)
                                if on_model_fitted is not None:
                                    on_model_fitted(
                                        estimator=est,
                                        record=dict(rec_base),
                                        train_table=train_control.copy(),
                                        test_table=test_control.copy(),
                                        meta=meta,
                                    )
                                yp = est.predict(views.test_x, meta)
                                row = self._point_metrics(task, views.test_y, yp)
                                lo, hi = est.predict_interval(views.test_x, meta)
                                row["picp"] = M.picp(views.test_y, lo, hi)
                                row.update(self._trajectory_metrics(task, views.test_x, yp))
                                if on_predictions is not None:
                                    on_predictions(
                                        record=dict(rec_base),
                                        sample_frame=sample_frame.copy(),
                                        y_pred=yp,
                                        interval_low=lo,
                                        interval_high=hi,
                                    )
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
