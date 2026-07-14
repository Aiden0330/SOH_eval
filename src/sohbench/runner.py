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
from .fabric import FeatureState, IdentityFeaturePipeline
import sohbench.featuresets  # noqa: F401 — ensure the 5 named sets are registered


def _instantiate_estimator(factory, seed):
    if isinstance(factory, type):
        params = inspect.signature(factory).parameters
        estimator = factory(random_state=seed) if "random_state" in params else factory()
    else:
        estimator = copy.deepcopy(factory)
    setter = getattr(estimator, "set_seed", None)
    if callable(setter):
        setter(seed)
    elif hasattr(estimator, "random_state"):
        estimator.random_state = seed
    return estimator


def _copy_fold_frame(frame):
    """Copy a fold frame safely without redoing fold construction."""
    if int(pd.__version__.split(".", 1)[0]) >= 3:
        return frame.copy(deep=False)
    if pd.get_option("mode.copy_on_write") is True:
        return frame.copy(deep=False)
    return frame.copy(deep=True)


def _copy_algorithm_meta(meta, target):
    if isinstance(meta, pd.DataFrame):
        return meta.drop(columns=[target], errors="ignore").copy(deep=True)
    return copy.deepcopy(meta)


def _copy_callback_meta(meta):
    if isinstance(meta, pd.DataFrame):
        return meta.copy(deep=True)
    return copy.deepcopy(meta)


def _transform_fold(pipeline, frame, state, meta, target):
    transformed = pipeline.transform(_copy_fold_frame(frame), state, _copy_callback_meta(meta))
    if not isinstance(transformed, pd.DataFrame):
        raise ValueError("feature pipeline did not return a DataFrame")
    if len(transformed) != len(frame):
        raise ValueError("feature pipeline changed row count")
    if not transformed.index.equals(frame.index):
        raise ValueError("feature pipeline changed row index")
    for column in ("dataset", "cell_id", "cycle_index"):
        if column in frame.columns and (column not in transformed.columns or not transformed[column].equals(frame[column])):
            raise ValueError(f"feature pipeline changed identity column {column!r}")
    return transformed.drop(columns=[target], errors="ignore").copy(deep=True)


def _control_table(features, target, values):
    control = features.copy(deep=True)
    control[target] = pd.Series(values, index=control.index, copy=True)
    return control


class BenchmarkRunner:
    def __init__(self, seeds=(0, 1, 2, 3, 4)):
        self.seeds = list(seeds)

    def run(self, task, table, meta, protocols, estimators,
            featuresets=None, feature_pipelines=None, on_model_fitted=None,
            on_predictions=None) -> pd.DataFrame:
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
        feature_pipelines = feature_pipelines or {}

        records = []

        for est_factory in estimators:
            est_name = getattr(est_factory, "name", est_factory.__class__.__name__)
            est_feature_mode = getattr(est_factory, "feature_mode", "unified")
            est_backend = getattr(est_factory, "backend", "sklearn")
            is_byo = est_feature_mode == "byo"

            for pname, proto in protocols.items():
                cross = getattr(proto, "cross_domain", False)
                for fi, fold in enumerate(proto.generate(table, meta)):
                    try:
                        views = make_fold_views(table, fold, task.target)
                        sample_frame = table.iloc[fold.test_idx].copy() if on_predictions is not None else None
                        view_error = None
                    except Exception as error:
                        views = None
                        sample_frame = None
                        view_error = error

                    byo_pipeline_name = next((name for name in featuresets if name in feature_pipelines), None)
                    fs_iterations = [("(byo)", "byo", feature_pipelines.get(byo_pipeline_name) or IdentityFeaturePipeline(), byo_pipeline_name)] if is_byo else [
                        (fs_name, "unified", feature_pipelines.get(fs_name) or IdentityFeaturePipeline(), fs_name if fs_name in feature_pipelines else None)
                        for fs_name in featuresets
                    ]
                    for fs_name, track, pipeline, pipeline_name in fs_iterations:
                        feature_state = FeatureState.create({"kind": "unavailable"})
                        try:
                            if view_error is not None:
                                raise view_error
                            feature_state = pipeline.fit(
                                _copy_fold_frame(views.train_x),
                                views.train_y.copy(),
                                _copy_callback_meta(meta),
                            )
                            transformed_train = _transform_fold(pipeline, views.train_x, feature_state, meta, task.target)
                            transformed_test = _transform_fold(pipeline, views.test_x, feature_state, meta, task.target)
                            if is_byo:
                                fit_train_x, fit_test_x = transformed_train, transformed_test
                            else:
                                fit_train_x = project(transformed_train, fs_name)
                                fit_test_x = project(transformed_test, fs_name)
                            train_control = _control_table(fit_train_x, task.target, views.train_y)
                            test_control = _control_table(fit_test_x, task.target, views.test_y)
                            feature_error = None
                        except Exception as error:
                            fit_train_x = fit_test_x = train_control = test_control = None
                            feature_error = error

                        for seed in self.seeds:
                            np.random.seed(seed)
                            rec_base = dict(
                                estimator=est_name, protocol=pname,
                                held_out_key=fold.held_out_key, fold=fi,
                                seed=seed, ci_source="conformal",
                                featureset=fs_name, track=track,
                                feature_mode=est_feature_mode,
                                backend=est_backend,
                                feature_state_digest=feature_state.digest,
                            )
                            try:
                                if feature_error is not None:
                                    raise feature_error
                                train_x = _copy_fold_frame(fit_train_x)
                                test_x = _copy_fold_frame(fit_test_x)
                                train_y = views.train_y.copy()
                                test_y = views.test_y.copy()
                                algorithm_meta = _copy_algorithm_meta(meta, task.target)
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
                                est.fit(train_x, train_y, algorithm_meta)
                                if on_model_fitted is not None:
                                    on_model_fitted(
                                        estimator=est,
                                        record=dict(rec_base),
                                        train_table=train_control.copy(deep=True),
                                        test_table=test_control.copy(deep=True),
                                        meta=_copy_callback_meta(meta),
                                        feature_state=feature_state,
                                        feature_pipeline=pipeline,
                                        feature_pipeline_name=pipeline_name,
                                    )
                                yp = est.predict(test_x, algorithm_meta)
                                row = self._point_metrics(task, test_y, yp)
                                lo, hi = est.predict_interval(test_x, algorithm_meta)
                                row["picp"] = M.picp(test_y, lo, hi)
                                row.update(self._trajectory_metrics(task, test_x, yp))
                                if on_predictions is not None:
                                    on_predictions(
                                        record=dict(rec_base),
                                        sample_frame=sample_frame.copy(deep=True),
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
