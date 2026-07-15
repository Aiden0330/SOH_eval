from __future__ import annotations

import copy
import inspect

import numpy as np
import pandas as pd

from . import metrics as M
from .conformal import ConformalWrapper, WeightedConformalWrapper
from .fabric import FeatureState, IdentityFeaturePipeline, clone_for_fold, scoped_meta_for_frame
from .fabric.views import make_fold_views
from .featuresets import project
import sohbench.featuresets  # noqa: F401


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
    if int(pd.__version__.split(".", 1)[0]) >= 3 or pd.get_option("mode.copy_on_write") is True:
        return frame.copy(deep=False)
    return frame.copy(deep=True)


def _transform_fold(pipeline, frame, state, meta, target):
    transformed = pipeline.transform(_copy_fold_frame(frame), state, scoped_meta_for_frame(meta, frame, target))
    if not isinstance(transformed, pd.DataFrame):
        raise ValueError("feature pipeline did not return a DataFrame")
    if len(transformed) != len(frame) or not transformed.index.equals(frame.index):
        raise ValueError("feature pipeline changed rows or index")
    for column in ("dataset", "cell_id", "cycle_index"):
        if column in frame.columns and (column not in transformed.columns or not transformed[column].equals(frame[column])):
            raise ValueError(f"feature pipeline changed identity column {column!r}")
    return transformed.drop(columns=[target], errors="ignore").copy(deep=True)


class BenchmarkRunner:
    def __init__(self, seeds=(0, 1, 2, 3, 4)):
        self.seeds = list(seeds)

    def run(self, task, table, meta, protocols, estimators, featuresets=None, feature_pipelines=None, on_model_fitted=None, on_predictions=None) -> pd.DataFrame:
        featuresets = ["full"] if featuresets is None else list(featuresets)
        feature_pipelines = feature_pipelines or {}
        records = []
        folds = [
            (pname, getattr(proto, "cross_domain", False), fi, fold)
            for pname, proto in protocols.items()
            for fi, fold in enumerate(proto.generate(table, meta))
        ]
        views_cache, raw_prepared_cache, prepared_cache = {}, {}, {}
        model_events, prediction_events = [], []

        def views_for(pname, fi, fold):
            key = (pname, fi)
            if key not in views_cache:
                try:
                    views_cache[key] = (make_fold_views(table, fold, task.target), None)
                except Exception as exc:
                    views_cache[key] = (None, exc)
            return views_cache[key]

        def raw_prepared_for(pname, fi, fold, pipeline_name):
            key = (pname, fi, pipeline_name)
            if key in raw_prepared_cache:
                return raw_prepared_cache[key]
            views, view_error = views_for(pname, fi, fold)
            state = FeatureState.create({"kind": "unavailable"})
            try:
                if view_error is not None:
                    raise view_error
                template = feature_pipelines.get(pipeline_name) if pipeline_name else IdentityFeaturePipeline()
                fitting_pipeline = clone_for_fold(template)
                train_meta = scoped_meta_for_frame(meta, views.train_x, task.target)
                state = fitting_pipeline.fit(_copy_fold_frame(views.train_x), views.train_y.copy(), train_meta)
                transformed_train = _transform_fold(clone_for_fold(template), views.train_x, state, meta, task.target)
                transformed_test = _transform_fold(clone_for_fold(template), views.test_x, state, meta, task.target)
                value = (transformed_train, transformed_test, state, None)
            except Exception as exc:
                value = (None, None, state, exc)
            raw_prepared_cache[key] = value
            return value

        def prepared_for(pname, fi, fold, fs_name, track, pipeline_name):
            key = (pname, fi, fs_name, track, pipeline_name)
            if key in prepared_cache:
                return prepared_cache[key]
            transformed_train, transformed_test, state, raw_error = raw_prepared_for(pname, fi, fold, pipeline_name)
            try:
                if raw_error is not None:
                    raise raw_error
                train_x, test_x = (
                    (transformed_train, transformed_test)
                    if track == "byo"
                    else (project(transformed_train, fs_name), project(transformed_test, fs_name))
                )
                value = (train_x, test_x, state, None)
            except Exception as exc:
                value = (None, None, state, exc)
            prepared_cache[key] = value
            return value

        for est_factory in estimators:
            est_name = getattr(est_factory, "name", est_factory.__class__.__name__)
            feature_mode = getattr(est_factory, "feature_mode", "unified")
            backend = getattr(est_factory, "backend", "sklearn")
            is_byo = feature_mode == "byo"
            byo_name = next((name for name in featuresets if name in feature_pipelines), None)
            fs_iterations = [("(byo)", "byo", byo_name)] if is_byo else [
                (name, "unified", name if name in feature_pipelines else None) for name in featuresets
            ]
            for pname, cross, fi, fold in folds:
                views, view_error = views_for(pname, fi, fold)
                for fs_name, track, pipeline_name in fs_iterations:
                    fit_train_x, fit_test_x, feature_state, feature_error = prepared_for(pname, fi, fold, fs_name, track, pipeline_name)
                    for seed in self.seeds:
                        np.random.seed(seed)
                        rec_base = {
                            "estimator": est_name, "protocol": pname, "held_out_key": fold.held_out_key,
                            "fold": fi, "seed": seed, "ci_source": "conformal", "featureset": fs_name,
                            "track": track, "feature_mode": feature_mode, "backend": backend,
                            "feature_state_digest": feature_state.digest,
                        }
                        try:
                            if feature_error is not None:
                                raise feature_error
                            train_x, test_x = _copy_fold_frame(fit_train_x), _copy_fold_frame(fit_test_x)
                            train_y, test_y = views.train_y.copy(), views.test_y.copy()
                            train_meta = scoped_meta_for_frame(meta, train_x, task.target)
                            test_meta = scoped_meta_for_frame(meta, test_x, task.target)
                            est = _instantiate_estimator(est_factory, seed)
                            if not hasattr(est, "predict_interval"):
                                alpha = 1 - task.acceptance["calibration"]["nominal"]
                                est = WeightedConformalWrapper(est, alpha=alpha, random_state=seed) if cross else ConformalWrapper(est, alpha=alpha)
                                rec_base["ci_source"] = "weighted_conformal" if cross else "conformal"
                            else:
                                rec_base["ci_source"] = "native"
                            est.fit(train_x, train_y, train_meta)
                            snapshot, snapshot_error = None, None
                            if on_model_fitted is not None:
                                try:
                                    snapshot = copy.deepcopy(est)
                                except Exception as exc:
                                    snapshot_error = str(exc)
                                    rec_base["model_snapshot_error"] = snapshot_error
                            yp = est.predict(test_x, test_meta)
                            row = self._point_metrics(task, test_y, yp)
                            lo, hi = est.predict_interval(test_x, test_meta)
                            row["picp"] = M.picp(test_y, lo, hi)
                            row.update(self._trajectory_metrics(task, test_x, yp))
                        except Exception as exc:
                            row = {metric: float("nan") for metric in (*task.point_metrics, *task.interval_metrics, *task.trajectory_metrics)}
                            rec_base["error"] = str(exc)
                            snapshot, snapshot_error = None, None
                        event_records = []
                        for metric, value in row.items():
                            metric_record = {**rec_base, "metric": metric, "value": value}
                            records.append(metric_record)
                            event_records.append(metric_record)
                        if "error" not in rec_base and on_model_fitted is not None:
                            model_events.append(
                                {
                                    "estimator": snapshot,
                                    "record": dict(rec_base),
                                    "train_table": train_x.copy(deep=True),
                                    "test_table": test_x.copy(deep=True),
                                    "meta": scoped_meta_for_frame(meta, train_x, task.target),
                                    "feature_state": feature_state,
                                    "pipeline_name": pipeline_name,
                                    "snapshot_error": snapshot_error,
                                    "records": event_records,
                                }
                            )
                        if "error" not in rec_base and on_predictions is not None:
                            prediction_events.append(
                                {
                                    "record": dict(rec_base),
                                    "sample_frame": table.iloc[fold.test_idx].copy(deep=True),
                                    "y_pred": np.array(yp, copy=True),
                                    "interval_low": np.array(lo, copy=True),
                                    "interval_high": np.array(hi, copy=True),
                                    "records": event_records,
                                }
                            )

        # Evidence callbacks run only after no remaining fold can be influenced.
        for event in model_events:
            if event["snapshot_error"] is not None:
                message = event["snapshot_error"]
                for metric_record in event["records"]:
                    metric_record["model_snapshot_error"] = message
                    metric_record["model_callback_error"] = "callback skipped because model snapshot failed"
                continue
            try:
                on_model_fitted(
                    estimator=event["estimator"], record=event["record"],
                    train_table=event["train_table"], test_table=event["test_table"], meta=event["meta"],
                    feature_state=event["feature_state"], pipeline_name=event["pipeline_name"],
                )
            except Exception as exc:
                for metric_record in event["records"]:
                    metric_record["model_callback_error"] = str(exc)
        for event in prediction_events:
            try:
                on_predictions(
                    record=event["record"], sample_frame=event["sample_frame"], y_pred=event["y_pred"],
                    interval_low=event["interval_low"], interval_high=event["interval_high"],
                )
            except Exception as exc:
                for metric_record in event["records"]:
                    metric_record["callback_error"] = str(exc)
        return pd.DataFrame.from_records(records)

    def _point_metrics(self, task, yt, yp):
        return {metric: getattr(M, metric)(yt, yp) for metric in task.point_metrics}

    def _trajectory_metrics(self, task, test_x, yp):
        order = np.argsort(test_x["cycle_index"].to_numpy())
        return {"monotonicity_violation_rate": M.monotonicity_violation_rate(yp[order]), "value_range_violation": M.value_range_violation(yp)}
