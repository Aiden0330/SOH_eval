from __future__ import annotations

import copy
import inspect
import random
import threading
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from joblib.externals import cloudpickle

from . import metrics as M
from .conformal import ConformalWrapper, WeightedConformalWrapper
from .contracts import (
    validate_finite_metrics,
    validate_finite_vector,
    validate_interval_bounds,
)
from .fabric import FeatureState, IdentityFeaturePipeline, clone_for_fold, scoped_meta_for_frame
from .fabric.failures import InputContractError, MetricEvaluationError, classify_exception
from .fabric.views import make_fold_views
from .featuresets import project
import sohbench.featuresets  # noqa: F401


_LEGACY_RANDOM_LOCK = threading.RLock()


class _DeferredCallbackSpool:
    """Bound callback payload memory by staging snapshots on local disk."""

    def __init__(self):
        self._directory = TemporaryDirectory(prefix="sohbench-callback-")
        self._counter = 0

    def dump(self, kind, value) -> Path:
        self._counter += 1
        path = Path(self._directory.name) / f"{kind}-{self._counter:08d}.pkl"
        try:
            with path.open("wb") as handle:
                cloudpickle.dump(value, handle, protocol=cloudpickle.DEFAULT_PROTOCOL)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def load(self, path: Path):
        with path.open("rb") as handle:
            return cloudpickle.load(handle)

    @staticmethod
    def discard(path: Path | None) -> None:
        if path is not None:
            path.unlink(missing_ok=True)

    def close(self) -> None:
        self._directory.cleanup()


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


@contextmanager
def _legacy_random_scope(seed: int):
    with _LEGACY_RANDOM_LOCK:
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


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


def _contextual_error(error_type, exc):
    if isinstance(exc, error_type):
        return exc
    contextual = error_type(str(exc))
    contextual.__cause__ = exc
    return contextual


def _root_error(exc):
    seen = set()
    current = exc
    while current.__cause__ is not None and id(current) not in seen:
        seen.add(id(current))
        current = current.__cause__
    return current


class BenchmarkRunner:
    def __init__(self, seeds=(0, 1, 2, 3, 4)):
        self.seeds = list(seeds)

    def run(self, task, table, meta, protocols, estimators, featuresets=None, feature_pipelines=None, on_model_fitted=None, on_predictions=None) -> pd.DataFrame:
        callback_spool = _DeferredCallbackSpool() if on_model_fitted is not None or on_predictions is not None else None
        try:
            return self._run(
                task, table, meta, protocols, estimators,
                featuresets=featuresets,
                feature_pipelines=feature_pipelines,
                on_model_fitted=on_model_fitted,
                on_predictions=on_predictions,
                callback_spool=callback_spool,
            )
        finally:
            if callback_spool is not None:
                callback_spool.close()

    def _run(self, task, table, meta, protocols, estimators, featuresets=None, feature_pipelines=None, on_model_fitted=None, on_predictions=None, callback_spool=None) -> pd.DataFrame:
        featuresets = ["full"] if featuresets is None else list(featuresets)
        feature_pipelines = feature_pipelines or {}
        records = []
        folds = [
            (pname, getattr(proto, "cross_domain", False), fi, fold)
            for pname, proto in protocols.items()
            for fi, fold in enumerate(proto.generate(table, meta))
        ]
        views_cache = {}
        model_events, prediction_events = [], []

        def views_for(pname, fi, fold):
            key = (pname, fi)
            if key not in views_cache:
                try:
                    views_cache[key] = (make_fold_views(table, fold, task.target), None)
                except Exception as exc:
                    views_cache[key] = (None, _contextual_error(InputContractError, exc))
            return views_cache[key]

        def prepared_candidate_for(views, view_error, fs_name, track, pipeline_name):
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
                train_x, test_x = (
                    (transformed_train, transformed_test)
                    if track == "byo"
                    else (project(transformed_train, fs_name), project(transformed_test, fs_name))
                )
                return train_x, test_x, state, None
            except Exception as exc:
                return None, None, state, _contextual_error(InputContractError, exc)

        for pname, cross, fi, fold in folds:
            views, view_error = views_for(pname, fi, fold)
            for est_factory in estimators:
                est_name = getattr(est_factory, "name", est_factory.__class__.__name__)
                feature_mode = getattr(est_factory, "feature_mode", "unified")
                backend = getattr(est_factory, "backend", "sklearn")
                is_byo = feature_mode == "byo"
                byo_name = next((name for name in featuresets if name in feature_pipelines), None)
                fs_iterations = [("(byo)", "byo", byo_name)] if is_byo else [
                    (name, "unified", name if name in feature_pipelines else None) for name in featuresets
                ]
                for fs_name, track, pipeline_name in fs_iterations:
                    for seed in self.seeds:
                        feature_state = FeatureState.create({"kind": "unavailable"})
                        rec_base = {
                            "estimator": est_name, "protocol": pname, "held_out_key": fold.held_out_key,
                            "fold": fi, "seed": seed, "ci_source": "conformal", "featureset": fs_name,
                            "track": track, "feature_mode": feature_mode, "backend": backend,
                            "feature_state_digest": feature_state.digest,
                            "unit_status": "succeeded", "error_category": "",
                            "error_type": "", "error_message": "",
                        }
                        snapshot_path, snapshot_error = None, None
                        fit_train_x = fit_test_x = train_x = test_x = None
                        try:
                            with _legacy_random_scope(seed):
                                fit_train_x, fit_test_x, feature_state, feature_error = prepared_candidate_for(
                                    views, view_error, fs_name, track, pipeline_name
                                )
                                rec_base["feature_state_digest"] = feature_state.digest
                                if feature_error is not None:
                                    raise feature_error
                                try:
                                    train_x, test_x = _copy_fold_frame(fit_train_x), _copy_fold_frame(fit_test_x)
                                    train_y, test_y = views.train_y.copy(), views.test_y.copy()
                                    train_meta = scoped_meta_for_frame(meta, train_x, task.target)
                                    test_meta = scoped_meta_for_frame(meta, test_x, task.target)
                                except Exception as exc:
                                    raise InputContractError(str(exc)) from exc
                                est = _instantiate_estimator(est_factory, seed)
                                if not hasattr(est, "predict_interval"):
                                    alpha = 1 - task.acceptance["calibration"]["nominal"]
                                    est = WeightedConformalWrapper(est, alpha=alpha, random_state=seed) if cross else ConformalWrapper(est, alpha=alpha)
                                    rec_base["ci_source"] = "weighted_conformal" if cross else "conformal"
                                else:
                                    rec_base["ci_source"] = "native"
                                est.fit(train_x, train_y, train_meta)
                                if on_model_fitted is not None:
                                    try:
                                        snapshot_path = callback_spool.dump("model", est)
                                    except Exception as exc:
                                        snapshot_error = str(exc)
                                        rec_base["model_snapshot_error"] = snapshot_error
                                yp = validate_finite_vector(
                                    est.predict(test_x, test_meta),
                                    name="predictions",
                                    n_samples=len(test_x),
                                )
                                lo, hi = est.predict_interval(test_x, test_meta)
                                lo = validate_finite_vector(
                                    lo,
                                    name="interval_low",
                                    n_samples=len(test_x),
                                )
                                hi = validate_finite_vector(
                                    hi,
                                    name="interval_high",
                                    n_samples=len(test_x),
                                )
                                validate_interval_bounds(lo, hi)
                                row = self._point_metrics(task, test_y, yp)
                                row.update(self._interval_metrics(test_y, lo, hi))
                                row.update(self._trajectory_metrics(task, test_x, yp))
                        except Exception as exc:
                            if callback_spool is not None:
                                callback_spool.discard(snapshot_path)
                            snapshot_path = None
                            row = {metric: float("nan") for metric in (*task.point_metrics, *task.interval_metrics, *task.trajectory_metrics)}
                            reported_error = _root_error(exc)
                            rec_base["error"] = str(reported_error)
                            rec_base.update(
                                unit_status="failed",
                                error_category=classify_exception(exc).value,
                                error_type=type(reported_error).__name__,
                                error_message=str(reported_error),
                            )
                        event_records = []
                        for metric, value in row.items():
                            metric_record = {**rec_base, "metric": metric, "value": value}
                            records.append(metric_record)
                            event_records.append(metric_record)
                        if "error" not in rec_base and on_model_fitted is not None:
                            prepared_snapshot_path, prepared_snapshot_error = None, None
                            if snapshot_error is None:
                                try:
                                    prepared_snapshot_path = callback_spool.dump(
                                        "prepared",
                                        {
                                            "train_table": fit_train_x,
                                            "test_table": fit_test_x,
                                            "feature_state": feature_state,
                                        },
                                    )
                                except Exception as exc:
                                    prepared_snapshot_error = str(exc)
                            model_events.append(
                                {
                                    "snapshot_path": snapshot_path,
                                    "prepared_snapshot_path": prepared_snapshot_path,
                                    "prepared_snapshot_error": prepared_snapshot_error,
                                    "record": dict(rec_base),
                                    "pipeline_name": pipeline_name,
                                    "snapshot_error": snapshot_error,
                                    "records": event_records,
                                }
                            )
                        if "error" not in rec_base and on_predictions is not None:
                            prediction_path, prediction_snapshot_error = None, None
                            try:
                                prediction_path = callback_spool.dump(
                                    "prediction",
                                    {
                                        "y_pred": np.array(yp, copy=True),
                                        "interval_low": np.array(lo, copy=True),
                                        "interval_high": np.array(hi, copy=True),
                                    },
                                )
                            except Exception as exc:
                                prediction_snapshot_error = str(exc)
                            prediction_events.append(
                                {
                                    "record": dict(rec_base),
                                    "snapshot_path": prediction_path,
                                    "snapshot_error": prediction_snapshot_error,
                                    "fold_key": (pname, fi),
                                    "records": event_records,
                                }
                            )

            # The complete fold has run across every estimator. Evidence has
            # been staged, so large transformed views can now be released.
            views_cache.pop((pname, fi), None)
            views = fit_train_x = fit_test_x = train_x = test_x = None
            train_y = test_y = train_meta = test_meta = est = yp = lo = hi = None

        # Evidence callbacks run only after no remaining fold can be influenced.
        fold_lookup = {(pname, fi): fold for pname, _, fi, fold in folds}
        prepared_event_groups = {}
        for event in model_events:
            if event["snapshot_error"] is not None:
                message = event["snapshot_error"]
                for metric_record in event["records"]:
                    metric_record["model_snapshot_error"] = message
                    metric_record["model_callback_error"] = "callback skipped because model snapshot failed"
                continue
            if event["prepared_snapshot_error"] is not None:
                callback_spool.discard(event["snapshot_path"])
                for metric_record in event["records"]:
                    metric_record["prepared_snapshot_error"] = event["prepared_snapshot_error"]
                    metric_record["model_callback_error"] = "callback skipped because prepared evidence snapshot failed"
                continue
            prepared_event_groups.setdefault(event["prepared_snapshot_path"], []).append(event)

        for prepared_snapshot_path, events in prepared_event_groups.items():
            try:
                prepared = callback_spool.load(prepared_snapshot_path)
            except Exception as exc:
                for event in events:
                    callback_spool.discard(event["snapshot_path"])
                    for metric_record in event["records"]:
                        metric_record["model_callback_error"] = str(exc)
                continue
            try:
                for event in events:
                    try:
                        estimator = callback_spool.load(event["snapshot_path"])
                        on_model_fitted(
                            estimator=estimator, record=event["record"],
                            train_table=prepared["train_table"].copy(deep=True),
                            test_table=prepared["test_table"].copy(deep=True),
                            meta=scoped_meta_for_frame(meta, prepared["train_table"], task.target),
                            feature_state=prepared["feature_state"], pipeline_name=event["pipeline_name"],
                        )
                    except Exception as exc:
                        for metric_record in event["records"]:
                            metric_record["model_callback_error"] = str(exc)
                    finally:
                        callback_spool.discard(event["snapshot_path"])
                        estimator = None
            finally:
                callback_spool.discard(prepared_snapshot_path)
                prepared = None
        for event in prediction_events:
            if event["snapshot_error"] is not None:
                for metric_record in event["records"]:
                    metric_record["prediction_snapshot_error"] = event["snapshot_error"]
                    metric_record["callback_error"] = "callback skipped because prediction snapshot failed"
                continue
            try:
                prediction = callback_spool.load(event["snapshot_path"])
                fold = fold_lookup[event["fold_key"]]
                on_predictions(
                    record=event["record"], sample_frame=table.iloc[fold.test_idx].copy(deep=True),
                    y_pred=prediction["y_pred"], interval_low=prediction["interval_low"],
                    interval_high=prediction["interval_high"],
                )
            except Exception as exc:
                for metric_record in event["records"]:
                    metric_record["callback_error"] = str(exc)
            finally:
                callback_spool.discard(event["snapshot_path"])
        return pd.DataFrame.from_records(records)

    def _point_metrics(self, task, yt, yp):
        try:
            metrics = {}
            for metric in task.point_metrics:
                if (
                    metric == "eol_region_error"
                    and not np.any(np.asarray(yt, dtype=float) <= M.EOL_THRESHOLD)
                ):
                    continue
                metrics[metric] = getattr(M, metric)(yt, yp)
            return validate_finite_metrics(metrics)
        except Exception as exc:
            raise MetricEvaluationError(str(exc)) from exc

    def _interval_metrics(self, yt, lo, hi):
        try:
            return validate_finite_metrics({"picp": M.picp(yt, lo, hi)})
        except Exception as exc:
            raise MetricEvaluationError(str(exc)) from exc

    def _trajectory_metrics(self, task, test_x, yp):
        try:
            frame = test_x[["dataset", "cell_id", "cycle_index"]].copy()
            frame["prediction"] = np.asarray(yp, dtype=float)
            if frame.duplicated(["dataset", "cell_id", "cycle_index"]).any():
                raise ValueError("duplicate cycle identity")
            per_asset = []
            for _, asset in frame.groupby(["dataset", "cell_id"], sort=True, dropna=False):
                asset = asset.sort_values("cycle_index", kind="stable")
                if len(asset) < 2:
                    continue
                prediction = asset["prediction"].to_numpy()
                per_asset.append(
                    {
                        "monotonicity_violation_rate": M.monotonicity_violation_rate(prediction),
                        "value_range_violation": M.value_range_violation(prediction),
                    }
                )
            if not per_asset:
                raise ValueError("trajectory metrics require an asset with at least two cycles")
            return validate_finite_metrics(
                {
                    metric: float(
                        np.mean([values[metric] for values in per_asset])
                    )
                    for metric in task.trajectory_metrics
                }
            )
        except Exception as exc:
            raise MetricEvaluationError(str(exc)) from exc
