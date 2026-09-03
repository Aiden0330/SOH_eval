from __future__ import annotations
import copy
import inspect
import numpy as np, pandas as pd
from . import metrics as M
from .conformal import ConformalWrapper, WeightedConformalWrapper
from .contracts import feature_columns
from .feature_safety import is_label_source_feature
from .features import unified_features
from .featuresets import project
from .pipelines import prepare_fold
from .method_contracts import method_contract
from .registry import get_featureset
from . import featuresets as _featuresets  # noqa: F401 — ensure the named sets are registered


def _instantiate_estimator(factory, seed: int):
    """A fresh estimator seeded with the evaluation seed (Spec 01 §6).

    Classes receive ``random_state=seed`` or ``seed=seed`` when their constructor accepts
    it; instances are deep-copied per fit (no state shared across folds or seeds) and
    seeded through ``set_seed(seed)`` or a ``random_state`` / ``seed`` attribute.  Estimators
    with no seed hook stay deterministic, which is what the leaderboard std reveals.
    """
    if isinstance(factory, type):
        params = inspect.signature(factory).parameters
        if "random_state" in params:
            return factory(random_state=seed)
        if "seed" in params:
            return factory(seed=seed)
        estimator = factory()
    else:
        try:
            estimator = copy.deepcopy(factory)
        except Exception as exc:  # pragma: no cover - exotic estimator state
            raise TypeError(
                f"estimator instance {factory!r} must support deepcopy so seeds and folds "
                "do not share state"
            ) from exc
    setter = getattr(estimator, "set_seed", None)
    if callable(setter):
        setter(seed)
    elif hasattr(estimator, "random_state"):
        estimator.random_state = seed
    elif hasattr(estimator, "seed"):
        estimator.seed = seed
    return estimator


class BenchmarkRunner:
    def __init__(self, seeds=(0, 1, 2, 3, 4)):
        self.seeds = list(seeds)

    def run(self, task, table, meta, protocols, estimators,
            featuresets=None, on_model_fitted=None, feature_pipelines=None) -> pd.DataFrame:
        """Run the benchmark matrix over estimators × protocols × featuresets × seeds.

        Parameters
        ----------
        featuresets:
            List of featureset names to iterate over for *unified* estimators,
            e.g. ``["full", "voltage"]``.  Defaults to ``["full"]`` when *None*
            (backward-compatible).  BYO estimators ignore this parameter and
            always receive the full (unprojected) table.
        feature_pipelines:
            Optional ordered mapping ``name -> pipeline`` (``sohkernel.pipelines``).
            Each pipeline is fitted on the training rows of every fold only and
            applied to both partitions before projection; its output is audited.
            The fit is shared across estimators, featuresets and seeds of a fold.
        """
        if featuresets is None:
            featuresets = ["full"]
        pipelines = dict(feature_pipelines or {})
        fold_cache: dict = {}

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
                        pipeline_error = None
                        if pipelines:
                            # Fold-local pipelines: fitted on this fold's training rows
                            # of the *unprojected* table, applied to both partitions,
                            # then projected to the featureset (Task 7d).
                            tr, te, pipeline_error = self._prepared_fold(
                                fold_cache, (pname, fi), pipelines, table, fold, meta, task.target
                            )
                            if pipeline_error is None:
                                if track == "byo":
                                    keep = [c for c in tr.columns if not is_label_source_feature(c)]
                                    tr, te = tr[keep], te[keep]
                                else:
                                    tr, te = project(tr, fs_name), project(te, fs_name)
                        else:
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
                                if pipeline_error is not None:
                                    raise pipeline_error
                                est = _instantiate_estimator(est_factory, seed)
                                native_ci = hasattr(est, "predict_interval")
                                if not native_ci:
                                    alpha = 1 - task.acceptance["calibration"]["nominal"]
                                    if cross:
                                        est = WeightedConformalWrapper(est, alpha=alpha, random_state=seed)
                                        rec_base["ci_source"] = "weighted_conformal"
                                    else:
                                        est = ConformalWrapper(est, alpha=alpha)
                                        rec_base["ci_source"] = "conformal"
                                else:
                                    rec_base["ci_source"] = "native"
                                ytr = tr[task.target].to_numpy()
                                # The label never rides inside X: built-in
                                # estimators already select feature columns,
                                # but a plugin estimator must not be able to
                                # read the test-fold target off the frame.
                                X_tr = tr.drop(columns=[task.target], errors="ignore")
                                X_te = te.drop(columns=[task.target], errors="ignore")
                                est.fit(X_tr, ytr, meta)
                                if on_model_fitted is not None:
                                    on_model_fitted(
                                        estimator=est,
                                        record=dict(rec_base),
                                        train_table=tr.copy(),
                                        test_table=te.copy(),
                                        meta=meta,
                                    )
                                yte = te[task.target].to_numpy()
                                yp = est.predict(X_te, meta)
                                row = self._point_metrics(task, yte, yp)
                                lo, hi = est.predict_interval(X_te, meta)
                                rec_base["ci_fallback"] = bool(getattr(est, "last_fallback", False))
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

    @staticmethod
    def _prepared_fold(cache, key, pipelines, table, fold, meta, target):
        """Fit-once-per-fold pipeline application; errors are cached too so every
        unit of a failed fold reports the same reason instead of re-raising."""
        if key not in cache:
            try:
                train, test, _states = prepare_fold(
                    pipelines, table.iloc[fold.train_idx], table.iloc[fold.test_idx], meta, target=target
                )
                cache[key] = (train, test, None)
            except Exception as exc:  # reported per unit through the failure path
                cache[key] = (None, None, exc)
        return cache[key]

    def _point_metrics(self, task, yt, yp):
        out = {}
        for m in task.point_metrics:
            out[m] = getattr(M, m)(yt, yp)
        return out

    def _trajectory_metrics(self, task, te, yp):
        # Multi-cell test folds (LODO / LCHO) must be scored per cell: a
        # global cycle sort interleaves cells and counts every cross-cell
        # level change as a rebound, which made the physical-consistency
        # verdict unpassable regardless of model quality.
        if "cell_id" in te.columns:
            mono = M.per_cell_monotonicity_violation_rate(
                te["cell_id"].to_numpy(), te["cycle_index"].to_numpy(), yp
            )
        else:
            order = np.argsort(te["cycle_index"].to_numpy())
            mono = M.monotonicity_violation_rate(yp[order])
        return {"monotonicity_violation_rate": mono,
                "value_range_violation": M.value_range_violation(yp)}
