"""Fold-local feature pipelines (Task 7d, port of Codex 1fdf6ed/320b250).

A pipeline is fitted on the training rows of a fold only.  A transform that peeks at test
rows changes the score; the fold-isolated version cannot, and its output is audited before
any estimator sees it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sohkernel import registry as R
from sohkernel.estimators.baselines import RidgeEstimator
from sohkernel.featuresets import FeatureSet
from sohkernel.metadata import build_meta
from sohkernel.pipelines import StatelessPipeline, prepare_fold
from sohkernel.protocols import LeaveOneCellOut
from sohkernel.runner import BenchmarkRunner
from sohkernel.tasks import SOHEstimationTask


def _table(n_cells=4, cycles=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cell in range(n_cells):
        for k in range(cycles):
            soh = 1.0 - 0.25 * k / (cycles - 1) + rng.normal(0, 0.004)
            rows.append({"dataset": "d1", "cell_id": f"c{cell}", "cycle_index": k, "soh": soh,
                         "f1": soh + rng.normal(0, 0.08), "f2": rng.normal(0, 1.0),
                         "charge_capacity_max": 1.1 * soh})
    return pd.DataFrame(rows)


def _key(frame):
    return set(zip(frame["dataset"], frame["cell_id"], frame["cycle_index"]))


class NoisyLabelMemory:
    """Remembers a noisy copy of the label for every row it was *fitted* on and emits it as a
    feature — informative, passes the affine audit on training rows (R² < 0.995), but a leak
    for any test row it was allowed to see."""

    def __init__(self, noise=0.02, seed=1):
        self.noise, self.seed = noise, seed
        self.fit_rows: list[set] = []

    def fit(self, train_x, train_y, meta):
        rng = np.random.default_rng(self.seed)
        keys = list(zip(train_x["dataset"], train_x["cell_id"], train_x["cycle_index"]))
        self.fit_rows.append(set(keys))
        return {key: float(y + rng.normal(0, self.noise)) for key, y in zip(keys, train_y)}

    def transform(self, frame, state, meta):
        keys = zip(frame["dataset"], frame["cell_id"], frame["cycle_index"])
        return frame.assign(leak=[state.get(key, np.nan) for key in keys])


def _mae(results, estimator="ridge"):
    rows = results[(results["estimator"] == estimator) & (results["metric"] == "mae")]
    assert rows["value"].notna().all(), results.get("error")
    return float(rows["value"].mean())


def test_pipeline_is_fitted_on_training_rows_only_and_once_per_fold():
    table = _table()
    meta = build_meta(table)
    R.register_featureset(FeatureSet(name="_t_leak_fs", candidates=["leak", "f1"]))
    pipeline = NoisyLabelMemory()
    seen_test: list[pd.DataFrame] = []
    folds = list(LeaveOneCellOut().generate(table, meta))
    results = BenchmarkRunner(seeds=[0, 1]).run(
        SOHEstimationTask(), table=table, meta=meta, protocols={"loco": LeaveOneCellOut()},
        estimators=[RidgeEstimator, RidgeEstimator], featuresets=["_t_leak_fs"],
        feature_pipelines={"memory": pipeline},
        on_model_fitted=lambda **kw: seen_test.append(kw["test_table"]),
    )
    assert "error" not in results.columns or results["error"].isna().all()
    # one fit per fold, shared by 2 estimators x 2 seeds
    assert len(pipeline.fit_rows) == len(folds) == 4
    for fold, rows in zip(folds, pipeline.fit_rows):
        assert rows == _key(table.iloc[fold.train_idx])
        assert not rows & _key(table.iloc[fold.test_idx])
    # test rows never received the remembered label
    assert all(frame["leak"].isna().all() for frame in seen_test)


def test_peeking_transform_changes_the_score_but_the_isolated_pipeline_cannot():
    table = _table()
    meta = build_meta(table)
    R.register_featureset(FeatureSet(name="_t_leak_fs2", candidates=["leak", "f1"]))
    task = SOHEstimationTask()
    # (a) the same transform applied globally before scoring: it saw every row, test rows included
    peeking = NoisyLabelMemory()
    state = peeking.fit(table.drop(columns=["soh"]), table["soh"].to_numpy(), meta)
    peeked = peeking.transform(table.drop(columns=["soh"]), state, meta)
    peeked.insert(3, "soh", table["soh"].to_numpy())
    leaked = BenchmarkRunner(seeds=[0]).run(task, table=peeked, meta=meta, protocols={"loco": LeaveOneCellOut()},
                                             estimators=[RidgeEstimator], featuresets=["_t_leak_fs2"])
    # (b) fold-local: fitted on training rows only
    isolated = BenchmarkRunner(seeds=[0]).run(task, table=table, meta=meta, protocols={"loco": LeaveOneCellOut()},
                                               estimators=[RidgeEstimator], featuresets=["_t_leak_fs2"],
                                               feature_pipelines={"memory": NoisyLabelMemory()})
    # (c) reference without the feature at all
    R.register_featureset(FeatureSet(name="_t_plain", candidates=["f1"]))
    plain = BenchmarkRunner(seeds=[0]).run(task, table=table, meta=meta, protocols={"loco": LeaveOneCellOut()},
                                            estimators=[RidgeEstimator], featuresets=["_t_plain"])
    mae_leaked, mae_isolated, mae_plain = _mae(leaked), _mae(isolated), _mae(plain)
    assert mae_leaked < 0.5 * mae_plain          # peeking makes the score too good to be true
    assert mae_isolated >= 0.8 * mae_plain       # isolation: no better than not having the feature


def test_pipeline_output_is_audited_before_any_estimator_runs():
    table = _table()
    meta = build_meta(table)

    class CapacityCopy:
        def fit(self, train_x, train_y, meta):
            return None

        def transform(self, frame, state, meta):
            return frame.assign(cap_copy=frame["charge_capacity_max"] * 0.5)

    R.register_featureset(FeatureSet(name="_t_cap_copy", candidates=["cap_copy", "f1"]))
    fits = []
    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), table=table, meta=meta, protocols={"loco": LeaveOneCellOut()},
        estimators=[RidgeEstimator], featuresets=["_t_cap_copy"], feature_pipelines={"copy": CapacityCopy()},
        on_model_fitted=lambda **kw: fits.append(kw),
    )
    assert fits == []
    assert results["error"].str.contains("label-safety audit").all()
    assert results["value"].isna().all()


def test_stateless_adapter_feeds_projection_per_fold():
    table = _table()
    meta = build_meta(table)
    R.register_featureset(FeatureSet(name="_t_sq", candidates=["f1_sq"]))
    seen = []
    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), table=table, meta=meta, protocols={"loco": LeaveOneCellOut()},
        estimators=[RidgeEstimator], featuresets=["_t_sq"],
        feature_pipelines={"sq": StatelessPipeline(lambda frame, m: frame.assign(f1_sq=frame["f1"] ** 2), name="sq")},
        on_model_fitted=lambda **kw: seen.append(kw["train_table"]),
    )
    assert "error" not in results.columns or results["error"].isna().all()
    assert seen and all(list(frame.columns) == ["dataset", "cell_id", "cycle_index", "soh", "f1_sq"] for frame in seen)


def test_pipeline_contract_violations_fail_the_fold_with_a_reason():
    table = _table()
    meta = build_meta(table)

    class DropsRows:
        def fit(self, train_x, train_y, meta):
            return None

        def transform(self, frame, state, meta):
            return frame.iloc[:-1]

    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(), table=table, meta=meta, protocols={"loco": LeaveOneCellOut()},
        estimators=[RidgeEstimator], featuresets=["full"], feature_pipelines={"bad": DropsRows()},
    )
    assert results["error"].str.contains("changed the rows or index").all()

    class MakesLabel:
        def fit(self, train_x, train_y, meta):
            return None

        def transform(self, frame, state, meta):
            return frame.assign(soh=1.0)

    with pytest.raises(Exception, match="protected label"):
        prepare_fold({"bad": MakesLabel()}, table.iloc[:100], table.iloc[100:], meta)
