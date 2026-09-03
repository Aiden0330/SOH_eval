"""Seeds must reach the estimator, not only NumPy's global state (Spec 01 §6; Task 7b).

Two seeds give different predictions for a stochastic model (mlp) and identical ones for a
deterministic model (ridge); the leaderboard std over seeds is therefore > 0 for the former
and 0 for the latter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohkernel import runner as runner_module
from sohkernel.estimators.baselines import GBDTEstimator, RidgeEstimator
from sohkernel.estimators.mechanism import ExtraTreesEstimator, StressResidualEstimator
from sohkernel.estimators.neural import MLPEstimator
from sohkernel.metadata import build_meta
from sohkernel.protocols import LeaveOneCellOut, LeaveOneDatasetOut
from sohkernel.report import write_artifacts
from sohkernel.runner import BenchmarkRunner, _instantiate_estimator
from sohkernel.tasks import SOHEstimationTask


def _table(datasets=("d1",), n_cells=4, cycles=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in datasets:
        for cell in range(n_cells):
            for k in range(cycles):
                soh = 1.0 - 0.25 * k / (cycles - 1) + rng.normal(0, 0.004)
                rows.append({"dataset": dataset, "cell_id": f"{dataset}-c{cell}", "cycle_index": k, "soh": soh,
                             "f1": soh + rng.normal(0, 0.05), "f2": rng.normal(0, 1.0),
                             "f3": np.sqrt(k) + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_classes_receive_random_state_or_seed_through_their_constructor():
    gbdt = _instantiate_estimator(GBDTEstimator, 7)
    assert gbdt._random_state == 7 and gbdt._model.get_params()["random_state"] == 7
    mlp = _instantiate_estimator(MLPEstimator, 11)
    assert mlp._model.get_params()["random_state"] == 11

    class TorchLike:
        name = "torch_like"

        def __init__(self, seed: int = 0):
            self.seed = seed

    assert _instantiate_estimator(TorchLike, 5).seed == 5
    assert isinstance(_instantiate_estimator(RidgeEstimator, 3), RidgeEstimator)  # no seed hook: plain construction


def test_instances_are_deep_copied_and_seeded_through_set_seed_or_attributes():
    calls = []

    class WithSetter:
        def __init__(self):
            self.state = {"fits": 0}

        def set_seed(self, seed):
            calls.append(seed)

    original = WithSetter()
    copy_ = _instantiate_estimator(original, 9)
    assert copy_ is not original and copy_.state is not original.state and calls == [9]

    class WithAttribute:
        random_state = 0

    assert _instantiate_estimator(WithAttribute(), 4).random_state == 4 and WithAttribute.random_state == 0


def test_set_seed_reseeds_the_sklearn_models():
    for cls in (GBDTEstimator, ExtraTreesEstimator, MLPEstimator):
        est = cls()
        est.set_seed(21)
        assert est._random_state == 21 and est._model.get_params()["random_state"] == 21
    stress = StressResidualEstimator()
    stress.set_seed(8)
    assert stress._random_state == 8


def test_mlp_differs_across_seeds_while_ridge_is_identical(tmp_path):
    table = _table()
    meta = build_meta(table)
    results = BenchmarkRunner(seeds=[0, 1]).run(
        SOHEstimationTask(), table=table, meta=meta, protocols={"loco": LeaveOneCellOut()},
        estimators=[RidgeEstimator, MLPEstimator], featuresets=["full"],
    )
    assert "error" not in results.columns or results["error"].isna().all()
    mae = results[results["metric"] == "mae"].pivot_table(index=["estimator", "held_out_key"], columns="seed", values="value")
    ridge = mae.loc["ridge"]
    mlp = mae.loc["mlp"]
    assert np.allclose(ridge[0], ridge[1])
    assert not np.allclose(mlp[0], mlp[1])
    out = write_artifacts(results, qc_report=None, task=SOHEstimationTask(), config={}, out_dir=tmp_path)
    leaderboard = {entry["estimator"]: entry for entry in json.loads((Path(out) / "leaderboard.json").read_text(encoding="utf-8"))}
    assert leaderboard["mlp"]["metrics"]["mae"]["std"] > 0.0
    # ridge's std across folds is not zero (folds differ), but across seeds it adds nothing:
    per_seed = results[(results["estimator"] == "ridge") & (results["metric"] == "mae")].groupby("seed")["value"].mean()
    assert per_seed.nunique() == 1


def test_weighted_conformal_receives_the_evaluation_seed(monkeypatch):
    seen = []

    class Spy(runner_module.WeightedConformalWrapper):
        def __init__(self, base, alpha=0.1, **kwargs):
            seen.append(kwargs.get("random_state"))
            super().__init__(base, alpha=alpha, **kwargs)

    monkeypatch.setattr(runner_module, "WeightedConformalWrapper", Spy)
    table = _table(datasets=("d1", "d2"), n_cells=2, cycles=40)
    BenchmarkRunner(seeds=[3, 4]).run(
        SOHEstimationTask(), table=table, meta=build_meta(table), protocols={"lodo": LeaveOneDatasetOut()},
        estimators=[RidgeEstimator], featuresets=["full"],
    )
    assert sorted(set(seen)) == [3, 4]
