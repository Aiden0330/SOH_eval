"""Tests for sohbench.estimators.pinn — Phase 2 physics-informed network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sohbench.contracts import Estimator
from sohbench.estimators.baselines import SOH_HIGH, SOH_LOW
from sohbench.estimators.pinn import PINNEstimator
from sohbench.metrics import monotonicity_violation_rate
from sohbench.registry import get_estimator


def _noisy_degradation_frame(n_cells: int = 3, n_cycles: int = 36, seed: int = 4):
    rng = np.random.default_rng(seed)
    rows = []
    for cell in range(n_cells):
        cycles = np.arange(n_cycles, dtype=float)
        base = 1.04 - 0.006 * cycles - 0.015 * cell
        bump = np.where((cycles.astype(int) % 7) == 3, 0.035, 0.0)
        soh = np.clip(base + bump + rng.normal(0, 0.003, n_cycles), 0.0, 1.3)
        for i, cyc in enumerate(cycles):
            rows.append({
                "dataset": "synth",
                "cell_id": f"c{cell}",
                "cycle_index": int(cyc),
                "soh": float(soh[i]),
                "cycle_fraction": float(cyc / max(n_cycles - 1, 1)),
                "f_soh": float(soh[i]),
                "f_bump": float(bump[i]),
            })
    X = pd.DataFrame(rows)
    return X, X["soh"].to_numpy()


def _mean_cell_violation_rate(X: pd.DataFrame, pred: np.ndarray) -> float:
    rates = []
    tmp = X.copy()
    tmp["_pred"] = pred
    for _, grp in tmp.groupby(["dataset", "cell_id"], sort=False):
        ordered = grp.sort_values("cycle_index")
        rates.append(monotonicity_violation_rate(ordered["_pred"].to_numpy()))
    return float(np.mean(rates))


def test_pinn_is_estimator_protocol():
    est = PINNEstimator()
    assert isinstance(est, Estimator)
    assert est.name == "pinn"
    assert est.feature_mode == "unified"
    assert est.backend == "pytorch"


def test_pinn_fits_and_predicts_in_range():
    X, y = _noisy_degradation_frame()
    est = PINNEstimator(hidden=24, epochs=80, lr=5e-3, seed=0, n_max=500)
    est.fit(X, y, meta=None)
    pred = est.predict(X, meta=None)
    assert pred.shape == (len(X),)
    assert np.all(pred >= SOH_LOW)
    assert np.all(pred <= SOH_HIGH)
    assert float(np.mean(np.abs(pred - y))) < 0.18


def test_physics_penalty_reduces_monotonicity_violations():
    X, y = _noisy_degradation_frame(n_cells=2, n_cycles=42)

    plain = PINNEstimator(hidden=32, epochs=100, lr=5e-3, seed=1,
                          lambda_mono=0.0, lambda_range=0.0,
                          enforce_monotone=False, n_max=500)
    phys = PINNEstimator(hidden=32, epochs=100, lr=5e-3, seed=1,
                         lambda_mono=20.0, lambda_range=1.0,
                         enforce_monotone=True, n_max=500)

    plain.fit(X, y, meta=None)
    phys.fit(X, y, meta=None)

    plain_pred = plain.predict(X, meta=None)
    phys_pred = phys.predict(X, meta=None)

    plain_rate = _mean_cell_violation_rate(X, plain_pred)
    phys_rate = _mean_cell_violation_rate(X, phys_pred)
    assert phys_rate < plain_rate


def test_pinn_registered():
    assert get_estimator("pinn") is PINNEstimator
