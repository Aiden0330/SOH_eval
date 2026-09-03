from __future__ import annotations

import numpy as np
import pandas as pd

from sohkernel.estimators.mechanism import (
    ElasticNetEstimator,
    ExtraTreesEstimator,
    PLSEstimator,
    StressResidualEstimator,
)
from sohkernel.featuresets import _mechanism_proxy_transform, get_featureset
from sohkernel.metadata import build_meta
from sohkernel.protocols import LeaveOneCellOut
from sohkernel.runner import BenchmarkRunner
from sohkernel.tasks import SOHEstimationTask


def _mechanism_table() -> pd.DataFrame:
    rows = []
    for cell_index, cell in enumerate(("c1", "c2", "c3")):
        for cycle in range(18):
            temperature = 24.0 + cell_index + 0.04 * cycle
            current_mean = 0.55 + 0.02 * cell_index
            current_std = 0.04 + 0.001 * cycle
            capacity = 2.1 * (1.0 - 0.008 * cycle - 0.002 * cell_index)
            rows.append(
                {
                    "dataset": "demo",
                    "cell_id": cell,
                    "cycle_index": cycle,
                    "soh": capacity / 2.1,
                    "cycle_index_feature": float(cycle),
                    "cycle_fraction": cycle / 17.0,
                    "charge_capacity_max": capacity,
                    "time_s_max": 3600.0 + 10.0 * cycle,
                    "voltage_mean": 3.75 - 0.006 * cycle,
                    "voltage_std": 0.04 + 0.001 * cycle,
                    "voltage_min": 3.20 - 0.002 * cycle,
                    "voltage_max": 4.20 - 0.001 * cycle,
                    "current_mean": current_mean,
                    "current_std": current_std,
                    "abs_current_mean": abs(current_mean),
                    "temperature_mean": temperature,
                    "q_win_ah": 0.55 - 0.003 * cycle,
                    "t_win_s": 900.0 + 4.0 * cycle,
                    "ic_peak_ah_per_v": 2.2 - 0.01 * cycle,
                    "ic_peak_voltage_v": 3.72 - 0.001 * cycle,
                }
            )
    return pd.DataFrame(rows)


def test_mechanism_proxy_transform_is_target_free_and_finite_when_sources_exist():
    table = _mechanism_table()
    transformed = _mechanism_proxy_transform(table)
    assert len(transformed) == len(table)
    assert transformed["soh"].equals(table["soh"])
    for column in (
        "cycle_sqrt",
        "cycle_log1p",
        "voltage_span_v",
        "current_rms_proxy_a",
        "current_variability_ratio",
        "temperature_over_25c",
        "arrhenius_stress_factor",
    ):
        assert column in transformed
        assert np.isfinite(transformed[column]).all()
    # Label-safety structural ban: proxies derived from charge_capacity_max
    # (the calibration capacity the SOH label is computed from) are no longer
    # produced at all.
    assert "charge_energy_proxy_wh" not in transformed
    assert "cumulative_capacity_proxy_ah" not in transformed


def test_full_featureset_is_stable_when_sibling_profile_adds_derived_columns():
    table = _mechanism_table()
    transformed = _mechanism_proxy_transform(table)
    original_full = get_featureset("full").columns(table)
    transformed_full = get_featureset("full").columns(transformed)
    assert transformed_full == original_full


def test_added_estimators_fit_and_predict_physical_soh():
    table = _mechanism_proxy_transform(_mechanism_table())
    target = table["soh"].to_numpy()
    for estimator in (
        ElasticNetEstimator(),
        ExtraTreesEstimator(n_estimators=24),
        PLSEstimator(n_components=3),
        StressResidualEstimator(max_iter=30),
    ):
        prediction = estimator.fit(table, target).predict(table)
        assert prediction.shape == target.shape
        assert np.isfinite(prediction).all()
        assert (prediction >= 0.0).all()
        assert (prediction <= 1.3).all()


def test_stress_residual_has_nonnegative_prior_and_monotone_trajectories():
    table = _mechanism_proxy_transform(_mechanism_table())
    estimator = StressResidualEstimator(max_iter=30).fit(table, table["soh"].to_numpy())
    prediction = estimator.predict(table)
    assert (estimator._prior_coeffs >= 0.0).all()
    for _, group in table.assign(prediction=prediction).groupby(["dataset", "cell_id"]):
        ordered = group.sort_values("cycle_index")["prediction"].to_numpy()
        assert (np.diff(ordered) <= 1e-12).all()


def test_runner_omits_incompatible_cartesian_cells_for_method_package():
    table = _mechanism_proxy_transform(_mechanism_table())
    results = BenchmarkRunner(seeds=[0]).run(
        SOHEstimationTask(),
        table=table,
        meta=build_meta(table),
        protocols={"loco": LeaveOneCellOut()},
        estimators=[StressResidualEstimator],
        featuresets=["full", "voltage"],
    )
    assert set(results["featureset"]) == {"full"}
    assert set(results["pairing_mode"]) == {"method_package"}
    assert set(results["method_family"]) == {"gray_box_degradation"}
    assert set(results["method_category"]) == {"joint_method"}
    assert set(results["combination_policy"]) == {"joint_package"}
    assert set(results["execution_contract"]) == {"joint_method/v1"}
