"""Inference-phase adaptation: operator math, channels, adapter mechanics (kernel level).

The end-to-end search and case-pipeline tests, which need EvaluationService, live in
tests/test_adaptation_workbench.py."""

import numpy as np
import pandas as pd
import pytest

from sohkernel.adaptation import (
    ADAPTATION_OPERATORS,
    CALIBRATION_SCENARIOS,
    CALIBRATION_SPLITS,
    OPERATOR_CHANNELS,
    InferenceAdapter,
    recalibration_loo_predictions,
    residual_loo_predictions,
    apply_coral,
    apply_monotone,
    apply_recalibration,
    apply_residual,
    calibration_split,
    apply_conformal_interval,
    fit_conformal_interval,
    fit_coral,
    fit_recalibration,
    fit_residual,
    wide_residual_columns,
)


class ConstantModel:
    """Picklable frozen model for the end-to-end adaptation test."""

    def predict(self, X, meta=None):
        return np.full(len(X), 0.9)


def test_coral_aligns_target_statistics_to_source():
    rng = np.random.default_rng(0)
    source = rng.normal([0.0, 1.0], [1.0, 0.5], size=(400, 2))
    target = rng.normal([5.0, -2.0], [3.0, 0.1], size=(400, 2))
    coral = fit_coral(source, target, ["a", "b"])
    table = pd.DataFrame(target, columns=["a", "b"])
    aligned = apply_coral(table, coral)[["a", "b"]].to_numpy()
    assert np.allclose(aligned.mean(axis=0), source.mean(axis=0), atol=0.15)
    assert np.allclose(aligned.std(axis=0), source.std(axis=0), atol=0.15)


def test_recalibration_removes_scale_and_bias():
    truth = np.linspace(1.0, 0.7, 50)
    biased = truth * 1.3 + 0.2
    recal = fit_recalibration(truth, biased)
    corrected = apply_recalibration(biased, recal)
    assert np.abs(corrected - truth).max() < 1e-9


def test_calibration_split_takes_earliest_cycles_round_robin():
    table = pd.DataFrame({
        "cell_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
        "cycle_index": [1, 2, 3, 1, 2, 3],
    })
    calibration, evaluation = calibration_split(table, 2)
    # One earliest cycle per cell, not two from the same cell.
    chosen = table[calibration]
    assert sorted(chosen["cell_id"]) == ["c1", "c2"]
    assert (chosen["cycle_index"] == 1).all()
    assert evaluation.sum() == 4


def test_every_operator_declares_its_channel():
    assert set(OPERATOR_CHANNELS) == set(ADAPTATION_OPERATORS)


def test_spread_calibration_split_covers_the_cycle_range():
    table = pd.DataFrame({
        "cell_id": ["c1"] * 10 + ["c2"] * 10,
        "cycle_index": list(range(1, 11)) * 2,
    })
    calibration, _evaluation = calibration_split(table, 6, "spread")
    chosen = table[calibration]
    for cell in ("c1", "c2"):
        cycles = sorted(chosen.loc[chosen["cell_id"] == cell, "cycle_index"])
        # Endpoints first, then a midpoint — never six-earliest.
        assert cycles[0] == 1 and cycles[-1] == 10
        assert len(cycles) == 3
    # Same table under "earliest" picks only the first cycles.
    earliest, _ = calibration_split(table, 6, "earliest")
    assert sorted(table[earliest]["cycle_index"]) == [1, 1, 2, 2, 3, 3]


def test_recalibration_guard_falls_back_on_degenerate_slope():
    rng = np.random.default_rng(3)
    truth = np.linspace(1.0, 0.7, 40)
    # Nearly-constant predictions: enough variance to fit, none to trust.
    pred = 0.9 + rng.normal(0.0, 1e-6, size=40)
    recal = fit_recalibration(truth, pred)
    assert recal["fallback"] == "intercept_only"
    corrected = apply_recalibration(pred, recal)
    assert np.abs(corrected - truth).max() < 0.5  # no scale explosion


def test_conformal_interval_covers_at_the_nominal_level():
    rng = np.random.default_rng(11)
    truth = np.linspace(1.0, 0.7, 2000)
    pred = truth + rng.normal(0, 0.01, 2000)
    calibration = np.zeros(2000, dtype=bool)
    calibration[::10] = True  # 200 exchangeable calibration samples
    conformal = fit_conformal_interval(truth[calibration], pred[calibration])
    lower, upper = apply_conformal_interval(pred[~calibration], conformal)
    y = truth[~calibration]
    coverage = float(np.mean((lower <= y) & (y <= upper)))
    assert 0.86 <= coverage <= 0.96          # nominal 90%, finite-sample slack
    assert conformal["quantile"] < 0.03      # width tracks the residual scale
    assert conformal["samples"] == 200


def test_conformal_interval_refuses_tiny_calibration():
    with pytest.raises(ValueError):
        fit_conformal_interval(np.ones(5), np.ones(5))


def test_wide_residual_columns_enforce_the_label_source_ban():
    table = pd.DataFrame({
        "cell_id": ["c1", "c2"],
        "soh": [1.0, 0.9],
        "charge_capacity_max": [1.1, 1.0],   # affine label source — banned
        "capacity_ah": [1.1, 1.0],            # banned alias
        "voltage_mean": [3.3, 3.31],
        "q_win_frac": [0.45, 0.44],
        "cycle_index": [1, 2],
    })
    columns = wide_residual_columns(table, ["q_win_frac"])
    assert columns[0] == "q_win_frac"          # model columns lead
    assert "voltage_mean" in columns and "cycle_index" in columns
    assert "soh" not in columns
    assert "charge_capacity_max" not in columns
    assert "capacity_ah" not in columns
    assert "cell_id" not in columns            # non-numeric excluded


def test_monotone_projection_enforces_per_cell_decline():
    rng = np.random.default_rng(7)
    cycles = np.arange(1, 61)
    table = pd.DataFrame({
        "cell_id": ["c1"] * 60 + ["c2"] * 60,
        "cycle_index": np.concatenate([cycles, cycles]),
    })
    trend = np.linspace(1.0, 0.8, 60)
    jittery = np.concatenate([
        trend + rng.normal(0, 0.01, 60),
        trend + rng.normal(0, 0.01, 60),
    ])
    smoothed, info = apply_monotone(table, jittery)
    for cell in ("c1", "c2"):
        values = smoothed[(table["cell_id"] == cell).to_numpy()]
        assert (np.diff(values) <= 1e-9).all()
    assert info["cells"] == 2
    assert info["adjusted_fraction"] > 0
    # The projection removes noise, it doesn't distort the trend.
    assert np.mean(np.abs(smoothed - np.concatenate([trend, trend]))) < 0.01


def test_residual_correction_restores_a_missed_trend():
    # Frozen model predicts a constant while truth declines with a feature
    # the model ignored — affine output recalibration cannot fix this.
    t = np.linspace(0.0, 1.0, 60)
    truth = 1.0 - 0.3 * t
    pred = np.full(60, 0.9)
    features = np.column_stack([t, np.ones(60)])
    calibration = np.zeros(60, dtype=bool)
    calibration[:: 60 // 12] = True
    model = fit_residual(
        features[calibration], pred[calibration], truth[calibration], ["t", "one"]
    )
    corrected = apply_residual(features, pred, model)
    baseline_mae = float(np.mean(np.abs(pred[~calibration] - truth[~calibration])))
    corrected_mae = float(np.mean(np.abs(corrected[~calibration] - truth[~calibration])))
    assert corrected_mae < baseline_mae / 3
    assert model["samples"] == int(calibration.sum())


def test_recalibration_refuses_plateau_only_calibration():
    rng = np.random.default_rng(5)
    truth = 1.0 + rng.normal(0, 0.005, 32)  # fresh plateau: no degradation span
    pred = 0.95 + rng.normal(0, 0.01, 32)
    recal = fit_recalibration(truth, pred)
    assert recal["fallback"] == "intercept_only"
    assert recal["fallback_reason"] == "flat_calibration_truth"
    assert recal["slope"] == 1.0
    # A declining trajectory must still decline after the correction.
    decline = np.linspace(0.95, 0.75, 20)
    corrected = apply_recalibration(decline, recal)
    assert corrected[0] > corrected[-1]


def test_recalibration_refuses_uncorrelated_predictions():
    rng = np.random.default_rng(9)
    truth = np.linspace(1.0, 0.7, 40)
    pred = 0.9 + rng.normal(0, 0.05, 40)  # spans a range, tracks nothing
    recal = fit_recalibration(truth, pred)
    assert recal["fallback"] == "intercept_only"
    assert recal["fallback_reason"] in {"weak_correlation", "slope_out_of_bounds"}


def test_recalibration_keeps_a_real_anti_correlated_fit():
    truth = np.linspace(1.0, 0.7, 40)
    pred = -truth + 2.0
    recal = fit_recalibration(truth, pred)
    assert "fallback" not in recal
    assert recal["slope"] < 0 and abs(recal["correlation"]) > 0.99


def test_recalibration_loo_matches_exact_refits():
    rng = np.random.default_rng(7)
    truth = np.linspace(1.0, 0.7, 24)
    pred = truth * 1.2 + 0.05 + rng.normal(0, 0.01, 24)
    recal = fit_recalibration(truth, pred)
    loo = recalibration_loo_predictions(truth, pred, recal)
    for i in range(24):
        mask = np.ones(24, dtype=bool)
        mask[i] = False
        slope, intercept = np.polyfit(pred[mask], truth[mask], 1)
        assert loo[i] == pytest.approx(pred[i] * slope + intercept, abs=1e-9)


def test_conformal_on_loo_residuals_covers_after_a_residual_correction():
    """In-sample residuals of a 16-feature ridge fitted on 32 rows are shrunken:
    a band built on them under-covers badly; the leave-one-out band does not."""
    in_sample, loo = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n, p, budget = 800, 16, 32
        X = rng.normal(size=(n, p))
        truth = 1.0 - 0.3 * (X[:, 0] - X[:, 0].min()) / np.ptp(X[:, 0])
        pred = truth + 0.05 * X[:, 1] + 0.03 * X[:, 2] + rng.normal(0, 0.02, n)
        cal = np.zeros(n, dtype=bool)
        cal[rng.choice(n, budget, replace=False)] = True
        cols = [f"f{i}" for i in range(p)]
        model = fit_residual(X[cal], pred[cal], truth[cal], cols)
        corrected = apply_residual(X, pred, model)
        err = np.abs(truth[~cal] - corrected[~cal])
        q_in = fit_conformal_interval(truth[cal], corrected[cal])["quantile"]
        loo_pred = residual_loo_predictions(X[cal], pred[cal], truth[cal], model)
        q_loo = fit_conformal_interval(truth[cal], loo_pred)["quantile"]
        in_sample.append(float(np.mean(err <= q_in)))
        loo.append(float(np.mean(err <= q_loo)))
    assert np.mean(in_sample) < 0.7
    assert 0.8 <= np.mean(loo) <= 0.97


class _LoadedStub:
    def feature_columns(self):
        return ["f1"]


def _target_table(n_cells=4, cycles=20):
    rows = [
        (f"c{cell}", k, 1.0 - 0.01 * k, float(k))
        for cell in range(n_cells)
        for k in range(cycles)
    ]
    return pd.DataFrame(rows, columns=["cell_id", "cycle_index", "soh", "f1"])


def test_adapter_holds_out_the_calibration_subset_for_label_free_operators():
    table = _target_table()
    plain = InferenceAdapter(service=None, operator="none", calibration_budget=8)
    labeled = InferenceAdapter(service=None, operator="recalibration", calibration_budget=8)
    _t1, mask_plain, info_plain = plain.prepare(_LoadedStub(), {}, table)
    _t2, mask_labeled, info_labeled = labeled.prepare(_LoadedStub(), {}, table)
    assert int(mask_plain.sum()) == 8
    assert np.array_equal(mask_plain, mask_labeled)  # identical evaluation subset
    assert info_plain["calibration_samples"] == info_labeled["calibration_samples"] == 8
    assert "evaluation_holdout" in info_plain
    assert "evaluation_holdout" not in info_labeled


def test_adapter_refuses_a_budget_that_leaves_no_evaluation_rows():
    table = _target_table(n_cells=1, cycles=12)
    adapter = InferenceAdapter(service=None, operator="recalibration", calibration_budget=12)
    with pytest.raises(ValueError, match="evaluation rows"):
        adapter.prepare(_LoadedStub(), {}, table)


def test_adapter_conformal_band_uses_loo_residuals_of_the_labeled_fit():
    rng = np.random.default_rng(3)
    table = _target_table()
    adapter = InferenceAdapter(
        service=None,
        operator="recalibration",
        calibration_budget=16,
        calibration_split="spread",
        interval="conformal",
    )
    _t, mask, _info = adapter.prepare(_LoadedStub(), {}, table)
    pred = table["soh"].to_numpy() * 1.1 + 0.02 + rng.normal(0, 0.01, len(table))
    corrected, result = adapter.recalibrate(table, pred, mask)
    assert result["conformal"]["residual_source"] == "leave_one_out"
    lower, upper = adapter.recalibrate_interval(None, result)
    assert np.all(upper - lower > 0)
    assert np.allclose((upper + lower) / 2, corrected)


def test_residual_correction_imputes_missing_feature_values():
    rng = np.random.default_rng(21)
    n = 40
    features = rng.normal(size=(n, 3))
    truth = np.linspace(1.0, 0.8, n)
    pred = truth + 0.05 * features[:, 0] + rng.normal(0, 0.005, n)
    features[::5, 1] = np.nan  # a partially missing wide column
    model = fit_residual(features, pred, truth, ["a", "b", "c"])
    assert model["samples"] == n  # rows with a NaN feature are kept, not dropped
    assert len(model["impute"]) == 3
    corrected = apply_residual(features, pred, model)
    assert np.isfinite(corrected).all()
    assert np.abs(corrected - truth).mean() < np.abs(pred - truth).mean()


def test_wide_residual_columns_skip_all_nan_columns():
    table = pd.DataFrame({
        "cell_id": ["c"] * 4,
        "cycle_index": [1, 2, 3, 4],
        "soh": [1.0, 0.99, 0.98, 0.97],
        "voltage_mean": [3.6, 3.59, 3.58, 3.57],
        "q_win_ah": [np.nan] * 4,  # transform column the target cannot derive
        "charge_capacity_max": [1.1, 1.09, 1.08, 1.07],  # banned
    })
    assert wide_residual_columns(table, ["voltage_mean"]) == ["voltage_mean", "cycle_index"]


# ---------------------------------------------------------------------------
# calibration split default and labelling (Task 7c)
# ---------------------------------------------------------------------------

def test_earliest_is_the_default_calibration_split_everywhere():
    import inspect

    assert inspect.signature(calibration_split).parameters["strategy"].default == "earliest"
    assert inspect.signature(InferenceAdapter).parameters["calibration_split"].default == "earliest"
    assert CALIBRATION_SPLITS == ("earliest", "spread")
    assert set(CALIBRATION_SCENARIOS) == set(CALIBRATION_SPLITS)


@pytest.mark.parametrize("split, scenario, forward", [
    ("earliest", "forward_deployment", True),
    ("spread", "periodic_checkup_backtest", False),
])
def test_provenance_labels_the_calibration_split_explicitly(split, scenario, forward):
    table = _target_table()
    adapter = InferenceAdapter(service=None, operator="recalibration", calibration_budget=8,
                               calibration_split=split)
    _t, _mask, info = adapter.prepare(_LoadedStub(), {}, table)
    assert info["calibration_split"] == split
    assert info["calibration_scenario"] == scenario
    assert info["supports_forward_claim"] is forward


def test_zero_budget_records_no_calibration_split():
    table = _target_table()
    _t, mask, info = InferenceAdapter(service=None, operator="none", calibration_budget=0).prepare(_LoadedStub(), {}, table)
    assert int(mask.sum()) == 0 and "calibration_split" not in info and "calibration_scenario" not in info
