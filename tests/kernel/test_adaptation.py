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


# ---------------------------------------------------------------------------
# the operator seam (Spec 12 §9, K13) and the calibration budget in capacity tests
# ---------------------------------------------------------------------------

from sohkernel.adaptation import (  # noqa: E402
    CALIBRATION_UNITS,
    INTERVAL_SOURCES,
    apply_operator,
    capacity_test_split,
    capacity_test_summary,
    fit_operator,
)


def _seam_table(n_cells=4, cycles=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cell in range(n_cells):
        for k in range(cycles):
            soh = 1.0 - 0.008 * k - 0.002 * cell
            rows.append((f"c{cell}", k, soh, soh + rng.normal(0, 0.01), rng.normal(0.5, 0.1)))
    return pd.DataFrame(rows, columns=["cell_id", "cycle_index", "soh", "f1", "f2"])


def _frozen_predict(rows):
    """A frozen model that is scaled and biased: what the operators correct."""
    return 1.1 * rows["f1"].to_numpy(dtype=float) + 0.03


def _frozen_interval(rows):
    pred = _frozen_predict(rows)
    return pred - 0.05, pred + 0.05


class _LoadedTwo:
    name = "frozen"
    manifest = {"trained_on": []}

    def feature_columns(self):
        return ["f1", "f2"]


@pytest.mark.parametrize("interval", ["model", "conformal"])
@pytest.mark.parametrize("operator", [op for op in ADAPTATION_OPERATORS if "coral" not in op])
def test_seam_reproduces_the_adapter_for_every_operator(operator, interval):
    """The ladder (InferenceAdapter) and serving (fit_operator / apply_operator) share one implementation."""
    table = _seam_table()
    adapter = InferenceAdapter(service=None, operator=operator, calibration_budget=24, calibration_split="spread",
                               interval=interval)
    adapted, mask, _info = adapter.prepare(_LoadedTwo(), {}, table)
    pred = _frozen_predict(adapted)
    bounds = _frozen_interval(adapted)
    params = None
    if adapter.uses_recalibration:
        pred, params = adapter.recalibrate(adapted, pred, mask)
    if params is not None:
        bounds = adapter.recalibrate_interval(bounds, params)
    columns = ["f1", "f2"]
    correction_columns = wide_residual_columns(table, columns) if "residual_wide" in operator else columns
    fitted = fit_operator(operator, table, mask, _frozen_predict, columns=columns, interval=interval,
                          correction_columns=correction_columns)
    seam_pred, seam_bounds, record = apply_operator(fitted, table, _frozen_predict, _frozen_interval)
    assert np.allclose(seam_pred, pred)
    assert np.allclose(seam_bounds[0], bounds[0]) and np.allclose(seam_bounds[1], bounds[1])
    assert record["interval_source"] in INTERVAL_SOURCES and record["calibration_rows"] == 24
    expected = "not_applicable" if operator == "none" and interval == "model" else "applied"
    assert record["verdict"] == expected and not record["refused"]
    if interval == "conformal":
        assert record["interval_source"] == "calibration_conformal"
        assert record["fitted"]["conformal"]["residual_source"] == ("leave_one_out" if params and ("slope" in params or "coef" in params) else "in_sample")


def test_seam_fits_coral_from_the_training_sample_and_reports_a_missing_one():
    rng = np.random.default_rng(5)
    source = np.column_stack([rng.normal(0.9, 0.05, 200), rng.normal(0.5, 0.1, 200)])
    target = _seam_table()
    target["f1"] = target["f1"] * 2.0 + 1.0   # the target's feature distribution is scaled and shifted
    mask, _ = calibration_split(target, 16, "spread")
    fitted = fit_operator("coral+recalibration", target, mask, _frozen_predict, columns=["f1", "f2"], source_features=source)
    assert fitted.applied[0] == "coral" and fitted.coral is not None
    aligned = apply_coral(target, fitted.coral)
    assert abs(aligned["f1"].mean() - source[:, 0].mean()) < 0.05   # CORAL's contract: target statistics match the source
    pred, bounds, record = apply_operator(fitted, target, _frozen_predict, _frozen_interval)
    assert record["verdict"] == "applied" and record["interval_source"] == "model_affine"
    assert set(record["fitted"]) == {"coral", "recalibration"} and record["fitted"]["coral"]["columns"] == ["f1", "f2"]
    loose = fit_operator("coral+recalibration", target, mask, _frozen_predict, columns=["f1", "f2"], strict=False)
    assert loose.coral is None and loose.refused[0].startswith("coral: CORAL needs the training feature sample")
    assert loose.applied == ["recalibration"] and loose.verdict == "partial"
    with pytest.raises(ValueError, match="CORAL needs the training feature sample"):
        fit_operator("coral", target, mask, _frozen_predict, columns=["f1", "f2"])


def test_seam_records_refusals_instead_of_raising_when_serving():
    table = _seam_table(n_cells=1, cycles=12)
    mask = np.zeros(len(table), dtype=bool)
    mask[:2] = True   # two calibration rows: below every labelled guard
    fitted = fit_operator("residual+monotone", table, mask, _frozen_predict, columns=["f1", "f2"], interval="conformal",
                          strict=False)
    assert fitted.correction is None and fitted.conformal is None and fitted.monotone
    assert [r.split(":")[0] for r in fitted.refused] == ["residual", "conformal"]
    assert fitted.verdict == "partial" and fitted.interval_source == "model_shifted"
    pred, bounds, record = apply_operator(fitted, table, _frozen_predict, _frozen_interval)
    assert record["refused"] == fitted.refused and record["rows"] == 12 and bounds is not None
    assert np.allclose(bounds[1] - bounds[0], 0.10)   # the model's band, shifted with the projection
    with pytest.raises(ValueError, match="residual correction needs at least 4"):
        fit_operator("residual+monotone", table, mask, _frozen_predict, columns=["f1", "f2"], interval="conformal")
    none = fit_operator("recalibration", table, np.zeros(len(table), dtype=bool), _frozen_predict, columns=["f1"], strict=False)
    assert none.refused == ["recalibration: needs calibration rows with labels; none were passed"] and none.verdict == "refused"
    bare, served, rec = apply_operator(none, table, _frozen_predict)
    assert np.allclose(bare, _frozen_predict(table)) and served is None and rec["interval_source"] == "none"


def test_seam_applies_a_fit_to_new_query_rows_with_the_conformal_band():
    table = _seam_table()
    mask, _ = calibration_split(table, 16, "spread")
    fitted = fit_operator("recalibration", table, mask, _frozen_predict, columns=["f1", "f2"], interval="conformal")
    query = _seam_table(n_cells=1, cycles=6, seed=9)
    query["cell_id"] = "new-unit"
    pred, (lo, hi), record = apply_operator(fitted, query, _frozen_predict, _frozen_interval)
    assert np.allclose(pred, apply_recalibration(_frozen_predict(query), fitted.correction))
    assert np.allclose(hi - lo, 2 * fitted.conformal["quantile"]) and record["interval_source"] == "calibration_conformal"
    assert record["calibration_rows"] == 16 and record["rows"] == 6 and record["verdict"] == "applied"
    assert set(record["fitted"]) == {"recalibration", "conformal"}
    truth = query["soh"].to_numpy()
    assert np.abs(pred - truth).mean() < np.abs(_frozen_predict(query) - truth).mean()


def _segment_table():
    # unit A: tests at 100, 200, 300 s; unit B: one test at 100 s
    rows = [
        ("A", 0, np.nan, 50.0, np.nan),     # before the first test: unlabelled
        ("A", 1, 0.99, 120.0, 100.0),
        ("A", 2, 0.99, 150.0, 100.0),
        ("A", 3, 0.98, 200.0, 200.0),       # starts at the second test: labelled by it
        ("A", 4, 0.98, 250.0, 200.0),
        ("A", 5, 0.97, 320.0, 300.0),
        ("B", 0, 0.995, 120.0, 100.0),
        ("B", 1, 0.995, 150.0, 100.0),
    ]
    return pd.DataFrame(rows, columns=["cell_id", "cycle_index", "soh", "segment_start_s", "label_time_s"])


def test_capacity_test_split_cuts_each_unit_at_its_kth_test():
    table = _segment_table()
    calibration, evaluation = calibration_split(table, 2, unit="capacity_tests")
    # A: the segments labelled by its first two tests; B (one test): all its labelled segments; the unlabelled
    # segment before A's first test and the segment labelled by A's third test are the evaluation side
    assert calibration.tolist() == [False, True, True, True, True, False, True, True]
    assert np.array_equal(evaluation, ~calibration)
    assert capacity_test_summary(table, calibration, 2) == {
        "calibration_unit": "capacity_tests", "calibration_tests": 2, "units": 2, "units_with_fewer_tests": 1,
        "calibration_labeled": 6, "evaluation_labeled": 1, "evaluation_unlabeled": 1,
    }
    one, _ = capacity_test_split(table, 1)   # K = 1: the segments labelled by each unit's first test
    assert one.tolist() == [False, True, True, False, False, False, True, True]
    # forward by construction: every evaluation row of A starts after every calibration row of A
    a = table["cell_id"] == "A"
    assert table.loc[a & ~calibration & table["soh"].notna(), "segment_start_s"].min() > table.loc[a & calibration, "segment_start_s"].max()
    zero, rest = calibration_split(table, 0, unit="capacity_tests")
    assert not zero.any() and rest.all()
    with pytest.raises(ValueError, match="forward cut"):
        capacity_test_split(table, 2, "spread")
    with pytest.raises(ValueError, match="label_time_s"):
        calibration_split(_target_table(), 2, unit="capacity_tests")
    with pytest.raises(ValueError, match="unknown calibration unit"):
        calibration_split(table, 2, unit="cycles")


def test_adapter_counts_the_budget_in_capacity_tests():
    rows = []
    for unit in ("s14", "s15", "s17"):
        ordinal = 0
        for test in (100.0, 200.0, 300.0):
            for k in range(5):
                rows.append((unit, ordinal, 1.0 - test / 4000.0, test + 10.0 + 20.0 * k, test, 0.9))
                ordinal += 1
    table = pd.DataFrame(rows, columns=["cell_id", "cycle_index", "soh", "segment_start_s", "label_time_s", "f1"])
    adapter = InferenceAdapter(service=None, operator="recalibration", calibration_budget=2, calibration_unit="capacity_tests")
    _t, mask, info = adapter.prepare(_LoadedStub(), {}, table)
    assert int(mask.sum()) == 30 and info["calibration_samples"] == 30   # two tests × five segments × three units
    assert info["calibration_unit"] == "capacity_tests" and info["calibration_tests"] == 2
    assert info["units"] == 3 and info["units_with_fewer_tests"] == 0
    assert info["calibration_labeled"] == 30 and info["evaluation_labeled"] == 15
    assert info["calibration_scenario"] == "forward_deployment" and info["supports_forward_claim"] is True
    with pytest.raises(ValueError, match="forward cut"):
        InferenceAdapter(service=None, operator="recalibration", calibration_budget=2, calibration_split="spread",
                         calibration_unit="capacity_tests")
    import inspect

    assert CALIBRATION_UNITS == ("rows", "capacity_tests")
    assert inspect.signature(calibration_split).parameters["unit"].default == "rows"
    assert inspect.signature(InferenceAdapter).parameters["calibration_unit"].default == "rows"
