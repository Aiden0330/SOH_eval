"""K12: literature partial-charge health indicators from raw charge curves, and the partial_charge_hi featureset."""
import numpy as np
import pandas as pd
import pytest

from sohkernel import health_indicators as hi
from sohkernel.feature_safety import PARTIAL_WINDOW_CAPACITY_COLUMNS, audit_feature_frame
from sohkernel.featuresets import _partial_charge_hi_transform
from sohkernel.metadata import DATASET_CHEMISTRY
from sohkernel.registry import get_featureset


def _linear_curve(cell="c1", cycle=0, v_lo=3.0, v_hi=3.6, q_total=1.0, t_total=3600.0, n=601):
    v = np.linspace(v_lo, v_hi, n)
    q = np.linspace(0.0, q_total, n)
    t = np.linspace(0.0, t_total, n)
    return (cell, cycle, v, q, t)


def test_windows_resolve_every_registered_chemistry_and_tile_their_span():
    for chemistry in set(DATASET_CHEMISTRY.values()):
        windows = hi.windows_for(chemistry)
        assert len(windows) == hi.N_WINDOWS
        assert all(windows[i][1] == windows[i + 1][0] for i in range(len(windows) - 1))   # contiguous
    assert hi.windows_for("LCO/NCO") == hi.windows_for("LCO")
    assert hi.windows_for("LFP")[0][0] == 3.30 and hi.windows_for("LFP")[-1][1] == 3.40   # covered by every LFP lab
    with pytest.raises(KeyError, match="no health-indicator windows"):
        hi.windows_for("sodium")


def test_indicators_of_a_linear_curve_are_exact():
    out = hi.hi_from_curves([_linear_curve()], "LFP")
    assert list(out.columns) == ["cell_id", "cycle_index", *hi.RAW_COLUMNS] and len(out) == 1
    row = out.iloc[0]
    # 1.0 Ah over 0.6 V: a 25 mV window holds 1/24 Ah, the 100 mV span 1/6 Ah; 3600 s over 0.6 V: 150 s and 600 s
    assert row["hi_dq_w1_ah"] == pytest.approx(1 / 24, rel=1e-6) and row["hi_dq_w4_ah"] == pytest.approx(1 / 24, rel=1e-6)
    assert row["hi_span_dq_ah"] == pytest.approx(1 / 6, rel=1e-6)
    assert row["hi_dt_w1_s"] == pytest.approx(150.0, rel=1e-6) and row["hi_span_dt_s"] == pytest.approx(600.0, rel=1e-6)
    assert row["hi_ic_peak_ah_per_v"] == pytest.approx(1 / 0.6, rel=1e-6)   # flat IC: every bin ties
    assert 3.30 < row["hi_ic_peak_v"] < 3.40
    assert row["hi_v_q25"] == pytest.approx(3.325, abs=1e-6) and row["hi_v_q50"] == pytest.approx(3.35, abs=1e-6)
    assert row["hi_v_q75"] == pytest.approx(3.375, abs=1e-6)
    assert row["hi_dvdq_v_per_ah"] == pytest.approx(0.6, rel=1e-6)
    assert row["hi_i_mean_a"] == pytest.approx(1.0, rel=1e-6) and row["hi_i_cv"] == pytest.approx(0.0, abs=1e-9)


def test_the_incremental_capacity_peak_sits_where_the_charge_density_doubles():
    v = np.linspace(3.0, 3.6, 601)                       # 1 mV steps
    step = np.full(600, 1.0 / 600.0)
    step[(v[:-1] >= 3.36 - 1e-9) & (v[:-1] < 3.37 - 1e-9)] *= 2.0   # twice the charge per mV inside 3.36–3.37 V
    q = np.concatenate([[0.0], np.cumsum(step)])
    t = np.linspace(0.0, 3600.0, 601)
    row = hi.hi_from_curves([("c1", 0, v, q, t)], "LFP").iloc[0]
    assert row["hi_ic_peak_v"] == pytest.approx(3.365, abs=1e-9)
    assert row["hi_ic_peak_ah_per_v"] == pytest.approx(2 / 0.6, rel=1e-6)


def test_a_curve_that_stops_inside_the_span_is_nan_not_wrong():
    short = _linear_curve(cycle=1, v_lo=3.0, v_hi=3.36, q_total=0.6)
    out = hi.hi_from_curves([_linear_curve(), short], "LFP").set_index("cycle_index")
    assert np.isfinite(out.loc[0, "hi_span_dq_ah"])
    assert np.isnan(out.loc[1, "hi_dq_w3_ah"]) and np.isnan(out.loc[1, "hi_span_dq_ah"]) and np.isnan(out.loc[1, "hi_v_q50"])
    assert np.isfinite(out.loc[1, "hi_dq_w1_ah"]) and np.isfinite(out.loc[1, "hi_dq_w2_ah"])
    assert np.isfinite(out.loc[1, "hi_ic_peak_ah_per_v"]) and out.loc[1, "hi_ic_peak_v"] < 3.36
    assert np.isfinite(out.loc[1, "hi_i_cv"])   # two windows carry a current


def test_a_curve_with_voltage_blips_still_reads_as_monotone_charge():
    cell, cycle, v, q, t = _linear_curve()
    noisy_v = v + np.where(np.arange(len(v)) % 50 == 0, -0.004, 0.0)   # small voltage dips (relaxation blips)
    out = hi.hi_from_curves([(cell, cycle, noisy_v, q, t)], "LFP").iloc[0]
    assert out["hi_span_dq_ah"] == pytest.approx(1 / 6, rel=0.05)
    empty = hi.hi_from_curves([(cell, cycle, np.array([3.3]), np.array([0.0]), np.array([0.0]))], "LFP").iloc[0]
    assert np.isnan(empty["hi_span_dq_ah"]) and np.isnan(empty["hi_ic_peak_v"])


def test_attach_joins_on_identity_and_leaves_unmatched_cycles_nan():
    indicators = hi.hi_from_curves([_linear_curve(cycle=3)], "LFP")
    table = pd.DataFrame({"dataset": ["hust"] * 2, "cell_id": ["c1", "c1"], "cycle_index": [3, 4], "soh": [1.0, 0.99],
                          "time_s_max": [3600.0, 3600.0]})
    joined = hi.attach(table, indicators)
    assert joined.loc[0, "hi_span_dq_ah"] == pytest.approx(1 / 6, rel=1e-6) and np.isnan(joined.loc[1, "hi_span_dq_ah"])
    assert hi.coverage_summary(joined, 1.1)["available"]["hi_span_dq_ah"] == 0.5
    assert hi.coverage_summary(joined, 1.1)["span_coverage_median"] == pytest.approx(round(1 / 6 / 1.1, 4))


def test_featureset_projects_dimensionless_indicators_from_rated_capacity():
    indicators = hi.hi_from_curves([_linear_curve(cell="c1"), _linear_curve(cell="c9", q_total=50.0, t_total=7200.0)], "LFP")
    table = pd.DataFrame({"dataset": ["a", "b"], "cell_id": ["c1", "c9"], "cycle_index": [0, 0], "soh": [1.0, 1.0],
                          "time_s_max": [3600.0, 7200.0]})
    table = hi.attach(table, indicators)
    meta = pd.DataFrame({"dataset": ["a", "b"], "cell_id": ["c1", "c9"], "c_rated_ah": [1.0, 50.0]})
    out = _partial_charge_hi_transform(table, meta)
    # a 1 Ah cell and a 50 Ah cell with the same curve shape land on the same dimensionless values
    for column in ("hi_dq_w1_frac", "hi_span_dq_frac", "hi_dt_w1_frac", "hi_ic_peak_frac_per_v", "hi_dvdq_v_per_frac", "hi_i_cv"):
        assert out[column].iloc[0] == pytest.approx(out[column].iloc[1], rel=1e-6), column
    assert out["hi_span_dq_frac"].iloc[0] == pytest.approx(1 / 6, rel=1e-6)
    # the C-rate is a descriptor, not a shape feature: the 50 Ah cell charged in twice the time sits at half the rate
    assert out["hi_i_mean_c"].iloc[0] == pytest.approx(1.0, rel=1e-6) and out["hi_i_mean_c"].iloc[1] == pytest.approx(0.5, rel=1e-6)
    featureset = get_featureset("partial_charge_hi")
    assert featureset.columns(out) == list(featureset.candidates) and len(featureset.candidates) == 18
    missing = _partial_charge_hi_transform(table, meta.iloc[:1])
    assert missing["hi_span_dq_frac"].isna().tolist() == [False, True]
    full = get_featureset("full")
    assert not set(featureset.candidates) & set(full.columns(out))   # request-scoped derived columns stay out of "full"


def test_indicator_charges_are_window_capacities_under_the_dod_rule():
    assert set(hi.WINDOW_CAPACITY_COLUMNS) <= PARTIAL_WINDOW_CAPACITY_COLUMNS
    rng = np.random.default_rng(0)
    n = 200
    soh = np.linspace(1.0, 0.8, n)
    table = pd.DataFrame({"dataset": ["hust"] * n, "cell_id": ["c1"] * n, "cycle_index": np.arange(n), "soh": soh,
                          "charge_capacity_max": 1.1 * soh,
                          "hi_dq_w1_ah": 0.09 * soh + rng.normal(0, 0.002, n),
                          "hi_span_dq_ah": 0.45 * soh + rng.normal(0, 0.005, n)})
    clean = audit_feature_frame(table, ["hi_dq_w1_ah", "hi_span_dq_ah"])
    assert clean["status"] == "passed" and not clean["violations"]
    leaky = table.assign(hi_span_dq_ah=1.15 * soh)   # a "window" that is the whole charge and more
    audit = audit_feature_frame(leaky, ["hi_span_dq_ah"])
    assert audit["violations"] and any("hi_span_dq_ah" in str(item) for item in audit["violations"])
