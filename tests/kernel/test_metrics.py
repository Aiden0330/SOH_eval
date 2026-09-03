import numpy as np
import pytest
from sohkernel import metrics as M

yt = np.array([1.00, 0.90, 0.80, 0.70])
yp = np.array([0.98, 0.92, 0.78, 0.69])

def test_rmse_mae_max():
    assert M.mae(yt, yp) == pytest.approx(0.0175)
    assert M.rmse(yt, yp) == pytest.approx(np.sqrt(np.mean((yt-yp)**2)))
    assert M.max_error(yt, yp) == pytest.approx(0.02)

def test_mape():
    assert M.mape(yt, yp) == pytest.approx(np.mean(np.abs((yt-yp)/yt)))

def test_mean_bias_error_sign():
    # pred systematically higher than true => positive bias (over-estimation)
    assert M.mean_bias_error(np.array([0.8, 0.8]), np.array([0.9, 0.9])) == pytest.approx(0.1)
    assert M.mean_bias_error(np.array([0.8, 0.8]), np.array([0.7, 0.7])) == pytest.approx(-0.1)

def test_overestimation_rate():
    assert M.overestimation_rate(np.array([0.8, 0.8, 0.8]),
                                 np.array([0.9, 0.7, 0.9])) == pytest.approx(2/3)

def test_alpha_accuracy():
    # rel errors: 0.02,0.0222,0.025,0.0143 all < 0.15 => 1.0
    assert M.alpha_accuracy(yt, yp, alpha=0.15) == pytest.approx(1.0)

def test_eol_region_error_filters():
    # only samples with yt<=0.8 count: indices 2,3
    exp = np.sqrt(np.mean((yt[2:]-yp[2:])**2))
    assert M.eol_region_error(yt, yp, eol_threshold=0.8) == pytest.approx(exp)

def test_picp():
    lo = np.array([0.95, 0.85, 0.75, 0.65]); hi = np.array([1.05, 0.95, 0.85, 0.75])
    assert M.picp(yt, lo, hi) == pytest.approx(1.0)
    lo2 = np.array([0.99, 0.85, 0.75, 0.65])  # 1.00 still inside [0.99,1.05]
    assert M.picp(yt, lo2, hi) == pytest.approx(1.0)

def test_monotonicity_violation_rate():
    # SOH should be non-increasing; a rise > tol is a violation
    ordered = np.array([1.0, 0.99, 1.02, 0.98])  # one rise 0.99->1.02 (+0.03 > 0.005)
    assert M.monotonicity_violation_rate(ordered, rebound_tol=0.005) == pytest.approx(1/3)

def test_value_range_violation():
    preds = np.array([0.8, 1.25, 1.35, -0.1])  # 1.35 and -0.1 out of [0,1.3)
    assert M.value_range_violation(preds, lower=0.0, upper=1.3) == pytest.approx(0.5)


def test_per_cell_monotonicity_ignores_cross_cell_level_jumps():
    import numpy as np
    import pytest
    from sohkernel.metrics import per_cell_monotonicity_violation_rate, monotonicity_violation_rate
    cells = ["a"] * 5 + ["b"] * 5
    cycles = list(range(5)) * 2
    pred = [1.0, 0.99, 0.98, 0.97, 0.96, 0.8, 0.79, 0.78, 0.77, 0.76]
    assert per_cell_monotonicity_violation_rate(cells, cycles, pred) == 0.0
    # A global cycle sort interleaves a/b at every cycle -> bogus rebounds.
    order = np.argsort(cycles, kind="stable")
    assert monotonicity_violation_rate(np.asarray(pred)[order]) > 0.0
    # A real within-cell rebound is still counted (1 of 8 transitions).
    pred[2] = 1.05
    assert per_cell_monotonicity_violation_rate(cells, cycles, pred) == pytest.approx(1 / 8)
