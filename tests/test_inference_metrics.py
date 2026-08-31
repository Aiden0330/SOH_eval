"""Extended inference metrics: pooled MAE hides where the error lives."""

import numpy as np
import pandas as pd

from sohbench.inference import _metrics


def _declining_group():
    # Two cells: c1 degrades through EOL (0.80), c2 stays healthy.
    cycles = np.arange(1, 41)
    c1_soh = np.linspace(1.0, 0.70, 40)
    c2_soh = np.linspace(1.0, 0.92, 40)
    return pd.DataFrame({
        "cell_id": ["c1"] * 40 + ["c2"] * 40,
        "cycle_index": np.concatenate([cycles, cycles]),
        "soh": np.concatenate([c1_soh, c2_soh]),
    })


def test_band_cell_and_eol_metrics_expose_a_flat_predictor():
    group = _declining_group()
    pred = np.full(80, 0.93)  # flat prediction: fine early, blind late
    out = _metrics(group, pred)

    bands = out["band_metrics"]
    assert bands["soh_ge_90"]["n"] > 0 and bands["soh_lt_80"]["n"] > 0
    # The failure concentrates below 80% SOH — pooled MAE hides this.
    assert bands["soh_lt_80"]["mae"] > 3 * bands["soh_ge_90"]["mae"]
    assert bands["soh_lt_80"]["bias"] > 0.1

    assert out["cell_mae_worst_cell"] == "c1"
    assert out["cell_mae_worst"] > out["mae"]
    assert out["cells_evaluated"] == 2

    eol = out["eol"]
    assert eol["threshold"] == 0.80
    assert eol["cells_with_true_eol"] == 1
    # The flat predictor never crosses 80%: undetected, no cycle error.
    assert eol["undetected_cells"] == 1
    assert eol["mean_abs_cycle_error"] is None


def test_eol_cycle_error_measured_when_prediction_crosses():
    group = _declining_group()
    # Prediction tracks c1's decline but crosses 0.80 five cycles late.
    c1_true = group.loc[group["cell_id"] == "c1", "soh"].to_numpy()
    pred = np.concatenate([np.roll(c1_true, 5), np.full(40, 0.93)])
    pred[:5] = c1_true[:5]
    out = _metrics(group, pred)
    eol = out["eol"]
    assert eol["cells_predicted"] == 1
    assert eol["undetected_cells"] == 0
    assert 3 <= eol["mean_abs_cycle_error"] <= 7


def test_interval_metrics_report_picp_and_width():
    group = _declining_group()
    pred = np.full(80, 0.93)
    lower = pred - 0.05
    upper = pred + 0.05
    out = _metrics(group, pred, interval=(lower, upper))
    truth = group["soh"].to_numpy()
    expected = float(np.mean((lower <= truth) & (truth <= upper)))
    assert abs(out["picp"] - expected) < 1e-12
    assert abs(out["mean_interval_width"] - 0.10) < 1e-9


def test_eval_mask_slices_interval_too():
    group = _declining_group()
    pred = np.full(80, 0.93)
    mask = np.ones(80, dtype=bool)
    mask[:10] = False  # pretend the first rows are calibration samples
    out = _metrics(group, pred, eval_mask=mask, interval=(pred - 0.05, pred + 0.05))
    assert out["band_metrics"]["soh_ge_90"]["n"] < 70
    assert out["picp"] is not None
