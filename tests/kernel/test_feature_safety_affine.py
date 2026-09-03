import numpy as np
import pandas as pd

from sohkernel.feature_safety import audit_feature_frame


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    soh = np.linspace(1.0, 0.7, n) + rng.normal(0.0, 0.002, n)
    return pd.DataFrame(
        {
            "dataset": ["demo"] * n,
            "cell_id": ["c1"] * n,
            "soh": soh,
            # Affine label copy: the calibration capacity the label is built from.
            "capacity_like": soh * 1.1 + rng.normal(0.0, 1e-5, n),
            # Strong but legitimate physical signal (~R2 well below 0.995).
            "physical_like": soh * 0.8 + rng.normal(0.0, 0.03, n),
        }
    )


def test_affine_label_reconstruction_is_blocked():
    audit = audit_feature_frame(_frame(), ["capacity_like", "physical_like"])
    assert audit["status"] == "blocked"
    kinds = {(item["column"], item["kind"]) for item in audit["violations"]}
    assert ("capacity_like", "affine_label_reconstruction") in kinds
    assert all(column != "physical_like" for column, _ in kinds)


def test_noisy_physical_signal_passes():
    audit = audit_feature_frame(_frame(), ["physical_like"])
    assert audit["status"] == "passed"


def test_small_samples_are_skipped():
    # Below the 64-sample floor the affine check must not fire (smoke fixtures).
    audit = audit_feature_frame(_frame(n=40), ["capacity_like"])
    assert audit["status"] == "passed"


def _window_frame(coverage: float, n=400, seed=0):
    """Table where q_win_ah integrates *coverage* of the full capacity."""
    rng = np.random.default_rng(seed)
    soh = np.linspace(1.0, 0.7, n) + rng.normal(0.0, 0.002, n)
    capacity = soh * 1.1
    return pd.DataFrame(
        {
            "dataset": ["demo"] * n,
            "cell_id": ["c1"] * n,
            "soh": soh,
            "charge_capacity_max": capacity,
            # Enough relative noise to stay under the affine R² threshold —
            # the DOD rule, not the affine fit, must catch the full window.
            "q_win_ah": capacity * coverage + rng.normal(0.0, 0.05, n),
        }
    )


def test_full_range_ah_window_is_blocked():
    # A "partial" window integrating ~97% of full capacity IS the label source.
    audit = audit_feature_frame(_window_frame(0.97), ["q_win_ah"])
    assert audit["status"] == "blocked"
    kinds = {(item["column"], item["kind"]) for item in audit["violations"]}
    assert ("q_win_ah", "full_range_ah_window") in kinds


def test_typical_partial_window_passes():
    # 45% DOD is normal practice (LFP: plateau region) — no finding at all.
    audit = audit_feature_frame(_window_frame(0.45), ["q_win_ah"])
    assert audit["status"] == "passed"
    assert not audit["warnings"]


def test_wide_partial_window_warns_but_runs():
    # Between typical (40–50%) and the 90% bound: warn, do not block.
    audit = audit_feature_frame(_window_frame(0.72), ["q_win_ah"])
    assert audit["status"] == "passed"
    kinds = {(item["column"], item["kind"]) for item in audit["warnings"]}
    assert ("q_win_ah", "wide_ah_window") in kinds


# ---------------------------------------------------------------------------
# coverage denominator: rated capacity first (D5 / report finding G6)
# ---------------------------------------------------------------------------

def _known_dataset_frame(dataset, cell, q_win, capacity_column=20.0, n=200, seed=0):
    """A dataset with a *distorted* capacity column (file-cumulative, like calce) and a
    window integral that is a fixed fraction of the nameplate capacity."""
    rng = np.random.default_rng(seed)
    soh = np.linspace(1.0, 0.8, n) + rng.normal(0, 0.003, n)
    return pd.DataFrame({
        "dataset": [dataset] * n,
        "cell_id": [cell] * n,
        "cycle_index": np.arange(n),
        "soh": soh,
        "charge_capacity_max": capacity_column,               # distorted: not per cycle
        "q_win_ah": q_win + rng.normal(0.0, 0.01, n),           # not an affine label copy
    })


def test_window_coverage_uses_rated_capacity_for_known_datasets():
    # hust is a 1.1 Ah cell: a 1.0 Ah window is 91 % of rated even though the distorted
    # capacity column would make it look like 5 % (the G6 "falsely clean" case)
    audit = audit_feature_frame(_known_dataset_frame("hust", "cell_1", q_win=1.0), ["q_win_ah"])
    assert audit["status"] == "blocked"
    (violation,) = audit["violations"]
    assert violation["kind"] == "full_range_ah_window" and violation["reference"] == "rated capacity"
    assert "of rated capacity" in violation["reason"]


def test_window_coverage_honours_cell_prefix_overrides():
    # calce CX2 cells are 1.35 Ah: a 1.05 Ah window is 78 % (warning) of the CX2 nameplate,
    # while the dataset default of 1.1 Ah would have blocked it at 95 %
    audit = audit_feature_frame(_known_dataset_frame("calce_cs2", "CX2_33", q_win=1.05), ["q_win_ah"])
    assert audit["status"] == "passed"
    assert [w["kind"] for w in audit["warnings"]] == ["wide_ah_window"]
    blocked = audit_feature_frame(_known_dataset_frame("calce_cs2", "CS2_33", q_win=1.05), ["q_win_ah"])
    assert blocked["status"] == "blocked"


def test_window_coverage_falls_back_to_the_capacity_column_for_unknown_datasets():
    audit = audit_feature_frame(_window_frame(0.95), ["q_win_ah"])
    assert audit["status"] == "blocked"
    assert audit["violations"][0]["reference"] == "full capacity (charge_capacity_max)"
    # no reference at all -> no coverage finding
    frame = _window_frame(0.95).drop(columns=["charge_capacity_max"])
    assert audit_feature_frame(frame, ["q_win_ah"])["status"] == "passed"
