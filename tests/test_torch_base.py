"""Tests for sohbench.estimators.torch_base.

Gated with pytest.importorskip("torch") so the suite stays green even when
PyTorch is not installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sohbench.estimators.torch_base import (
    build_windows,
    clamp,
    set_seed,
    subsample_idx,
    train_mlp_like,
)
from sohbench.estimators.baselines import SOH_LOW, SOH_HIGH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_cell_frame(n_cycles: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Frame with cellA and cellB, each having ``n_cycles`` cycles and 2 features."""
    records = []
    for cell, ds_val in [("cellA", "ds1"), ("cellB", "ds1")]:
        for c in range(n_cycles):
            # Feature values encode the cell so we can assert no leakage:
            # cellA: f1=100+c, f2=200+c
            # cellB: f1=c,     f2=c*10
            if cell == "cellA":
                f1, f2 = 100.0 + c, 200.0 + c
            else:
                f1, f2 = float(c), float(c * 10)
            records.append({
                "dataset": ds_val,
                "cell_id": cell,
                "cycle_index": c,
                "soh": 1.0 - 0.01 * c,
                "f1": f1,
                "f2": f2,
            })
    df = pd.DataFrame(records)
    return df, ["f1", "f2"]


def _single_cell_frame(n_cycles: int = 6) -> tuple[pd.DataFrame, list[str]]:
    """Single-cell frame for basic shape / padding tests."""
    cycles = np.arange(n_cycles)
    df = pd.DataFrame({
        "dataset": ["ds1"] * n_cycles,
        "cell_id": ["c1"] * n_cycles,
        "cycle_index": cycles,
        "soh": 1.0 - 0.01 * cycles,
        "f1": cycles.astype(float),
        "f2": (cycles * 2).astype(float),
    })
    return df, ["f1", "f2"]


# ---------------------------------------------------------------------------
# build_windows — shape
# ---------------------------------------------------------------------------

class TestBuildWindowsShape:
    def test_output_shape(self):
        df, feature_cols = _single_cell_frame(n_cycles=6)
        W, F = 3, 2
        wins = build_windows(df, feature_cols, W)
        assert wins.shape == (len(df), W, F)

    def test_shape_with_two_cells(self):
        df, feature_cols = _two_cell_frame(n_cycles=5)
        W = 4
        wins = build_windows(df, feature_cols, W)
        assert wins.shape == (len(df), W, len(feature_cols))

    def test_dtype_is_float32(self):
        df, feature_cols = _single_cell_frame()
        wins = build_windows(df, feature_cols, W=3)
        assert wins.dtype == np.float32


# ---------------------------------------------------------------------------
# build_windows — no cross-cell leakage
# ---------------------------------------------------------------------------

class TestBuildWindowsNoCellLeakage:
    def test_first_row_of_cellB_has_only_cellB_values(self):
        """The window for cellB at cycle 0 must be left-padded with cellB's
        first feature vector, NOT with any cellA values."""
        n_cycles = 5
        df, feature_cols = _two_cell_frame(n_cycles=n_cycles)
        W = 4

        wins = build_windows(df, feature_cols, W)

        # Locate the row in df that is cellB / cycle 0
        mask = (df["cell_id"] == "cellB") & (df["cycle_index"] == 0)
        row_idx = df.index[mask][0]

        window = wins[row_idx]  # shape (W, F)

        # cellB cycle 0: f1=0, f2=0  — all W rows must equal this (left-pad)
        # cellA values have f1 >= 100; any value >= 100 means leakage.
        assert np.all(window[:, 0] < 100), (
            f"cellA f1 values (>=100) leaked into cellB window:\n{window}"
        )

    def test_cellB_cycle0_window_is_left_padded_with_cellB_first_cycle(self):
        """For cellB cycle 0, all W rows in the window must equal the first
        cycle of cellB (f1=0, f2=0)."""
        df, feature_cols = _two_cell_frame(n_cycles=5)
        W = 3

        wins = build_windows(df, feature_cols, W)

        mask = (df["cell_id"] == "cellB") & (df["cycle_index"] == 0)
        row_idx = df.index[mask][0]
        window = wins[row_idx]  # (W, F)

        # cellB first cycle: f1=0.0, f2=0.0
        expected = np.zeros((W, 2), dtype=np.float32)
        np.testing.assert_array_almost_equal(
            window, expected,
            err_msg="Left-padding for cellB cycle 0 should repeat its first cycle (f1=0, f2=0).",
        )

    def test_cellA_cycle0_window_is_left_padded_with_cellA_first_cycle(self):
        """For cellA cycle 0, all W rows must equal cellA's first cycle."""
        df, feature_cols = _two_cell_frame(n_cycles=5)
        W = 3

        wins = build_windows(df, feature_cols, W)

        mask = (df["cell_id"] == "cellA") & (df["cycle_index"] == 0)
        row_idx = df.index[mask][0]
        window = wins[row_idx]  # (W, F)

        # cellA first cycle: f1=100.0, f2=200.0
        expected_row = np.array([100.0, 200.0], dtype=np.float32)
        for w_i in range(W):
            np.testing.assert_array_almost_equal(window[w_i], expected_row)


# ---------------------------------------------------------------------------
# build_windows — left-padding semantics
# ---------------------------------------------------------------------------

class TestBuildWindowsPadding:
    def test_left_padding_repeats_first_cycle(self):
        """At cycle t=0 with W=3, all 3 rows of the window must equal cycle 0."""
        df, feature_cols = _single_cell_frame(n_cycles=6)
        W = 3
        wins = build_windows(df, feature_cols, W)

        # Row 0 corresponds to cycle 0 (the single cell, first row)
        mask = (df["cell_id"] == "c1") & (df["cycle_index"] == 0)
        row_idx = df.index[mask][0]
        window = wins[row_idx]  # (3, 2)

        # cycle 0 features: f1=0, f2=0
        expected = np.zeros((W, 2), dtype=np.float32)
        np.testing.assert_array_almost_equal(window, expected)

    def test_window_at_later_cycle_contains_history(self):
        """At cycle t=2 with W=3, the window should be cycles [0, 1, 2]."""
        df, feature_cols = _single_cell_frame(n_cycles=6)
        W = 3
        wins = build_windows(df, feature_cols, W)

        mask = (df["cell_id"] == "c1") & (df["cycle_index"] == 2)
        row_idx = df.index[mask][0]
        window = wins[row_idx]  # (3, 2)

        # cycle 0: f1=0,f2=0; cycle 1: f1=1,f2=2; cycle 2: f1=2,f2=4
        expected = np.array([[0., 0.], [1., 2.], [2., 4.]], dtype=np.float32)
        np.testing.assert_array_almost_equal(window, expected)

    def test_window_at_first_cycle_equal_to_first_cycle_feature(self):
        """Single-cycle frame: the entire window should be that one feature row."""
        df = pd.DataFrame({
            "dataset": ["ds1"],
            "cell_id": ["c1"],
            "cycle_index": [0],
            "soh": [1.0],
            "f1": [3.14],
            "f2": [2.72],
        })
        wins = build_windows(df, ["f1", "f2"], W=4)
        assert wins.shape == (1, 4, 2)
        # All W rows should be the single cycle
        expected = np.array([[3.14, 2.72]] * 4, dtype=np.float32)
        np.testing.assert_array_almost_equal(wins[0], expected)


# ---------------------------------------------------------------------------
# subsample_idx
# ---------------------------------------------------------------------------

class TestSubsampleIdx:
    def test_returns_all_when_n_le_n_max(self):
        idx = subsample_idx(50, 100, seed=0)
        np.testing.assert_array_equal(idx, np.arange(50))

    def test_caps_to_n_max(self):
        idx = subsample_idx(1000, 100, seed=0)
        assert len(idx) == 100

    def test_unique_indices(self):
        idx = subsample_idx(1000, 100, seed=0)
        assert len(set(idx.tolist())) == 100

    def test_indices_in_range(self):
        idx = subsample_idx(1000, 100, seed=42)
        assert np.all(idx >= 0)
        assert np.all(idx < 1000)

    def test_exact_boundary(self):
        """n == n_max should return all indices."""
        idx = subsample_idx(100, 100, seed=7)
        np.testing.assert_array_equal(idx, np.arange(100))

    def test_reproducible_with_same_seed(self):
        idx1 = subsample_idx(1000, 200, seed=42)
        idx2 = subsample_idx(1000, 200, seed=42)
        np.testing.assert_array_equal(idx1, idx2)


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

class TestSetSeed:
    def test_set_seed_does_not_raise(self):
        set_seed(0)

    def test_numpy_reproducible_after_set_seed(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_clamp_within_range_unchanged(self):
        arr = np.array([0.5, 1.0, 0.8])
        result = clamp(arr)
        np.testing.assert_array_almost_equal(result, arr)

    def test_clamp_below_zero(self):
        arr = np.array([-0.5, 0.0, 0.3])
        result = clamp(arr)
        assert result[0] == pytest.approx(SOH_LOW)

    def test_clamp_above_high(self):
        arr = np.array([1.5, 2.0, 0.5])
        result = clamp(arr)
        assert result[0] == pytest.approx(SOH_HIGH)
        assert result[1] == pytest.approx(SOH_HIGH)

    def test_clamp_extreme_ood(self):
        arr = np.array([-1e6, 1e6])
        result = clamp(arr)
        assert result[0] == pytest.approx(SOH_LOW)
        assert result[1] == pytest.approx(SOH_HIGH)


# ---------------------------------------------------------------------------
# train_mlp_like — minimal smoke test
# ---------------------------------------------------------------------------

class TestTrainMlpLike:
    def test_returns_model_and_loss_decreases(self):
        """A tiny MLP should fit a trivial task via train_mlp_like."""
        import torch
        import torch.nn as nn

        set_seed(0)
        N, F = 200, 3
        Xt = torch.randn(N, F)
        yt = (Xt[:, 0] * 0.5 + 0.6).clamp(0.0, 1.3)

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(nn.Linear(F, 16), nn.ReLU(), nn.Linear(16, 1))
            def forward(self, x):
                return self.fc(x)

        model = TinyNet()
        trained = train_mlp_like(model, Xt, yt, epochs=40, lr=1e-2, batch=64)
        assert trained is model  # returns the same object

        model.eval()
        with torch.no_grad():
            preds = model(Xt).squeeze(-1).numpy()
        mae = float(np.mean(np.abs(preds - yt.numpy())))
        assert mae < 0.3, f"train_mlp_like did not reduce loss: MAE={mae:.4f}"
