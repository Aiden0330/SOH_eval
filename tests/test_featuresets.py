"""Tests for sohbench.featuresets — TDD: written before implementation."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest

from sohbench.contracts import ID_COLUMNS, feature_columns
from sohbench import registry as R
from sohbench.featuresets import FeatureSet, project, register_featureset, get_featureset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table(**extra_cols):
    """Return a small DataFrame with all standard ID cols + any extras."""
    n = 5
    base = {
        "dataset": ["ds"] * n,
        "cell_id": ["c1"] * n,
        "cycle_index": range(n),
        "soh": np.linspace(1.0, 0.8, n),
    }
    base.update({k: np.ones(n) * i for i, k in enumerate(extra_cols)})
    for k, v in extra_cols.items():
        if v is not None:
            base[k] = v if hasattr(v, "__len__") else np.ones(n) * v
    return pd.DataFrame(base)


def _full_feature_table():
    """Table with all the feature cols referenced in the 5 named feature sets."""
    n = 5
    return pd.DataFrame({
        "dataset": ["ds"] * n,
        "cell_id": ["c1"] * n,
        "cycle_index": range(n),
        "soh": np.linspace(1.0, 0.8, n),
        # voltage set
        "voltage_mean": np.random.default_rng(0).random(n),
        "voltage_std": np.random.default_rng(1).random(n),
        "voltage_min": np.random.default_rng(2).random(n),
        "voltage_max": np.random.default_rng(3).random(n),
        # ic_dva set
        "ic_peak_ah_per_v": np.random.default_rng(4).random(n),
        "ic_peak_voltage_v": np.random.default_rng(5).random(n),
        # partial_window set
        "q_win_ah": np.random.default_rng(6).random(n),
        "t_win_s": np.random.default_rng(7).random(n),
        # capacity_time set
        "charge_capacity_max": np.random.default_rng(8).random(n),
        "time_s_max": np.random.default_rng(9).random(n),
        "cycle_index_feature": np.arange(n, dtype=float),
        "cycle_fraction": np.linspace(0, 1, n),
        # extra feature col not in any named set
        "extra_col": np.ones(n),
    })


# ---------------------------------------------------------------------------
# FeatureSet class
# ---------------------------------------------------------------------------

class TestFeatureSetClass:
    def test_name_attribute(self):
        fs = FeatureSet(name="test", candidates=["voltage_mean", "voltage_std"])
        assert fs.name == "test"

    def test_columns_returns_intersection_with_present(self):
        table = _full_feature_table()
        fs = FeatureSet(name="voltage", candidates=["voltage_mean", "voltage_std", "voltage_min", "voltage_max"])
        cols = fs.columns(table)
        assert set(cols) == {"voltage_mean", "voltage_std", "voltage_min", "voltage_max"}

    def test_columns_excludes_absent_candidates(self):
        table = _full_feature_table().drop(columns=["voltage_std"])
        fs = FeatureSet(name="voltage", candidates=["voltage_mean", "voltage_std", "voltage_min", "voltage_max"])
        cols = fs.columns(table)
        assert "voltage_std" not in cols
        assert "voltage_mean" in cols

    def test_columns_only_returns_feature_columns_not_id(self):
        table = _full_feature_table()
        fs = FeatureSet(name="all", candidates=list(table.columns))
        cols = fs.columns(table)
        for idc in ID_COLUMNS:
            if idc in table.columns:
                assert idc not in cols, f"ID column {idc!r} must not appear in FeatureSet columns"

    def test_empty_intersection_returns_empty_list(self):
        table = _make_table()  # only ID cols
        fs = FeatureSet(name="voltage", candidates=["voltage_mean", "voltage_std"])
        cols = fs.columns(table)
        assert cols == []


# ---------------------------------------------------------------------------
# Registry functions
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_register_and_get_featureset(self):
        fs = FeatureSet(name="my_set", candidates=["voltage_mean"])
        register_featureset(fs)
        retrieved = get_featureset("my_set")
        assert retrieved is fs

    def test_get_featureset_unknown_raises(self):
        with pytest.raises(KeyError):
            get_featureset("__nonexistent__")

    def test_module_level_register_featureset_is_same_as_registry(self):
        # register_featureset imported from featuresets must delegate to registry
        fs = FeatureSet(name="my_set2", candidates=["voltage_mean"])
        register_featureset(fs)
        assert R.get_featureset("my_set2") is fs


# ---------------------------------------------------------------------------
# Named feature sets registered at import time
# ---------------------------------------------------------------------------

class TestNamedFeatureSets:
    def test_full_set_registered(self):
        fs = get_featureset("full")
        assert fs.name == "full"

    def test_voltage_set_registered(self):
        fs = get_featureset("voltage")
        assert fs.name == "voltage"

    def test_ic_dva_set_registered(self):
        fs = get_featureset("ic_dva")
        assert fs.name == "ic_dva"

    def test_partial_window_set_registered(self):
        fs = get_featureset("partial_window")
        assert fs.name == "partial_window"

    def test_capacity_time_set_registered(self):
        fs = get_featureset("capacity_time")
        assert fs.name == "capacity_time"

    def test_full_returns_all_feature_columns_except_label_sources(self):
        table = _full_feature_table()
        fs = get_featureset("full")
        cols = fs.columns(table)
        # label-source columns (the quantity SOH is computed from) are
        # structurally banned as model inputs, even for "full"
        expected = set(feature_columns(table)) - {"charge_capacity_max"}
        assert set(cols) == expected

    def test_voltage_returns_only_voltage_cols_present(self):
        table = _full_feature_table()
        fs = get_featureset("voltage")
        cols = fs.columns(table)
        assert set(cols) == {"voltage_mean", "voltage_std", "voltage_min", "voltage_max"}

    def test_ic_dva_columns(self):
        table = _full_feature_table()
        fs = get_featureset("ic_dva")
        cols = fs.columns(table)
        assert set(cols) == {"ic_peak_ah_per_v", "ic_peak_voltage_v"}

    def test_partial_window_columns(self):
        table = _full_feature_table()
        fs = get_featureset("partial_window")
        cols = fs.columns(table)
        assert set(cols) == {"q_win_ah", "t_win_s"}

    def test_capacity_time_columns(self):
        table = _full_feature_table()
        fs = get_featureset("capacity_time")
        cols = fs.columns(table)
        # charge_capacity_max is the label source and must never be selected
        assert set(cols) == {"time_s_max", "cycle_index_feature", "cycle_fraction"}

    def test_no_featureset_can_select_label_source_columns(self):
        table = _full_feature_table()
        # even a featureset that explicitly lists banned columns cannot
        # select them — the ban is enforced in the selector itself
        fs = FeatureSet(name="sneaky", candidates=[
            "charge_capacity_max", "capacity_ah", "rated_capacity", "soh",
            "voltage_mean",
        ])
        assert fs.columns(table) == ["voltage_mean"]


# ---------------------------------------------------------------------------
# project()
# ---------------------------------------------------------------------------

class TestProject:
    def test_project_keeps_id_cols_and_selected_features(self):
        table = _full_feature_table()
        result = project(table, "voltage")
        expected_cols = [c for c in ID_COLUMNS if c in table.columns] + ["voltage_mean", "voltage_std", "voltage_min", "voltage_max"]
        assert set(result.columns) == set(expected_cols)

    def test_project_drops_non_selected_features(self):
        table = _full_feature_table()
        result = project(table, "voltage")
        # ic_dva, partial_window, capacity_time, extra cols should be gone
        assert "ic_peak_ah_per_v" not in result.columns
        assert "extra_col" not in result.columns

    def test_project_full_keeps_all_feature_columns_except_label_sources(self):
        table = _full_feature_table()
        result = project(table, "full")
        assert set(result.columns) == set(table.columns) - {"charge_capacity_max"}

    def test_project_preserves_row_count(self):
        table = _full_feature_table()
        result = project(table, "voltage")
        assert len(result) == len(table)

    def test_project_absent_set_warns_and_returns_id_cols(self):
        table = _make_table()  # no voltage cols present
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = project(table, "voltage")
        assert len(w) == 1
        assert "voltage" in str(w[0].message).lower() or "zero" in str(w[0].message).lower() or "no" in str(w[0].message).lower()
        # Should still return id columns
        for idc in ID_COLUMNS:
            if idc in table.columns:
                assert idc in result.columns

    def test_project_partial_table_id_cols_subset(self):
        """project works even if table is missing some ID columns (e.g. no 'cycle_index')."""
        table = _full_feature_table().drop(columns=["cycle_index"])
        result = project(table, "voltage")
        assert "cycle_index" not in result.columns
        assert "dataset" in result.columns


def test_partial_window_norm_produces_dimensionless_transferable_his():
    import pandas as pd

    from sohbench.featuresets import _partial_window_norm_transform
    from sohbench.registry import get_featureset

    table = pd.DataFrame({
        "dataset": ["a", "a", "b"],
        "cell_id": ["c1", "c1", "c9"],
        "cycle_index": [0, 1, 0],
        "soh": [1.0, 0.99, 1.0],
        "q_win_ah": [0.50, 0.49, 25.0],
        "t_win_s": [900.0, 890.0, 1800.0],
        "time_s_max": [3600.0, 3600.0, 7200.0],
    })
    meta = pd.DataFrame({
        "dataset": ["a", "b"],
        "cell_id": ["c1", "c9"],
        "c_rated_ah": [1.1, 50.0],
    })
    out = _partial_window_norm_transform(table, meta)
    # Different absolute scales (1.1Ah cell vs 50Ah cell) land on the same
    # dimensionless range — the transferable-HI point.
    assert out["q_win_frac"].round(4).tolist() == [0.4545, 0.4455, 0.5]
    assert out["t_win_frac"].round(4).tolist() == [0.25, 0.2472, 0.25]

    featureset = get_featureset("partial_window_norm")
    assert featureset.columns(out) == ["q_win_frac", "t_win_frac"]

    # Missing rated capacity degrades to NaN, never to a wrong scale.
    missing = _partial_window_norm_transform(table, meta.iloc[:1])
    assert missing["q_win_frac"].isna().tolist() == [False, False, True]


def test_full_featureset_excludes_partial_window_derived_columns():
    import pandas as pd

    from sohbench.featuresets import _partial_window_norm_transform
    from sohbench.registry import get_featureset

    table = pd.DataFrame({
        "dataset": ["a"] * 4,
        "cell_id": ["c1"] * 4,
        "cycle_index": [0, 1, 2, 3],
        "soh": [1.0, 0.99, 0.98, 0.97],
        "voltage_mean": [3.5, 3.49, 3.48, 3.47],
        "q_win_ah": [0.5, 0.49, 0.48, 0.47],
        "t_win_s": [900.0] * 4,
        "time_s_max": [3600.0] * 4,
    })
    meta = pd.DataFrame({"dataset": ["a"], "cell_id": ["c1"], "c_rated_ah": [1.1]})
    transformed = _partial_window_norm_transform(table, meta)
    full = get_featureset("full")
    assert full.columns(transformed) == full.columns(table)
