"""Trusted-inference label gate (Task 8): a frozen model's manifest columns are audited before predict."""
import numpy as np
import pandas as pd
import pytest

from sohkernel.feature_safety import FeatureSafetyError
from sohkernel.inference_gate import assert_manifest_label_safe


def _table(n=100, seed=0):
    rng = np.random.default_rng(seed)
    soh = np.linspace(1.0, 0.8, n)
    return pd.DataFrame({
        "dataset": ["d"] * n, "cell_id": ["c"] * n, "cycle_index": np.arange(n), "soh": soh,
        "voltage_mean": 3.6 + 0.2 * soh + rng.normal(0, 0.02, n),
        "charge_capacity_max": 1.1 * soh,
        "cap_copy": 0.5 * soh + 0.1,
    })


def test_structural_ban_blocks_label_source_columns_before_any_value_check():
    with pytest.raises(FeatureSafetyError, match="inference blocked .*structurally banned.*charge_capacity_max"):
        assert_manifest_label_safe(["voltage_mean", "charge_capacity_max"], _table(), model_name="m")


def test_value_probe_blocks_an_affine_label_copy():
    with pytest.raises(FeatureSafetyError, match="inference blocked .*cap_copy.*reconstructs soh"):
        assert_manifest_label_safe(["voltage_mean", "cap_copy"], _table(), model_name="m")


def test_clean_manifest_passes_and_returns_the_audit():
    audit = assert_manifest_label_safe(["voltage_mean"], _table(), model_name="m")
    assert audit["status"] == "passed" and audit["checked_columns"] == ["voltage_mean"]
