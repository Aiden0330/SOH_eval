import numpy as np
import pandas as pd

from sohbench.feature_safety import audit_feature_frame


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
