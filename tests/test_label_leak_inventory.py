"""Standing label-leak inventory over the real processed datasets.

This test is the going-forward guard for leak-class issues: it runs the
label-safety audit over every supervised dataset on disk and pins the
expected findings.  It fails in BOTH directions —

* a NEW column that reconstructs the label appears (new leak, or a new
  dataset carrying its label source unannotated), or
* a KNOWN label-source column stops being flagged (detection regression).

Update ``EXPECTED_LABEL_SOURCE_COLUMNS`` only with a reviewed justification;
that edit is the audit trail.
"""
from pathlib import Path

import pandas as pd
import pytest

from sohbench.feature_safety import audit_feature_frame

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Reviewed 2026-08-20: soh labels on these datasets are computed from the
# listed capacity columns (calibration capacity / rated capacity), confirmed
# by per-dataset affine fits (R² >= 0.998).  nasa / calce_cs2 / snl_lfp use a
# separate calibration measurement and carry no reconstruction column.
EXPECTED_LABEL_SOURCE_COLUMNS = {
    "hust": ["charge_capacity_max"],
    "oxford": ["charge_capacity_max"],
    "uconn": ["charge_capacity_max"],
    "rwth_isea_public": ["capacity_ah"],
    "nasa": [],
    "calce_cs2": [],
    "snl_lfp": [],
}

ID_COLUMNS = ("dataset", "cell_id", "cycle_index", "soh")


def _dataset_paths():
    if not DATA_DIR.is_dir():
        return []
    return sorted(DATA_DIR.glob("*_features.pkl"))


@pytest.mark.skipif(not _dataset_paths(), reason="processed datasets not present")
def test_label_leak_inventory_matches_expectations():
    unexpected = {}
    missing_detection = {}
    for path in _dataset_paths():
        name = path.name.replace("_features.pkl", "")
        table = pd.read_pickle(path)
        if "soh" not in table.columns:
            continue
        feature_columns = [c for c in table.columns if c not in ID_COLUMNS]
        audit = audit_feature_frame(table, feature_columns)
        found = sorted({item["column"] for item in audit["violations"]})
        expected = sorted(EXPECTED_LABEL_SOURCE_COLUMNS.get(name, []))
        if name in EXPECTED_LABEL_SOURCE_COLUMNS:
            extra = [c for c in found if c not in expected]
            absent = [c for c in expected if c not in found]
            if extra:
                unexpected[name] = extra
            if absent:
                missing_detection[name] = absent
        elif found:
            # A dataset outside the inventory must not silently carry a
            # label-source column — add it to the inventory with review.
            unexpected[name] = found
    assert not unexpected, (
        "new label-reconstruction columns detected (review and either fix the "
        f"dataset or extend the inventory with justification): {unexpected}"
    )
    assert not missing_detection, (
        f"known label-source columns no longer detected (audit regression): {missing_detection}"
    )
