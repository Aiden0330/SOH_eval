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

from sohkernel.feature_safety import audit_feature_frame
from sohbench.paths import data_root

DATA_DIR = data_root() / "processed"

# Reviewed 2026-08-20: soh labels on these datasets are computed from the
# listed capacity columns (calibration capacity / rated capacity), confirmed
# by per-dataset affine fits (R² >= 0.998).  nasa / calce_cs2 / snl_lfp use a
# separate calibration measurement and carry no reconstruction column.
#
# Reviewed 2026-08-21 (Ah-window coverage rule): q_win_ah was flagged where
# the "partial" window integrated >= 90% of full capacity (hust 97.7%,
# snl_lfp 94.4%, uconn 95.2% median coverage) — at that DOD the window
# integral is the calibration capacity under another name.
#
# Reviewed 2026-08-25 (window rebuild): q_win_ah/t_win_s on hust, snl_lfp and
# uconn were re-integrated from raw curves with genuinely partial LFP-plateau
# windows (scripts/rebuild_partial_window_features.py — hust [3.30,3.52]V on
# the 5C CC charge, snl_lfp [3.30,3.38]V at 0.5C, uconn [3.30,3.37]V on the
# 1C RPT ref charge).  Median coverage now 48.7% / 50.7% / 52.2%, so the
# q_win_ah findings are gone by construction.  oxford (71.7%) remains a
# warning.
#
# Reviewed 2026-08-27 (calce correction): calce's "2.8% coverage" was a
# denominator artifact — its charge_capacity_max is the file-cumulative
# Arbin Charge_Capacity(Ah), not a per-cycle quantity, so the coverage
# audit was fooled ("false clean"); the old [3.6, 4.1] window actually
# integrated ~75% of the cycle charge.  q_win rebuilt on [3.85, 4.05] V
# (true coverage ~43% of rated).  The audit's reference-robustness gap is
# a documented TODO in the label-safety doc §6.3.
#
# Reviewed 2026-09-03 (D5, report finding G6): the window-coverage audit now divides
# by the datasheet rated capacity from sohkernel.metadata (cell-prefix overrides
# included) and falls back to the table's capacity column only for datasets without
# metadata, so a distorted charge_capacity_max can no longer make it "falsely clean".
# Median q_win / rated capacity on this date (raw source labels such as CALCE/SNL
# resolve through metadata.canonical_dataset_name): calce_cs2 0.371; hust 0.486; nasa 0.276; oxford 0.594; snl_lfp 0.449; uconn 0.536. Findings unchanged.
#
# Reviewed 2026-09-04 (D1, chemistry breadth): snl_nmc (21 LG Chem 18650HG2 cells) and snl_nca
# (18 of 22 Panasonic NCR18650B cells; the four 20-80 % SOC cells carry no calibration capacity
# and are not ingested) were built from the BatteryLife mirror by
# scripts/ingest_batterylife_snl.py with the same charge-segment extractor that reproduces
# snl_lfp (gated columns >= 99.99 % row match).  Windows calibrated on the 0.5C charge segment:
# snl_nmc [3.65, 4.10] V, median q_win / 3.0 Ah = 0.433 (max 0.516, affine R² 0.88);
# snl_nca [3.70, 4.10] V, median q_win / 3.2 Ah = 0.440 (max 0.561, affine R² 0.87).
# Findings: snl_nmc charge_capacity_max reconstructs soh affinely (R² 0.998 — the charge
# throughput of a 0-100 % cycle equals the discharge capacity that anchors the label);
# on snl_nca the same column stays below the 0.995 threshold, so no finding is pinned there.
EXPECTED_LABEL_SOURCE_COLUMNS = {
    "hust": ["charge_capacity_max"],
    "oxford": ["charge_capacity_max"],
    "uconn": ["charge_capacity_max"],
    "rwth_isea_public": ["capacity_ah"],
    "nasa": [],
    "calce_cs2": [],
    "snl_lfp": [],
    "snl_nmc": ["charge_capacity_max"],
    "snl_nca": [],
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
