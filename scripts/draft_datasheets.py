"""Generate the Spec 06 datasheets for the datasets under the data root (D6).

Every field without a primary source is written as ``VERIFY`` for the human to resolve (Spec 06 §2).
Sources: sohkernel.metadata (chemistry label, rated capacity), data/manifest.json (digests), the standing
leak inventory (label-source columns), the window calibration notes (bounds), the shipped raw readmes, and
the 2026-09-03 web review of the primary dataset records (D4, owner-delegated) recorded per dataset below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sohkernel.metadata import DATASET_CHEMISTRY, DATASET_C_RATED_AH  # noqa: E402

V = "VERIFY"
REVIEW_DATE = "2026-09-03"
REVIEWER = "owner-delegated web review 2026-09-03 (Claude Fable 5.1); sources in chemistry.evidence"
UNSIGNED = {"by": V, "on": V}
SIGNED = {"by": REVIEWER, "on": REVIEW_DATE}

FACTS = {
    "nasa": {
        "citation": "Saha, B. and Goebel, K. (2007). Battery Data Set. NASA Ames Prognostics Data Repository (PCoE), NASA Ames Research Center.",
        "url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
        "license": "public (NASA PCoE repository)", "retrieved": f"{REVIEW_DATE} (NASA PCoE / DASHlink / data.nasa.gov descriptions via web review)",
        "cathode": V, "anode": "graphite (assumed; VERIFY)", "verified_by": V, "verified_on": V,
        "format": "18650 (commercial, manufacturer not named)", "rated_source": "NASA description (rated 2 Ah; EOL at 30 % fade, 2 -> 1.4 Ah)",
        "chemistry_evidence": (
            "Web review 2026-09-03: NASA's own description (PCoE repository, DASHlink resource 133, data.nasa.gov) says "
            "'commercially available Li-ion 18650 sized rechargeable batteries', rated 2 Ah, three operational profiles "
            "(charge, discharge, EIS) at room temperature; neither chemistry nor manufacturer is named, and the shipped "
            "READMEs (data/raw/nasa_full/README*.txt) only describe the .mat structure. Secondary literature is split: most "
            "SOH papers and the Kaggle mirror call the cells LiCoO2; a minority calls them 'Gen 2' NCA/graphite cells, but "
            "the DOE Gen 2 18650 cells were ~1 Ah while these are rated 2 Ah. The working label LCO is kept for fold grouping "
            "only; no chemistry claim may rest on nasa until a primary source is found (VERIFY)."
        ),
        "segments": [{"id": "cc_cv_charge", "rate": "1.5 A CC to 4.2 V, CV to 20 mA (NASA description)"},
                     {"id": "discharge", "rate": "2 A CC to 2.7 / 2.5 / 2.2 / 2.5 V for B0005 / B0006 / B0007 / B0018 (NASA description)"}],
        "temperature_c": [24], "reference_test": "capacity of each full discharge (NASA description)",
        "measured": ["discharge capacity (Ah)"], "window": None,
        "known_issues": ["chemistry unverified (D4, 2026-09-03) — lcho folds holding out nasa warn",
                         "cell B0018 is the phase 2.2 golden case; 18 non-positive SOH rows dropped by QC"],
        "signed": UNSIGNED, "status": "draft — chemistry unverified after the 2026-09-03 web review; unsigned",
    },
    "calce_cs2": {
        "citation": ("CALCE Battery Research Group, University of Maryland: CS2 prismatic cells (Battery Data page); "
                     "He, W., Williard, N., Osterman, M., Pecht, M. (2011). Prognostics of lithium-ion batteries based on "
                     "Dempster-Shafer theory and the Bayesian Monte Carlo method. J. Power Sources 196(23), 10314-10321."),
        "url": "https://calce.umd.edu/battery-data", "license": "academic use; the page requests a citation", "retrieved": f"{REVIEW_DATE} (page fetched)",
        "cathode": "LCO", "anode": "graphite (assumed; the CALCE page states only the cathode)", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": "prismatic, 5.4 x 33.6 x 50.6 mm (CALCE page)", "rated_source": "CALCE page (CS2 'Capacity Rating: 1100 mAh'; CX2 1350 mAh)",
        "chemistry_evidence": ("CALCE Battery Data page, CS2: 'Cell Chemistry: LiCoO2 cathode, EDS results also showed trace elements of "
                               "Manganese'; 'Capacity Rating: 1100 mAh'; charging 'constant current rate of 0.5C until the voltage reached "
                               "4.2V and then 4.2V was sustained until the charging current dropped to below 0.05A'; 'discharge cut off "
                               "voltage for these batteries was 2.7V'."),
        "segments": [{"id": "cc_charge", "rate": "0.5C CC to 4.2 V, CV to 0.05 A (CALCE page)"},
                     {"id": "discharge", "rate": "0.5C-1C per cell to 2.7 V (CALCE page: cut-off 2.7 V)"}],
        "temperature_c": [25], "reference_test": "capacity of each discharge (Arbin file)",
        "measured": ["discharge capacity (Ah)"],
        "window": {"segment": "cc_charge", "v_lo": 3.85, "v_hi": 4.05, "coverage_median": 0.371, "calibrated_on": "2026-08-27"},
        "known_issues": ["charge_capacity_max in the processed table is the file-cumulative Arbin quantity, not per cycle (report finding G6)"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); remaining VERIFY fields are non-chemistry",
    },
    "hust": {
        "citation": ("Ma, G., Xu, S., Jiang, B., Cheng, C., Yang, X., Shen, Y., Yang, T., Huang, Y., Ding, H., Yuan, Y. (2022). Real-time "
                     "personalized health status prediction of lithium-ion batteries using deep transfer learning. Energy & Environmental "
                     "Science 15(10), 4083-4094. Dataset: Mendeley Data v2 (2022-05-24), DOI 10.17632/nsc7hnsg4s.2."),
        "url": "https://data.mendeley.com/datasets/nsc7hnsg4s/2", "license": "CC BY 4.0 (Mendeley record)", "retrieved": f"{REVIEW_DATE} (record fetched)",
        "cathode": "LFP", "anode": "graphite", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": "18650, 1.1 Ah / 3.3 V nominal (record); cell model reported as A123 APR18650M1A in the paper (model VERIFY)",
        "rated_source": "Mendeley record (1.1 Ah nominal capacity)",
        "chemistry_evidence": ("Mendeley Data record nsc7hnsg4s v2: '77 LFP/graphite cells (1.1 Ah nominal capacity and 3.3 V nominal voltage)' "
                               "with an 'identical charge protocol but different multi-stage discharge protocols at a constant temperature of 30°C'."),
        "segments": [{"id": "cc_charge", "rate": "identical fast-charge protocol for all cells (record); stages VERIFY"},
                     {"id": "discharge", "rate": "77 distinct multi-stage discharge protocols (record); rates VERIFY"}],
        "temperature_c": [30], "reference_test": "capacity of each discharge", "measured": ["charge_capacity_max"],
        "window": {"segment": "cc_charge", "v_lo": 3.30, "v_hi": 3.52, "coverage_median": 0.486, "calibrated_on": "2026-08-25"}, "known_issues": [],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); remaining VERIFY fields are non-chemistry",
    },
    "oxford": {
        "citation": ("Howey, D. and Birkl, C. (2017). Oxford Battery Degradation Dataset 1. University of Oxford, ORA; Birkl, C. R. (2017). "
                     "Diagnosis and Prognosis of Degradation in Lithium-Ion Batteries. DPhil thesis, University of Oxford."),
        "url": "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac", "license": "ODC-ODbL (database), DbCL (contents) — readme",
        "retrieved": f"{REVIEW_DATE} (ORA record and Readme.txt fetched)",
        "cathode": "LCO/NCO", "anode": "graphite", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": "pouch, 8 x Kokam SLPB533459H4, 740 mAh (readme)", "rated_source": "ORA Readme.txt ('Test subjects: 8 x Kokam CO LTD, SLPB533459H4, 740mAh')",
        "chemistry_evidence": ("ORA Readme.txt: 8 x Kokam SLPB533459H4 740 mAh pouch cells, 40 degC thermal chamber, CC-CV charge, urban Artemis "
                               "drive-cycle discharge, characterisation every 100 cycles at 1C (740 mA) and pseudo-OCV (40 mA). Chemistry: dos Reis, "
                               "G. et al. (2021), Lithium-ion battery data and where to find it, Energy and AI 5, 100081 (peer-reviewed data review): "
                               "the Oxford cells have a 'lithium cobalt oxide (LCO)/lithium nickel cobalt oxide (NCO) positive electrode and graphite "
                               "negative electrode', consistent with Birkl 2017; the primary thesis/paper text was not fetchable in this review "
                               "(ORA downloads blocked by network policy)."),
        "segments": [{"id": "cc_charge", "rate": "1C (740 mA) characterisation every 100 drive cycles (readme)"},
                     {"id": "pseudo_ocv", "rate": "C/18.5 (40 mA) (readme)"}],
        "temperature_c": [40], "reference_test": "1C characterisation discharge every 100 Artemis drive cycles (readme)", "measured": ["charge_capacity_max"],
        "window": V, "known_issues": ["519 characterisation points only; coverage median 0.594 of rated (wide window warning)"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); remaining VERIFY fields are non-chemistry",
    },
    "snl_lfp": {
        "citation": ("Preger, Y., Barkholtz, H. M., Fresquez, A., Campbell, D. L., Juba, B. W., Roman-Kustas, J., Ferreira, S. R., Chalamala, B. "
                     "(2020). Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions. J. Electrochem. Soc. "
                     "167(12), 120532, DOI 10.1149/1945-7111/abae37. Data via BatteryArchive (snl_study) / BatteryLife mirror (data/raw/snl_batterylife)."),
        "url": "https://doi.org/10.1149/1945-7111/abae37", "license": "VERIFY (the BatteryArchive study page states no license)",
        "retrieved": f"{REVIEW_DATE} (JES record via web review; BatteryArchive snl_study page fetched)",
        "cathode": "LFP", "anode": "graphite", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": "18650 (A123 Systems APR18650M1A)", "rated_source": "JES 167 120532 (LFP: A123 APR18650M1A, 1.1 Ah)",
        "chemistry_evidence": ("JES 167(12) 120532: the LFP cells are A123 Systems APR18650M1A, 1.1 Ah (the study's NCA cells are Panasonic NCR18650B "
                               "3.2 Ah and its NMC cells LG Chem 18650HG2 3 Ah); all cells charged at 0.5C, discharged at 0.5C / 1C / 2C / 3C over "
                               "0-100, 20-80 or 40-60 % SOC at 15 / 25 / 35 degC. The BatteryArchive snl_study page confirms 'All cells were charged "
                               "at a rate of 0.5C' and the three SOC windows."),
        "segments": [{"id": "cc_charge", "rate": "0.5C CC (JES)"}, {"id": "discharge", "rate": "0.5C-3C (JES); cut-off voltage VERIFY"}],
        "temperature_c": [15, 25, 35], "reference_test": "capacity of each discharge", "measured": ["separate calibration measurement (no label-source column)"],
        "window": {"segment": "cc_charge", "v_lo": 3.30, "v_hi": 3.38, "coverage_median": 0.449, "calibrated_on": "2026-08-25"}, "known_issues": [],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); remaining VERIFY fields are non-chemistry",
    },
    "uconn": {
        "citation": ("Nowacki, B., Ramamurthy, J., Thelen, A., Tischer, C., Pint, C. L., Hu, C. (2025). UConn-ISU-ILCC LFP/Gr Battery Aging Dataset. "
                     "UConn Digital Commons, REIL datasets 1; paper: Rapid estimation of lithium-ion battery capacity and resistances from short "
                     "duration current pulses. J. Power Sources 628, 235813 (Feb. 2025)."),
        "url": "https://digitalcommons.lib.uconn.edu/reil_datasets/1/", "license": "CC BY 4.0 (record)", "retrieved": f"{REVIEW_DATE} (record fetched)",
        "cathode": "LFP", "anode": "graphite", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": "18650 (LithiumWerks APR18650M1B)", "rated_source": "LithiumWerks APR18650M1B specification (1.1 Ah nominal, 3.3 V); data/curves/uconn.json",
        "chemistry_evidence": ("UConn Digital Commons record: '64 LithiumWerks APR18650M1B cells' cycled under 11 unique conditions to 80 % remaining "
                               "capacity, with 100-second CC pulses at three states of charge; the dataset is titled LFP/Gr. LithiumWerks APR18650M1B "
                               "is a LiFePO4/graphite 18650 cell, 1.1 Ah, 3.3 V nominal (vendor specification)."),
        "segments": [{"id": "rpt_charge", "rate": "1C reference charge"}], "temperature_c": [V], "reference_test": "RPT reference discharge", "measured": ["charge_capacity_max"],
        "window": {"segment": "rpt_charge", "v_lo": 3.30, "v_hi": 3.37, "coverage_median": 0.536, "calibrated_on": "2026-08-25"},
        "known_issues": ["64 rows in 32 duplicated (cell, cycle) identities are dropped by QC since 2026-09-03"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); remaining VERIFY fields are non-chemistry",
    },
    "rwth_isea_public": {
        "citation": ("Li, W., Sengupta, N., Dechent, P., Howey, D., Annaswamy, A., Sauer, D. U. (2021). One-shot battery degradation trajectory "
                     "prediction with deep learning. J. Power Sources 506, 230024. Dataset: Time-series cyclic aging data on 48 commercial NMC/graphite "
                     "Sanyo/Panasonic UR18650E cylindrical cells, RWTH Publications record 818642, DOI 10.18154/RWTH-2021-04545."),
        "url": "https://publications.rwth-aachen.de/record/818642", "license": "VERIFY (carried in the record's Contents.pdf; the GitHub mirror shows no license)",
        "retrieved": f"{REVIEW_DATE} (RWTH record via web review; GitHub mirror readme via API)",
        "cathode": "NMC", "anode": "graphite", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": "18650 (Sanyo/Panasonic UR18650E)", "rated_source": "RWTH-2021-04545 record (nominal 1.85 Ah); table per-cell maximum 1.83-1.85 Ah",
        "chemistry_evidence": ("RWTH Publications record 818642: 'Time-series cyclic aging data on 48 commercial NMC/graphite Sanyo/Panasonic UR18650E "
                               "cylindrical cells', nominal capacity 1.85 Ah. scripts/validate_trusted_plugin_workflow.py builds this table from "
                               "'Pre-processing Codes/Degradation_Prediction_Dataset_ISEA.mat' in github.com/isea-rwth-aachen/battery-degradation-"
                               "trajectory-prediction (mirror of git.rwth-aachen.de/isea/..., '48 cells during ageing tests') keeping 8 cells; the "
                               "per-cell maximum capacities (1.83-1.85 Ah) match. The previous label 'LFP (Fortelion residential storage)' was wrong: "
                               "it described the separate rwth_hss field data."),
        "segments": [{"id": V, "rate": V}], "temperature_c": [V],
        "reference_test": "regular characterisation capacity tests (mirror readme: 'capacity test with various C-rates'); the C-rate behind capacity_ah VERIFY",
        "measured": ["capacity_ah"], "window": V,
        "known_issues": ["47 of the 48 cells ingested on 2026-09-04 (D1; until then 8, validate_trusted_plugin_workflow max_cells=8; the builder skips one cell with fewer than 8 capacity points): the 8-cell subset reproduces exactly; artifacts trained on the 8-cell table are not comparable with 47-cell results",
                         "raw time series (V/I/T) are not ingested — only per-cycle capacity histories from the ISEA .mat; no window or curve features",
                         "manifest curve_path curves/rwth_curves.json holds rwth_hss home-storage field curves from another source, not these cells",
                         "chemistry corrected from LFP to NMC and rated capacity set to 1.85 Ah on 2026-09-03 (D4)"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); remaining VERIFY fields are non-chemistry",
    },
}
SIGNED_D1 = {"by": "owner-delegated review 2026-09-04 (Claude Fable 5.1); sources in chemistry.evidence", "on": "2026-09-04"}
FACTS["snl_nmc"] = {
    "citation": ("Preger, Y. et al. (2020). Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions. "
                 "J. Electrochem. Soc. 167(12), 120532, DOI 10.1149/1945-7111/abae37 — NMC subset; data via BatteryArchive / BatteryLife mirror "
                 "(data/raw/snl_batterylife), ingested by scripts/ingest_batterylife_snl.py (D1, 2026-09-04)."),
    "url": "https://doi.org/10.1149/1945-7111/abae37", "license": "VERIFY (the BatteryArchive study page states no license)",
    "retrieved": "2026-09-04 (built from the local BatteryLife mirror)",
    "cathode": "NMC", "anode": "graphite", "verified_by": SIGNED_D1["by"], "verified_on": "2026-09-04",
    "format": "18650 (LG Chem 18650HG2)", "rated_source": "JES 167 120532 (NMC: LG Chem 18650HG2, 3 Ah); pickle nominal_capacity_in_Ah 3 (the mirror derates 15 degC cells to 2.7 Ah; not used)",
    "chemistry_evidence": ("BatteryLife pickle metadata: cathode_material 'NMC', anode_material 'graphite', form_factor cylindrical_18650, nominal 3 Ah; "
                           "JES 167(12) 120532 names the NMC cells LG Chem 18650HG2, 3 Ah. The extractor was verified against the existing snl_lfp "
                           "table before use (soh 100 % row match; charge-segment statistics >= 99.998 %)."),
    "segments": [{"id": "cc_charge", "rate": "0.5C CC to 4.2 V (JES); charge segment = current > 0.01 A"},
                 {"id": "discharge", "rate": "0.5C-3C to 2.0 V (JES; the rate is in the cell name)"}],
    "temperature_c": [15, 25, 35], "reference_test": "maximum discharge capacity of each cycle (BatteryLife discharge_capacity_in_Ah) over the 3.0 Ah nameplate",
    "measured": ["discharge capacity (Ah)"],
    "window": {"segment": "cc_charge", "v_lo": 3.65, "v_hi": 4.10, "coverage_median": 0.433, "calibrated_on": "2026-09-04"},
    "known_issues": ["charge_capacity_max reconstructs soh affinely (R² 0.998) on 0-100 % cycles: pinned in the leak inventory and banned as an input by name",
                     "21 cells, 14,283 rows; raw curves stay in the mirror (not copied into data/curves)"],
    "signed": SIGNED_D1, "status": "signed 2026-09-04 (chemistry verified from the pickle metadata and JES 2020, owner-delegated); remaining VERIFY fields are non-chemistry",
}
FACTS["snl_nca"] = {
    "citation": ("Preger, Y. et al. (2020). Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions. "
                 "J. Electrochem. Soc. 167(12), 120532, DOI 10.1149/1945-7111/abae37 — NCA subset; data via BatteryArchive / BatteryLife mirror "
                 "(data/raw/snl_batterylife), ingested by scripts/ingest_batterylife_snl.py (D1, 2026-09-04)."),
    "url": "https://doi.org/10.1149/1945-7111/abae37", "license": "VERIFY (the BatteryArchive study page states no license)",
    "retrieved": "2026-09-04 (built from the local BatteryLife mirror)",
    "cathode": "NCA", "anode": "graphite", "verified_by": SIGNED_D1["by"], "verified_on": "2026-09-04",
    "format": "18650 (Panasonic NCR18650B)", "rated_source": "JES 167 120532 (NCA: Panasonic NCR18650B, 3.2 Ah); pickle nominal_capacity_in_Ah 3.2 (the mirror derates 15 degC cells to 2.88 Ah; not used)",
    "chemistry_evidence": ("BatteryLife pickle metadata: cathode_material 'NCA', anode_material 'graphite', form_factor cylindrical_18650, nominal 3.2 Ah; "
                           "JES 167(12) 120532 names the NCA cells Panasonic NCR18650B, 3.2 Ah. Same verified extractor as snl_nmc."),
    "segments": [{"id": "cc_charge", "rate": "0.5C CC to 4.2 V (JES); charge segment = current > 0.01 A"},
                 {"id": "discharge", "rate": "0.5C-2C to 2.5 V (JES; the rate is in the cell name)"}],
    "temperature_c": [15, 25, 35], "reference_test": "maximum discharge capacity of each cycle (BatteryLife discharge_capacity_in_Ah) over the 3.2 Ah nameplate",
    "measured": ["discharge capacity (Ah)"],
    "window": {"segment": "cc_charge", "v_lo": 3.70, "v_hi": 4.10, "coverage_median": 0.440, "calibrated_on": "2026-09-04"},
    "known_issues": ["the four 20-80 % SOC cells carry no calibration capacity and are not ingested (18 of 22 cells)",
                     "two cells (25C_0-100_0.5-0.5C_b and 15C_0-100_0.5-2C_a) fall to soh ~0.02 at end of test; the rows stay inside the QC bounds [0, 1.3) — review before any EOL-band claim",
                     "18 cells, 12,080 rows; raw curves stay in the mirror"],
    "signed": SIGNED_D1, "status": "signed 2026-09-04 (chemistry verified from the pickle metadata and JES 2020, owner-delegated); remaining VERIFY fields are non-chemistry",
}
LEAK_INVENTORY = {"hust": ["charge_capacity_max"], "oxford": ["charge_capacity_max"], "uconn": ["charge_capacity_max"],
                  "rwth_isea_public": ["capacity_ah"], "nasa": [], "calce_cs2": [], "snl_lfp": [],
                  "snl_nmc": ["charge_capacity_max"], "snl_nca": []}


def main() -> int:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    entries = {item["name"]: item for item in manifest.get("datasets", [])}
    out_dir = ROOT / "data" / "datasheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, facts in FACTS.items():
        entry = entries.get(name, {})
        rated = DATASET_C_RATED_AH.get(name)
        window = facts["window"]
        sheet = {
            "name": name,
            "status": facts["status"],
            "source": {"citation": facts["citation"], "url": facts["url"], "license": facts["license"], "retrieved": facts["retrieved"]},
            "chemistry": {"cathode": facts["cathode"], "anode": facts["anode"], "metadata_label": DATASET_CHEMISTRY.get(name, "unknown"),
                          "verified_by": facts["verified_by"], "verified_on": facts["verified_on"], "evidence": facts["chemistry_evidence"]},
            "cell": {"format": facts["format"], "rated_capacity_ah": rated if rated == rated else V, "rated_source": facts["rated_source"]},
            "protocol": {"segments": facts["segments"], "temperature_c": facts["temperature_c"]},
            "label_anchor": {"formula": "soh = calibration_capacity / rated_capacity_ah", "measured_quantities": facts["measured"],
                             "reference_test": facts["reference_test"], "label_source": "calibration_capacity"},
            "sampling": {"rate_hz": V, "channels": ["V", "I", "T"]},
            "windows": [{"feature": "q_win_ah", **window}] if isinstance(window, dict) else V,
            "digests": {"raw": entry.get("raw_sha256") or V, "processed": entry.get("processed_sha256") or (entry.get("processed") or {}).get("sha256") or V,
                        "curves": entry.get("curve_sha256") or V},
            "leak_scan": {"label_source_columns": LEAK_INVENTORY.get(name, V), "scanned_on": "2026-09-03",
                          "reference": "tests/test_label_leak_inventory.py"},
            "held_out_ready": bool(entry.get("held_out_ready", True)),
            "known_issues": facts["known_issues"],
            "signed": dict(facts["signed"]),
        }
        (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(sheet, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print("wrote", out_dir / f"{name}.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
