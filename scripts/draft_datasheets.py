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
SURVEY_DATE = "2026-09-05"   # D12: sampling, provenance, licence text and protocol cut-offs from the archives and record pages
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
            "only; no chemistry claim may rest on nasa until a primary source is found (VERIFY). D12 survey 2026-09-05: the "
            "README.txt shipped inside the archive (data/raw/nasa_battery_dataset.zip -> '5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip', "
            "dated 2008-09-30) says 'A set of four Li-ion batteries (# 5, 6, 7 and 18) were run through 3 different operational profiles "
            "(charge, discharge and impedance) at room temperature. Charging was carried out in a constant current (CC) mode at 1.5A until "
            "the battery voltage reached 4.2V' and names no chemistry or manufacturer either."
        ),
        "segments": [{"id": "cc_cv_charge", "rate": "1.5 A CC to 4.2 V, CV to 20 mA (NASA description)"},
                     {"id": "discharge", "rate": "2 A CC to 2.7 / 2.5 / 2.2 / 2.5 V for B0005 / B0006 / B0007 / B0018 (NASA description)"}],
        "temperature_c": [24], "reference_test": "capacity of each full discharge (NASA description)",
        "measured": ["discharge capacity (Ah)"], "window": None,
        "known_issues": ["chemistry unverified (D4, 2026-09-03) — lcho folds holding out nasa warn",
                         "cell B0018 is the phase 2.2 golden case; 18 non-positive SOH rows dropped by QC",
                         "raw provenance (D12): data/raw/nasa_battery_dataset.zip is the PCoE '5. Battery Data Set' (six nested BatteryAgingARC zips with "
                         "README.txt files); the B00xx.mat files are unpacked under data/raw/nasa and data/raw/nasa_full; logging is variable-rate "
                         "(the Time vectors step about 9 s in the first charges, measured 2026-09-05)"],
        "signed": UNSIGNED, "status": "draft — chemistry unverified after the 2026-09-03 web review and the 2026-09-05 archive survey; unsigned",
    },
    "calce_cs2": {
        "citation": ("CALCE Battery Research Group, University of Maryland: CS2 prismatic cells (Battery Data page); "
                     "He, W., Williard, N., Osterman, M., Pecht, M. (2011). Prognostics of lithium-ion batteries based on "
                     "Dempster-Shafer theory and the Bayesian Monte Carlo method. J. Power Sources 196(23), 10314-10321."),
        "url": "https://calce.umd.edu/battery-data",
        "license": ("academic use with citation — CALCE Battery Data page (fetched 2026-09-05): 'Use of this data for publication purposes should "
                    "include references to the CALCE article(s) that describe the experiments conducted to generating the data'; no licence text"),
        "retrieved": f"{REVIEW_DATE} (page fetched); {SURVEY_DATE} (terms and CS2 description re-fetched)",
        "cathode": "LCO", "anode": "graphite (assumed; the CALCE page states only the cathode)", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": "prismatic, 5.4 x 33.6 x 50.6 mm (CALCE page); tester Arbin BT2000, CADEX for CS2_8 and CS2_21 (CALCE page)",
        "rated_source": "CALCE page (CS2 'Capacity Rating: 1100 mAh'; CX2 1350 mAh)",
        "chemistry_evidence": ("CALCE Battery Data page, CS2: 'Cell Chemistry: LiCoO2 cathode, EDS results also showed trace elements of "
                               "Manganese'; 'Capacity Rating: 1100 mAh'; charging 'constant current rate of 0.5C until the voltage reached "
                               "4.2V and then 4.2V was sustained until the charging current dropped to below 0.05A'; 'discharge cut off "
                               "voltage for these batteries was 2.7V'."),
        "segments": [{"id": "cc_charge", "rate": "0.5C CC to 4.2 V, CV to 0.05 A (CALCE page); measured 0.55 A = 0.5C, 4.20 V max in CS2_35 (2026-09-05)"},
                     {"id": "discharge", "rate": "0.5C-1C per cell to 2.7 V (CALCE page: cut-off 2.7 V)"}],
        "temperature_c": [25], "reference_test": "capacity of each discharge (Arbin file)",
        "measured": ["discharge capacity (Ah)"],
        "window": {"segment": "cc_charge", "v_lo": 3.85, "v_hi": 4.05, "coverage_median": 0.371, "calibrated_on": "2026-08-27"},
        "known_issues": ["charge_capacity_max in the processed table is the file-cumulative Arbin quantity, not per cycle (report finding G6)",
                         "raw provenance (D12): data/raw/calce/CS2_*.zip — Arbin xlsx per test date (30 s logging, measured 2026-09-05); CS2_8 and "
                         "CS2_21 are CADEX text logs the raw-curve iterator skips"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); D12 survey 2026-09-05 filled sampling, provenance, licence text and cut-offs",
    },
    "hust": {
        "citation": ("Ma, G., Xu, S., Jiang, B., Cheng, C., Yang, X., Shen, Y., Yang, T., Huang, Y., Ding, H., Yuan, Y. (2022). Real-time "
                     "personalized health status prediction of lithium-ion batteries using deep transfer learning. Energy & Environmental "
                     "Science 15(10), 4083-4094. Dataset: Mendeley Data v2 (2022-05-24), DOI 10.17632/nsc7hnsg4s.2."),
        "url": "https://data.mendeley.com/datasets/nsc7hnsg4s/2", "license": "CC BY 4.0 (Mendeley record page, 'Licence CC BY 4.0', re-read 2026-09-05)",
        "retrieved": f"{REVIEW_DATE} (record fetched); {SURVEY_DATE} (record page re-read; raw pickles measured)",
        "cathode": "LFP", "anode": "graphite", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": ("18650, 1.1 Ah / 3.3 V nominal (record); cell model VERIFY — the record page names none and the EES paper's text was not "
                   "fetchable in the 2026-09-05 survey (BatteryLife/BatteryML describe the cells as A123 18650 LFP; secondary)"),
        "rated_source": "Mendeley record (1.1 Ah nominal capacity)",
        "chemistry_evidence": ("Mendeley Data record nsc7hnsg4s v2: '77 LFP/graphite cells (1.1 Ah nominal capacity and 3.3 V nominal voltage)' "
                               "with an 'identical charge protocol but different multi-stage discharge protocols at a constant temperature of 30°C'."),
        "segments": [{"id": "cc_charge", "rate": ("identical fast-charge protocol for all cells (record): CC at 5C — 5.50 A median measured in cycle 2 of "
                                                    "cell 1-1 (2026-09-05) — to about 3.54 V, then a stage the raw frames label 'Constant current-constant "
                                                    "voltage charge' (about 21 min at 5 s logging)")},
                     {"id": "discharge", "rate": ("77 distinct multi-stage CC discharge protocols (record): four CC discharge stages per cycle in the raw "
                                                   "frames (statuses Constant current discharge_0..3); cell 1-1 cycle 2 reaches 1.94 V (measured)")}],
        "temperature_c": [30], "reference_test": "capacity of each discharge", "measured": ["charge_capacity_max"],
        "window": {"segment": "cc_charge", "v_lo": 3.30, "v_hi": 3.52, "coverage_median": 0.486, "calibrated_on": "2026-08-25"},
        "known_issues": ["raw provenance (D12): data/raw/hust/our_data/*.pkl from the record's our_data.zip (1.11 GB); one pickle per cell with per-cycle "
                         "frames (Status, Voltage (V), Current (mA), Capacity (mAh), Time (s)) at 5 s logging (measured 2026-09-05)"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); D12 survey 2026-09-05 filled sampling, provenance and the charge stages; the cell model stays unverified",
    },
    "oxford": {
        "citation": ("Howey, D. and Birkl, C. (2017). Oxford Battery Degradation Dataset 1. University of Oxford, ORA; Birkl, C. R. (2017). "
                     "Diagnosis and Prognosis of Degradation in Lithium-Ion Batteries. DPhil thesis, University of Oxford."),
        "url": "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac", "license": "ODC-ODbL (database), DbCL (contents) — readme",
        "retrieved": f"{REVIEW_DATE} (ORA record and Readme.txt fetched); {SURVEY_DATE} (Readme.txt on disk re-read; .mat measured)",
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
        "window": None,
        "known_issues": ["519 characterisation points only; coverage median 0.594 of rated (wide window warning)",
                         "no partial window calibrated: the processed table holds the characterisation cycles (1C charges every 100 Artemis drive cycles), "
                         "and the window scan has no raw-curve source registered for oxford (D12)",
                         "raw provenance (D12): data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat + Readme.txt (Howey & Birkl, June 2017; tests "
                         "started 2015-01-08, Bio-Logic MPG-205, Binder chamber at 40 degC); time is stored as a fraction of a day — 1.0 s steps "
                         "(measured 2026-09-05)"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); D12 survey 2026-09-05 filled sampling and provenance",
    },
    "snl_lfp": {
        "citation": ("Preger, Y., Barkholtz, H. M., Fresquez, A., Campbell, D. L., Juba, B. W., Roman-Kustas, J., Ferreira, S. R., Chalamala, B. "
                     "(2020). Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions. J. Electrochem. Soc. "
                     "167(12), 120532, DOI 10.1149/1945-7111/abae37. Data via BatteryArchive (snl_study) / BatteryLife mirror (data/raw/snl_batterylife)."),
        "url": "https://doi.org/10.1149/1945-7111/abae37",
        "license": ("no licence text (survey 2026-09-05): neither the BatteryArchive study page nor JES 167 120532 states one; BatteryArchive asks for "
                    "attribution to the study page and the JES paper ('double attribution'), and the BatteryLife mirror README asks to cite "
                    "BatteryLife, BatteryML and BatteryArchive — attribution terms, no formal licence"),
        "retrieved": f"{REVIEW_DATE} (JES record via web review; BatteryArchive snl_study page fetched); {SURVEY_DATE} (study page and mirror README re-fetched; pickles measured)",
        "cathode": "LFP", "anode": "graphite", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": "18650 (A123 Systems APR18650M1A)", "rated_source": "JES 167 120532 (LFP: A123 APR18650M1A, 1.1 Ah)",
        "chemistry_evidence": ("JES 167(12) 120532: the LFP cells are A123 Systems APR18650M1A, 1.1 Ah (the study's NCA cells are Panasonic NCR18650B "
                               "3.2 Ah and its NMC cells LG Chem 18650HG2 3 Ah); all cells charged at 0.5C, discharged at 0.5C / 1C / 2C / 3C over "
                               "0-100, 20-80 or 40-60 % SOC at 15 / 25 / 35 degC. The BatteryArchive snl_study page confirms 'All cells were charged "
                               "at a rate of 0.5C' and the three SOC windows."),
        "segments": [{"id": "cc_charge", "rate": "0.5C CC to 3.60 V (JES; measured 0.549 A and 3.600 V max in SNL_18650_LFP_15C_0-100_0.5-2C_b, 2026-09-05)"},
                     {"id": "discharge", "rate": "0.5C-3C (JES) to 2.0 V (measured 1.995 V min; the rate is in the cell name)"}],
        "temperature_c": [15, 25, 35], "reference_test": "capacity of each discharge", "measured": ["separate calibration measurement (no label-source column)"],
        "window": {"segment": "cc_charge", "v_lo": 3.30, "v_hi": 3.38, "coverage_median": 0.449, "calibrated_on": "2026-08-25"},
        "known_issues": ["raw provenance (D12): data/raw/snl_batterylife/SNL/SNL_18650_LFP_*.pkl — the BatteryLife mirror's pickles (cycle_data with "
                         "time_in_s, voltage_in_V, current_in_A, charge/discharge_capacity_in_Ah) at 120 s logging (measured 2026-09-05), not "
                         "BatteryArchive's original CSV"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); D12 survey 2026-09-05 filled sampling, provenance, licence terms and cut-offs",
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
        "segments": [{"id": "rpt_charge", "rate": "1C reference charge to 3.6 V (measured 1.20 A = 1.09C and 3.601 V max in rpt_cell_02_part0.csv, 2026-09-05)"}],
        "temperature_c": [V], "reference_test": "RPT reference discharge every ~100 cycles (about 3.5 days; record)", "measured": ["charge_capacity_max"],
        "window": {"segment": "rpt_charge", "v_lo": 3.30, "v_hi": 3.37, "coverage_median": 0.536, "calibrated_on": "2026-08-25"},
        "known_issues": ["64 rows in 32 duplicated (cell, cycle) identities are dropped by QC since 2026-09-03",
                         "raw provenance (D12): data/raw/uconn_lfp/{cycling_part1.zip, cycling_part2.zip, rpt_data.zip} from the UConn Digital Commons "
                         "record (CC BY 4.0); the record states 1 Hz sampling for cycling and RPT (measured 1.00 s, 2026-09-05)",
                         "test temperatures stay unverified: the record names '11 unique conditions' without listing them (survey 2026-09-05)"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); D12 survey 2026-09-05 filled sampling, provenance and the RPT charge; temperatures stay unverified",
    },
    "rwth_isea_public": {
        "citation": ("Li, W., Sengupta, N., Dechent, P., Howey, D., Annaswamy, A., Sauer, D. U. (2021). One-shot battery degradation trajectory "
                     "prediction with deep learning. J. Power Sources 506, 230024. Dataset: Time-series cyclic aging data on 48 commercial NMC/graphite "
                     "Sanyo/Panasonic UR18650E cylindrical cells, RWTH Publications record 818642, DOI 10.18154/RWTH-2021-04545."),
        "url": "https://publications.rwth-aachen.de/record/818642",
        "license": ("VERIFY — the record states 'The file Content.pdf contains information about the test setup, data, and license', but the PDF is "
                    "served as a download only and the GitHub mirror carries no LICENSE (GitHub API license null; survey 2026-09-05): the owner opens "
                    "Content_RWTH-2021-04545.pdf to close this field"),
        "retrieved": f"{REVIEW_DATE} (RWTH record via web review; GitHub mirror readme via API); {SURVEY_DATE} (record page read in a browser; mirror README via the GitHub API)",
        "cathode": "NMC", "anode": "graphite", "verified_by": REVIEWER, "verified_on": REVIEW_DATE,
        "format": "18650 (Sanyo/Panasonic UR18650E)", "rated_source": "RWTH-2021-04545 record (nominal 1.85 Ah); table per-cell maximum 1.83-1.85 Ah",
        "chemistry_evidence": ("RWTH Publications record 818642: 'Time-series cyclic aging data on 48 commercial NMC/graphite Sanyo/Panasonic UR18650E "
                               "cylindrical cells', nominal capacity 1.85 Ah. scripts/validate_trusted_plugin_workflow.py builds this table from "
                               "'Pre-processing Codes/Degradation_Prediction_Dataset_ISEA.mat' in github.com/isea-rwth-aachen/battery-degradation-"
                               "trajectory-prediction (mirror of git.rwth-aachen.de/isea/..., '48 cells during ageing tests') keeping 8 cells; the "
                               "per-cell maximum capacities (1.83-1.85 Ah) match. The previous label 'LFP (Fortelion residential storage)' was wrong: "
                               "it described the separate rwth_hss field data."),
        "segments": [{"id": "cycling", "rate": ("VERIFY — 'aged with the same profile under equal conditions' (record); the profile's C-rates are in Content.pdf, "
                                                  "which was not readable in the survey")}],
        "temperature_c": [V],
        "reference_test": ("regular reference parameter tests (record: 'aging reference parameter tests (RPT) were conducted in regular intervals'; mirror "
                           "README: initial characterisation with 'capacity test with various C-rates', qOCV and EIS); the C-rate behind capacity_ah VERIFY"),
        "measured": ["capacity_ah"], "window": None,
        "known_issues": ["all 47 cells of the public .mat ingested on 2026-09-04 (D1; until then 8, validate_trusted_plugin_workflow max_cells=8). The RWTH record describes 48 cells; the mirror's Degradation_Prediction_Dataset_ISEA.mat carries cell ids 1-47, so one cell of the record is absent from the public file (VERIFY which). The 8-cell subset reproduces exactly; artifacts trained on the 8-cell table are not comparable with 47-cell results",
                         "raw time series (V/I/T) are not ingested — only per-cycle capacity histories from the ISEA .mat; no window or curve features",
                         "manifest curve_path curves/rwth_curves.json holds rwth_hss home-storage field curves from another source, not these cells",
                         "chemistry corrected from LFP to NMC and rated capacity set to 1.85 Ah on 2026-09-03 (D4)",
                         "raw provenance (D12): work/public_sources/rwth-battery-degradation/Pre-processing Codes/Degradation_Prediction_Dataset_ISEA.mat "
                         "(the isea-rwth-aachen GitHub mirror of git.rwth-aachen.de; TDS = 10,686 (cell, sample) capacity-history records over 48 cells); "
                         "no time series ingested, so no sampling rate applies; no partial window (no curves)"],
        "signed": SIGNED, "status": "signed 2026-09-03 (chemistry verified by owner-delegated web review); D12 survey 2026-09-05 filled provenance and the reference-test text; licence, cycling profile and temperature stay unverified (Content.pdf)",
    },
}
SIGNED_D1 = {"by": "owner-delegated review 2026-09-04 (Claude Fable 5.1); sources in chemistry.evidence", "on": "2026-09-04"}
FACTS["snl_nmc"] = {
    "citation": ("Preger, Y. et al. (2020). Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions. "
                 "J. Electrochem. Soc. 167(12), 120532, DOI 10.1149/1945-7111/abae37 — NMC subset; data via BatteryArchive / BatteryLife mirror "
                 "(data/raw/snl_batterylife), ingested by scripts/ingest_batterylife_snl.py (D1, 2026-09-04)."),
    "url": "https://doi.org/10.1149/1945-7111/abae37",
    "license": ("no licence text (survey 2026-09-05): neither the BatteryArchive study page nor JES 167 120532 states one; BatteryArchive asks for "
                    "attribution to the study page and the JES paper ('double attribution'), and the BatteryLife mirror README asks to cite "
                    "BatteryLife, BatteryML and BatteryArchive — attribution terms, no formal licence"),
    "retrieved": "2026-09-04 (built from the local BatteryLife mirror); 2026-09-05 (study page and mirror README re-fetched; pickles measured)",
    "cathode": "NMC", "anode": "graphite", "verified_by": SIGNED_D1["by"], "verified_on": "2026-09-04",
    "format": "18650 (LG Chem 18650HG2)", "rated_source": "JES 167 120532 (NMC: LG Chem 18650HG2, 3 Ah); pickle nominal_capacity_in_Ah 3 (the mirror derates 15 degC cells to 2.7 Ah; not used)",
    "chemistry_evidence": ("BatteryLife pickle metadata: cathode_material 'NMC', anode_material 'graphite', form_factor cylindrical_18650, nominal 3 Ah; "
                           "JES 167(12) 120532 names the NMC cells LG Chem 18650HG2, 3 Ah. The extractor was verified against the existing snl_lfp "
                           "table before use (soh 100 % row match; charge-segment statistics >= 99.998 %)."),
    "segments": [{"id": "cc_charge", "rate": "0.5C CC to 4.2 V (JES; measured 1.496 A and 4.200 V max in SNL_18650_NMC_15C_0-100_0.5-1C_a, 2026-09-05); charge segment = current > 0.01 A"},
                 {"id": "discharge", "rate": "0.5C-3C to 2.0 V (JES; measured 1.999 V min; the rate is in the cell name)"}],
    "temperature_c": [15, 25, 35], "reference_test": "maximum discharge capacity of each cycle (BatteryLife discharge_capacity_in_Ah) over the 3.0 Ah nameplate",
    "measured": ["discharge capacity (Ah)"],
    "window": {"segment": "cc_charge", "v_lo": 3.65, "v_hi": 4.10, "coverage_median": 0.433, "calibrated_on": "2026-09-04"},
    "known_issues": ["charge_capacity_max reconstructs soh affinely (R² 0.998) on 0-100 % cycles: pinned in the leak inventory and banned as an input by name",
                     "21 cells, 14,283 rows; raw curves stay in the mirror (not copied into data/curves)",
                     "raw provenance (D12): data/raw/snl_batterylife/SNL/SNL_18650_NMC_*.pkl (BatteryLife mirror pickles, 120 s logging measured 2026-09-05)"],
    "signed": SIGNED_D1, "status": "signed 2026-09-04 (chemistry verified from the pickle metadata and JES 2020, owner-delegated); D12 survey 2026-09-05 filled sampling, provenance, licence terms and cut-offs",
}
FACTS["snl_nca"] = {
    "citation": ("Preger, Y. et al. (2020). Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions. "
                 "J. Electrochem. Soc. 167(12), 120532, DOI 10.1149/1945-7111/abae37 — NCA subset; data via BatteryArchive / BatteryLife mirror "
                 "(data/raw/snl_batterylife), ingested by scripts/ingest_batterylife_snl.py (D1, 2026-09-04)."),
    "url": "https://doi.org/10.1149/1945-7111/abae37",
    "license": ("no licence text (survey 2026-09-05): neither the BatteryArchive study page nor JES 167 120532 states one; BatteryArchive asks for "
                    "attribution to the study page and the JES paper ('double attribution'), and the BatteryLife mirror README asks to cite "
                    "BatteryLife, BatteryML and BatteryArchive — attribution terms, no formal licence"),
    "retrieved": "2026-09-04 (built from the local BatteryLife mirror); 2026-09-05 (study page and mirror README re-fetched; pickles measured)",
    "cathode": "NCA", "anode": "graphite", "verified_by": SIGNED_D1["by"], "verified_on": "2026-09-04",
    "format": "18650 (Panasonic NCR18650B)", "rated_source": "JES 167 120532 (NCA: Panasonic NCR18650B, 3.2 Ah); pickle nominal_capacity_in_Ah 3.2 (the mirror derates 15 degC cells to 2.88 Ah; not used)",
    "chemistry_evidence": ("BatteryLife pickle metadata: cathode_material 'NCA', anode_material 'graphite', form_factor cylindrical_18650, nominal 3.2 Ah; "
                           "JES 167(12) 120532 names the NCA cells Panasonic NCR18650B, 3.2 Ah. Same verified extractor as snl_nmc."),
    "segments": [{"id": "cc_charge", "rate": "0.5C CC to 4.2 V (JES; measured 1.624 A and 4.200 V max in SNL_18650_NCA_15C_0-100_0.5-1C_a, 2026-09-05); charge segment = current > 0.01 A"},
                 {"id": "discharge", "rate": "0.5C-2C to 2.5 V (JES; measured 2.499 V min; the rate is in the cell name)"}],
    "temperature_c": [15, 25, 35], "reference_test": "maximum discharge capacity of each cycle (BatteryLife discharge_capacity_in_Ah) over the 3.2 Ah nameplate",
    "measured": ["discharge capacity (Ah)"],
    "window": {"segment": "cc_charge", "v_lo": 3.70, "v_hi": 4.10, "coverage_median": 0.440, "calibrated_on": "2026-09-04"},
    "known_issues": ["the four 20-80 % SOC cells carry no calibration capacity and are not ingested (18 of 22 cells)",
                     "two cells (25C_0-100_0.5-0.5C_b and 15C_0-100_0.5-2C_a) fall to soh ~0.02 at end of test; the rows stay inside the QC bounds [0, 1.3) — review before any EOL-band claim",
                     "18 cells, 12,080 rows; raw curves stay in the mirror",
                     "raw provenance (D12): data/raw/snl_batterylife/SNL/SNL_18650_NCA_*.pkl (BatteryLife mirror pickles, 120 s logging measured 2026-09-05)"],
    "signed": SIGNED_D1, "status": "signed 2026-09-04 (chemistry verified from the pickle metadata and JES 2020, owner-delegated); D12 survey 2026-09-05 filled sampling, provenance, licence terms and cut-offs",
}
LEAK_INVENTORY = {"hust": ["charge_capacity_max"], "oxford": ["charge_capacity_max"], "uconn": ["charge_capacity_max"],
                  "rwth_isea_public": ["capacity_ah"], "nasa": [], "calce_cs2": [], "snl_lfp": [],
                  "snl_nmc": ["charge_capacity_max"], "snl_nca": []}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="draft the datasheets from the facts table, the manifest and the measured provenance")
    parser.add_argument("--data-dir", default=str(ROOT / "data"), help="data root with manifest.json (default: the repository's data/)")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "datasheets"), help="where the YAML sheets and _provenance.json live")
    args = parser.parse_args(argv)
    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text(encoding="utf-8"))
    entries = {item["name"]: item for item in manifest.get("datasets", [])}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = out_dir / "_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")).get("datasets", {}) if provenance_path.is_file() else {}
    for name, facts in FACTS.items():
        entry = entries.get(name, {})
        measured = provenance.get(name, {})
        interval = (measured.get("sampling_interval_s") or {})
        if interval.get("seconds"):
            sampling = {"rate_hz": round(1.0 / float(interval["seconds"]), 4), "interval_s": interval["seconds"],
                        "interval_p10_s": interval.get("p10"), "interval_p90_s": interval.get("p90"), "segments_measured": interval.get("segments"),
                        "method": f"{interval.get('method')} (scripts/measure_raw_provenance.py, {measured.get('measured_at', SURVEY_DATE)})",
                        "channels": ["V", "I", "T"]}
        elif interval.get("method"):
            sampling = {"rate_hz": "not applicable", "method": interval["method"], "channels": []}
        else:
            sampling = {"rate_hz": V, "channels": ["V", "I", "T"]}
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
            "sampling": sampling,
            "windows": [{"feature": "q_win_ah", **window}] if isinstance(window, dict) else ([] if window is None else V),
            "digests": {"raw": measured.get("raw_sha256") or entry.get("raw_sha256") or V,
                        "processed": entry.get("processed_sha256") or (entry.get("processed") or {}).get("sha256") or V,
                        "curves": entry.get("curve_sha256") or ("none (no curves file; the raw curves stay in the archive)" if not entry.get("curve_path") else V)},
            "provenance": {"raw_files": [{"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"]} for r in measured.get("raw_files", [])],
                           "measured_at": measured.get("measured_at", V if not measured else SURVEY_DATE),
                           "reference": "scripts/measure_raw_provenance.py -> data/datasheets/_provenance.json"} if measured else V,
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
