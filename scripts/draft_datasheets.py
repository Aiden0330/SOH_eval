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

# ---------------------------------------------------------------------------------------------------------
# Wave 6 (2026-09-05): D2 tongji_*, D8 forklift_lfp180, D9 matr, D7 home_storage_lfp, D3 turku_rc_*, D10
# field_lfp_28.  Drafted by the data-onboarding session from the primary records (papers, repository
# metadata, the raw files); every field without a primary source is VERIFY and the signature stays the
# owner's (unsigned until the owner reviews).  Field sets carry the Spec 12 sections (supervision, label
# rule, segments QC, units, verification).
# ---------------------------------------------------------------------------------------------------------
DRAFT_DATE = "2026-09-05"
UNSIGNED_W6 = {"by": "VERIFY (the owner's signature; drafted 2026-09-05 by the Wave 6 data-onboarding session)", "on": V}
STANDARD_CLASSES = {
    "identity": ["dataset", "cell_id", "cycle_index"],
    "label": ["soh"],
    "label_source": ["charge_capacity_max"],
    "feature": ["time_s_max", "voltage_mean", "voltage_std", "voltage_min", "voltage_max", "current_mean", "current_std",
                "abs_current_mean", "temperature_mean", "q_win_ah", "t_win_s", "ic_peak_ah_per_v", "ic_peak_voltage_v",
                "cycle_index_feature", "cycle_fraction", "hi_* (K12 indicators)"],
}
SEGMENT_CLASSES = {
    "identity": ["dataset", "cell_id (the unit)", "cycle_index (segment ordinal)", "segment_start_s", "segment_end_s", "segment_kind", "sample_period_s"],
    "label": ["soh"],
    "label_source": ["charge_capacity_max (ΔQ of the segment)", "label_source", "label_time_s", "label_age_s"],
    "descriptor": ["window_coverage", "v_start", "v_end", "current_mean_c", "current_cv", "rest_before_s", "rest_after_s", "temp_min", "temp_max"],
    "feature": ["time_s_max", "voltage_mean", "voltage_std", "voltage_min", "voltage_max", "current_mean", "current_std",
                "abs_current_mean", "temperature_mean", "q_win_ah", "t_win_s", "ic_peak_ah_per_v", "ic_peak_voltage_v", "hi_* (K12 indicators)"],
}
# >>> dataset-block: tongji
TONGJI_CITATION = ("Zhu, J., Wang, Y., Huang, Y., Bhushan Gopaluni, R., Cao, Y., Heere, M., Mühlbauer, M. J., Mereacre, L., Dai, H., Liu, X., "
                   "Senyshyn, A., Wei, X., Knapp, M., Ehrenberg, H. (2022). Data-driven capacity estimation of commercial lithium-ion batteries "
                   "from voltage relaxation. Nature Communications 13, 2261, DOI 10.1038/s41467-022-29837-w. Dataset: Zenodo 6405084 "
                   "(DOI 10.5281/zenodo.6405084; the paper cites the concept record 10.5281/zenodo.6379165); ingested by scripts/ingest_tongji.py (D2, 2026-09-05).")
TONGJI_PROTOCOL = ("Nature Communications 13, 2261, Methods: 'constant current (CC) charging to 4.2 V with current rates ranging from 0.25 C (0.875 A) "
                   "to 1 C (3.5 A), followed by a constant voltage (CV) charging step at 4.2 V until a current of 0.05 C is reached. Constant current "
                   "is then employed for the discharge to 2.65 V for the NCA cells and 2.5 V for the NCM and NCM + NCA cells'; 'The CC discharging "
                   "capacity is treated as the battery residual capacity during cycling'; relaxation 30 min after charge (120 s sampling) for NCA and "
                   "NCM, 60 min (30 s sampling) for NCM + NCA; BioLogic BCS-815 cycler, BINDER chambers (±0.2 °C).")

def _tongji(name, *, zip_name, md5, cathode, cathode_text, model, rated, cutoff_low, cells, conditions, window, coverage, rows, soh_first,
            low_rows, extra_issues):
    return {
        "citation": TONGJI_CITATION, "url": "https://doi.org/10.5281/zenodo.6405084",
        "license": "CC BY 4.0 (Zenodo record 6405084 metadata: cc-by-4.0)",
        "retrieved": f"{DRAFT_DATE} (Zenodo API record; {zip_name} md5 {md5} verified against the record)",
        "cathode": cathode, "anode": "graphite with ~2 wt% Si (paper Methods, citing Sorensen et al.)" if name != "tongji_blend" else "graphite (paper Methods)",
        "verified_by": V, "verified_on": V,
        "format": f"18650 ({model})", "rated_capacity_ah": rated,
        "rated_source": f"paper Table 1 (nominal capacity {rated} Ah; 1C = {rated} A)",
        "chemistry_evidence": (f"Nature Communications 13, 2261, Methods ('Cell selection and cycling'): {cathode_text} " + TONGJI_PROTOCOL),
        "segments": [{"id": "cc_cv_charge", "rate": "CC 0.25C / 0.5C / 1C to 4.2 V, CV at 4.2 V to 0.05C (paper); charge segment = samples with <I>/mA > 10 mA (CC and CV steps)"},
                     {"id": "relaxation", "rate": "rest after charge: 30 min at 120 s sampling (NCA, NCM), 60 min at 30 s (NCM+NCA); not yet a feature source"},
                     {"id": "discharge", "rate": f"CC 1C to {cutoff_low} V" + (" (2C and 4C groups in Dataset_3)" if name == "tongji_blend" else "")}],
        "temperature_c": [25, 35, 45] if name != "tongji_blend" else [25],
        "reference_test": "the CC discharge capacity of each cycle: max 'Q discharge/mA.h' of the cycle (the paper's residual capacity) over the nominal capacity",
        "measured": ["Q discharge/mA.h (per-cycle CC discharge capacity)"],
        "label_source": "calibration_capacity", "supervision": "soh_supervised",
        "rate_hz": "0.1 during charge and discharge (10 s), 1/120 (NCA, NCM) or 1/30 (blend) in relaxation — CSV time/s",
        "channels": ["V", "I", "Q charge", "Q discharge"],
        "window": {"segment": "cc_cv_charge", "v_lo": window[0], "v_hi": window[1], "coverage_median": coverage, "calibrated_on": DRAFT_DATE,
                   "note": "scan of [3.4, 4.2] V at 0.05 V steps on every 10th cycle; the mid-charge window shared with the SNL NMC/NCA sets"},
        "units": {"unit": "cell", "cells": cells, "cells_per_condition (chamber_charge_discharge)": conditions},
        "verification": {"cells_per_condition_vs_paper_table_1": "identical (paper Table 1 counts reproduced by the archive inventory)",
                         "rows": rows, "first_cycle_soh": soh_first,
                         "raw_hand_check": "one cycle of a CSV read by hand: rest, CC charge at the named rate to 4.2 V, CV, 28 min rest at 120 s, CC discharge to the cut-off, rest (Dataset_1 CY25-1_1-#3 cycle 5)"},
        "column_classes": STANDARD_CLASSES, "scanned_on": DRAFT_DATE,
        "known_issues": ["no temperature channel in the CSVs: temperature_mean is the chamber setpoint from the cell name (CY25/35/45)",
                         f"{low_rows}",
                         "relaxation features (the paper's own feature source) are not extracted; K12 §4.6 lists them for the segment work",
                         "signature pending: the owner's review of chemistry (quoted from the paper) and rated capacity"] + extra_issues,
        "signed": dict(UNSIGNED_W6), "status": f"draft {DRAFT_DATE} (D2) — chemistry and rated capacity quoted from the paper; unsigned until the owner reviews",
    }

FACTS["tongji_nca"] = _tongji("tongji_nca", zip_name="Dataset_1_NCA_battery.zip", md5="cbd6d7eab2a296b52385e025e8d0f42a", cathode="NCA",
    cathode_text="'The positive electrode compositions of the INR18650-35E battery and INR18650-MJ1 battery are LiNi0.86Co0.11Al0.03O2 and Li(Ni0.83Co0.11Mn0.07)O2 respectively' … 'The INR18650-35E battery is named as NCA battery.'",
    model="LG INR18650-35E, 3.5 Ah, 3.6 V", rated=3.5, cutoff_low=2.65, cells=66,
    conditions={"25C_0.25C_1C": 7, "25C_0.5C_1C": 19, "25C_1C_1C": 9, "35C_0.5C_1C": 3, "45C_0.5C_1C": 28},
    window=(3.65, 4.10), coverage=0.432, rows=22913, soh_first="0.879–0.961 (median 0.936: the 1C discharge capacity of a 3.5 Ah nominal cell)",
    low_rows="268 rows with soh < 0.5: cells of the 35 °C and 45 °C groups collapse at end of test (soh down to 0.004); inside the QC bounds [0, 1.3) — review before any EOL-band claim",
    extra_issues=["the 25 °C 1C-charge cells (CY25-1_1) fail within 29–60 cycles (the paper: 71 % after ~50 cycles)"])
FACTS["tongji_ncm"] = _tongji("tongji_ncm", zip_name="Dataset_2_NCM_battery.zip", md5="a05c485df92f4af560d1d8d7f659ce1b", cathode="NCM (NMC)",
    cathode_text="'… the INR18650-MJ1 battery … Li(Ni0.83Co0.11Mn0.07)O2' … 'The INR18650-MJ1 is named as NCM battery.'",
    model="Samsung INR18650-MJ1, 3.5 Ah, 3.6 V", rated=3.5, cutoff_low=2.5, cells=55,
    conditions={"25C_0.5C_1C": 23, "35C_0.5C_1C": 4, "45C_0.5C_1C": 28},
    window=(3.65, 4.10), coverage=0.436, rows=28003, soh_first="0.899–0.957 (median 0.939)",
    low_rows="195 rows with soh < 0.5: the four 35 °C cells and two 45 °C cells collapse at end of test (soh down to 0.006); inside the QC bounds",
    extra_issues=["the kernel label is NMC (metadata) for the authors' NCM"])
FACTS["tongji_blend"] = _tongji("tongji_blend", zip_name="Dataset_3_NCM_NCA_battery.zip", md5="d11e68e410a638058906af5e2f5f60f3", cathode="NCM+NCA blend (42 wt% Li(NiCoMn)O2 / 58 wt% Li(NiCoAl)O2)",
    cathode_text="'The positive electrode of the INR18650-25R battery is the blend of 42 (3) wt.% Li(NiCoMn)O2 - 58 (3) wt.% Li(NiCoAl)O2, and the negative electrode is graphite' … 'The INR18650-25R is named as NCM + NCA battery according to the positive electrode.'",
    model="Samsung INR18650-25R, 2.5 Ah, 3.6 V", rated=2.5, cutoff_low=2.5, cells=9,
    conditions={"25C_0.5C_1C": 3, "25C_0.5C_2C": 3, "25C_0.5C_4C": 3},
    window=(3.70, 4.10), coverage=0.402, rows=8764, soh_first="0.988–1.001 (median 0.995)",
    low_rows="45 rows with soh > 1.1 (18 of them ≥ 1.3, dropped by QC): five cycles per cell where 'Q discharge' exceeds the nominal capacity by 10–44 % — VERIFY (capacity-check or logging cycles)",
    extra_issues=["the kernel label is NMC/NCA (metadata); the K12 indicator windows resolve on NMC (3.70–4.10 V)",
                  "affine R² of q_win_ah against soh is 0.89 on this window (the authors report a linear fade); windows above 3.55 V reach 0.99 and were rejected by the scan",
                  "cycled in 2020 by a different experimenter than Datasets 1–2 (paper); 60 min relaxation at 30 s sampling"])
# <<< dataset-block: tongji

# >>> dataset-block: forklift_lfp180
FACTS["forklift_lfp180"] = {
    "citation": ("Vilsen, S. B., Stroe, D.-I. (2024). Dataset of lithium-ion battery degradation based on a forklift mission profile for "
                 "state-of-health estimation and lifetime prediction. Data in Brief 52, 109861 (PMC10749244). Dataset: 'Lithium-ion battery degradation "
                 "dataset based on a realistic forklift operation profile', Mendeley Data v2 (2023-10-13), DOI 10.17632/yz4pttm73n.2 (yz4pttm73n-2.zip, "
                 "321 files, 4.89 GB uncompressed); ingested by scripts/ingest_forklift_lfp180.py (D8, 2026-09-05) through sohkernel.segments (K13)."),
    "url": "https://doi.org/10.17632/yz4pttm73n.2", "license": "CC BY 4.0 (Mendeley record)",
    "retrieved": f"{DRAFT_DATE} (zip integrity tested: no errors; the record publishes no checksum — VERIFY)",
    "cathode": "LFP", "anode": "graphite", "verified_by": V, "verified_on": V,
    "format": "prismatic, 180 Ah, 3.2 V nominal, 2.5–3.65 V (paper Table 3); manufacturer and model not named (VERIFY)",
    "rated_capacity_ah": 180.0, "rated_source": "paper Table 3 ('Nominal capacity 180 Ah', 'Nominal voltage 3.2 V', 'Maximum voltage 3.65 V', 'Minimum voltage 2.5 V')",
    "chemistry_evidence": ("Data in Brief 52, 109861, §3: 'The Li-ion batteries used in this study are prismatic cells using lithium iron phosphate (LFP) and "
                           "graphite as cathode and anode active materials, respectively'; Table 3 'Chemistry LFP/C', 180 Ah, 3.2 V. Ageing: 'realistic forklift "
                           "load profile, with a length of approximately two weeks … distilled from a four-month forklift field operation', idle periods removed, "
                           "currents limited to 50 A; 45 °C (Cell 1), 40 °C (Cell 2), 35 °C (Cell 3); Digatron MCT 50-06-24, Memmert ICP-600 chambers. "
                           "RPT (25 °C): 45 A discharge to 2.5 V, 1 h, 90 A CC to 3.65 V / CV to 9 A, 1 h, 90 A discharge to 2.5 V, the same at 45 A, "
                           "recharge, resistance pulses at 90/70/50/30 % SOC, 45 A discharge, charge to 90 % SOC; 'capacity measurements performed at C/4 (i.e., 45 A)'."),
    "segments": [{"id": "ageing_charge", "rate": "charging segments of the forklift profile (Spec 12 §2 defaults: current > 3.6 A, rests < 600 s bridged, ≥ 300 s, ≥ 9 Ah); mean charge current ≈ 23.5 A (0.13 C), coefficient of variation ≈ 0.04, charges end below 3.40 V"},
                 {"id": "rpt", "rate": "reference performance test after every round (label source); its CC-CV charges are not segments"}],
    "temperature_c": [45, 40, 35],
    "reference_test": "the RPT's C/4 capacity: Ah of the 45 A constant-current discharge from the fully charged state (preceding CC-CV charge reached ≥ 3.60 V) to the 2.5 V cut-off (paper §3, Fig. 5); the C/2 capacity is recorded alongside in the extractor report",
    "measured": ["RPT C/4 (45 A) discharge capacity, Coulomb-counted from RPT.csv (Ah)"],
    "label_formula": "soh = RPT_C4_capacity_ah / 180 Ah", "label_source": "rpt", "supervision": "sparse_field",
    "label_rule": "forward-only (Spec 12 §3): every segment carries the last RPT of its cell that ended before the segment started; the segments of round N carry the RPT of round N-1 (Round00 is the beginning-of-life RPT); a lost RPT makes the following round's label one round older; no age bound (the age is one round, median 7.4 days)",
    "rate_hz": 1, "channels": ["V", "I", "T", "E (Wh)"],
    "window": {"segment": "ageing_charge", "v_lo": 3.30, "v_hi": 3.40, "coverage_median": 0.181, "calibrated_on": DRAFT_DATE,
               "note": "the K12 LFP indicator span; the ageing charges end below 3.40 V so the window holds the whole segment charge — coverage is the charged fraction of 180 Ah, never the calibration capacity"},
    "segments_qc": {"rounds_with_ageing_files": {"cell1": 56, "cell2": 56, "cell3": 48}, "rpts_with_c4_capacity": {"cell1": 55, "cell2": 56, "cell3": 50},
                    "segments": 17761, "segments_per_round": {"cell1": 110.4, "cell2": 109.8, "cell3": 113.2},
                    "candidates_found": 637399, "dropped_too_short": 617420, "dropped_low_coverage": 2218,
                    "note": "the profile's regenerative pulses are the short candidates; the paper's rounds 58/58/53 minus the lost parts (Table 2: round 29 lost for every cell, RPTs lost in rounds 28/37/51 (cell 1), 37/51 (cell 2), 25/33 (cell 3), cell 3 round 28 entirely)"},
    "units": {"unit": "cell", "cells": 3, "c_rated_ah": 180.0, "time_axis": "test time in seconds since the cell's first sample (files restart Time; parts and RPTs concatenated in round order with a 1 s step) — wall-clock dates are not in the data (VERIFY)"},
    "verification": {"rpt_capacity_range_ah": {"cell1": "190.66 → 147.94", "cell2": "189.47 → 154.86", "cell3": "189.29 → 159.84"},
                     "raw_hand_check": "Cell1/Round01/RPT.csv read by hand: the C/4 step is 45.0 A for 14,804 s from 3.44 V to 2.50 V after the CC-CV charge to 3.65 V = 185.05 Ah; the 90 A step gives 183.57 Ah; the first 45 A step (135.9 Ah from the ageing end state) is not a capacity",
                     "archive_vs_paper_table_2": "the missing parts and RPTs in the archive reproduce the paper's Table 2",
                     "spec_12_acceptance_1": "rounds per cell with an RPT 55/56/50 (paper 58/58/53 minus the lost RPTs); segments per round recorded above"},
    "column_classes": SEGMENT_CLASSES, "scanned_on": DRAFT_DATE,
    "known_issues": ["soh > 1 at beginning of life (C/4 capacity 190.7 Ah vs the 180 Ah nominal; QC keeps rows below 1.3)",
                     "segments never span a file part; rests at part boundaries are unknown (NaN)",
                     "the RPT temperature is 25 °C while the segments are at 45/40/35 °C",
                     "signature pending: the owner's review of chemistry (paper Table 3) and the C/4 label rule"],
    "signed": dict(UNSIGNED_W6), "status": f"draft {DRAFT_DATE} (D8) — chemistry and rated capacity quoted from the paper; unsigned until the owner reviews",
}
# <<< dataset-block: forklift_lfp180

# >>> dataset-block: matr
FACTS["matr"] = {
    "citation": ("Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen, M. H., Aykol, M., Herring, P. K., Fraggedakis, D., "
                 "Bazant, M. Z., Harris, S. J., Chueh, W. C., Braatz, R. D. (2019). Data-driven prediction of battery cycle life before capacity "
                 "degradation. Nature Energy 4, 383–391, DOI 10.1038/s41560-019-0356-8 (batches 2017-05-12, 2017-06-30, 2018-04-12); Attia, P. M. et al. "
                 "(2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. Nature 578, 397–402, "
                 "DOI 10.1038/s41586-020-1994-5 (batch 2019-01-24). Data: data.matr.io (MATLAB v7.3 batches MATR_batch_<date>.mat); ingested by "
                 "scripts/ingest_matr.py (D9, 2026-09-05)."),
    "url": "https://data.matr.io/1/", "license": "VERIFY (data.matr.io terms: cite the papers; no licence text found on the record pages)",
    "retrieved": f"{DRAFT_DATE} (four batches, 10.9 GB; the site publishes no checksums — VERIFY)",
    "cathode": "LFP", "anode": "graphite", "verified_by": V, "verified_on": V,
    "format": "18650 (A123 Systems APR18650M1A, 1.1 Ah nominal; Severson 2019 Methods)", "rated_capacity_ah": 1.1,
    "rated_source": "Severson et al. 2019 (A123 APR18650M1A, nominal capacity 1.1 Ah; cycle life defined at 0.88 Ah = 80 %)",
    "chemistry_evidence": ("Severson et al. 2019, Methods: commercial A123 Systems APR18650M1A LFP/graphite cells (1.1 Ah nominal, 3.3 V), cycled in a "
                           "30 °C chamber with one- and two-step fast-charging policies to 80 % SOC followed by 1C CC-CV to 3.6 V, 4C discharge to 2.0 V; "
                           "Attia et al. 2020 use the same cell and cycler for the closed-loop batches. The batch files carry the policy strings "
                           "(e.g. '3.6C(80%)-3.6C') per cell. Cell identity and chemistry are stated in the papers, not in the files (VERIFY)."),
    "segments": [{"id": "fast_charge", "rate": "one- or two-step CC fast charge (3.6C–8C) to 80 % SOC, then 1C CC to 3.6 V and CV (papers); charge segment = samples with I > 0.01 A"},
                 {"id": "discharge", "rate": "4C CC to 2.0 V (papers)"}],
    "temperature_c": [30],
    "reference_test": "the batch summary's QDischarge of each cycle (the 4C discharge capacity) over the 1.1 Ah nominal; equals the raw discharge array's maximum on every cycle (extractor --verify: 139,730 of 139,730 rows within 0.001 Ah)",
    "measured": ["summary QDischarge (Ah)"], "label_source": "calibration_capacity", "supervision": "soh_supervised",
    "rate_hz": "irregular (cycler logging; ~1,100 samples per ~55 min cycle) — the per-cycle arrays carry t in minutes",
    "channels": ["V", "I", "Qc", "Qd", "T"],
    "window": {"segment": "fast_charge", "v_lo": 3.40, "v_hi": 3.50, "coverage_median": 0.803, "calibrated_on": DRAFT_DATE,
               "note": "no lattice window of the scan ([3.0, 3.6] V, 0.05 V steps, every 25th cycle) has a median coverage in the 40–50 % band: at 3.6C and above the terminal voltage sits above 3.4 V for most of the charge (3.30–3.40 V holds a median 14 %). [3.40, 3.50] V is the narrowest window with max coverage below 0.90 (0.895); the audit warns 'wide window' and allows it"},
    "units": {"unit": "cell", "cells": 169, "cells_per_batch": {"2017-05-12": 41, "2017-06-30": 43, "2018-04-12": 40, "2019-01-24": 45},
              "naming": "b<batch date>c<index> (the literature's b1c0 … b3c45 are b20170512c00 … b20180412c45)"},
    "verification": {"raw_discharge_vs_summary": "max(Qd) of every raw cycle equals the summary QDischarge (match 100 %, 139,730 rows)",
                     "cycle_life": "recomputed cycle life (first cycle below 80 %, else last cycle + 1) equals the batch cycle_life field for 167 of 169 cells (within one cycle); b20190124c38 and c40 differ (recorded 1166 / 1089, computed 1056 / 872: an early transient dip below 80 % — VERIFY the authors' definition for the 2019 batch)",
                     "cleaning_vs_authors_notebook": "batch-2 channels 7, 8, 9, 15, 16 continue batch-1 cells 0–4 (cycle counts 662/981/1060/208/482 equal the notebook's add_len); batch-1 cells 8, 10, 12, 13, 22 and batch-3 cells 2, 23, 32, 37, 38, 39 excluded per the authors' 'Load Data' notebook (VERIFY the noisy-channel list against the repository)"},
    "column_classes": STANDARD_CLASSES, "scanned_on": DRAFT_DATE,
    "known_issues": ["4 rows with soh ≥ 1.3 (up to 2.62: logging glitches, dropped by QC); the 2019 batch runs far past 80 % (soh down to 0.23)",
                     "the first summary entry of every batch-1 cell is zero (dropped: no label)",
                     "q_win_ah covers ≈ 80 % of rated (wide-window warning) because no plateau window fits the fast-charge protocols; the K12 LFP span 3.30–3.40 V covers a median 14 %",
                     "only one of the Attia et al. 2020 batches (2019-01-24, 45 cells) is on disk; the plan's 233-cell count needs the other batches",
                     "signature pending: the owner's review of chemistry (papers) and the licence (data.matr.io terms)"],
    "signed": dict(UNSIGNED_W6), "status": f"draft {DRAFT_DATE} (D9) — chemistry and rated capacity from the papers; licence VERIFY; unsigned until the owner reviews",
}
# <<< dataset-block: matr
# >>> dataset-block: turku
TURKU_CITATION = ("Heydarzadeh, M., Toivola, T., Vega-Garita, V., Immonen, E. (2025). Dataset of lithium-ion cell degradation under randomized "
                  "current profiles for NMC, NCA, and LFP chemistries. Data in Brief 60, 111531, DOI 10.1016/j.dib.2025.111531 (PMC12175228). "
                  "Dataset: 'NMC, NCA, and LFP Cells Degradation Under Randomized Current Profiles', FairData / Etsin (Metax id "
                  "f60a7388-99b6-4637-96d5-8610a7f71612), DOI 10.23729/b07462e8-9e31-428c-bdd5-deb1890c608b, issued 2024-12-11 — DATASET/<CHEM>/"
                  "<CHEM>_CELL_<n>.mat (the 4.7 GB 'Raw Data.7z' cycler export is not fetched); ingested by scripts/ingest_turku_rc.py (D3 re-scoped, 2026-09-05).")
TURKU_EVIDENCE = ("Data in Brief 60, 111531: 'four ICR18650-26J (NCA), two INR21700-40T (NMC), and two JGPFR26650 (LFP) which are tested at room "
                  "temperature'; Table 1 (manufacturer Samsung / Samsung / JGNE; nominal capacity 4.0 / 2.55 / 3.0 Ah for NMC / NCA / LFP; charge "
                  "cut-off 4.2 / 4.2 / 3.6 V; discharge cut-off 2.50 / 2.75 / 2.50 V; cut-off current 200 / 130 / 150 mA); 'A cycle is defined by "
                  "the following Steps 1 to 4: 1) CC-CV charge at a constant 1C. 2) CC discharge at 1C. 3) HPPC discharge at 0.5C, 1C, and 2C. "
                  "4) User-defined (RC) discharge profile to reach the cut-off voltage'; 'After every set of 20 cycles, HPPC tests are performed … "
                  "Following the HPPC tests, reference constant current discharges at 1C is served as benchmarks'; 'every 200 cycles, the RC data is "
                  "updated'; 1 Hz sampling, Chroma battery test system. The anode material is not stated in the article (VERIFY).")

def _turku(name, *, chem, cathode, model, fmt, rated, cutoff, cells, window, coverage, rows, labeled, refs_per_cell, ref_range, soh_range, extra):
    return {
        "citation": TURKU_CITATION, "url": "https://etsin.fairdata.fi/dataset/f60a7388-99b6-4637-96d5-8610a7f71612",
        "license": "CC BY-NC 3.0 (the FairData record's access-rights licence; the Data in Brief article is CC BY 4.0). The Wave 6 plan assumed CC BY 4.0: non-commercial terms — VERIFY redistribution of derived tables",
        "retrieved": f"{DRAFT_DATE} (Etsin download authorization per file; the sha256 of every DATASET file matches the Metax file list)",
        "cathode": cathode, "anode": "VERIFY (not stated in the article; graphite assumed)", "verified_by": V, "verified_on": V,
        "format": f"{fmt} ({model})", "rated_capacity_ah": rated, "rated_source": f"paper Table 1 (Q_nom {rated} Ah)",
        "chemistry_evidence": TURKU_EVIDENCE,
        "segments": [{"id": "cc_cv_charge", "rate": f"1C CC to {cutoff[1]} V, CV to the cut-off current (paper Table 1–2); the row's feature source = the cycle entry's largest charging span (Spec 12 §2 spans, samples above 0.01 A)"},
                     {"id": "rc_discharge", "rate": f"randomized setpoints of one minute each (0.5–5C; NCA up to 2C) from the full state to {cutoff[0]} V (paper)"},
                     {"id": "reference", "rate": "1C constant-current discharge to the cut-off after the HPPC tests, every ~20 cycles (label source)"}],
        "temperature_c": ["VERIFY (room temperature, paper; the cell temperature is measured and stored in temperature_mean)"],
        "reference_test": "the '1C Reference' entry's first constant-current discharge at ~1C (mean current within ±25 % of 1C) from the full state left by the previous entry's charge to the cut-off, Coulomb-counted at 1 Hz",
        "measured": ["1C reference discharge capacity (Ah)"],
        "label_formula": f"soh = reference_1C_capacity_ah / {rated} Ah", "label_source": "rpt", "supervision": "sparse_field",
        "label_rule": "forward-only (Spec 12 §3): every randomized-current cycle carries the last 1C reference of its cell that ended before the cycle's charge started; the cycles before the first reference (≈ 20 per cell) carry no label; no age bound",
        "rate_hz": 1, "channels": ["V", "I", "T"],
        "window": {"segment": "cc_cv_charge", "v_lo": window[0], "v_hi": window[1], "coverage_median": coverage, "calibrated_on": DRAFT_DATE,
                   "note": "scan on every 5th labeled cycle; " + extra.get("window_note", "")},
        "units": {"unit": "cell", "cells": cells, "c_rated_ah": rated, "time_axis": "test time in seconds counted from the cell's first 1 Hz sample (the entries' start times are MATLAB string objects scipy cannot decode) — wall-clock dates VERIFY"},
        "segments_qc": {"rows": rows, "labeled_rows": labeled, "references_per_cell": refs_per_cell, "reference_capacity_ah_first_to_last": ref_range},
        "verification": {"entry_structure_read_by_hand": "LFP_CELL_1 entry 59 ('Random Walking'): randomized discharge 1242 s at −7.8 A mean to 2.50 V, 1200 s rest, 1C CC charge 3131 s to 3.60 V, 61 s, CV 364 s, 1800 s rest; entry 58 ('1C Reference'): 1C discharge 3345 s from 3.40 V to 2.50 V = 2.782 Ah, then the 1C CC-CV charge",
                         "cells_vs_paper_table_1": "4 NCA / 2 NMC / 2 LFP as in the paper",
                         "soh_range": soh_range},
        "column_classes": SEGMENT_CLASSES, "scanned_on": DRAFT_DATE,
        "known_issues": ["non-commercial repository licence (CC BY-NC 3.0) — redistribution of derived tables needs the owner's decision",
                         "the first ~20 cycles of every cell are unlabeled (before the first 1C reference)",
                         "identity / label-time columns correlate with the monotone fade (R² up to 0.97 on the two NMC cells) — never features"] + extra.get("issues", []) +
                        ["signature pending: the owner's review of chemistry (paper Table 1), the anode and the licence"],
        "signed": dict(UNSIGNED_W6), "status": f"draft {DRAFT_DATE} (D3 re-scoped) — chemistry from the paper's Table 1; licence CC BY-NC 3.0 (VERIFY); unsigned until the owner reviews",
    }

FACTS["turku_rc_nca"] = _turku("turku_rc_nca", chem="NCA", cathode="NCA", model="Samsung ICR18650-26J, 2.55 Ah, 3.63 V", fmt="18650", rated=2.55, cutoff=(2.75, 4.2), cells=4,
    window=(3.70, 4.10), coverage=0.430, rows=2008, labeled=1850, refs_per_cell={"NCA_CELL_1": 23, "NCA_CELL_2": 19, "NCA_CELL_3": 22, "NCA_CELL_4": 27},
    ref_range={"NCA_CELL_1": "2.524 → 1.692", "NCA_CELL_2": "2.508 → 2.174", "NCA_CELL_3": "2.484 → 2.100", "NCA_CELL_4": "2.522 → 0.466"},
    soh_range="0.20–0.99 (NCA_CELL_4 collapses to 0.466 Ah at the end)",
    extra={"window_note": "the mid-charge window of the SNL NCA set (max coverage 0.63, affine R² 0.49)",
           "issues": ["three cells carry one extra 'Reference Cycles' entry (skipped)", "NCA_CELL_4 ends at soh 0.20 (inside the QC bounds)"]})
FACTS["turku_rc_nmc"] = _turku("turku_rc_nmc", chem="NMC", cathode="NMC", model="Samsung INR21700-40T, 4.0 Ah, 3.6 V", fmt="21700", rated=4.0, cutoff=(2.5, 4.2), cells=2,
    window=(3.80, 4.15), coverage=0.408, rows=1031, labeled=980, refs_per_cell={"NMC_CELL_1": 25, "NMC_CELL_2": 24},
    ref_range={"NMC_CELL_1": "3.902 → 3.339", "NMC_CELL_2": "3.914 → 3.342"}, soh_range="0.83–0.98",
    extra={"window_note": "[3.65, 4.10] V (the SNL NMC window) reaches affine R² 0.97 on these two cells with a near-linear fade; [3.80, 4.15] V (R² 0.35, max coverage 0.44) keeps margin below the 0.995 threshold",
           "issues": ["two cells only: every fold that holds one out trains on one"]})
FACTS["turku_rc_lfp"] = _turku("turku_rc_lfp", chem="LFP", cathode="LFP", model="JGNE JGPFR26650, 3.0 Ah, 3.2 V", fmt="26650", rated=3.0, cutoff=(2.5, 3.6), cells=2,
    window=(3.30, 3.40), coverage=0.446, rows=1114, labeled=1054, refs_per_cell={"LFP_CELL_1": 26, "LFP_CELL_2": 27},
    ref_range={"LFP_CELL_1": "2.783 → 2.525", "LFP_CELL_2": "2.771 → 2.454"}, soh_range="0.82–0.93 (the 3.0 Ah nominal is above the cells' 1C capacity)",
    extra={"window_note": "the K12 LFP indicator span (max coverage 0.73, affine R² 0.63)",
           "issues": ["two cells only", "ic_peak_ah_per_v reaches affine R² 0.96 against soh on these two cells (a feature; below the threshold, watched in the leak inventory)"]})
# <<< dataset-block: turku
LEAK_INVENTORY = {"hust": ["charge_capacity_max"], "oxford": ["charge_capacity_max"], "uconn": ["charge_capacity_max"],
                  "rwth_isea_public": ["capacity_ah"], "nasa": [], "calce_cs2": [], "snl_lfp": [],
                  "snl_nmc": ["charge_capacity_max"], "snl_nca": [],
                  "tongji_nca": [],
                  "tongji_ncm": [],
                  "tongji_blend": [],
                  "forklift_lfp180": [],
                  "matr": [],
                  "turku_rc_nca": [],
                  "turku_rc_nmc": [],
                  "turku_rc_lfp": [],
                  }


def main(argv=None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="draft the datasheets from the facts table, the manifest and the measured provenance")
    parser.add_argument("--data-dir", default=os.environ.get("SOHBENCH_DATA_DIR") or str(ROOT / "data"),
                        help="data root with manifest.json (default: SOHBENCH_DATA_DIR or the repository's data/)")
    parser.add_argument("--out-dir", default=None, help="where the YAML sheets and _provenance.json live (default: <data-dir>/datasheets)")
    parser.add_argument("--only", nargs="+", default=None, help="write only these datasheets (default: every dataset in FACTS)")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = {item["name"]: item for item in manifest.get("datasets", [])}
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "datasheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = out_dir / "_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")).get("datasets", {}) if provenance_path.is_file() else {}
    for name, facts in FACTS.items():
        if args.only and name not in args.only:
            continue
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
        if not measured and (facts.get("rate_hz") is not None or facts.get("channels")):   # Wave 6 sets: sampling from the paper / extractor
            sampling = {"rate_hz": facts.get("rate_hz", V), "channels": facts.get("channels", ["V", "I", "T"])}
        rated = DATASET_C_RATED_AH.get(name)
        window = facts["window"]
        sheet = {
            "name": name,
            "status": facts["status"],
            "supervision": entry.get("supervision") or facts.get("supervision") or "soh_supervised",
            "source": {"citation": facts["citation"], "url": facts["url"], "license": facts["license"], "retrieved": facts["retrieved"]},
            "chemistry": {"cathode": facts["cathode"], "anode": facts["anode"], "metadata_label": DATASET_CHEMISTRY.get(name, "unknown"),
                          "verified_by": facts["verified_by"], "verified_on": facts["verified_on"], "evidence": facts["chemistry_evidence"]},
            "cell": {"format": facts["format"], "rated_capacity_ah": facts.get("rated_capacity_ah", rated if rated == rated else V), "rated_source": facts["rated_source"]},
            "protocol": {"segments": facts["segments"], "temperature_c": facts["temperature_c"]},
            "label_anchor": {"formula": facts.get("label_formula", "soh = calibration_capacity / rated_capacity_ah"),
                             "measured_quantities": facts["measured"], "reference_test": facts["reference_test"],
                             "label_source": facts.get("label_source", "calibration_capacity"),
                             **({"label_rule": facts["label_rule"]} if facts.get("label_rule") else {})},
            "sampling": sampling,
            "windows": [{"feature": "q_win_ah", **window}] if isinstance(window, dict) else ([] if window is None else V),
            **({"segments": facts["segments_qc"]} if facts.get("segments_qc") else {}),
            **({"units": facts["units"]} if facts.get("units") else {}),
            **({"verification": facts["verification"]} if facts.get("verification") else {}),
            "digests": {"raw": measured.get("raw_sha256") or entry.get("raw_sha256") or facts.get("raw_digest") or V,
                        "processed": entry.get("processed_sha256") or (entry.get("processed") or {}).get("sha256") or V,
                        "curves": entry.get("curve_sha256") or ("none (no curves file; the raw curves stay in the archive)" if not entry.get("curve_path") else V)},
            "provenance": {"raw_files": [{"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"]} for r in measured.get("raw_files", [])],
                           "measured_at": measured.get("measured_at", V if not measured else SURVEY_DATE),
                           "reference": "scripts/measure_raw_provenance.py -> data/datasheets/_provenance.json"} if measured else V,
            "leak_scan": {"label_source_columns": LEAK_INVENTORY.get(name, V), "scanned_on": facts.get("scanned_on", "2026-09-03"),
                          "reference": "tests/test_label_leak_inventory.py",
                          **({"column_classes": facts["column_classes"]} if facts.get("column_classes") else {})},
            "held_out_ready": bool(facts.get("held_out_ready", entry.get("held_out_ready", True))),
            "known_issues": facts["known_issues"],
            "signed": dict(facts["signed"]),
        }
        (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(sheet, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print("wrote", out_dir / f"{name}.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
