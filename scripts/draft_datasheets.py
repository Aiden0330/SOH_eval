"""Draft the Spec 06 datasheets for the datasets under the data root (D6, agent draft).

Every field without a primary source is written as ``VERIFY`` for the human to resolve
(Spec 06 §2).  Sources: sohkernel.metadata (chemistry, rated capacity), data/manifest.json
(license, source, digests), the standing leak inventory (label-source columns), the window
calibration notes (bounds), and the shipped raw readmes where they exist.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sohkernel.metadata import DATASET_CHEMISTRY, DATASET_C_RATED_AH  # noqa: E402

TODAY = date.today().isoformat()
V = "VERIFY"
# what the agent could ground, per dataset; anything absent becomes VERIFY
FACTS = {
    "nasa": {
        "citation": "Saha, B. and Goebel, K. (2007). Battery Data Set. NASA Ames Prognostics Data Repository.",
        "url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
        "cathode": V, "anode": V, "format": "18650", "rated_source": "documentation (2 Ah nominal)",
        "chemistry_evidence": (
            "VERIFY: the shipped PCoE READMEs (data/raw/nasa_full/README*.txt) name neither chemistry nor "
            "manufacturer; metadata.py and data/manifest.json say LCO (literature attribution) while "
            "data/curves/nasa.json says 'NCA (Gen2 18650)'. Resolve against a primary source before any chemistry claim."
        ),
        "segments": [{"id": "cc_cv_charge", "rate": "1.5 A CC to 4.2 V, CV to 20 mA"}, {"id": "discharge", "rate": "2 A (B0005-7, B0018) to 2.7/2.5/2.2 V"}],
        "temperature_c": [24], "reference_test": "capacity from each full discharge (documented per README)",
        "measured": ["discharge capacity (Ah)"], "window": None, "known_issues": ["cell B0018 is the phase 2.2 golden case; 18 non-positive SOH rows dropped by QC"],
    },
    "calce_cs2": {
        "citation": "CALCE Battery Research Group, University of Maryland: CS2/CX2 prismatic LiCoO2 cells; He, W. et al. (2011) Journal of Power Sources 196(23).",
        "url": "https://calce.umd.edu/battery-data", "cathode": "LCO", "anode": "graphite", "format": "prismatic",
        "rated_source": "CALCE data page (CS2 1100 mAh; CX2 1350 mAh)",
        "chemistry_evidence": "CALCE data page states LiCoO2 cathode for CS2/CX2 (VERIFY: confirm the page and sign)",
        "segments": [{"id": "cc_charge", "rate": "0.5C CC to 4.2 V, CV to 0.05 A"}, {"id": "discharge", "rate": "1C to 2.7 V"}],
        "temperature_c": [25], "reference_test": "capacity of each 1C discharge (Arbin file)",
        "measured": ["discharge capacity (Ah)"], "window": {"segment": "cc_charge", "v_lo": 3.85, "v_hi": 4.05, "coverage_median": 0.371, "calibrated_on": "2026-08-27"},
        "known_issues": ["charge_capacity_max in the processed table is the file-cumulative Arbin quantity, not per cycle (report finding G6)"],
    },
    "hust": {
        "citation": "Ma, G. et al. (2022). Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning. Energy & Environmental Science; dataset on Mendeley Data.",
        "url": V, "cathode": "LFP", "anode": "graphite", "format": "18650 (A123 APR18650M1A)", "rated_source": "publication (1.1 Ah)",
        "chemistry_evidence": "publication states A123 LFP 18650 cells (VERIFY: confirm and sign)",
        "segments": [{"id": "cc_charge", "rate": "5C CC (multi-stage) to 3.6 V"}, {"id": "discharge", "rate": "1C to 2.0 V"}],
        "temperature_c": [30], "reference_test": "capacity of each discharge", "measured": ["charge_capacity_max"],
        "window": {"segment": "cc_charge", "v_lo": 3.30, "v_hi": 3.52, "coverage_median": 0.486, "calibrated_on": "2026-08-25"}, "known_issues": [],
    },
    "oxford": {
        "citation": "Birkl, C. R. (2017). Diagnosis and Prognosis of Degradation in Lithium-Ion Batteries. PhD thesis, University of Oxford; Oxford Battery Degradation Dataset 1 (ORA).",
        "url": "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac", "cathode": "LCO/NCO", "anode": "graphite",
        "format": "pouch (Kokam SLPB533459H4, 740 mAh)", "rated_source": "shipped Readme.txt (Test subjects: 8 x Kokam SLPB533459H4, 740mAh)",
        "chemistry_evidence": "dataset description / Birkl 2017: LCO/NCO cathode, graphite anode (VERIFY: confirm and sign)",
        "segments": [{"id": "cc_charge", "rate": "1C (740 mA) characterisation every 100 drive cycles"}, {"id": "pseudo_ocv", "rate": "C/18.5 (40 mA)"}],
        "temperature_c": [40], "reference_test": "1C characterisation discharge every 100 Artemis drive cycles", "measured": ["charge_capacity_max"],
        "window": V, "known_issues": ["519 characterisation points only; coverage median 0.594 of rated (wide window warning)"],
    },
    "snl_lfp": {
        "citation": "Preger, Y. et al. (2020). Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions. J. Electrochem. Soc. 167; BatteryArchive / BatteryLife mirror.",
        "url": "https://www.batteryarchive.org/", "cathode": "LFP", "anode": "graphite", "format": "18650 (A123 APR18650M1A)", "rated_source": "publication (1.1 Ah)",
        "chemistry_evidence": "Preger 2020 LFP subset (VERIFY: confirm and sign)",
        "segments": [{"id": "cc_charge", "rate": "0.5C CC to 3.6 V"}, {"id": "discharge", "rate": "0.5C-3C to 2.0 V"}],
        "temperature_c": [15, 25, 35], "reference_test": "capacity of each discharge", "measured": ["separate calibration measurement (no label-source column)"],
        "window": {"segment": "cc_charge", "v_lo": 3.30, "v_hi": 3.38, "coverage_median": 0.449, "calibrated_on": "2026-08-25"}, "known_issues": [],
    },
    "uconn": {
        "citation": "Nowacki, B. et al. (2025). UConn-ISU-ILCC LFP/graphite ageing dataset.", "url": V, "cathode": "LFP", "anode": "graphite",
        "format": "18650 (LithiumWerks APR18650M1B)", "rated_source": "data/curves/uconn.json (1.1 Ah)",
        "chemistry_evidence": "dataset description: LithiumWerks APR18650M1B LFP (VERIFY: confirm and sign)",
        "segments": [{"id": "rpt_charge", "rate": "1C reference charge"}], "temperature_c": [V], "reference_test": "RPT reference discharge", "measured": ["charge_capacity_max"],
        "window": {"segment": "rpt_charge", "v_lo": 3.30, "v_hi": 3.37, "coverage_median": 0.536, "calibrated_on": "2026-08-25"},
        "known_issues": ["64 rows in 32 duplicated (cell, cycle) identities are dropped by QC since 2026-09-03"],
    },
    "rwth_isea_public": {
        "citation": "RWTH Aachen ISEA public residential storage battery data.", "url": V, "cathode": "LFP", "anode": "graphite",
        "format": V, "rated_source": V, "chemistry_evidence": "metadata.py: Fortelion LFP (VERIFY: nameplate unconfirmed)",
        "segments": [{"id": V, "rate": V}], "temperature_c": [V], "reference_test": V, "measured": ["capacity_ah"], "window": V,
        "known_issues": ["raw curves not local; nameplate capacity unknown (c_rated_ah NaN)"],
    },
}
LEAK_INVENTORY = {"hust": ["charge_capacity_max"], "oxford": ["charge_capacity_max"], "uconn": ["charge_capacity_max"],
                  "rwth_isea_public": ["capacity_ah"], "nasa": [], "calce_cs2": [], "snl_lfp": []}


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
            "status": "draft (agent, 2026-09-03) — fields marked VERIFY await the owner",
            "source": {"citation": facts["citation"], "url": facts["url"], "license": entry.get("license") or V, "retrieved": V},
            "chemistry": {"cathode": facts["cathode"], "anode": facts["anode"], "metadata_label": DATASET_CHEMISTRY.get(name, "unknown"),
                          "verified_by": V, "verified_on": V, "evidence": facts["chemistry_evidence"]},
            "cell": {"format": facts["format"], "rated_capacity_ah": rated if rated == rated else V, "rated_source": facts["rated_source"]},
            "protocol": {"segments": facts["segments"], "temperature_c": facts["temperature_c"]},
            "label_anchor": {"formula": "soh = calibration_capacity / rated_capacity_ah", "measured_quantities": facts["measured"],
                             "reference_test": facts["reference_test"], "label_source": "calibration_capacity"},
            "sampling": {"rate_hz": V, "channels": ["V", "I", "T"]},
            "windows": [{"feature": "q_win_ah", **window}] if isinstance(window, dict) else V,
            "digests": {"raw": entry.get("raw_sha256") or V, "processed": entry.get("processed_sha256") or entry.get("sha256") or V,
                        "curves": entry.get("curve_sha256") or V},
            "leak_scan": {"label_source_columns": LEAK_INVENTORY.get(name, V), "scanned_on": "2026-09-03",
                          "reference": "tests/test_label_leak_inventory.py"},
            "held_out_ready": bool(entry.get("held_out_ready", True)),
            "known_issues": facts["known_issues"],
            "signed": {"by": V, "on": V},
        }
        (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(sheet, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print("wrote", out_dir / f"{name}.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
