from __future__ import annotations
import pandas as pd

DATASET_CHEMISTRY = {
    "nasa": "LCO",       # NASA PCoE 18650, 2 Ah — literature attribution only, chemistry UNVERIFIED (datasheet D4 2026-09-03)
    "calce_cs2": "LCO",  # CALCE CS2 prismatic LiCoO2
    "hust": "LFP",       # HUST A123 18650 LFP
    "oxford": "LCO/NCO", # Oxford ORA Kokam pouch, graphite/LCO-NCO (was "NMC")
    "snl_lfp": "LFP",    # SNL LFP subset from BatteryLife/BatteryArchive mirror
    "uconn": "LFP",      # UConn-ISU-ILCC LFP/graphite dataset
    "rwth_isea_public": "NMC",  # RWTH ISEA one-shot dataset: Sanyo/Panasonic UR18650E NMC/graphite (DOI 10.18154/RWTH-2021-04545); was mislabelled LFP until 2026-09-03
    "snl_nmc": "NMC",    # SNL NMC subset (LG Chem 18650HG2) from the BatteryLife/BatteryArchive mirror; pickle cathode_material NMC + JES 167 120532 (D1, 2026-09-04)
    "snl_nca": "NCA",    # SNL NCA subset (Panasonic NCR18650B) from the BatteryLife/BatteryArchive mirror; pickle cathode_material NCA + JES 167 120532 (D1, 2026-09-04)
    "tongji_nca": "NCA",   # Zhu et al. 2022 Dataset_1: LG INR18650-35E, LiNi0.86Co0.11Al0.03O2 (paper Methods; D2, 2026-09-05)
    "tongji_ncm": "NMC",   # Zhu et al. 2022 Dataset_2: Samsung INR18650-MJ1, Li(Ni0.83Co0.11Mn0.07)O2 — the authors write NCM (D2)
    "tongji_blend": "NMC/NCA",   # Zhu et al. 2022 Dataset_3: Samsung INR18650-25R, 42 % NCM + 58 % NCA blend cathode (D2); HI windows resolve on NMC
    "forklift_lfp180": "LFP",    # Vilsen & Stroe 2024: prismatic LFP/graphite 180 Ah (paper Table 3; D8, 2026-09-05)
    "matr": "LFP",               # Severson 2019 / Attia 2020: A123 APR18650M1A LFP/graphite 1.1 Ah (D9, 2026-09-05)
    "turku_rc_nca": "NCA",       # Heydarzadeh 2025 Table 1: Samsung ICR18650-26J (NCA) (D3, 2026-09-05)
    "turku_rc_nmc": "NMC",       # Heydarzadeh 2025 Table 1: Samsung INR21700-40T (NMC) (D3)
    "turku_rc_lfp": "LFP",       # Heydarzadeh 2025 Table 1: JGNE JGPFR26650 (LFP) (D3)
}
DATASET_C_RATED_AH = {
    "nasa": 2.0,      # NASA PCoE 18650 LiCoO2, 2 Ah
    "calce_cs2": 1.1, # CALCE CS2 prismatic LiCoO2
    "hust": 1.1,      # HUST A123 18650 LFP
    "oxford": 0.74,   # Oxford ORA Kokam pouch, graphite/LCO-NCO
    "snl_lfp": 1.1,   # SNL/BatteryLife A123 18650 LFP (data/curves/snl_lfp.json)
    "uconn": 1.1,     # UConn-ISU-ILCC LithiumWerks APR18650M1B LFP (data/curves/uconn.json)
    "rwth_isea_public": 1.85,  # UR18650E nominal 1.85 Ah (RWTH-2021-04545 record; table max 1.845 Ah)
    "snl_nmc": 3.0,   # LG Chem 18650HG2 nominal 3.0 Ah (JES 167 120532; BatteryLife pickle nominal_capacity_in_Ah, 15 degC cells derated to 2.7 there — not used)
    "snl_nca": 3.2,   # Panasonic NCR18650B nominal 3.2 Ah (JES 167 120532; BatteryLife pickle nominal_capacity_in_Ah, 15 degC cells derated to 2.88 there — not used)
    "tongji_nca": 3.5,       # Zhu et al. 2022 Table 1 (1C = 3.5 A)
    "tongji_ncm": 3.5,       # Zhu et al. 2022 Table 1 (1C = 3.5 A)
    "tongji_blend": 2.5,     # Zhu et al. 2022 Table 1 (1C = 2.5 A)
    "forklift_lfp180": 180.0,   # Vilsen & Stroe 2024 Table 3 (nominal 180 Ah)
    "matr": 1.1,             # A123 APR18650M1A nominal 1.1 Ah (Severson 2019)
    "turku_rc_nca": 2.55,    # Heydarzadeh 2025 Table 1 (Q_nom 2.55 Ah)
    "turku_rc_nmc": 4.0,     # Heydarzadeh 2025 Table 1 (Q_nom 4.0 Ah)
    "turku_rc_lfp": 3.0,     # Heydarzadeh 2025 Table 1 (Q_nom 3.0 Ah)
}
DATASET_LABEL_SOURCE = {  # all reference-discharge calibrated capacity
    "nasa": "calibration_capacity", "calce_cs2": "calibration_capacity",
    "hust": "calibration_capacity", "oxford": "calibration_capacity",
    "snl_lfp": "calibration_capacity", "uconn": "calibration_capacity",
    "rwth_isea_public": "calibration_capacity",
    "snl_nmc": "calibration_capacity", "snl_nca": "calibration_capacity",
    "tongji_nca": "calibration_capacity", "tongji_ncm": "calibration_capacity", "tongji_blend": "calibration_capacity",
    "matr": "calibration_capacity",
    "forklift_lfp180": "rpt",                 # forklift_lfp180: RPT capacities attached forward-only to segments (Spec 12 §3)
    "turku_rc_nca": "rpt", "turku_rc_nmc": "rpt", "turku_rc_lfp": "rpt",   # turku_rc: 1C reference discharges attached forward-only
}
# Supervision kinds (Spec 12 §4, K13).  ``soh_supervised``: every row labeled (the laboratory
# sets); ``sparse_field``: labels from capacity tests / RPTs attached forward-only to a subset of
# the segments; ``unlabeled_field``: no labels at all — inference, audit and drift only, refused
# by the training preflight and never claimed.
SUPERVISION_KINDS = ("soh_supervised", "sparse_field", "unlabeled_field")
SUPERVISION_ROLES = {
    "soh_supervised": ["training", "same_dataset_holdout", "inference", "audit"],
    "sparse_field": ["training", "inference", "audit", "drift"],
    "unlabeled_field": ["inference", "audit", "drift"],
}
DATASET_SUPERVISION: dict[str, str] = {   # every dataset not listed is soh_supervised
    "forklift_lfp180": "sparse_field",    # D8: RPT after every ageing round; every segment labeled by the preceding RPT
    "turku_rc_nca": "sparse_field", "turku_rc_nmc": "sparse_field", "turku_rc_lfp": "sparse_field",   # turku_rc: a 1C reference every ~20 cycles
}


def supervision_kind(name: object) -> str:
    return DATASET_SUPERVISION.get(canonical_dataset_name(name), "soh_supervised")


# Raw ingestion frames label datasets the way their source did (``CALCE``, ``SNL``,
# ``HUST`` ...); the loader rewrites the column to the canonical file-derived name, and
# metadata lookups on raw frames go through :func:`canonical_dataset_name`.
DATASET_ALIASES = {"calce": "calce_cs2", "snl": "snl_lfp"}


def canonical_dataset_name(name: object) -> str:
    key = str(name).strip().lower()
    return DATASET_ALIASES.get(key, key)


# Cell-prefix overrides for datasets that mix cell formats: CALCE CX2 cells
# are 1.35 Ah prismatic vs the CS2 series' 1.1 Ah.
CELL_PREFIX_C_RATED_AH = {
    ("calce_cs2", "CX2_"): 1.35,
}

def build_meta(table: pd.DataFrame) -> pd.DataFrame:
    cells = table[["dataset", "cell_id"]].drop_duplicates().reset_index(drop=True)
    cells["chemistry"] = cells["dataset"].map(DATASET_CHEMISTRY).fillna("unknown")
    cells["c_rated_ah"] = cells["dataset"].map(DATASET_C_RATED_AH)
    for (dataset, prefix), rated in CELL_PREFIX_C_RATED_AH.items():
        mask = (cells["dataset"] == dataset) & cells["cell_id"].astype(str).str.startswith(prefix)
        cells.loc[mask, "c_rated_ah"] = rated
    cells["label_source"] = cells["dataset"].map(DATASET_LABEL_SOURCE).fillna("unknown")
    cells["supervision"] = cells["dataset"].map(supervision_kind)
    return cells
