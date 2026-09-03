from __future__ import annotations
import pandas as pd

DATASET_CHEMISTRY = {
    "nasa": "LCO",       # NASA PCoE 18650 LiCoO2, 2 Ah
    "calce_cs2": "LCO",  # CALCE CS2 prismatic LiCoO2
    "hust": "LFP",       # HUST A123 18650 LFP
    "oxford": "LCO/NCO", # Oxford ORA Kokam pouch, graphite/LCO-NCO (was "NMC")
    "snl_lfp": "LFP",    # SNL LFP subset from BatteryLife/BatteryArchive mirror
    "uconn": "LFP",      # UConn-ISU-ILCC LFP/graphite dataset
    "rwth_isea_public": "LFP",  # RWTH ISEA residential storage cells (Fortelion LFP)
}
DATASET_C_RATED_AH = {
    "nasa": 2.0,      # NASA PCoE 18650 LiCoO2, 2 Ah
    "calce_cs2": 1.1, # CALCE CS2 prismatic LiCoO2
    "hust": 1.1,      # HUST A123 18650 LFP
    "oxford": 0.74,   # Oxford ORA Kokam pouch, graphite/LCO-NCO
    "snl_lfp": 1.1,   # SNL/BatteryLife A123 18650 LFP (data/curves/snl_lfp.json)
    "uconn": 1.1,     # UConn-ISU-ILCC LithiumWerks APR18650M1B LFP (data/curves/uconn.json)
    "rwth_isea_public": float("nan"),  # field storage cells; nameplate unconfirmed
}
DATASET_LABEL_SOURCE = {  # all reference-discharge calibrated capacity
    "nasa": "calibration_capacity", "calce_cs2": "calibration_capacity",
    "hust": "calibration_capacity", "oxford": "calibration_capacity",
    "snl_lfp": "calibration_capacity", "uconn": "calibration_capacity",
    "rwth_isea_public": "calibration_capacity",
}
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
    return cells
