from __future__ import annotations
import pandas as pd

DATASET_CHEMISTRY = {
    "nasa": "LCO",       # NASA PCoE 18650 LiCoO2, 2 Ah
    "calce_cs2": "LCO",  # CALCE CS2 prismatic LiCoO2
    "hust": "LFP",       # HUST A123 18650 LFP
    "oxford": "LCO/NCO", # Oxford ORA Kokam pouch, graphite/LCO-NCO (was "NMC")
}
DATASET_C_RATED_AH = {
    "nasa": 2.0,      # NASA PCoE 18650 LiCoO2, 2 Ah
    "calce_cs2": 1.1, # CALCE CS2 prismatic LiCoO2
    "hust": 1.1,      # HUST A123 18650 LFP
    "oxford": 0.74,   # Oxford ORA Kokam pouch, graphite/LCO-NCO
}
DATASET_LABEL_SOURCE = {  # all reference-discharge calibrated capacity
    "nasa": "calibration_capacity", "calce_cs2": "calibration_capacity",
    "hust": "calibration_capacity", "oxford": "calibration_capacity",
}

def build_meta(table: pd.DataFrame) -> pd.DataFrame:
    cells = table[["dataset", "cell_id"]].drop_duplicates().reset_index(drop=True)
    cells["chemistry"] = cells["dataset"].map(DATASET_CHEMISTRY)
    cells["c_rated_ah"] = cells["dataset"].map(DATASET_C_RATED_AH)
    cells["label_source"] = cells["dataset"].map(DATASET_LABEL_SOURCE).fillna("unknown")
    return cells
