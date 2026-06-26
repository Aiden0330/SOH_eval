from __future__ import annotations
import pandas as pd

# VERIFY each against source docs before trusting chemistry-out results.
DATASET_CHEMISTRY = {
    "nasa": "LCO",       # NASA PCoE 18650 (LiCoO2)  # VERIFY
    "calce_cs2": "LCO",  # CALCE CS2 prismatic LCO    # VERIFY
    "hust": "LFP",       # HUST A123 18650 LFP        # VERIFY
    "oxford": "NMC",     # Oxford Kokam pouch NMC      # VERIFY
}
DATASET_C_RATED_AH = {"nasa": 2.0, "calce_cs2": 1.1, "hust": 1.1, "oxford": 0.74}  # VERIFY
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
