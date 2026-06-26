import numpy as np
import pandas as pd
from sohbench.contracts import (
    ID_COLUMNS, feature_columns, CellMeta, Fold, QCReport, Verdict, Estimator,
)

def test_feature_columns_excludes_ids():
    df = pd.DataFrame({"dataset": ["a"], "cell_id": ["c1"], "cycle_index": [0],
                       "soh": [1.0], "f1": [0.5], "f2": [0.3]})
    assert feature_columns(df) == ["f1", "f2"]

def test_cellmeta_fields():
    m = CellMeta(dataset="a", cell_id="c1", chemistry="LFP", c_rated_ah=1.1,
                 label_source="calibration_capacity")
    assert m.chemistry == "LFP" and m.c_rated_ah == 1.1

def test_estimator_protocol_runtime_checkable():
    class Dummy:
        name = "d"; feature_mode = "unified"; backend = "sklearn"
        def fit(self, X, y, meta): ...
        def predict(self, X, meta): return np.zeros(len(X))
    assert isinstance(Dummy(), Estimator)
