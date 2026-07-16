import numpy as np
import pandas as pd
import pytest
from sohbench.contracts import (
    ID_COLUMNS, feature_columns, CellMeta, Fold, QCReport, Verdict, Estimator,
    validate_finite_metrics,
)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        "0.1",
        0.1 + 0j,
        np.array([]),
        np.array([0.1]),
        np.array([0.1, 0.2]),
        np.array([[0.1]]),
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_finite_metrics_rejects_non_real_scalar_values(value):
    with pytest.raises(ValueError, match="finite non-boolean real scalar"):
        validate_finite_metrics({"mae": value})


@pytest.mark.parametrize("value", [np.float32(0.25), np.int64(2)])
def test_validate_finite_metrics_normalizes_numpy_real_scalars(value):
    metrics = validate_finite_metrics({"mae": value})

    assert metrics == {"mae": float(value)}
    assert type(metrics["mae"]) is float

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
