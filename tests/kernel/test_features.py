import pandas as pd
from sohkernel.features import unified_features

def test_unified_features_excludes_ids():
    df = pd.DataFrame({"dataset":["a"],"cell_id":["c"],"cycle_index":[0],
                       "soh":[1.0],"f1":[0.5],"f2":[0.3]})
    X, names = unified_features(df)
    assert names == ["f1","f2"]
    assert list(X.columns) == ["f1","f2"]
