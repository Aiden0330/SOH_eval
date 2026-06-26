import pytest
from sohbench import registry as R

@pytest.fixture(autouse=True)
def _isolate_registry():
    snap = (dict(R._ESTIMATORS), dict(R._DATASETS), dict(R._TASKS), dict(R._FEATURESETS))
    yield
    for store, snapped in zip((R._ESTIMATORS, R._DATASETS, R._TASKS, R._FEATURESETS), snap):
        store.clear(); store.update(snapped)
