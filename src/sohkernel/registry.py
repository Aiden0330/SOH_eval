from __future__ import annotations

_ESTIMATORS: dict[str, type] = {}
_DATASETS: dict[str, object] = {}
_TASKS: dict[str, type] = {}
_FEATURESETS: dict[str, object] = {}

def _register(store, key, value):
    if key in store:
        raise ValueError(f"duplicate registration: {key!r}")
    store[key] = value
    return value

def register_estimator(cls):
    return _register(_ESTIMATORS, cls.name, cls)

def get_estimator(name): return _ESTIMATORS[name]

def register_dataset(name):
    def deco(obj): return _register(_DATASETS, name, obj)
    return deco

def get_dataset(name): return _DATASETS[name]

def register_task(cls):
    return _register(_TASKS, cls.name, cls)

def get_task(name): return _TASKS[name]

def register_featureset(fs):
    """Register a FeatureSet instance by its .name attribute."""
    return _register(_FEATURESETS, fs.name, fs)

def get_featureset(name):
    """Retrieve a registered FeatureSet by name."""
    return _FEATURESETS[name]

def clear():
    _ESTIMATORS.clear(); _DATASETS.clear(); _TASKS.clear(); _FEATURESETS.clear()
