import json
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sohbench import registry as R
import sohbench.trusted_plugins as trusted_plugins_module
from sohbench.data import DatasetLoader
from sohbench.registry import get_estimator, get_featureset
from sohbench.trusted_plugins import TrustedPluginError, TrustedPluginStore, _load_module


def _dataset_frame(name="public_demo", cells=("c1", "c2", "c3"), n=8):
    rows = []
    for cell in cells:
        for cycle in range(n):
            soh = 1.0 - 0.01 * cycle
            rows.append(
                {
                    "dataset": name,
                    "cell_id": cell,
                    "cycle_index": cycle,
                    "soh": soh,
                    "voltage_mean": 3.4 - 0.01 * cycle,
                    "resistance_proxy": 0.02 + 0.001 * cycle,
                }
            )
    return pd.DataFrame(rows)


def test_upload_dataset_validates_and_persists_normalized_pickle(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    payload = _dataset_frame("ignored_name").to_csv(index=False).encode("utf-8")

    registered = store.upload_dataset(
        name="public_demo",
        filename="demo.csv",
        payload=payload,
    )

    saved_path = tmp_path / "data" / "processed" / "public_demo_features.pkl"
    saved = pd.read_pickle(saved_path)
    assert registered["name"] == "public_demo"
    assert registered["rows"] == len(saved)
    assert saved["dataset"].unique().tolist() == ["public_demo"]
    assert DatasetLoader(tmp_path / "data").discover_datasets() == ["public_demo"]


def test_upload_and_catalog_preserve_unlabeled_inference_dataset(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    payload = _dataset_frame("ignored_name").drop(columns=["soh"]).to_csv(index=False).encode("utf-8")

    registered = store.upload_dataset(
        name="unlabeled_demo",
        filename="unlabeled.csv",
        payload=payload,
    )
    catalog_item = next(
        item for item in store.catalog()["datasets"] if item["name"] == "unlabeled_demo"
    )

    assert registered["has_labels"] is False
    assert catalog_item["has_labels"] is False
    assert DatasetLoader(tmp_path / "data").discover_datasets() == []
    assert DatasetLoader(tmp_path / "data").discover_datasets(require_labels=False) == [
        "unlabeled_demo"
    ]


def test_dataset_replacement_waits_for_execution_lease(
    tmp_path, monkeypatch
):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    original = _dataset_frame("lease_demo")
    replacement = original.copy()
    replacement["voltage_mean"] = 9.9
    resource = store.upload_dataset(
        "lease_demo",
        "lease.csv",
        original.to_csv(index=False).encode("utf-8"),
    )
    dataset_path = Path(resource["path"])
    write_entered = threading.Event()
    release_write = threading.Event()
    replacement_started = threading.Event()
    replacement_finished = threading.Event()
    errors = []
    real_to_pickle = pd.DataFrame.to_pickle

    def blocking_to_pickle(frame, path, *args, **kwargs):
        result = real_to_pickle(frame, path, *args, **kwargs)
        write_entered.set()
        if not release_write.wait(timeout=5):
            raise TimeoutError("replacement pickle write was not released")
        return result

    monkeypatch.setattr(pd.DataFrame, "to_pickle", blocking_to_pickle)

    def replace_dataset():
        replacement_started.set()
        try:
            store.upload_dataset(
                "lease_demo",
                "lease.csv",
                replacement.to_csv(index=False).encode("utf-8"),
                replace=True,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            replacement_finished.set()

    worker = threading.Thread(target=replace_dataset)
    entered_while_leased = None
    observed_while_leased = None
    finished_while_leased = None
    try:
        with store.execution_lease():
            worker.start()
            assert replacement_started.wait(timeout=5)
            entered_while_leased = write_entered.wait(timeout=0.4)
            observed_while_leased = pd.read_pickle(dataset_path)
            finished_while_leased = replacement_finished.is_set()
            release_write.set()
    finally:
        release_write.set()
        if worker.ident is not None:
            worker.join(timeout=5)

    assert not worker.is_alive()
    assert errors == []
    assert entered_while_leased is False
    assert finished_while_leased is False
    assert observed_while_leased["voltage_mean"].equals(
        original["voltage_mean"]
    )
    saved = pd.read_pickle(dataset_path)
    assert saved["voltage_mean"].eq(9.9).all()


def test_dataset_replacement_publishes_pickle_atomically(tmp_path, monkeypatch):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    original = _dataset_frame("atomic_demo")
    replacement = original.copy()
    replacement["voltage_mean"] = 8.8
    resource = store.upload_dataset(
        "atomic_demo",
        "atomic.csv",
        original.to_csv(index=False).encode("utf-8"),
    )
    dataset_path = Path(resource["path"])
    partial_written = threading.Event()
    release_write = threading.Event()
    errors = []
    real_to_pickle = pd.DataFrame.to_pickle

    def partial_then_pickle(frame, path, *args, **kwargs):
        Path(path).write_bytes(b"partial-pickle")
        partial_written.set()
        if not release_write.wait(timeout=5):
            raise TimeoutError("partial pickle write was not released")
        return real_to_pickle(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_pickle", partial_then_pickle)

    def replace_dataset():
        try:
            store.upload_dataset(
                "atomic_demo",
                "atomic.csv",
                replacement.to_csv(index=False).encode("utf-8"),
                replace=True,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=replace_dataset)
    try:
        worker.start()
        assert partial_written.wait(timeout=5)
        try:
            observed = pd.read_pickle(dataset_path)
        except BaseException as exc:
            observed = exc
    finally:
        release_write.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert errors == []
    assert isinstance(observed, pd.DataFrame)
    assert observed["voltage_mean"].equals(original["voltage_mean"])
    saved = pd.read_pickle(dataset_path)
    assert saved["voltage_mean"].eq(8.8).all()


def test_concurrent_dataset_catalog_updates_are_not_lost(
    tmp_path, monkeypatch
):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    first_staged = threading.Event()
    second_finished = threading.Event()
    call_lock = threading.Lock()
    call_count = 0
    errors = []
    real_stage_catalog = store._stage_catalog

    def delayed_stage_catalog(document):
        nonlocal call_count
        staged = real_stage_catalog(document)
        with call_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_staged.set()
            second_finished.wait(timeout=0.6)
        return staged

    monkeypatch.setattr(store, "_stage_catalog", delayed_stage_catalog)

    def upload(name):
        try:
            store.upload_dataset(
                name,
                f"{name}.csv",
                _dataset_frame(name).to_csv(index=False).encode("utf-8"),
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            if name == "second_demo":
                second_finished.set()

    first = threading.Thread(target=upload, args=("first_demo",))
    second = threading.Thread(target=upload, args=("second_demo",))
    first.start()
    assert first_staged.wait(timeout=5)
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    persisted = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    assert {item["name"] for item in persisted["datasets"]} == {
        "first_demo",
        "second_demo",
    }


def test_dataset_catalog_activation_failure_rolls_back_replacement(
    tmp_path, monkeypatch
):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    original = _dataset_frame("rollback_demo")
    replacement = original.copy()
    replacement["voltage_mean"] = 7.7
    resource = store.upload_dataset(
        "rollback_demo",
        "rollback.csv",
        original.to_csv(index=False).encode("utf-8"),
    )
    dataset_path = Path(resource["path"])
    original_pickle = dataset_path.read_bytes()
    original_catalog = store.catalog_path.read_bytes()
    real_replace = trusted_plugins_module.os.replace

    def fail_catalog_activation(source, destination):
        if Path(destination) == store.catalog_path:
            raise OSError("injected dataset catalog activation failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        trusted_plugins_module.os, "replace", fail_catalog_activation
    )

    with pytest.raises(
        OSError, match="injected dataset catalog activation failure"
    ):
        store.upload_dataset(
            "rollback_demo",
            "rollback.csv",
            replacement.to_csv(index=False).encode("utf-8"),
            replace=True,
        )

    assert dataset_path.read_bytes() == original_pickle
    assert store.catalog_path.read_bytes() == original_catalog
    assert pd.read_pickle(dataset_path)["voltage_mean"].equals(
        original["voltage_mean"]
    )


@pytest.mark.parametrize(
    "failure_location",
    [
        pytest.param("dataset_directory", id="dataset-directory-fsync"),
        pytest.param("catalog_directory", id="catalog-directory-fsync"),
    ],
)
def test_dataset_directory_fsync_failure_rolls_back_complete_transaction(
    tmp_path, monkeypatch, failure_location
):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    original = _dataset_frame("durability_demo")
    replacement = original.copy()
    replacement["voltage_mean"] = 6.6
    resource = store.upload_dataset(
        "durability_demo",
        "durability.csv",
        original.to_csv(index=False).encode("utf-8"),
    )
    dataset_path = Path(resource["path"])
    original_pickle = dataset_path.read_bytes()
    original_catalog_bytes = store.catalog_path.read_bytes()
    original_catalog = json.loads(json.dumps(store._catalog))
    original_estimators = dict(R._ESTIMATORS)
    original_featuresets = dict(R._FEATURESETS)
    original_feature_plugins = dict(store._feature_plugins)
    failure_target = (
        dataset_path.parent
        if failure_location == "dataset_directory"
        else store.base_dir
    )
    fsync_calls = []

    def fail_selected_directory(path):
        canonical = Path(path)
        fsync_calls.append(canonical)
        if canonical == failure_target:
            raise OSError(
                "injected post-replacement directory fsync failure"
            )

    monkeypatch.setattr(
        trusted_plugins_module,
        "_fsync_directory",
        fail_selected_directory,
    )

    with pytest.raises(
        OSError, match="post-replacement directory fsync failure"
    ):
        store.upload_dataset(
            "durability_demo",
            "durability.csv",
            replacement.to_csv(index=False).encode("utf-8"),
            replace=True,
        )

    assert failure_target in fsync_calls
    assert dataset_path.read_bytes() == original_pickle
    assert store.catalog_path.read_bytes() == original_catalog_bytes
    assert store._catalog == original_catalog
    assert R._ESTIMATORS == original_estimators
    assert R._FEATURESETS == original_featuresets
    assert store._feature_plugins == original_feature_plugins
    assert pd.read_pickle(dataset_path)["voltage_mean"].equals(
        original["voltage_mean"]
    )
    assert not list(
        dataset_path.parent.glob(f".{dataset_path.name}.rollback-*")
    )
    assert not list(
        store.base_dir.glob(f".{store.catalog_path.name}.rollback-*")
    )
    assert not list(
        dataset_path.parent.glob(f".{dataset_path.name}.tmp-*")
    )
    assert not list(
        store.base_dir.glob(f".{store.catalog_path.name}.tmp-*")
    )


@pytest.mark.parametrize(
    "bad_cycle",
    [1.2, np.nan, np.inf, -np.inf, True, -1, 2**63, float(2**63)],
)
def test_upload_dataset_rejects_non_integral_cycle_index(tmp_path, bad_cycle):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data",
    )
    frame = _dataset_frame()
    frame["cycle_index"] = frame["cycle_index"].astype(object)
    frame.loc[1, "cycle_index"] = bad_cycle

    with pytest.raises(
        TrustedPluginError, match="cycle_index must contain finite integers",
    ):
        store.upload_dataset(
            "invalid_cycles",
            "invalid.csv",
            frame.to_csv(index=False).encode("utf-8"),
        )


def test_upload_estimator_plugin_smoke_tests_and_registers(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    code = b'''
import numpy as np

class Estimator:
    name = "trusted_mean_estimator"
    feature_mode = "unified"
    backend = "external"

    def fit(self, X, y, meta=None):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X, meta=None):
        return np.full(len(X), self.mean_, dtype=float)
'''

    registered = store.upload_estimator_plugin(
        name="mean_estimator",
        filename="mean_estimator.py",
        payload=code,
    )

    cls = get_estimator("trusted_mean_estimator")
    est = cls()
    sample = pd.DataFrame({"dataset": ["d"] * 2, "cell_id": ["c"] * 2, "cycle_index": [0, 1], "soh": [1.0, 0.9], "f": [1, 2]})
    est.fit(sample, sample["soh"].to_numpy(), None)
    assert registered["name"] == "trusted_mean_estimator"
    assert est.predict(sample, None).shape == (2,)
    assert any(item["name"] == "trusted_mean_estimator" for item in store.catalog()["estimators"])


def test_upload_estimator_plugin_rejects_non_vector_predictions(tmp_path):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    code = b'''
import numpy as np

class Estimator:
    name = "trusted_column_prediction"

    def fit(self, X, y, meta=None):
        return self

    def predict(self, X, meta=None):
        return np.full((len(X), 1), 0.9)
'''

    with pytest.raises(TrustedPluginError, match="one-dimensional"):
        store.upload_estimator_plugin(
            name="column_prediction",
            filename="column_prediction.py",
            payload=code,
        )


def test_upload_estimator_plugin_rejects_non_finite_native_intervals(tmp_path):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    code = b'''
import numpy as np

class Estimator:
    name = "trusted_non_finite_interval"

    def fit(self, X, y, meta=None):
        return self

    def predict(self, X, meta=None):
        return np.full(len(X), 0.9)

    def predict_interval(self, X, meta=None):
        return np.full(len(X), 0.8), np.full(len(X), np.inf)
'''

    with pytest.raises(TrustedPluginError, match="interval_high.*finite"):
        store.upload_estimator_plugin(
            name="non_finite_interval",
            filename="non_finite_interval.py",
            payload=code,
        )


def test_upload_estimator_plugin_rejects_inverted_native_intervals(tmp_path):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    code = b'''
import numpy as np

class Estimator:
    name = "trusted_inverted_interval"

    def fit(self, X, y, meta=None):
        return self

    def predict(self, X, meta=None):
        return np.full(len(X), 0.9)

    def predict_interval(self, X, meta=None):
        low = np.full(len(X), 0.8)
        low[-1] = 1.1
        return low, np.full(len(X), 1.0)
'''

    with pytest.raises(
        TrustedPluginError,
        match="interval_low must not exceed interval_high",
    ):
        store.upload_estimator_plugin(
            name="inverted_interval",
            filename="inverted_interval.py",
            payload=code,
        )


def test_upload_featureset_plugin_registers_transforming_feature_set(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    code = b'''
FEATURESET_NAME = "trusted_trend_features"
FEATURE_COLUMNS = ["cycle_feature"]

def transform(table, meta=None):
    out = table.copy()
    out["cycle_feature"] = out["cycle_index"].astype(float)
    return out
'''

    registered = store.upload_featureset_plugin(
        name="trend_features",
        filename="trend_features.py",
        payload=code,
    )
    transformed = store.apply_feature_transforms(
        _dataset_frame(),
        meta=pd.DataFrame(),
        featuresets=["trusted_trend_features"],
    )

    fs = get_featureset("trusted_trend_features")
    assert registered["name"] == "trusted_trend_features"
    assert fs.columns(transformed) == ["cycle_feature"]
    assert transformed["cycle_feature"].equals(transformed["cycle_index"].astype(float))
    assert any(item["name"] == "trusted_trend_features" for item in store.catalog()["featuresets"])


def test_legacy_pipeline_rejects_a_state_from_changed_plugin_source(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    store.upload_featureset_plugin(
        name="stable_source",
        filename="stable_source.py",
        payload=b'''
FEATURESET_NAME = "stable_source"
FEATURE_COLUMNS = ["stable_feature"]

def transform(table, meta=None):
    out = table.copy()
    assert "soh" not in out.columns
    out["stable_feature"] = out["cycle_index"]
    return out
''',
    )
    pipeline = store.feature_pipeline("stable_source")
    frame = _dataset_frame().drop(columns=["soh"])
    state = pipeline.fit(frame, np.empty(len(frame)), pd.DataFrame())
    path = tmp_path / "plugins" / "featuresets" / "stable_source.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="plugin digest"):
        pipeline.transform(frame, state, pd.DataFrame())


def test_legacy_pipeline_reloads_module_globals_for_each_transform(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    store.upload_featureset_plugin(
        name="fresh_globals",
        filename="fresh_globals.py",
        payload=b'''
FEATURESET_NAME = "fresh_globals"
FEATURE_COLUMNS = ["fresh_counter"]
COUNTER = 0

def transform(table, meta=None):
    global COUNTER
    COUNTER += 1
    out = table.copy()
    out["fresh_counter"] = COUNTER
    return out
''',
    )
    pipeline = store.feature_pipeline("fresh_globals")
    frame = _dataset_frame().drop(columns=["soh"])
    state = pipeline.fit(frame, np.empty(len(frame)), pd.DataFrame())

    first = pipeline.transform(frame, state, pd.DataFrame())
    second = pipeline.transform(frame, state, pd.DataFrame())

    assert first["fresh_counter"].eq(1).all()
    assert second["fresh_counter"].eq(1).all()


def test_plugin_registration_executes_replacement_source_with_same_size_and_mtime(tmp_path):
    path = tmp_path / "registration.py"
    old_source = "REGISTRATION_MARKER = 1\n"
    new_source = "REGISTRATION_MARKER = 2\n"
    assert len(old_source.encode("utf-8")) == len(new_source.encode("utf-8"))
    path.write_text(old_source, encoding="utf-8")
    original_mtime = path.stat().st_mtime_ns

    assert _load_module(path, "featureset").REGISTRATION_MARKER == 1
    path.write_text(new_source, encoding="utf-8")
    os.utime(path, ns=(original_mtime, original_mtime))

    assert _load_module(path, "featureset").REGISTRATION_MARKER == 2


def test_catalog_is_persisted_for_uploaded_resources(tmp_path):
    store = TrustedPluginStore(base_dir=tmp_path / "plugins", data_dir=tmp_path / "data")
    store.upload_dataset("persisted_demo", "demo.csv", _dataset_frame().to_csv(index=False).encode("utf-8"))

    catalog_path = tmp_path / "plugins" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert catalog["datasets"][0]["name"] == "persisted_demo"


def test_plugin_catalog_paths_reload_from_different_working_directory(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    store = TrustedPluginStore(base_dir="plugins", data_dir="data")
    store.upload_featureset_plugin(
        name="reload_features",
        filename="reload_features.py",
        payload=b'''
FEATURESET_NAME = "reload_features"
FEATURE_COLUMNS = ["reload_cycle"]

def transform(table, meta=None):
    out = table.copy()
    out["reload_cycle"] = out["cycle_index"]
    return out
''',
    )
    catalog = json.loads((project / "plugins" / "catalog.json").read_text(encoding="utf-8"))
    assert Path(catalog["featuresets"][0]["path"]).is_absolute()

    R._FEATURESETS.pop("reload_features", None)
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    TrustedPluginStore(base_dir=project / "plugins", data_dir=project / "data")

    assert get_featureset("reload_features").candidates == ["reload_cycle"]


def test_invalid_estimator_replacement_preserves_last_known_good_state(tmp_path):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    valid_source = b'''
import numpy as np

class Estimator:
    name = "replacement_safe_estimator"
    feature_mode = "unified"
    backend = "external"

    def fit(self, X, y, meta=None):
        self.value = float(np.mean(y))
        return self

    def predict(self, X, meta=None):
        return np.full(len(X), self.value)
'''
    resource = store.upload_estimator_plugin(
        "replacement_safe_estimator",
        "replacement_safe_estimator.py",
        valid_source,
    )
    path = Path(resource["path"])
    original_class = R._ESTIMATORS["replacement_safe_estimator"]
    original_catalog = store.catalog_path.read_bytes()

    with pytest.raises(TrustedPluginError, match="plugin import failed"):
        store.upload_estimator_plugin(
            "replacement_safe_estimator",
            "replacement_safe_estimator.py",
            b"class Estimator:\n    this is invalid python\n",
            replace=True,
        )

    assert path.read_bytes() == valid_source
    assert R._ESTIMATORS["replacement_safe_estimator"] is original_class
    assert store.catalog_path.read_bytes() == original_catalog
    estimator = original_class().fit(
        _dataset_frame().iloc[:8],
        _dataset_frame().iloc[:8]["soh"].to_numpy(),
        pd.DataFrame(),
    )
    assert np.isfinite(estimator.predict(_dataset_frame().iloc[8:], None)).all()


def test_invalid_feature_replacement_preserves_pipeline_and_catalog(tmp_path):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    valid_source = b'''
FEATURESET_NAME = "replacement_safe_features"
FEATURE_COLUMNS = ["replacement_cycle"]

def transform(table, meta=None):
    out = table.copy()
    out["replacement_cycle"] = out["cycle_index"].astype(float)
    return out
'''
    resource = store.upload_featureset_plugin(
        "replacement_safe_features",
        "replacement_safe_features.py",
        valid_source,
    )
    path = Path(resource["path"])
    original_featureset = R._FEATURESETS["replacement_safe_features"]
    original_plugin = store._feature_plugins["replacement_safe_features"]
    original_catalog = store.catalog_path.read_bytes()
    pipeline = store.feature_pipeline("replacement_safe_features")
    frame = _dataset_frame().drop(columns=["soh"])
    state = pipeline.fit(frame, np.empty(len(frame)), pd.DataFrame())

    with pytest.raises(TrustedPluginError, match="did not return a DataFrame"):
        store.upload_featureset_plugin(
            "replacement_safe_features",
            "replacement_safe_features.py",
            b'''
FEATURESET_NAME = "replacement_safe_features"
FEATURE_COLUMNS = ["replacement_cycle"]

def transform(table, meta=None):
    return table.to_numpy()
''',
            replace=True,
        )

    assert path.read_bytes() == valid_source
    assert R._FEATURESETS["replacement_safe_features"] is original_featureset
    assert store._feature_plugins["replacement_safe_features"] is original_plugin
    assert store.catalog_path.read_bytes() == original_catalog
    transformed = pipeline.transform(frame, state, pd.DataFrame())
    assert transformed["replacement_cycle"].equals(
        transformed["cycle_index"].astype(float)
    )


def test_catalog_activation_failure_rolls_back_plugin_replacement(
    tmp_path, monkeypatch
):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    original_source = b'''
import numpy as np
class Estimator:
    name = "catalog_safe_estimator"
    def fit(self, X, y, meta=None):
        self.value = 0.9
        return self
    def predict(self, X, meta=None):
        return np.full(len(X), self.value)
'''
    replacement_source = original_source.replace(b"0.9", b"0.8")
    resource = store.upload_estimator_plugin(
        "catalog_safe_estimator", "catalog_safe_estimator.py", original_source
    )
    path = Path(resource["path"])
    original_class = R._ESTIMATORS["catalog_safe_estimator"]
    original_catalog = store.catalog_path.read_bytes()
    real_replace = trusted_plugins_module.os.replace

    def fail_catalog_activation(source, destination):
        if Path(destination) == store.catalog_path:
            raise OSError("injected catalog activation failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        trusted_plugins_module.os, "replace", fail_catalog_activation
    )

    with pytest.raises(OSError, match="injected catalog activation failure"):
        store.upload_estimator_plugin(
            "catalog_safe_estimator",
            "catalog_safe_estimator.py",
            replacement_source,
            replace=True,
        )

    assert path.read_bytes() == original_source
    assert R._ESTIMATORS["catalog_safe_estimator"] is original_class
    assert store.catalog_path.read_bytes() == original_catalog


def test_registry_activation_failure_rolls_back_file_catalog_and_runtime(
    tmp_path,
):
    store = TrustedPluginStore(
        base_dir=tmp_path / "plugins", data_dir=tmp_path / "data"
    )
    name = "callback_safe_estimator"
    original_source = b"original plugin bytes"
    replacement_source = b"replacement plugin bytes"
    path = store.estimator_dir / f"{name}.py"
    path.write_bytes(original_source)
    original_class = type("OriginalEstimator", (), {})
    replacement_class = type("ReplacementEstimator", (), {})
    original_resource = {
        "name": name,
        "kind": "estimator",
        "path": str(path),
        "backend": "original",
    }
    store._catalog = store._catalog_with_upsert(
        "estimators", original_resource
    )
    store.catalog_path.write_text(
        json.dumps(store._catalog, indent=2), encoding="utf-8"
    )
    original_catalog_bytes = store.catalog_path.read_bytes()
    previous_registry_value = R._ESTIMATORS.get(name)
    R._ESTIMATORS[name] = original_class
    original_catalog_document = store.catalog()
    staged = store._stage_plugin_payload(path, replacement_source)
    next_catalog = store._catalog_with_upsert(
        "estimators", {**original_resource, "backend": "replacement"}
    )
    catalog_staging = store._stage_catalog(next_catalog)

    def fail_after_runtime_mutation():
        R._ESTIMATORS[name] = replacement_class
        raise RuntimeError("injected registry activation failure")

    try:
        with pytest.raises(RuntimeError, match="registry activation failure"):
            store._activate_plugin_transaction(
                path,
                staged,
                catalog_staging,
                next_catalog,
                fail_after_runtime_mutation,
            )

        assert path.read_bytes() == original_source
        assert store.catalog_path.read_bytes() == original_catalog_bytes
        assert store.catalog() == original_catalog_document
        assert R._ESTIMATORS[name] is original_class
        assert not list(store.estimator_dir.glob(f".{path.name}.rollback-*"))
    finally:
        if previous_registry_value is None:
            R._ESTIMATORS.pop(name, None)
        else:
            R._ESTIMATORS[name] = previous_registry_value
