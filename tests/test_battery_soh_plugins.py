import json
import os
import subprocess
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from sohbench import registry as R
from sohbench.api_server import create_handler
from sohbench.battery_soh_plugins import register_battery_soh_plugins
from sohbench.evaluation_service import EvaluationService
from sohbench.trusted_plugins import TrustedPluginStore


def _write_fake_battery_soh_root(root):
    model_dir = root / "soh_core" / "models"
    model_dir.mkdir(parents=True)
    (root / "soh_core" / "__init__.py").write_text("", encoding="utf-8")
    (model_dir / "__init__.py").write_text("", encoding="utf-8")
    (model_dir / "tabular_ensemble.py").write_text(
        """
import numpy as np


class TabularSOHEnsemble:
    def __init__(self, random_state=2026):
        self.random_state = random_state
        self.mean_ = 0.95

    def fit(self, features, target):
        self.mean_ = float(np.mean(target))
        return self

    def predict(self, features, calibration_offset=0.0):
        return np.full(features.shape[0], self.mean_ + calibration_offset)

    def structure_signature(self):
        return ("fake",)
""".strip(),
        encoding="utf-8",
    )


def _sample_frame():
    rows = []
    for cell in ["c1", "c2"]:
        for cycle in range(6):
            rows.append(
                {
                    "dataset": "demo",
                    "cell_id": cell,
                    "cycle_index": cycle,
                    "soh": 1.0 - 0.05 * cycle,
                    "voltage_mean": 3.7 - 0.02 * cycle,
                    "resistance_proxy": 0.02 + 0.001 * cycle,
                }
            )
    return pd.DataFrame(rows)


def _post_json(base_url, path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _serve(service):
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_register_battery_soh_plugins_adds_catalog_resources_and_runs(tmp_path):
    battery_root = tmp_path / "battery_SOH"
    _write_fake_battery_soh_root(battery_root)
    store = TrustedPluginStore(tmp_path / "plugins", tmp_path / "data")

    result = register_battery_soh_plugins(store, battery_root=battery_root, replace=True)

    assert result["estimator"]["name"] == "battery_soh_tabular_ensemble"
    assert result["featureset"]["name"] == "battery_soh_operational"
    catalog = store.catalog()
    assert any(item["name"] == "battery_soh_tabular_ensemble" for item in catalog["estimators"])
    assert any(item["name"] == "battery_soh_operational" for item in catalog["featuresets"])

    table = _sample_frame()
    transformed = store.apply_feature_transforms(table, pd.DataFrame(), ["battery_soh_operational"])
    assert "battery_soh_voltage_sag" in transformed.columns
    assert "battery_soh_cycle_norm" not in transformed.columns
    assert "battery_soh_resistance_growth" not in transformed.columns

    estimator_cls = R.get_estimator("battery_soh_tabular_ensemble")
    estimator = estimator_cls().fit(transformed, transformed["soh"].to_numpy(), pd.DataFrame())
    pred = estimator.predict(transformed.iloc[:3], pd.DataFrame())
    assert pred.shape == (3,)
    assert np.isfinite(pred).all()


def test_battery_voltage_sag_uses_fixed_reference_without_asset_baseline(tmp_path):
    battery_root = tmp_path / "battery_SOH"
    _write_fake_battery_soh_root(battery_root)
    store = TrustedPluginStore(tmp_path / "plugins", tmp_path / "data")
    register_battery_soh_plugins(store, battery_root=battery_root, replace=True)
    frame = _sample_frame().iloc[[5, 0]].copy()
    frame.loc[frame.index[0], "cycle_index"] = np.nan

    transformed = store.apply_feature_transforms(frame, pd.DataFrame(), ["battery_soh_operational"])

    np.testing.assert_allclose(
        transformed["battery_soh_voltage_sag"],
        3.7 - transformed["voltage_mean"],
    )


def test_register_battery_soh_plugins_normalizes_relative_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    battery_root = Path("battery_SOH")
    _write_fake_battery_soh_root(battery_root)
    store = TrustedPluginStore(tmp_path / "plugins", tmp_path / "data")

    result = register_battery_soh_plugins(store, battery_root=battery_root, replace=True)
    estimator_catalog = json.loads((tmp_path / "plugins" / "catalog.json").read_text(encoding="utf-8"))[
        "estimators"
    ]

    assert Path(result["battery_soh_root"]).is_absolute()
    dependency_root = Path(result["dependency_snapshot"])
    assert dependency_root.is_absolute()
    assert dependency_root.is_relative_to(tmp_path / "plugins" / "dependencies")
    assert repr(str(dependency_root)) in (
        tmp_path / "plugins" / "estimators" / "battery_soh_tabular_ensemble.py"
    ).read_text(encoding="utf-8")
    assert Path(estimator_catalog[0]["path"]).is_absolute()


def test_battery_dependency_change_updates_candidate_and_run_identity(tmp_path):
    battery_root = tmp_path / "battery_SOH"
    _write_fake_battery_soh_root(battery_root)
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.import_battery_soh_plugins(
        battery_soh_root=battery_root, replace=True
    )
    table = _sample_frame()
    request = service._normalize_request(
        {
            "datasets": ["demo"],
            "estimators": ["battery_soh_tabular_ensemble"],
            "featuresets": ["battery_soh_operational"],
            "protocols": ["loco"],
            "seeds": [0],
        }
    )
    first = service._build_run_spec(
        run_type="training",
        request=request,
        datasets=["demo"],
        table=table,
        qc={},
    )

    dependency_file = (
        battery_root / "soh_core" / "models" / "tabular_ensemble.py"
    )
    dependency_file.write_text(
        dependency_file.read_text(encoding="utf-8") + "\nDEPENDENCY_REVISION = 2\n",
        encoding="utf-8",
    )
    service.import_battery_soh_plugins(
        battery_soh_root=battery_root, replace=True
    )
    second = service._build_run_spec(
        run_type="training",
        request=request,
        datasets=["demo"],
        table=table,
        qc={},
    )

    first_dependencies = [
        artifact
        for artifact in first.candidates[0].artifacts
        if artifact.kind == "executable_dependency_tree"
    ]
    second_dependencies = [
        artifact
        for artifact in second.candidates[0].artifacts
        if artifact.kind == "executable_dependency_tree"
    ]
    assert len(first_dependencies) == len(second_dependencies) == 1
    assert first_dependencies[0].spec_id == second_dependencies[0].spec_id
    assert first_dependencies[0].digest != second_dependencies[0].digest
    assert first.candidates[0].digest != second.candidates[0].digest
    assert first.digest != second.digest


def test_battery_inference_prediction_and_identity_follow_managed_dependency(
    tmp_path,
):
    battery_root = tmp_path / "battery_SOH"
    _write_fake_battery_soh_root(battery_root)
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.import_battery_soh_plugins(
        battery_soh_root=battery_root, replace=True
    )
    table = _sample_frame()
    service.store.upload_dataset(
        "demo",
        "demo.csv",
        table.to_csv(index=False).encode("utf-8"),
        replace=True,
    )
    training_job = service.create_job(
        {
            "datasets": ["demo"],
            "estimators": ["battery_soh_tabular_ensemble"],
            "featuresets": ["battery_soh_operational"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    training = service.run_job_inline(training_job["id"])
    model_name = training["saved_models"][0]
    request = service._normalize_inference_request(
        {"datasets": ["demo"], "models": [model_name]}
    )

    def prediction_and_spec():
        with service.model_store.read_lease([model_name]) as snapshots:
            loaded = snapshots[model_name].loaded
            prepared = service._prepare_inference_table(
                model_name,
                table,
                pd.DataFrame(),
                loaded_model=loaded,
            )
            prediction = loaded.predict(prepared, pd.DataFrame())
            spec = service._build_run_spec(
                run_type="inference",
                request=request,
                datasets=["demo"],
                table=table,
                qc={},
                model_snapshots=snapshots,
            )
        return prediction, spec

    first_prediction, first_spec = prediction_and_spec()
    dependency_file = (
        battery_root / "soh_core" / "models" / "tabular_ensemble.py"
    )
    dependency_file.write_text(
        dependency_file.read_text(encoding="utf-8").replace(
            "self.mean_ + calibration_offset",
            "self.mean_ - 0.20 + calibration_offset",
        ),
        encoding="utf-8",
    )
    service.import_battery_soh_plugins(
        battery_soh_root=battery_root, replace=True
    )
    second_prediction, second_spec = prediction_and_spec()

    artifact_kinds = {
        artifact.kind for artifact in second_spec.candidates[0].artifacts
    }
    assert "estimator_plugin" in artifact_kinds
    assert "executable_dependency_tree" in artifact_kinds
    assert not np.allclose(first_prediction, second_prediction)
    assert first_spec.candidates[0].digest != second_spec.candidates[0].digest
    assert first_spec.digest != second_spec.digest


def test_battery_dependency_snapshot_excludes_unhashed_bytecode(tmp_path):
    battery_root = tmp_path / "battery_SOH"
    _write_fake_battery_soh_root(battery_root)
    bytecode_dir = battery_root / "soh_core" / "models" / "__pycache__"
    bytecode_dir.mkdir()
    (bytecode_dir / "tabular_ensemble.cpython-312.pyc").write_bytes(
        b"unbound-bytecode"
    )
    store = TrustedPluginStore(tmp_path / "plugins", tmp_path / "data")

    result = register_battery_soh_plugins(
        store, battery_root=battery_root, replace=True
    )

    snapshot = Path(result["dependency_snapshot"])
    assert not any(
        "__pycache__" in item.relative_to(snapshot).parts
        or item.suffix.lower() == ".pyc"
        for item in snapshot.rglob("*")
    )


def test_api_import_battery_soh_plugins_registers_resources(tmp_path):
    battery_root = tmp_path / "battery_SOH"
    _write_fake_battery_soh_root(battery_root)
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    server, base_url = _serve(service)
    try:
        response = _post_json(
            base_url,
            "/api/import/battery-soh-plugins",
            {"battery_soh_root": str(battery_root), "replace": True},
        )
    finally:
        server.shutdown()

    assert response["ok"] is True
    assert response["plugins"]["estimator"]["name"] == "battery_soh_tabular_ensemble"
    assert response["plugins"]["featureset"]["name"] == "battery_soh_operational"


def test_inference_job_replays_featureset_transforms_for_trained_models(tmp_path):
    battery_root = tmp_path / "battery_SOH"
    _write_fake_battery_soh_root(battery_root)
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.import_battery_soh_plugins(battery_soh_root=battery_root, replace=True)
    service.store.upload_dataset(
        "demo",
        "demo.csv",
        _sample_frame().to_csv(index=False).encode("utf-8"),
        replace=True,
    )
    train_job = service.create_job(
        {
            "datasets": ["demo"],
            "estimators": ["battery_soh_tabular_ensemble"],
            "featuresets": ["battery_soh_operational"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    train_job = service.run_job_inline(train_job["id"])
    assert train_job["status"] == "succeeded"
    assert train_job["saved_models"]

    inference_job = service.create_inference_job(
        {"datasets": ["demo"], "models": train_job["saved_models"]},
        start=False,
    )
    inference_job = service.run_inference_job_inline(inference_job["id"])

    assert inference_job["status"] == "succeeded"
    payload = json.loads((tmp_path / inference_job["raw_run_dir"] / "inference_benchmark.json").read_text())
    assert payload["summary"]["n_succeeded"] == len(train_job["saved_models"])
    assert {record["status"] for record in payload["records"]} == {"succeeded"}


def test_trusted_plugin_model_pickle_loads_in_fresh_process(tmp_path):
    battery_root = tmp_path / "battery_SOH"
    _write_fake_battery_soh_root(battery_root)
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.import_battery_soh_plugins(battery_soh_root=battery_root, replace=True)
    service.store.upload_dataset(
        "demo",
        "demo.csv",
        _sample_frame().to_csv(index=False).encode("utf-8"),
        replace=True,
    )
    train_job = service.create_job(
        {
            "datasets": ["demo"],
            "estimators": ["battery_soh_tabular_ensemble"],
            "featuresets": ["battery_soh_operational"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    train_job = service.run_job_inline(train_job["id"])
    model_name = train_job["saved_models"][0]

    code = f"""
import json
import pandas as pd
from sohbench.evaluation_service import EvaluationService

svc = EvaluationService(
    data_dir={str(tmp_path / 'data')!r},
    work_dir={str(tmp_path / 'work')!r},
    runs_dir={str(tmp_path / 'runs')!r},
    frontend_data_dir={str(tmp_path / 'frontend_data')!r},
)
loaded = svc.model_store.load_model({model_name!r})
frame = pd.DataFrame({{
    'dataset': ['demo', 'demo'],
    'cell_id': ['c1', 'c1'],
    'cycle_index': [0, 1],
    'soh': [1.0, 0.99],
    'battery_soh_voltage_sag': [0.0, 0.02],
}})
pred = loaded.predict(frame, pd.DataFrame())
print(json.dumps({{'n': len(pred)}}))
"""
    env = {**os.environ, "PYTHONPATH": str((Path(__file__).parents[1] / "src").resolve())}
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["n"] == 2


def test_replacing_battery_soh_plugins_twice_still_saves_training_models(tmp_path):
    battery_root = tmp_path / "battery_SOH"
    _write_fake_battery_soh_root(battery_root)
    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    service.import_battery_soh_plugins(battery_soh_root=battery_root, replace=True)

    restarted = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    restarted.import_battery_soh_plugins(battery_soh_root=battery_root, replace=True)
    restarted.store.upload_dataset(
        "demo",
        "demo.csv",
        _sample_frame().to_csv(index=False).encode("utf-8"),
        replace=True,
    )

    train_job = restarted.create_job(
        {
            "datasets": ["demo"],
            "estimators": ["battery_soh_tabular_ensemble"],
            "featuresets": ["battery_soh_operational"],
            "protocols": ["loco"],
            "seeds": [0],
        },
        start=False,
    )
    train_job = restarted.run_job_inline(train_job["id"])

    assert train_job["status"] == "succeeded"
    assert train_job["saved_models"]
    assert train_job["model_warnings"] == []
