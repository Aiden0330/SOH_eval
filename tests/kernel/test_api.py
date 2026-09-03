"""sohkernel.api — the surface any tool calls (Spec 01 §9; extraction plan Task 6).

The preflight is an admission oracle: label-source columns can never be selected, a
transform that reconstructs the label blocks its combinations, a full-range Ah window
blocks and a wide one warns, and a transform never sees the label column.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sohkernel.estimators as E
from sohkernel import api
from sohkernel import registry as R
from sohkernel.feature_safety import _affine_reconstruction
from sohkernel.featuresets import FeatureSet
from sohkernel.protocols import PROTOCOLS
from sohkernel.tasks import SOHEstimationTask

E.register_all(include_torch=False)


def _table(n_cells: int = 4, cycles: int = 100, dataset: str = "d1", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cell in range(n_cells):
        for k in range(cycles):
            soh = 1.0 - 0.3 * k / (cycles - 1) + rng.normal(0.0, 0.003)
            rows.append(
                {
                    "dataset": dataset,
                    "cell_id": f"c{cell}",
                    "cycle_index": k,
                    "soh": soh,
                    "voltage_mean": 3.6 + 0.2 * soh + rng.normal(0.0, 0.02),
                    "voltage_std": 0.05 + rng.normal(0.0, 0.005),
                    # the calibration capacity the label is computed from (label source)
                    "charge_capacity_max": 1.1 * soh,
                }
            )
    return pd.DataFrame(rows)


def _meta(table: pd.DataFrame) -> pd.DataFrame:
    from sohkernel.metadata import build_meta

    return build_meta(table)


def _request(**overrides) -> dict:
    base = {"datasets": ["d1"], "estimators": ["ridge"], "featuresets": ["voltage"],
            "protocols": ["loco"], "seeds": [0]}
    base.update(overrides)
    return base


def test_protocols_menu_is_exported_by_the_kernel():
    assert set(PROTOCOLS) >= {"loco", "lodo", "lcho"}
    assert api.PROTOCOLS is PROTOCOLS


def test_normalize_request_defaults_and_validation(tmp_path):
    normalized = api.normalize_request({})
    assert normalized == {"datasets": ["all"], "estimators": ["ridge"], "featuresets": ["full"],
                          "protocols": ["loco"], "seeds": [0]}
    with pytest.raises(ValueError, match="unknown protocols"):
        api.normalize_request({"protocols": ["random_split"]})
    with pytest.raises(KeyError):
        api.normalize_request({"estimators": ["no_such_estimator"]})
    with pytest.raises(ValueError, match="unknown datasets"):
        api.normalize_request({"datasets": ["nope"]}, data_dir=tmp_path)
    # "all" stays admissible; it is resolved by discovery later.
    assert api.normalize_request({"datasets": ["all"]}, data_dir=tmp_path)["datasets"] == ["all"]


def test_preflight_never_selects_a_label_source_column():
    R.register_featureset(FeatureSet(name="_t_label_source_only", candidates=["charge_capacity_max"]))
    table = _table()
    result = api.preflight(_request(featuresets=["_t_label_source_only"]), table=table, meta=_meta(table))
    assert not result.ready
    (combo,) = result.combinations
    assert combo["status"] == "blocked"
    assert "charge_capacity_max" not in combo["selected_columns"]
    assert any("selects zero usable columns" in reason for reason in combo["reasons"])
    assert result.reasons and "ridge × _t_label_source_only × loco" in result.reasons[0]


def test_preflight_blocks_a_transform_that_reconstructs_the_label():
    def leaky(frame, meta):  # a scaled copy of the calibration capacity
        return frame.assign(cap_copy=frame["charge_capacity_max"] * 0.5)

    R.register_featureset(FeatureSet(name="_t_leaky", candidates=["cap_copy"], transform=leaky))
    table = _table()
    result = api.preflight(_request(featuresets=["_t_leaky"]), table=table, meta=_meta(table))
    assert not result.ready
    (combo,) = result.combinations
    assert combo["status"] == "blocked"
    assert any("label-safety audit" in reason and "cap_copy" in reason for reason in combo["reasons"])
    # the offending column never reached the table the models would see
    assert "cap_copy" not in result.table.columns


@pytest.mark.parametrize(
    "coverage, expected_status, marker",
    [
        ((0.90, 1.00), "blocked", "≥90%"),
        ((0.62, 0.78), "warning", "wider than the typical"),
        ((0.40, 0.50), "ready", None),
    ],
)
def test_preflight_gates_the_ah_window_by_coverage(coverage, expected_status, marker):
    rng = np.random.default_rng(7)
    table = _table()
    table["q_win_ah"] = table["charge_capacity_max"] * rng.uniform(*coverage, size=len(table))
    # precondition: the window column is not an affine copy of the label, so the
    # coverage rule (not the affine rule) is what decides
    fit = _affine_reconstruction(table["q_win_ah"].to_numpy(), table["soh"].to_numpy())
    assert fit is not None and fit[0] < 0.995
    R.register_featureset(FeatureSet(name="_t_window_only", candidates=["q_win_ah"]))
    result = api.preflight(_request(featuresets=["_t_window_only"]), table=table, meta=_meta(table))
    (combo,) = result.combinations
    assert combo["status"] == expected_status
    assert result.ready is (expected_status != "blocked")
    if marker:
        pool = combo["reasons"] if expected_status == "blocked" else combo["warnings"]
        assert any(marker in text for text in pool)


def test_transform_never_receives_the_label_column():
    seen: list[list[str]] = []

    def spy(frame, meta):
        seen.append(list(frame.columns))
        return frame.assign(spy_feature=frame["voltage_mean"] * 2.0)

    R.register_featureset(FeatureSet(name="_t_spy", candidates=["spy_feature", "never_derived"], transform=spy))
    table = _table()
    out = api.transform(table, _meta(table), ["_t_spy", "voltage"])  # "voltage" has no transform
    assert seen and all("soh" not in columns for columns in seen)
    assert "soh" in out.columns and len(out) == len(table)
    assert "spy_feature" in out.columns
    assert out["never_derived"].isna().all()  # undeliverable candidates are materialised as NaN


def test_score_verdicts_and_write_artifacts_roundtrip(tmp_path):
    import jsonschema
    import sohkernel.report

    table = _table()
    meta = _meta(table)
    results = api.score(None, table, meta, ["loco"], ["ridge"], ["voltage"], seeds=[0])
    assert set(results["protocol"]) == {"loco"} and set(results["estimator"]) == {"ridge"}
    assert results["held_out_key"].nunique() == 4
    flat = results.groupby("metric")["value"].mean().to_dict()
    names = [v.name for v in api.verdicts(flat)]
    assert names == ["generalization", "calibration", "physical_consistency"]
    out = api.write_artifacts(results, None, SOHEstimationTask(), {"seeds": [0]}, tmp_path)
    leaderboard = json.loads((Path(out) / "leaderboard.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (Path(sohkernel.report.__file__).with_name("schemas") / "leaderboard.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(leaderboard, schema)
    assert leaderboard[0]["estimator"] == "ridge" and leaderboard[0]["featureset"] == "voltage"


def test_preflight_loads_and_resolves_datasets_from_a_data_root(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    _table(dataset="synthetic").to_pickle(processed / "synthetic_features.pkl")
    result = api.preflight({"datasets": ["all"], "estimators": ["ridge"], "featuresets": ["voltage"],
                            "protocols": ["loco"], "seeds": [0, 1]}, data_dir=tmp_path)
    assert result.request["datasets"] == ["synthetic"]
    assert result.ready and result.summary["run_instances"] == 2
    assert result.qc_report is not None and result.summary["qc_pass_rate"] == pytest.approx(1.0)
    table, meta, qc = api.load("synthetic", data_dir=tmp_path)
    assert len(table) == 400 and qc.n_out == 0 and set(meta["cell_id"]) == {"c0", "c1", "c2", "c3"}
    monkeypatch.delenv("SOHBENCH_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="SOHBENCH_DATA_DIR"):
        api.resolve_data_dir(None)
    monkeypatch.setenv("SOHBENCH_DATA_DIR", str(tmp_path))
    assert api.resolve_data_dir(None) == tmp_path
