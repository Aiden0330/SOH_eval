"""K13 (Spec 12 §2–§4): charging segments, descriptors, forward-only labels, supervision kinds."""
import numpy as np
import pandas as pd
import pytest

import sohkernel.estimators as E
from sohkernel import api, segments as S
from sohkernel.contracts import IDENTITY_ADJACENT_COLUMNS, feature_columns
from sohkernel.feature_safety import is_label_source_feature
from sohkernel.metadata import DATASET_SUPERVISION, SUPERVISION_ROLES, build_meta, supervision_kind
from sohkernel.qc import qc_gate

E.register_all(include_torch=False)

C_RATED = 100.0
T0 = 1_700_000_000.0


def _record():
    """1 Hz record of a 100 Ah unit: rest, a 1 h charge at 20 A with a 4 min rest inside (bridged),
    30 min rest, a 20 min discharge, a 3 min charge (too short), a 20 min charge at 3 A (low
    coverage), a 40 min charge at 50 A, rest.  Voltage steps mark the pieces."""
    pieces = [
        (600, 0.0, 3.30), (1800, 20.0, 3.32), (240, 0.0, 3.34), (1800, 20.0, 3.35), (1800, 0.0, 3.36),
        (1200, -30.0, 3.30), (180, 20.0, 3.31), (700, 0.0, 3.31), (1200, 3.0, 3.31), (700, 0.0, 3.31),
        (2400, 50.0, 3.40), (600, 0.0, 3.38),
    ]
    i = np.concatenate([np.full(n, a) for n, a, _ in pieces])
    v = np.concatenate([np.full(n, x) for n, _, x in pieces])
    temp = np.linspace(20.0, 30.0, len(i))
    t = T0 + np.arange(len(i), dtype=float)
    return t, v, i, temp


def test_charging_spans_bridge_short_rests_and_end_at_discharges_or_long_rests():
    t, v, i, _ = _record()
    spans = S.charging_spans(t, i, C_RATED)
    starts = [600, 600 + 3840 + 1800 + 1200, 600 + 3840 + 1800 + 1200 + 180 + 700, 600 + 3840 + 1800 + 1200 + 180 + 700 + 1200 + 700]
    assert [s for s, _ in spans] == starts
    assert spans[0] == (600, 600 + 3840 - 1)            # the 240 s rest inside is bridged
    assert spans[1][1] - spans[1][0] == 179              # the 3 min charge stands alone (700 s rest after it)
    assert spans[3] == (starts[3], starts[3] + 2399)
    # a rest of exactly rest_split_s still bridges; one second more splits
    short = S.SegmentConfig(rest_split_s=239.0)
    assert len(S.charging_spans(t, i, C_RATED, short)) == 5
    assert S.charging_spans(t, np.zeros_like(i), C_RATED) == []
    with pytest.raises(ValueError, match="non-decreasing"):
        S.charging_spans(t[::-1], i, C_RATED)


def test_segment_table_keeps_the_accepted_segments_with_descriptors_and_qc_counts():
    t, v, i, temp = _record()
    table, qc = S.segment_table("unit-1", t, v, i, temp, c_rated_ah=C_RATED, dataset="fixture", window=(3.31, 3.36))
    assert qc == pytest.approx({"segments_found": 4, "too_short": 1, "low_coverage": 1, "accepted": 2,
                                "accepted_charge_ah": 20.0 + 2400 * 50 / 3600, "record_start_s": T0,
                                "record_end_s": T0 + len(t) - 1}, rel=1e-3)
    assert list(table.columns) == list(S.SEGMENT_TABLE_COLUMNS)
    assert table["cycle_index"].tolist() == [0, 1] and table["cell_id"].tolist() == ["unit-1"] * 2
    first = table.iloc[0]
    assert first["time_s_max"] == 3839.0 and first["segment_start_s"] == T0 + 600 and first["segment_kind"] == "charge"
    assert first["charge_capacity_max"] == pytest.approx(20.0, rel=1e-3)
    assert first["window_coverage"] == pytest.approx(0.2, rel=1e-3)
    assert first["current_mean"] == pytest.approx(20.0) and first["current_cv"] == pytest.approx(0.0)
    assert first["current_mean_c"] == pytest.approx(0.2)
    assert first["v_start"] == pytest.approx(3.32) and first["v_end"] == pytest.approx(3.35)
    assert np.isnan(first["rest_before_s"])                      # the record starts at rest: unknown
    assert first["rest_after_s"] == pytest.approx(1801.0)         # to the first discharge sample
    assert first["sample_period_s"] == 1.0 and first["temp_min"] < first["temperature_mean"] < first["temp_max"]
    assert np.isnan(first["soh"]) and pd.isna(first["label_source"])
    second = table.iloc[1]
    assert second["rest_before_s"] == pytest.approx(701.0) and np.isnan(second["rest_after_s"])   # record ends at rest
    assert second["charge_capacity_max"] == pytest.approx(2400 * 50 / 3600, rel=1e-3)
    # the q_win window [3.31, 3.36] covers the first segment entirely, none of the second (3.40 V)
    assert first["q_win_ah"] == pytest.approx(20.0, rel=1e-3) and np.isnan(second["q_win_ah"])


def test_mean_cell_voltage_for_a_string():
    t, v, i, _ = _record()
    pack, _ = S.segment_table("string", t, v * 16.0, i, c_rated_ah=C_RATED, cells_in_series=16)
    cell, _ = S.segment_table("string", t, v, i, c_rated_ah=C_RATED)
    for column in ("voltage_mean", "voltage_min", "voltage_max", "v_start", "v_end"):
        assert pack[column].to_numpy() == pytest.approx(cell[column].to_numpy())
    assert "dataset" not in pack.columns


def _segments(starts, unit="u"):
    return pd.DataFrame({"cell_id": [unit] * len(starts), "cycle_index": range(len(starts)),
                         "segment_start_s": list(starts), "soh": np.nan})


def test_forward_only_labels_never_look_ahead():
    segments = _segments([100.0, 200.0, 300.0])
    tests = pd.DataFrame({"cell_id": ["u", "u"], "label_time_s": [150.0, 250.0], "capacity_ah": [90.0, 80.0]})
    labeled = S.attach_forward_labels(segments, tests, c_rated_ah=100.0)
    assert labeled["soh"].tolist() == pytest.approx([np.nan, 0.9, 0.8], nan_ok=True)
    assert labeled["label_age_s"].tolist() == pytest.approx([np.nan, 50.0, 50.0], nan_ok=True)
    assert pd.isna(labeled["label_source"].iloc[0]) and labeled["label_source"].tolist()[1:] == ["capacity_test", "capacity_test"]
    # Spec 12 §7.2: a later test changes no earlier segment's label
    later = pd.concat([tests, pd.DataFrame({"cell_id": ["u"], "label_time_s": [400.0], "capacity_ah": [70.0]})])
    again = S.attach_forward_labels(segments, later, c_rated_ah=100.0)
    pd.testing.assert_frame_equal(again, labeled)
    # a test completed at the segment start labels it; a bounded age drops old labels
    exact = S.attach_forward_labels(_segments([250.0]), tests, c_rated_ah=100.0)
    assert exact["soh"].iloc[0] == pytest.approx(0.8) and exact["label_age_s"].iloc[0] == 0.0
    bounded = S.attach_forward_labels(segments, tests, c_rated_ah=100.0, max_label_age_s=40.0)
    assert bounded["soh"].isna().all() and bounded["label_source"].isna().all()
    # rated capacity per unit and the label source column
    two = pd.concat([_segments([1000.0], "a"), _segments([1000.0], "b")], ignore_index=True)
    rpt = pd.DataFrame({"cell_id": ["a", "b"], "label_time_s": [0.0, 0.0], "capacity_ah": [90.0, 180.0], "label_source": ["rpt", "rpt"]})
    out = S.attach_forward_labels(two, rpt, c_rated_ah={"a": 100.0, "b": 200.0})
    assert out["soh"].tolist() == pytest.approx([0.9, 0.9]) and set(out["label_source"]) == {"rpt"}
    assert S.tests_per_unit(labeled) == {"u": 2} and S.tests_per_unit(out) == {"a": 1, "b": 1}
    with pytest.raises(ValueError, match="missing columns"):
        S.attach_forward_labels(segments, tests.drop(columns=["capacity_ah"]), c_rated_ah=100.0)


def test_column_classes_for_the_leak_inventory():
    table = pd.DataFrame({column: [0.0] for column in S.SEGMENT_TABLE_COLUMNS})
    features = feature_columns(table)
    assert not set(features) & set(IDENTITY_ADJACENT_COLUMNS) and set(S.SEGMENT_IDENTITY_COLUMNS) == set(IDENTITY_ADJACENT_COLUMNS)
    assert all(is_label_source_feature(column) for column in S.LABEL_COLUMNS)
    assert all(is_label_source_feature(column) for column in ("capacity_test_ah", "rpt_capacity_ah", "charge_capacity_max"))
    assert set(S.SEGMENT_DESCRIPTOR_COLUMNS) <= set(features)       # descriptors are covariates a program may read
    assert not any(is_label_source_feature(column) for column in S.SEGMENT_DESCRIPTOR_COLUMNS)


def test_qc_counts_unlabeled_rows_and_keeps_them_only_on_request():
    table = pd.DataFrame({"dataset": ["d"] * 4, "cell_id": ["u"] * 4, "cycle_index": [0, 1, 2, 3],
                          "soh": [np.nan, 0.95, 1.35, 0.9], "f1": [1.0, 2.0, 3.0, 4.0]})
    meta = build_meta(table)
    clean, report = qc_gate(table, meta)
    assert clean["cycle_index"].tolist() == [1, 3]
    assert report.drop_reasons["soh_unlabeled"] == 1 and report.drop_reasons["soh_out_of_range"] == 1
    kept, report = qc_gate(table, meta, keep_unlabeled=True)
    assert kept["cycle_index"].tolist() == [0, 1, 3] and report.drop_reasons["soh_unlabeled"] == 0
    assert report.drop_reasons["soh_out_of_range"] == 1 and report.n_out == 1


def test_supervision_kinds_and_the_training_refusal(monkeypatch):
    assert supervision_kind("hust") == "soh_supervised" and "supervision" in build_meta(pd.DataFrame({"dataset": ["hust"], "cell_id": ["c"]})).columns
    assert set(SUPERVISION_ROLES) == {"soh_supervised", "sparse_field", "unlabeled_field"}
    assert "training" not in SUPERVISION_ROLES["unlabeled_field"] and "drift" in SUPERVISION_ROLES["sparse_field"]
    monkeypatch.setitem(DATASET_SUPERVISION, "fieldx", "unlabeled_field")
    monkeypatch.setitem(DATASET_SUPERVISION, "fieldy", "sparse_field")
    assert api.supervision_errors(["hust", "fieldy"]) == []
    errors = api.supervision_errors(["fieldx", "hust"])
    assert len(errors) == 1 and "unlabeled_field" in errors[0] and "Spec 12" in errors[0]
    # a preflight over an unlabeled_field set blocks every combination with that reason
    rng = np.random.default_rng(0)
    rows = []
    for cell in range(3):
        for k in range(40):
            soh = 1.0 - 0.2 * k / 39
            rows.append({"dataset": "fieldy", "cell_id": f"c{cell}", "cycle_index": k, "soh": soh,
                         "voltage_mean": 3.3 + 0.1 * soh + rng.normal(0, 0.01), "voltage_std": 0.02 + rng.normal(0, 0.002),
                         "label_time_s": float(1000 * (k // 10)), "label_age_s": float(k % 10)})
    table = pd.DataFrame(rows)
    meta = build_meta(table)
    ok = api.preflight({"datasets": ["fieldy"], "estimators": ["ridge"], "featuresets": ["voltage"], "protocols": ["loco"], "seeds": [0]}, table=table, meta=meta)
    assert ok.ready and ok.datasets[0]["supervision"] == "sparse_field" and ok.datasets[0]["labeled_rows"] == 120
    assert ok.datasets[0]["tests_per_unit"] == {"c0": 4, "c1": 4, "c2": 4}
    assert "label_time_s" not in ok.combinations[0]["selected_columns"] and "label_age_s" not in ok.combinations[0]["selected_columns"]
    unlabeled = table.assign(dataset="fieldx")
    refused = api.preflight({"datasets": ["fieldx"], "estimators": ["ridge"], "featuresets": ["voltage"], "protocols": ["loco"], "seeds": [0]}, table=unlabeled, meta=build_meta(unlabeled))
    assert not refused.ready and all(item["status"] == "blocked" for item in refused.combinations)
    assert any("unlabeled_field" in reason for reason in refused.reasons)
