"""D1 (2026-09-04): the BatteryLife SNL extractor — charge-segment statistics, label rules, dedup, IC peak."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ingest():
    spec = importlib.util.spec_from_file_location("ingest_batterylife_snl", ROOT / "scripts" / "ingest_batterylife_snl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cycle(number, *, discharge=2.7, n=200, rest=5, dq_scale=1.0):
    """Rest -> 0.5C CC charge 3.4->4.2 V -> discharge; time 10 s per sample."""
    n_chg = n
    time = 10.0 * np.arange(rest + n_chg + 20)
    current = np.concatenate([np.zeros(rest), np.full(n_chg, 1.5), np.full(20, -1.5)])
    voltage = np.concatenate([np.full(rest, 3.4), np.linspace(3.4, 4.2, n_chg), np.linspace(4.1, 3.0, 20)])
    charge = np.concatenate([np.zeros(rest), np.linspace(0.0, 2.9 * dq_scale, n_chg), np.full(20, 2.9 * dq_scale)])
    dis = np.concatenate([np.zeros(rest + n_chg), np.linspace(0.0, discharge, 20)])
    return {"cycle_number": number, "current_in_A": current.tolist(), "voltage_in_V": voltage.tolist(),
            "charge_capacity_in_Ah": charge.tolist(), "discharge_capacity_in_Ah": dis.tolist(),
            "time_in_s": time.tolist(), "temperature_in_C": np.full(len(time), 25.0).tolist(),
            "internal_resistance_in_ohm": None}


def _payload(cell_id, cycles, soc=(0, 1), nominal=3.0):
    return {"cell_id": cell_id, "nominal_capacity_in_Ah": nominal, "SOC_interval": list(soc), "cycle_data": cycles,
            "cathode_material": "NMC", "anode_material": "graphite"}


def test_cycle_row_uses_the_charge_segment_and_the_rated_capacity(ingest):
    payload = _payload("c1", [_cycle(1, discharge=2.7)])
    row = ingest.cycle_row(payload, payload["cycle_data"][0], (3.65, 4.10), "snl_nmc", rated_ah=3.0)
    assert row["soh"] == pytest.approx(2.7 / 3.0)
    assert row["time_s_max"] == pytest.approx(10.0 * 199)          # re-based to the first charge sample
    assert row["voltage_min"] == pytest.approx(3.4) and row["voltage_max"] == pytest.approx(4.2)
    assert row["current_mean"] == pytest.approx(1.5) and row["current_std"] == pytest.approx(0.0)
    assert row["charge_capacity_max"] == pytest.approx(2.9)
    assert 0.0 < row["q_win_ah"] < 2.9 and row["t_win_s"] > 0
    assert np.isfinite(row["ic_peak_ah_per_v"]) and 3.4 < row["ic_peak_voltage_v"] < 4.2
    # the pickle nominal is the fallback denominator only
    assert ingest.cycle_soh(payload, payload["cycle_data"][0], None) == pytest.approx(2.7 / 3.0)
    assert ingest.cycle_soh(_payload("c", [], nominal=2.7), payload["cycle_data"][0], None) == pytest.approx(1.0)


def test_unlabelled_and_partial_soc_cycles_are_not_ingested(ingest, tmp_path):
    good = _payload("full", [_cycle(1), _cycle(2, discharge=0.0), _cycle(3), _cycle(3)])
    partial = _payload("partial", [_cycle(1), _cycle(2)], soc=(0.2, 0.8))
    paths = []
    for payload in (good, partial):
        path = tmp_path / f"SNL_18650_NMC_{payload['cell_id']}.pkl"
        pd.to_pickle(payload, path)
        paths.append(str(path))
    table, stats = ingest.build_table(paths, "snl_nmc", (3.65, 4.10), rated_ah=3.0)
    assert stats["partial_soc_cells_skipped"] == ["SNL_18650_NMC_partial"]
    assert set(table["cell_id"]) == {"full"}
    assert table["cycle_index"].tolist() == [1, 3]                   # soh == 0 dropped; exact duplicate collapsed
    assert stats["exact_duplicate_rows_dropped"] == 1 and stats["ambiguous_identity_rows_dropped"] == 0
    assert list(table.columns) == ingest.COLUMNS
    assert table["cycle_fraction"].tolist() == pytest.approx([1 / 3, 1.0])


def test_ambiguous_cycle_identities_are_dropped_whole(ingest, tmp_path):
    payload = _payload("amb", [_cycle(1), _cycle(1, discharge=2.6), _cycle(2)])
    path = tmp_path / "SNL_18650_NMC_amb.pkl"
    pd.to_pickle(payload, path)
    table, stats = ingest.build_table([str(path)], "snl_nmc", (3.65, 4.10), rated_ah=3.0)
    assert table["cycle_index"].tolist() == [2] and stats["ambiguous_identity_rows_dropped"] == 2


def test_ic_peak_finds_the_steepest_charge_region(ingest):
    v = np.linspace(3.4, 4.2, 800)
    # charge accumulates fastest around 3.8 V (a gaussian dQ/dV bump on a flat background)
    ic_true = 1.0 + 8.0 * np.exp(-((v - 3.8) / 0.03) ** 2)
    q = np.cumsum(ic_true) * (v[1] - v[0])
    value, voltage = ingest.ic_peak(v, q)
    assert 3.78 < voltage < 3.82 and 1.5 < value < 9.0
    assert ingest.ic_peak(v[:3], q[:3]) == (pytest.approx(np.nan, nan_ok=True), pytest.approx(np.nan, nan_ok=True))


def test_compare_tables_reports_match_rates_and_ungated_ic_columns(ingest):
    payload = _payload("c1", [_cycle(1), _cycle(2), _cycle(3)])
    frame = ingest.extract_cell.__wrapped__ if hasattr(ingest.extract_cell, "__wrapped__") else None
    rows = [ingest.cycle_row(payload, c, (3.65, 4.10), "snl_nmc", 3.0) for c in payload["cycle_data"]]
    ref = pd.DataFrame(rows)
    ref["cycle_fraction"] = ref["cycle_index"] / 3
    new = ref.copy()
    new.loc[0, "voltage_mean"] += 0.01
    new.loc[1, "ic_peak_ah_per_v"] *= 2
    report = ingest.compare_tables(new, ref)
    assert report["reference_rows_found"] == 3 and report["rebuilt_rows_not_in_reference"] == 0
    assert report["columns"]["voltage_mean"]["match_rate"] == pytest.approx(2 / 3)
    assert report["columns"]["soh"]["match_rate"] == 1.0
    assert report["columns"]["ic_peak_ah_per_v"]["gated"] is False
    assert report["all_gated_columns_at_least_99pct"] is False
