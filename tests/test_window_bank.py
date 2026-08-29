"""Window bank: lattice integration, label-free bounds selection, substitution."""

import numpy as np
import pandas as pd
import pytest

from sohbench.window_bank import (
    apply_bank_window,
    bank_from_curves,
    lattice_edges,
    select_bounds,
)


def _linear_charge_curve(cell, cycle, capacity):
    # Voltage rises linearly 3.0 -> 3.6 while charge accumulates linearly to
    # `capacity`, so a window [lo, hi] integrates capacity * (hi-lo)/0.6.
    v = np.linspace(3.0, 3.6, 61)
    q = np.linspace(0.0, capacity, 61)
    t = np.linspace(0.0, 3600.0, 61)
    return (cell, cycle, v, q, t)


def test_bank_from_curves_integrates_the_lattice():
    edges = lattice_edges("hust")
    curves = [_linear_charge_curve("c1", 1, 1.2)]
    bank = bank_from_curves(curves, edges)
    windows = len(edges) * (len(edges) - 1) // 2
    assert len(bank) == windows
    row = bank[(bank["lo_mv"] == 3300) & (bank["hi_mv"] == 3500)].iloc[0]
    assert row["q_win_ah"] == pytest.approx(1.2 * (0.2 / 0.6), rel=1e-6)
    assert row["t_win_s"] == pytest.approx(3600.0 * (0.2 / 0.6), rel=1e-6)
    # Full-range window integrates everything.
    full = bank[(bank["lo_mv"] == 3000) & (bank["hi_mv"] == 3600)].iloc[0]
    assert full["q_win_ah"] == pytest.approx(1.2, rel=1e-6)


def test_select_bounds_hits_the_coverage_target_label_free():
    edges = lattice_edges("hust")
    curves = [_linear_charge_curve("c1", cycle, 1.2) for cycle in range(80)]
    bank = bank_from_curves(curves, edges)
    rated = {"c1": 1.2}
    chosen = select_bounds(bank, rated, 0.45)
    # 45% of a linear 3.0-3.6 curve = 0.27 V span; nearest lattice span is
    # 0.25 V -> median fraction ~0.4167.
    assert (chosen["hi_mv"] - chosen["lo_mv"]) == 250
    assert chosen["median_fraction"] == pytest.approx(0.25 / 0.6, rel=1e-3)
    assert chosen["rows"] == 80


def test_apply_bank_window_substitutes_only_that_dataset():
    edges = lattice_edges("hust")
    bank = bank_from_curves([_linear_charge_curve("c1", 1, 1.2)], edges)
    table = pd.DataFrame({
        "dataset": ["hust", "other"],
        "cell_id": ["c1", "c1"],
        "cycle_index": [1, 1],
        "soh": [1.0, 1.0],
        "q_win_ah": [9.9, 7.7],
        "t_win_s": [1.0, 2.0],
    })
    out = apply_bank_window(table, "hust", bank, 3300, 3500)
    assert out.loc[0, "q_win_ah"] == pytest.approx(0.4, rel=1e-6)
    assert out.loc[1, "q_win_ah"] == 7.7  # untouched foreign dataset row
    # Cycles absent from the bank become NaN, never stale values.
    table2 = table.assign(cycle_index=[42, 1])
    out2 = apply_bank_window(table2, "hust", bank, 3300, 3500)
    assert np.isnan(out2.loc[0, "q_win_ah"])


def test_tuned_window_candidate_without_bank_is_gate_excluded(tmp_path):
    from pathlib import Path

    from sohbench.agent_search import AgentSearch, Candidate, GridPolicy
    from sohbench.evaluation_service import EvaluationService

    data_dir = Path(__file__).resolve().parents[1] / "data"
    if not (data_dir / "processed" / "soh_quickstart_demo_features.pkl").is_file():
        pytest.skip("demo datasets not present")
    service = EvaluationService(
        data_dir=data_dir,
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    search = AgentSearch(
        service,
        inner_datasets=["soh_quickstart_demo"],
        outer_dataset="smoke_eval_demo",
        top_k=1,
    )
    policy = GridPolicy(["ridge"], ["voltage"], window_targets=[0.45])
    report = search.run(policy, log=lambda *_: None)
    assert [item["candidate"] for item in report["ranking"]] == ["ridge×voltage"]
    # The tuned candidate must be hard-gated out (here by preflight: the demo
    # dataset has no window columns at all), never silently scored.
    tuned = next(
        item for item in report["excluded"] if "@w45%" in item["candidate"]
    )
    assert tuned["reasons"]
