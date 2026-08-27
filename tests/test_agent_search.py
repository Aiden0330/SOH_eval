"""Protocol-gated agent search: loop structure, gating, and outer integrity."""

import numpy as np
from pathlib import Path

import pytest

from sohbench.agent_search import AgentSearch, Candidate, GridPolicy
from sohbench.evaluation_service import EvaluationService

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _datasets_present(*names: str) -> bool:
    return all((DATA_DIR / "processed" / f"{name}_features.pkl").is_file() for name in names)


@pytest.mark.skipif(
    not _datasets_present("soh_quickstart_demo", "smoke_eval_demo"),
    reason="demo datasets not present",
)
def test_grid_search_gates_scores_and_claims_once(tmp_path):
    service = EvaluationService(
        data_dir=DATA_DIR,
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
    # partial_window selects zero usable columns on the demo dataset — the
    # preflight gate must exclude it with a reason instead of running it.
    policy = GridPolicy(["ridge"], ["voltage", "partial_window"])
    report = search.run(policy, log=lambda *_: None)

    assert [item["candidate"] for item in report["ranking"]] == ["ridge×voltage"]
    assert report["excluded"] and "partial_window" in report["excluded"][0]["candidate"]
    assert report["excluded"][0]["reasons"]

    winner = report["ranking"][0]
    assert np.isfinite(winner["mae_worst"])
    assert winner["per_dataset"]["soh_quickstart_demo"]["failed_folds"] == 0

    # Single-shot outer claim on the held-out dataset only.
    assert len(report["outer_claims"]) == 1
    claim = report["outer_claims"][0]
    assert claim["outer_dataset"] == "smoke_eval_demo"
    assert np.isfinite(claim["outer_mae"])
    assert report["integrity"]["outer_dataset_visible_during_search"] is False


def test_outer_dataset_must_stay_out_of_the_inner_pool(tmp_path):
    service = EvaluationService(
        data_dir=DATA_DIR,
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    with pytest.raises(ValueError, match="outer dataset"):
        AgentSearch(
            service,
            inner_datasets=["soh_quickstart_demo", "smoke_eval_demo"],
            outer_dataset="smoke_eval_demo",
        )


def test_grid_policy_converges_after_one_round():
    policy = GridPolicy(["ridge"], ["voltage"])

    class _History:
        rounds = []

    first = policy.propose(_History())
    assert first == [Candidate("ridge", "voltage")]
    _History.rounds = [{"round": 0}]
    assert policy.propose(_History()) == []
