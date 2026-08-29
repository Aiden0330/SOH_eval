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


def test_llm_policy_parses_validates_and_converges():
    from sohbench.agent_search import SearchHistory
    from sohbench.llm_policy import LLMPolicy

    responses = iter([
        # Round 1: two valid proposals, one unknown name, one duplicate.
        '{"proposals": [{"estimator": "ridge", "featureset": "voltage"},'
        ' {"estimator": "gbdt", "featureset": "partial_window_norm", "window_target": 0.45},'
        ' {"estimator": "nope", "featureset": "voltage"},'
        ' {"estimator": "ridge", "featureset": "voltage"}],'
        ' "reasoning": "start broad", "stop": false}',
        # Round 2: converged.
        '{"proposals": [], "reasoning": "plateaued", "stop": true}',
    ])
    policy = LLMPolicy(
        estimators=["ridge", "gbdt"],
        featuresets=["voltage", "partial_window_norm"],
        inner_datasets=["a", "b"],
        transport=lambda prompt: next(responses),
    )
    history = SearchHistory()
    first = policy.propose(history)
    assert [c.key() for c in first] == ["ridge×voltage", "gbdt×partial_window_norm@w45%"]
    assert policy.log_entries[0]["rejected"]
    history.rounds.append({"round": 0, "scores": [], "excluded": []})
    assert policy.propose(history) == []
    assert policy.log_entries[1]["stop"] is True


def test_llm_policy_transport_failure_converges_not_crashes():
    from sohbench.agent_search import SearchHistory
    from sohbench.llm_policy import LLMPolicy

    def broken(prompt):
        raise RuntimeError("no credentials")

    policy = LLMPolicy(
        estimators=["ridge"], featuresets=["voltage"], inner_datasets=["a"],
        transport=broken,
    )
    assert policy.propose(SearchHistory()) == []
    assert "no credentials" in policy.log_entries[0]["error"]


def test_llm_search_job_requires_available_transport(tmp_path, monkeypatch):
    from pathlib import Path

    from sohbench import llm_policy
    from sohbench.evaluation_service import EvaluationService

    data_dir = Path(__file__).resolve().parents[1] / "data"
    service = EvaluationService(
        data_dir=data_dir,
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    monkeypatch.setattr(
        llm_policy, "llm_transport_status",
        lambda: {"available": False, "transport": None, "detail": "no Claude credentials"},
    )
    with pytest.raises(ValueError, match="AI 策略不可用"):
        service.create_search_job(
            {
                "inner_datasets": ["soh_quickstart_demo"],
                "outer_dataset": "smoke_eval_demo",
                "estimators": ["ridge"],
                "featuresets": ["voltage"],
                "policy": "llm",
            },
            start=False,
        )
