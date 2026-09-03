"""Window bank × agent search: a tuned-window candidate without a bank is gate-excluded.

Workbench-level test (needs EvaluationService + AgentSearch); the kernel-level window
bank tests live in tests/kernel/test_window_bank.py.
"""

import pytest

from sohbench.paths import data_root


def test_tuned_window_candidate_without_bank_is_gate_excluded(tmp_path):
    from sohbench.agent_search import AgentSearch, Candidate, GridPolicy
    from sohbench.evaluation_service import EvaluationService

    data_dir = data_root()
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
