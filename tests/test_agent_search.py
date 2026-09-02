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
        lambda config=None: {"available": False, "transport": None, "detail": "no Claude credentials"},
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


def test_llm_config_roundtrip_masks_and_persists_key(tmp_path):
    from sohbench.evaluation_service import EvaluationService
    from sohbench.llm_policy import load_llm_config

    service = EvaluationService(
        data_dir=tmp_path / "data",
        work_dir=tmp_path / "work",
        runs_dir=tmp_path / "runs",
        frontend_data_dir=tmp_path / "frontend_data",
    )
    initial = service.get_llm_config()
    assert initial["config"]["has_api_key"] is False

    saved = service.set_llm_config({
        "model": "claude-opus-5",
        "base_url": "https://gateway.example/v1",
        "api_key": "sk-test-123",
    })
    assert saved["config"]["has_api_key"] is True
    assert "sk-test" not in str(saved)  # the key is never echoed
    assert saved["status"]["available"] is True
    assert load_llm_config(tmp_path / "work")["api_key"] == "sk-test-123"

    # Empty key keeps the stored one; clear_api_key wipes it.
    kept = service.set_llm_config({"model": "claude-opus-5", "api_key": ""})
    assert kept["config"]["has_api_key"] is True
    cleared = service.set_llm_config({"clear_api_key": True})
    assert cleared["config"]["has_api_key"] is False


def test_inner_ranking_penalizes_failed_folds():
    from sohbench.agent_search import inner_ranking_key

    clean_slow = {"candidate": "a", "mae_worst": 0.05, "failed_folds": 0}
    fast_but_crashy = {"candidate": "b", "mae_worst": 0.01, "failed_folds": 3}
    unscored = {"candidate": "c", "mae_worst": float("nan"), "failed_folds": 0}
    ranked = sorted([unscored, fast_but_crashy, clean_slow], key=inner_ranking_key)
    assert [item["candidate"] for item in ranked] == ["a", "b", "c"]


def test_llm_adaptation_policy_lists_and_enforces_admitted_operators():
    import json

    from sohbench.agent_search import SearchHistory
    from sohbench.llm_policy import AdaptationLLMPolicy

    prompts = []

    def transport(prompt):
        prompts.append(prompt)
        return json.dumps({
            "proposals": [
                {"model": "m1", "operator": "residual_wide+monotone"},
                {"model": "m1", "operator": "coral"},
            ],
            "reasoning": "r",
            "stop": False,
        })

    policy = AdaptationLLMPolicy(
        models=[{"name": "m1", "estimator": "gpr", "featureset": "partial_window_norm",
                 "trained_on": ["hust"]}],
        target_dataset="snl_lfp",
        calibration_budget=64,
        transport=transport,
        allowed_operators=["none", "residual_wide+monotone"],
    )
    proposals = policy.propose(SearchHistory())
    assert [c.key() for c in proposals] == ["m1 · residual_wide+monotone"]
    assert '"residual_wide+monotone"' in prompts[0]
    assert '"coral"' not in prompts[0]
    assert any("not admitted" in item for item in policy.log_entries[-1].get("rejected", []))


def test_llm_config_file_is_created_private_and_atomically(tmp_path):
    import os
    import sys

    from sohbench.llm_policy import CONFIG_FILENAME, save_llm_config

    save_llm_config(tmp_path, {"provider": "anthropic", "api_key": "sk-test"})
    path = tmp_path / CONFIG_FILENAME
    assert path.is_file()
    if sys.platform != "win32":
        assert os.stat(path).st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".*.tmp"))  # temp file swapped in, not left behind


def test_search_reports_are_json_safe():
    import json

    from sohbench.agent_search import json_safe

    report = json_safe({
        "ranking": [{"mae_worst": float("nan"), "mae_mean": np.float64(0.1), "folds": np.int64(3)}],
        "flag": np.bool_(True),
        "nested": [float("inf"), 1.5],
    })
    assert report["ranking"][0]["mae_worst"] is None
    assert report["ranking"][0]["mae_mean"] == 0.1 and report["ranking"][0]["folds"] == 3
    assert report["flag"] is True and report["nested"] == [None, 1.5]
    json.dumps(report, allow_nan=False)  # what the API does — must not raise


def test_openai_compatible_transport_posts_chat_completions(monkeypatch):
    """The OpenAI-compatible provider must work against any ChatCompletions
    gateway: bearer auth, /v1 join, content parsing — no SDK involved."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from sohbench.llm_policy import (
        default_transport,
        llm_transport_status,
        openai_chat_completions_url,
    )

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(length))
            payload = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": (
                    'ok {"proposals": [], "reasoning": "r", "stop": true}'
                )}}],
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):  # keep pytest output clean
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = {"provider": "openai", "model": "demo-model", "base_url": f"http://{host}:{port}", "api_key": "sk-demo"}
        status = llm_transport_status(config)
        assert status["available"] is True and status["transport"] == "openai"
        text = default_transport("hello policy", config)
        assert "proposals" in text
        assert seen["path"] == "/v1/chat/completions"
        assert seen["auth"] == "Bearer sk-demo"
        assert seen["body"]["model"] == "demo-model"
        assert seen["body"]["messages"] == [{"role": "user", "content": "hello policy"}]
    finally:
        server.shutdown()
        server.server_close()

    assert openai_chat_completions_url("https://host/v1") == "https://host/v1/chat/completions"
    assert openai_chat_completions_url("https://host/") == "https://host/v1/chat/completions"
    assert openai_chat_completions_url("") == "https://api.openai.com/v1/chat/completions"


def test_openai_provider_status_requires_key_and_model(monkeypatch, tmp_path):
    from sohbench.llm_policy import llm_transport_status, save_llm_config

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_transport_status({"provider": "openai", "model": "m"})["available"] is False
    missing_model = llm_transport_status({"provider": "openai", "api_key": "k"})
    assert missing_model["available"] is False and "model" in missing_model["detail"]
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    env_status = llm_transport_status({"provider": "openai", "model": "m"})
    assert env_status["available"] is True and "OPENAI_API_KEY" in env_status["detail"]
    saved = save_llm_config(tmp_path, {"provider": "openai", "model": "m", "api_key": "k"})
    assert saved["provider"] == "openai" and "openai" in saved["providers"]
    with pytest.raises(ValueError, match="unknown AI provider"):
        save_llm_config(tmp_path, {"provider": "gemini"})
