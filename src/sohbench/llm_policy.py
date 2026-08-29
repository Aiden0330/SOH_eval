"""LLM-driven search policy for the protocol-gated agent search.

The policy proposes candidates; every hard rule stays outside it — the
preflight/label-safety/window gates admit or exclude, the runner scores
under protocol, and the outer dataset remains invisible.  A wrong or
malicious proposal can therefore waste budget but never corrupt a verdict.

Transports (first available wins):

1. Anthropic Python SDK (``claude-opus-5``) — used when credentials resolve
   (``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` env, or an
   ``ant auth login`` profile on disk).
2. ``claude`` CLI in headless mode (``claude -p --output-format json``) —
   uses the local Claude Code login.

``llm_transport_status()`` reports availability so job submission can fail
fast with an actionable message instead of dying mid-search.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

LLM_MODEL = "claude-opus-5"
MAX_CANDIDATES_PER_ROUND = 4
DEFAULT_MAX_ROUNDS = 3

PROMPT_TEMPLATE = """You are the search policy inside a battery SOH benchmark's \
protocol-gated model-selection loop.

Goal: find the algorithm x featureset (x optional window coverage target) \
combination with the LOWEST WORST-CASE inner-dataset LOCO MAE that will also \
transfer to an unseen dataset (a held-out outer dataset you never observe — \
prefer dimensionless / physics-grounded features over dataset-scale ones).

Hard context (you cannot change these):
- Every proposal passes label-safety and window-DOD gates before scoring; \
excluded candidates are reported back to you with reasons.
- Inner datasets: {inner_datasets}. Scoring: per-dataset LOCO, ranked by \
worst-case MAE.
- Allowed estimators: {estimators}
- Allowed featuresets: {featuresets}
- window_target: null, or a fraction 0.20-0.60 (only meaningful with \
featureset "partial_window_norm"; requires a window bank for every inner \
dataset).

Search history so far (empty means this is round 1 — start broad):
{history}

Respond with ONLY a JSON object, no other text:
{{"proposals": [{{"estimator": "...", "featureset": "...", "window_target": null}}, ...],
 "reasoning": "one short paragraph",
 "stop": false}}

Rules: at most {max_candidates} proposals; never repeat an already-scored or \
already-excluded candidate; set "stop": true with empty proposals when you \
judge the search converged (scores plateaued or the space is exhausted)."""


def _profile_on_disk() -> bool:
    return (Path.home() / ".config" / "anthropic").is_dir()


def llm_transport_status() -> dict:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return {"available": True, "transport": "sdk", "detail": "environment credentials"}
    if _profile_on_disk():
        return {"available": True, "transport": "sdk", "detail": "ant auth profile"}
    cli = shutil.which("claude")
    if cli:
        return {"available": True, "transport": "cli", "detail": cli}
    return {
        "available": False,
        "transport": None,
        "detail": (
            "no Claude credentials: set ANTHROPIC_API_KEY (or run `ant auth login`) "
            "in the API server's environment, or install the claude CLI"
        ),
    }


def _invoke_sdk(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _invoke_cli(prompt: str) -> str:
    completed = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {completed.stderr.strip()[:300]}")
    payload = json.loads(completed.stdout)
    return str(payload.get("result") or "")


def default_transport(prompt: str) -> str:
    status = llm_transport_status()
    if not status["available"]:
        raise RuntimeError(f"LLM policy unavailable: {status['detail']}")
    if status["transport"] == "cli":
        return _invoke_cli(prompt)
    return _invoke_sdk(prompt)


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in policy response")
    return json.loads(match.group(0))


@dataclass
class LLMPolicy:
    """Claude-backed :class:`sohbench.agent_search.Policy`.

    ``transport`` is injectable for tests; the default resolves SDK/CLI at
    call time.  Any transport or parse failure converges the search (empty
    proposal) and is recorded in ``log_entries`` — the loop never crashes on
    the policy.
    """

    estimators: list[str]
    featuresets: list[str]
    inner_datasets: list[str]
    window_targets_allowed: bool = True
    max_rounds: int = DEFAULT_MAX_ROUNDS
    transport: object | None = None
    log_entries: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.transport is None:
            self.transport = default_transport

    def _history_payload(self, history) -> str:
        scored = list(history.scored().values())
        excluded = []
        for entry in history.rounds:
            excluded.extend(entry.get("excluded") or [])
        compact = {
            "scored": [
                {
                    "candidate": item["candidate"],
                    "mae_worst": item.get("mae_worst"),
                    "mae_mean": item.get("mae_mean"),
                    "per_dataset": {
                        name: values.get("mae")
                        for name, values in (item.get("per_dataset") or {}).items()
                    },
                }
                for item in scored
            ],
            "excluded": excluded,
        }
        return json.dumps(compact, ensure_ascii=False)

    def propose(self, history):
        from .agent_search import Candidate

        if len(history.rounds) >= self.max_rounds:
            return []
        prompt = PROMPT_TEMPLATE.format(
            inner_datasets=", ".join(self.inner_datasets),
            estimators=", ".join(sorted(self.estimators)),
            featuresets=", ".join(sorted(self.featuresets)),
            history=self._history_payload(history),
            max_candidates=MAX_CANDIDATES_PER_ROUND,
        )
        try:
            raw = self.transport(prompt)
            payload = _extract_json(raw)
        except Exception as exc:  # noqa: BLE001 - policy failure = converge
            self.log_entries.append({"round": len(history.rounds), "error": str(exc)})
            return []
        reasoning = str(payload.get("reasoning") or "")
        seen = set(history.scored())
        for entry in history.rounds:
            for item in entry.get("excluded") or []:
                seen.add(item.get("candidate"))
        proposals: list[Candidate] = []
        rejected: list[str] = []
        for item in payload.get("proposals") or []:
            estimator = str(item.get("estimator") or "")
            featureset = str(item.get("featureset") or "")
            target = item.get("window_target")
            if estimator not in self.estimators or featureset not in self.featuresets:
                rejected.append(f"{estimator}×{featureset} (unknown name)")
                continue
            if target is not None:
                if not self.window_targets_allowed or featureset != "partial_window_norm":
                    rejected.append(f"{estimator}×{featureset} (window_target not allowed here)")
                    continue
                try:
                    target = float(target)
                except (TypeError, ValueError):
                    rejected.append(f"{estimator}×{featureset} (bad window_target)")
                    continue
                if not 0.2 <= target <= 0.6:
                    rejected.append(f"{estimator}×{featureset} (window_target out of range)")
                    continue
            candidate = Candidate(estimator, featureset, target)
            if candidate.key() in seen or any(c.key() == candidate.key() for c in proposals):
                rejected.append(f"{candidate.key()} (duplicate)")
                continue
            proposals.append(candidate)
            if len(proposals) >= MAX_CANDIDATES_PER_ROUND:
                break
        self.log_entries.append(
            {
                "round": len(history.rounds),
                "reasoning": reasoning,
                "proposed": [c.key() for c in proposals],
                "rejected": rejected,
                "stop": bool(payload.get("stop")),
            }
        )
        if payload.get("stop"):
            return []
        return proposals
