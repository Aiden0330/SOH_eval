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


CONFIG_FILENAME = "llm_config.json"
CONFIG_FIELDS = ("provider", "model", "base_url", "api_key")


def load_llm_config(work_dir: str | Path) -> dict:
    path = Path(work_dir) / CONFIG_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {key: str(payload.get(key) or "") for key in CONFIG_FIELDS}


def save_llm_config(work_dir: str | Path, payload: dict) -> dict:
    """Merge-and-persist connection settings; the key is write-only.

    An empty/absent ``api_key`` keeps the stored one; ``clear_api_key`` wipes
    it.  The file is chmod 0600 — it holds a credential.
    """
    current = load_llm_config(work_dir)
    updated = dict(current)
    for key in ("provider", "model", "base_url"):
        if key in payload:
            updated[key] = str(payload.get(key) or "").strip()
    if payload.get("clear_api_key"):
        updated["api_key"] = ""
    elif str(payload.get("api_key") or "").strip():
        updated["api_key"] = str(payload["api_key"]).strip()
    path = Path(work_dir) / CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return masked_llm_config(updated)


def masked_llm_config(config: dict) -> dict:
    return {
        "provider": str(config.get("provider") or "anthropic"),
        "model": str(config.get("model") or ""),
        "base_url": str(config.get("base_url") or ""),
        "has_api_key": bool(str(config.get("api_key") or "").strip()),
    }


def llm_transport_status(config: dict | None = None) -> dict:
    config = config or {}
    if str(config.get("api_key") or "").strip():
        target = str(config.get("base_url") or "") or "api.anthropic.com"
        return {"available": True, "transport": "sdk", "detail": f"workbench key → {target}"}
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
            "no Claude credentials: save an API key in the panel's AI 连接设置, "
            "set ANTHROPIC_API_KEY (or run `ant auth login`) in the API server's "
            "environment, or install the claude CLI"
        ),
    }


def _invoke_sdk(prompt: str, config: dict | None = None) -> str:
    import anthropic

    config = config or {}
    kwargs = {}
    if str(config.get("api_key") or "").strip():
        kwargs["api_key"] = str(config["api_key"]).strip()
    if str(config.get("base_url") or "").strip():
        kwargs["base_url"] = str(config["base_url"]).strip()
    client = anthropic.Anthropic(**kwargs)
    response = client.messages.create(
        model=str(config.get("model") or "").strip() or LLM_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _invoke_cli(prompt: str, config: dict | None = None) -> str:
    command = ["claude", "-p", prompt, "--output-format", "json"]
    model = str((config or {}).get("model") or "").strip()
    if model:
        command[1:1] = ["--model", model]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {completed.stderr.strip()[:300]}")
    payload = json.loads(completed.stdout)
    return str(payload.get("result") or "")


def default_transport(prompt: str, config: dict | None = None) -> str:
    status = llm_transport_status(config)
    if not status["available"]:
        raise RuntimeError(f"LLM policy unavailable: {status['detail']}")
    if status["transport"] == "cli":
        return _invoke_cli(prompt, config)
    return _invoke_sdk(prompt, config)


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
    config: dict | None = None
    log_entries: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.transport is None:
            connection = self.config
            self.transport = lambda prompt: default_transport(prompt, connection)

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


ADAPTATION_PROMPT_TEMPLATE = """You are the adaptation policy inside a battery SOH \
benchmark's inference-phase search.

Setting: trained model artifacts are FROZEN (algorithm and features cannot \
change). Your job is to pair models with deployment-legal adaptation \
operators to minimize MAE on the target dataset "{target_dataset}".

Operators (information channels are enforced and reported by the harness):
- "none": the frozen model as-is.
- "coral": align target feature distribution to the model's training \
distribution (unlabeled statistics only).
- "recalibration": two-parameter output correction fitted on {budget} labeled \
calibration samples (excluded from evaluation).
- "coral+recalibration": both.

Candidate models (name · estimator · featureset · trained_on):
{models}

Search history so far (empty = round 1; prefer starting with a "none" \
baseline for each promising model so deltas are interpretable):
{history}

Respond with ONLY a JSON object:
{{"proposals": [{{"model": "...", "operator": "none|coral|recalibration|coral+recalibration"}}, ...],
 "reasoning": "one short paragraph",
 "stop": false}}

Rules: at most {max_candidates} proposals; never repeat an already-scored or \
excluded candidate; models trained on chemistry/scale similar to the target \
usually transfer best; set "stop": true when converged."""


@dataclass
class AdaptationLLMPolicy:
    """Claude-backed policy for the inference-phase adaptation search."""

    models: list[dict]  # catalog-style entries: name/estimator/featureset/trained_on
    target_dataset: str
    calibration_budget: int
    max_rounds: int = DEFAULT_MAX_ROUNDS
    transport: object | None = None
    config: dict | None = None
    log_entries: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.transport is None:
            connection = self.config
            self.transport = lambda prompt: default_transport(prompt, connection)

    def propose(self, history):
        from .adaptation import ADAPTATION_OPERATORS
        from .agent_search import AdaptationCandidate

        if len(history.rounds) >= self.max_rounds:
            return []
        model_lines = "\n".join(
            f'- {item.get("name")} · {item.get("estimator")} · {item.get("featureset")}'
            f' · trained_on={",".join(item.get("trained_on") or [])}'
            for item in self.models
        )
        scored = history.scored()
        excluded_keys = set()
        for entry in history.rounds:
            for item in entry.get("excluded") or []:
                excluded_keys.add(item.get("candidate"))
        compact_history = json.dumps(
            {
                "scored": [
                    {
                        "candidate": item["candidate"],
                        "target_mae": item.get("target_mae"),
                        "target_bias": item.get("target_bias"),
                    }
                    for item in scored.values()
                ],
                "excluded": sorted(excluded_keys),
            },
            ensure_ascii=False,
        )
        prompt = ADAPTATION_PROMPT_TEMPLATE.format(
            target_dataset=self.target_dataset,
            budget=self.calibration_budget,
            models=model_lines,
            history=compact_history,
            max_candidates=MAX_CANDIDATES_PER_ROUND,
        )
        try:
            payload = _extract_json(self.transport(prompt))
        except Exception as exc:  # noqa: BLE001 - policy failure = converge
            self.log_entries.append({"round": len(history.rounds), "error": str(exc)})
            return []
        known_models = {str(item.get("name")) for item in self.models}
        seen = set(scored) | excluded_keys
        proposals = []
        rejected = []
        for item in payload.get("proposals") or []:
            model = str(item.get("model") or "")
            operator = str(item.get("operator") or "")
            if model not in known_models:
                rejected.append(f"{model} (unknown model)")
                continue
            if operator not in ADAPTATION_OPERATORS:
                rejected.append(f"{model} · {operator} (unknown operator)")
                continue
            candidate = AdaptationCandidate(model, operator)
            if candidate.key() in seen or any(c.key() == candidate.key() for c in proposals):
                rejected.append(f"{candidate.key()} (duplicate)")
                continue
            proposals.append(candidate)
            if len(proposals) >= MAX_CANDIDATES_PER_ROUND:
                break
        self.log_entries.append(
            {
                "round": len(history.rounds),
                "reasoning": str(payload.get("reasoning") or ""),
                "proposed": [c.key() for c in proposals],
                "rejected": rejected,
                "stop": bool(payload.get("stop")),
            }
        )
        return [] if payload.get("stop") else proposals
