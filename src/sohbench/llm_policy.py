"""LLM-driven search policy for the protocol-gated agent search.

The policy proposes candidates; every hard rule stays outside it — the
preflight/label-safety/window gates admit or exclude, the runner scores
under protocol, and the outer dataset remains invisible.  A wrong or
malicious proposal can therefore waste budget but never corrupt a verdict.

Providers / transports (``provider`` in the workbench AI settings):

- ``anthropic`` (default): Anthropic Python SDK (``claude-opus-5``) when
  credentials resolve (workbench key, ``ANTHROPIC_API_KEY`` /
  ``ANTHROPIC_AUTH_TOKEN`` env, or an ``ant auth login`` profile on disk);
  otherwise the ``claude`` CLI in headless mode (local Claude Code login).
- ``openai``: any OpenAI-compatible ChatCompletions endpoint (OpenAI itself,
  DeepSeek, Qwen/DashScope, Moonshot, vLLM/Ollama gateways, ...) over plain
  HTTP — ``POST {base_url}/chat/completions`` with a bearer key; no SDK
  dependency.  Credentials: workbench key or ``OPENAI_API_KEY`` (+ optional
  ``OPENAI_BASE_URL``); the model name is mandatory because it is
  gateway-specific.

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
LLM_PROVIDERS = ("anthropic", "openai")
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_TIMEOUT_SECONDS = 300


def _provider(config: dict | None) -> str:
    return str((config or {}).get("provider") or "anthropic").strip().lower() or "anthropic"


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
    provider = str(updated.get("provider") or "anthropic").strip().lower()
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"unknown AI provider {provider!r}; expected one of {list(LLM_PROVIDERS)}")
    updated["provider"] = provider
    if payload.get("clear_api_key"):
        updated["api_key"] = ""
    elif str(payload.get("api_key") or "").strip():
        updated["api_key"] = str(payload["api_key"]).strip()
    path = Path(work_dir) / CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create the file 0600 from the start and swap it in atomically: a
    # plain write_text() followed by chmod leaves a world-readable window
    # (default umask) around a credential, and a crash mid-write would leave
    # a truncated config behind.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(updated, ensure_ascii=False, indent=2))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return masked_llm_config(updated)


def masked_llm_config(config: dict) -> dict:
    return {
        "provider": _provider(config),
        "providers": list(LLM_PROVIDERS),
        "model": str(config.get("model") or ""),
        "base_url": str(config.get("base_url") or ""),
        "has_api_key": bool(str(config.get("api_key") or "").strip()),
    }


def llm_transport_status(config: dict | None = None) -> dict:
    config = config or {}
    if _provider(config) == "openai":
        return _openai_status(config)
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


def _openai_status(config: dict) -> dict:
    model = str(config.get("model") or "").strip()
    base_url = str(config.get("base_url") or "").strip() or os.environ.get("OPENAI_BASE_URL", "").strip() or OPENAI_DEFAULT_BASE_URL
    has_key = bool(str(config.get("api_key") or "").strip() or os.environ.get("OPENAI_API_KEY"))
    if not has_key:
        return {
            "available": False,
            "transport": None,
            "detail": (
                "no OpenAI-compatible credentials: save an API key in the panel's "
                "AI 连接设置 or set OPENAI_API_KEY in the API server's environment"
            ),
        }
    if not model:
        return {
            "available": False,
            "transport": None,
            "detail": (
                "OpenAI-compatible provider needs a model name (gateway-specific, "
                "e.g. gpt-4o / deepseek-chat / qwen-plus) — fill in 模型 in AI 连接设置"
            ),
        }
    source = "workbench key" if str(config.get("api_key") or "").strip() else "OPENAI_API_KEY"
    return {"available": True, "transport": "openai", "detail": f"{source} → {base_url} · {model}"}


def openai_chat_completions_url(base_url: str) -> str:
    """Join a user-entered base URL with the chat-completions path.

    Gateways document their base as either ``https://host`` or
    ``https://host/v1``; both must resolve to ``.../v1/chat/completions``
    (a base that already ends in ``/chat/completions`` is used verbatim)."""
    base = str(base_url or "").strip().rstrip("/") or OPENAI_DEFAULT_BASE_URL
    if base.endswith("/chat/completions"):
        return base
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def _openai_message_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI-compatible response carries no choices")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # some gateways return content parts
        content = "".join(
            str(part.get("text") or "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenAI-compatible response has empty message content")
    return content


def _invoke_openai(prompt: str, config: dict | None = None) -> str:
    """OpenAI-compatible ChatCompletions over urllib (no SDK dependency)."""
    import urllib.error
    import urllib.request

    config = config or {}
    api_key = str(config.get("api_key") or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OpenAI-compatible provider: no API key configured")
    model = str(config.get("model") or "").strip()
    if not model:
        raise RuntimeError("OpenAI-compatible provider: model name is required")
    base_url = str(config.get("base_url") or "").strip() or os.environ.get("OPENAI_BASE_URL", "").strip()
    url = openai_chat_completions_url(base_url)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"OpenAI-compatible endpoint {url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible endpoint {url} unreachable: {exc.reason}") from exc
    return _openai_message_text(payload)


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
    if status["transport"] == "openai":
        return _invoke_openai(prompt, config)
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

Operators (information channels are enforced and reported by the harness; \
"K" below is the labeled calibration budget = {budget} target samples, always \
excluded from evaluation):
{operators}

Candidate models (name · estimator · featureset · trained_on):
{models}

Search history so far (empty = round 1; prefer starting with a "none" \
baseline for each promising model so deltas are interpretable):
{history}

Respond with ONLY a JSON object:
{{"proposals": [{{"model": "...", "operator": "<one of the operator names above>"}}, ...],
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
    # Operators the search request admits; the prompt lists exactly these
    # (with their declared channels) and proposals outside them are rejected.
    allowed_operators: tuple[str, ...] | list[str] | None = None
    log_entries: list[dict] = field(default_factory=list)

    def __post_init__(self):
        from .adaptation import ADAPTATION_OPERATORS

        if self.transport is None:
            connection = self.config
            self.transport = lambda prompt: default_transport(prompt, connection)
        requested = set(self.allowed_operators or ADAPTATION_OPERATORS)
        self.allowed_operators = tuple(
            op for op in ADAPTATION_OPERATORS if op in requested
        ) or tuple(ADAPTATION_OPERATORS)

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
        from .adaptation import OPERATOR_CHANNELS

        allowed = [op for op in ADAPTATION_OPERATORS if op in self.allowed_operators]
        operator_lines = "\n".join(
            f'- "{operator}": {OPERATOR_CHANNELS[operator]}' for operator in allowed
        )
        prompt = ADAPTATION_PROMPT_TEMPLATE.format(
            target_dataset=self.target_dataset,
            budget=self.calibration_budget,
            operators=operator_lines,
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
            if operator not in self.allowed_operators:
                rejected.append(f"{model} · {operator} (operator not admitted by this search)")
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
