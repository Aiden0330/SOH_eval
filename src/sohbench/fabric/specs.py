from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Literal

import jsonschema

from .ids import sha256_json

RunType = Literal["training", "adaptation", "inference"]
_RUN_TYPES = frozenset(("training", "adaptation", "inference"))
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_string(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")


@dataclass(frozen=True)
class SpecRef:
    kind: str
    spec_id: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        for field in ("kind", "spec_id", "version", "digest"):
            _require_string(getattr(self, field), field)
        if not _DIGEST_PATTERN.fullmatch(self.digest):
            raise ValueError("digest must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True)
class CandidateSpec:
    spec_id: str
    artifacts: tuple[SpecRef, ...]
    version: str = "1.0.0"
    kind: str = "candidate"

    def __post_init__(self) -> None:
        for field in ("kind", "spec_id", "version"):
            _require_string(getattr(self, field), field)
        if self.kind != "candidate":
            raise ValueError("candidate kind must be 'candidate'")
        if not isinstance(self.artifacts, (list, tuple)):
            raise TypeError("artifacts must be a list or tuple of SpecRef")
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ValueError("candidate artifacts must not be empty")
        if any(not isinstance(item, SpecRef) for item in artifacts):
            raise TypeError("artifacts must contain only SpecRef values")
        artifacts = tuple(
            sorted(
                artifacts,
                key=lambda item: (
                    item.kind,
                    item.spec_id,
                    item.version,
                    item.digest,
                ),
            )
        )
        identities = [
            (item.kind, item.spec_id, item.version) for item in artifacts
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("candidate artifact references must be unique")
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "kind": self.kind,
                "spec_id": self.spec_id,
                "version": self.version,
                "artifacts": [asdict(item) for item in self.artifacts],
            }
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "spec_id": self.spec_id,
            "version": self.version,
            "digest": self.digest,
            "artifacts": [asdict(item) for item in self.artifacts],
        }


@dataclass(frozen=True)
class RunSpec:
    run_type: RunType
    task: SpecRef
    dataset: SpecRef
    protocol: SpecRef
    metric: SpecRef
    policy: SpecRef
    candidates: tuple[SpecRef | CandidateSpec, ...]
    seeds: tuple[int, ...] = (0,)
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        _require_string(self.run_type, "run_type")
        if self.run_type not in _RUN_TYPES:
            raise ValueError(f"run_type must be one of {sorted(_RUN_TYPES)}")
        for field in ("task", "dataset", "protocol", "metric", "policy"):
            if not isinstance(getattr(self, field), SpecRef):
                raise TypeError(f"{field} must be a SpecRef")
        if not isinstance(self.candidates, (list, tuple)):
            raise TypeError("candidates must be a list or tuple of SpecRef")
        candidates = tuple(self.candidates)
        if any(
            not isinstance(item, (SpecRef, CandidateSpec))
            for item in candidates
        ):
            raise TypeError(
                "candidates must contain only SpecRef or CandidateSpec values"
            )
        object.__setattr__(self, "candidates", candidates)
        if not isinstance(self.seeds, (list, tuple)):
            raise TypeError("seeds must be a list or tuple of int")
        seeds = tuple(self.seeds)
        if any(type(seed) is not int for seed in seeds):
            raise TypeError("seeds must contain only int values")
        object.__setattr__(self, "seeds", seeds)
        _require_string(self.idempotency_key, "idempotency_key")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["candidates"] = [
            item.to_dict() if isinstance(item, CandidateSpec) else asdict(item)
            for item in self.candidates
        ]
        value["seeds"] = list(self.seeds)
        return value

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def comparison_key(self) -> str:
        return sha256_json({
            "task": asdict(self.task),
            "dataset": asdict(self.dataset),
            "protocol": asdict(self.protocol),
            "metric": asdict(self.metric),
            "policy": asdict(self.policy),
        })


def validate_run_spec_document(document: dict) -> None:
    schema_path = files("sohbench.fabric.schemas").joinpath("run-spec.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError as exc:
        location = ".".join(map(str, exc.absolute_path)) or "run_spec"
        raise ValueError(f"{location}: {exc.message}") from exc
    for index, candidate_document in enumerate(document["candidates"]):
        if "artifacts" not in candidate_document:
            continue
        candidate = CandidateSpec(
            spec_id=candidate_document["spec_id"],
            version=candidate_document["version"],
            artifacts=tuple(
                SpecRef(**artifact)
                for artifact in candidate_document["artifacts"]
            ),
        )
        if candidate_document["digest"] != candidate.digest:
            raise ValueError(
                f"candidates.{index}.digest conflicts with artifact references"
            )
        if candidate_document["artifacts"] != candidate.to_dict()["artifacts"]:
            raise ValueError(
                f"candidates.{index}.artifacts must use canonical order"
            )
