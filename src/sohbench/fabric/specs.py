from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Literal

import jsonschema

from .ids import sha256_json

RunType = Literal["training", "adaptation", "inference"]


@dataclass(frozen=True)
class SpecRef:
    kind: str
    spec_id: str
    version: str
    digest: str


@dataclass(frozen=True)
class RunSpec:
    run_type: RunType
    task: SpecRef
    dataset: SpecRef
    protocol: SpecRef
    metric: SpecRef
    policy: SpecRef
    candidates: tuple[SpecRef, ...]
    seeds: tuple[int, ...] = (0,)
    idempotency_key: str = ""

    def to_dict(self) -> dict:
        value = asdict(self)
        value["candidates"] = [asdict(item) for item in self.candidates]
        value["seeds"] = list(self.seeds)
        return value

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
