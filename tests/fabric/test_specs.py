from importlib.resources import files

import pytest

import sohbench.fabric.specs as specs_module
from sohbench.fabric.specs import (
    RunSpec,
    RunType,
    SpecRef,
    validate_run_spec_document,
)


def ref(kind: str, name: str) -> SpecRef:
    return SpecRef(kind=kind, spec_id=name, version="1.0.0", digest=("a" * 64))


def run_spec(candidate: str = "ridge") -> RunSpec:
    return RunSpec(
        run_type="training",
        task=ref("task", "soh"),
        dataset=ref("dataset_snapshot", "seven"),
        protocol=ref("protocol", "loco"),
        metric=ref("metric", "soh-core"),
        policy=ref("policy", "fair-v1"),
        candidates=(ref("candidate", candidate),),
        seeds=(0, 1, 2),
    )


def test_candidate_change_does_not_change_comparison_key():
    assert run_spec("ridge").comparison_key == run_spec("gbdt").comparison_key


def test_candidate_and_run_digests_bind_canonical_artifact_references():
    CandidateSpec = getattr(specs_module, "CandidateSpec", None)
    assert CandidateSpec is not None, "CandidateSpec is missing"
    first_artifact = SpecRef(
        kind="model_pickle",
        spec_id="model.pkl",
        version="sha256",
        digest="a" * 64,
    )
    second_artifact = SpecRef(
        kind="model_pickle",
        spec_id="model.pkl",
        version="sha256",
        digest="b" * 64,
    )
    first_candidate = CandidateSpec(
        spec_id="same-display-name", artifacts=(first_artifact,)
    )
    second_candidate = CandidateSpec(
        spec_id="same-display-name", artifacts=(second_artifact,)
    )
    base = run_spec()
    first_run = RunSpec(
        **{**base.__dict__, "candidates": (first_candidate,)}
    )
    second_run = RunSpec(
        **{**base.__dict__, "candidates": (second_candidate,)}
    )

    assert first_candidate.digest != second_candidate.digest
    assert first_run.digest != second_run.digest
    assert first_run.comparison_key == second_run.comparison_key
    validate_run_spec_document(first_run.to_dict())


def test_candidate_artifact_order_is_canonical():
    CandidateSpec = getattr(specs_module, "CandidateSpec", None)
    assert CandidateSpec is not None, "CandidateSpec is missing"
    model = SpecRef(
        kind="model_pickle",
        spec_id="model.pkl",
        version="sha256",
        digest="a" * 64,
    )
    manifest = SpecRef(
        kind="model_manifest",
        spec_id="model_manifest.json",
        version="sha256",
        digest="b" * 64,
    )

    first = CandidateSpec(
        spec_id="candidate", artifacts=(model, manifest)
    )
    second = CandidateSpec(
        spec_id="candidate", artifacts=(manifest, model)
    )

    assert first.artifacts == second.artifacts
    assert first.digest == second.digest


def test_policy_change_changes_comparison_key():
    first = run_spec()
    second = RunSpec(**{**first.__dict__, "policy": ref("policy", "other")})
    assert first.comparison_key != second.comparison_key


def test_comparison_key_ignores_candidate_run_type_seeds_and_idempotency_key():
    first = run_spec()
    second = RunSpec(
        run_type="inference",
        task=first.task,
        dataset=first.dataset,
        protocol=first.protocol,
        metric=first.metric,
        policy=first.policy,
        candidates=(ref("candidate", "other"),),
        seeds=(99,),
        idempotency_key="request-2",
    )
    assert first.comparison_key == second.comparison_key


def test_to_dict_serializes_tuples_and_is_immutable():
    spec = run_spec()
    document = spec.to_dict()
    assert document["seeds"] == [0, 1, 2]
    assert document["candidates"] == [spec.candidates[0].__dict__]
    with pytest.raises(AttributeError):
        spec.seeds = (4,)


def test_list_inputs_are_normalized_and_cannot_mutate_run_spec():
    candidates = [ref("candidate", "ridge")]
    seeds = [0, 1]
    spec = RunSpec(
        run_type="training",
        task=ref("task", "soh"),
        dataset=ref("dataset_snapshot", "seven"),
        protocol=ref("protocol", "loco"),
        metric=ref("metric", "soh-core"),
        policy=ref("policy", "fair-v1"),
        candidates=candidates,
        seeds=seeds,
    )
    candidates.append(ref("candidate", "gbdt"))
    seeds.append(2)
    assert isinstance(spec.candidates, tuple)
    assert isinstance(spec.seeds, tuple)
    assert len(spec.candidates) == 1
    assert spec.seeds == (0, 1)


@pytest.mark.parametrize("value", ["magic", 1, None])
def test_run_type_is_validated_at_construction(value):
    with pytest.raises((TypeError, ValueError), match="run_type"):
        RunSpec(
            run_type=value,
            task=ref("task", "soh"),
            dataset=ref("dataset_snapshot", "seven"),
            protocol=ref("protocol", "loco"),
            metric=ref("metric", "soh-core"),
            policy=ref("policy", "fair-v1"),
            candidates=(),
        )


def test_constructor_validates_spec_refs_digest_and_nested_types():
    with pytest.raises(TypeError, match="kind"):
        SpecRef(kind=1, spec_id="soh", version="1.0.0", digest="a" * 64)
    with pytest.raises(ValueError, match="digest"):
        SpecRef(kind="task", spec_id="soh", version="1.0.0", digest="A" * 64)
    with pytest.raises(TypeError, match="task"):
        RunSpec(
            run_type="training",
            task={"kind": "task"},
            dataset=ref("dataset_snapshot", "seven"),
            protocol=ref("protocol", "loco"),
            metric=ref("metric", "soh-core"),
            policy=ref("policy", "fair-v1"),
            candidates=(),
        )
    with pytest.raises(TypeError, match="candidates"):
        RunSpec(
            run_type="training",
            task=ref("task", "soh"),
            dataset=ref("dataset_snapshot", "seven"),
            protocol=ref("protocol", "loco"),
            metric=ref("metric", "soh-core"),
            policy=ref("policy", "fair-v1"),
            candidates=["ridge"],
        )


def test_constructor_requires_real_int_seeds_and_string_idempotency_key():
    with pytest.raises(TypeError, match="seeds"):
        RunSpec(**{**run_spec().__dict__, "seeds": [True]})
    with pytest.raises(TypeError, match="idempotency_key"):
        RunSpec(**{**run_spec().__dict__, "idempotency_key": None})


def test_run_type_is_publicly_exported():
    from sohbench.fabric import RunType as ExportedRunType

    assert ExportedRunType is RunType


def test_schema_resource_is_readable():
    schema = files("sohbench.fabric.schemas").joinpath("run-spec.schema.json")
    assert schema.is_file()
    assert '"additionalProperties": false' in schema.read_text(encoding="utf-8")


def test_schema_rejects_unknown_run_type():
    document = run_spec().to_dict()
    validate_run_spec_document(document)
    document["run_type"] = "magic"
    with pytest.raises(ValueError, match="run_type"):
        validate_run_spec_document(document)


def test_schema_rejects_unknown_properties_and_bad_digest():
    document = run_spec().to_dict()
    document["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        validate_run_spec_document(document)

    document = run_spec().to_dict()
    document["task"]["digest"] = "A" * 64
    with pytest.raises(ValueError, match="digest"):
        validate_run_spec_document(document)


def test_schema_rejects_unknown_nested_spec_ref_properties():
    document = run_spec().to_dict()
    document["task"]["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        validate_run_spec_document(document)


def test_public_exports_are_available_from_fabric_package():
    from sohbench.fabric import CandidateSpec as ExportedCandidateSpec
    from sohbench.fabric import RunSpec as ExportedRunSpec
    from sohbench.fabric import SpecRef as ExportedSpecRef
    from sohbench.fabric import validate_run_spec_document as exported_validator

    assert ExportedCandidateSpec is specs_module.CandidateSpec
    assert ExportedRunSpec is RunSpec
    assert ExportedSpecRef is SpecRef
    assert exported_validator is validate_run_spec_document
